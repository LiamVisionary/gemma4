"""
RunPod Serverless Handler for Gemma 4 26B-A4B-it via llama.cpp

Downloads the GGUF model on first boot (cached on network volume),
starts llama-server, and proxies OpenAI-compatible requests.

Patched: wraps init in try/except so any failure (HF download, llama-server
crash, volume permissions) is captured and returned in the job response
instead of silently killing the worker.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback

# Per-phase timing for cold-start diagnosis. All times monotonic seconds.
_MODULE_START = time.monotonic()
PHASE_TIMES = {"module_import": 0.0}

def _phase(name):
    """Record a timestamp relative to module import."""
    PHASE_TIMES[name] = round(time.monotonic() - _MODULE_START, 3)
    print(f"[PHASE {PHASE_TIMES[name]:.3f}s] {name}")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Allow xet downloads by default (override via env if needed)
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")

# RunPod "Cached Models" stages the HF repo at this path. With NO network
# volume attached, this resolves to host-local NVMe (fast). With a network
# volume attached, it lands on the (slow) network volume — so this endpoint
# must run WITHOUT a network volume to get the speedup.
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

import requests
import runpod
from huggingface_hub import hf_hub_download

MODEL_REPO = os.environ.get("MODEL_REPO", "ggml-org/gemma-4-26B-A4B-it-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "gemma-4-26B-A4B-it-Q4_K_M.gguf")
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
PARALLEL = int(os.environ.get("PARALLEL", "1"))

BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}"


def diagnostic_snapshot() -> dict:
    """Capture environment state for crash reports."""
    snap = {
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "model_dir": MODEL_DIR,
        "ctx_size": CTX_SIZE,
        "n_gpu_layers": N_GPU_LAYERS,
        "parallel": PARALLEL,
        "env_hf_xet_disable": os.environ.get("HF_HUB_DISABLE_XET"),
        "python": sys.version.split()[0],
    }
    # Disk
    try:
        usage = shutil.disk_usage("/")
        snap["disk_root_free_gb"] = round(usage.free / (1024**3), 2)
        snap["disk_root_total_gb"] = round(usage.total / (1024**3), 2)
    except Exception as e:
        snap["disk_root_err"] = str(e)
    # Volume
    snap["volume_exists"] = os.path.isdir("/runpod-volume")
    if snap["volume_exists"]:
        try:
            usage = shutil.disk_usage("/runpod-volume")
            snap["volume_free_gb"] = round(usage.free / (1024**3), 2)
            snap["volume_total_gb"] = round(usage.total / (1024**3), 2)
            snap["volume_writable"] = os.access("/runpod-volume", os.W_OK)
        except Exception as e:
            snap["volume_err"] = str(e)
    # CUDA
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        snap["nvidia_smi"] = r.stdout.strip()[:300]
    except Exception as e:
        snap["nvidia_smi_err"] = str(e)
    # llama-server binary
    snap["llama_server_exists"] = os.path.isfile("/app/llama-server")
    if snap["llama_server_exists"]:
        snap["llama_server_executable"] = os.access("/app/llama-server", os.X_OK)
    # RunPod Cached Models presence
    snap["hf_cache_root_exists"] = os.path.isdir(HF_CACHE_ROOT)
    if snap["hf_cache_root_exists"]:
        try:
            snap["hf_cache_contents"] = os.listdir(HF_CACHE_ROOT)[:10]
        except Exception as e:
            snap["hf_cache_err"] = str(e)
    return snap


def resolve_snapshot_path(model_id: str) -> str:
    """Resolve the local snapshot dir for a RunPod-cached model.
    Mirrors runpod-workers/model-store-cache-example.
    """
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be 'org/name'")
    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if not os.path.isdir(snapshots_dir):
        raise RuntimeError(f"snapshots dir not found: {snapshots_dir}")
    versions = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not versions:
        raise RuntimeError(f"no snapshots under {snapshots_dir}")
    versions.sort()
    return os.path.join(snapshots_dir, versions[0])


def ensure_model() -> str:
    """Resolve the GGUF path. Prefer RunPod's cached-models snapshot
    (host-local NVMe when no network volume is attached), then fall back to
    a one-time hf_hub_download onto container disk.
    """
    # Pass 1: RunPod Cached Models snapshot dir (the fast path).
    try:
        snap = resolve_snapshot_path(MODEL_REPO)
        candidate = os.path.join(snap, MODEL_FILE)
        if os.path.isfile(candidate):
            size_gb = os.path.getsize(candidate) / (1024 ** 3)
            print(f"[cache HIT] {candidate} ({size_gb:.1f} GB)")
            return candidate
        # GGUF may be symlinked under blobs/; resolve_snapshot_path returns the
        # snapshot dir which contains symlinks to blobs — os.path.isfile follows them.
        for f in os.listdir(snap):
            if f == MODEL_FILE:
                p = os.path.join(snap, f)
                print(f"[cache HIT via listdir] {p}")
                return p
        print(f"[cache MISS] snapshot dir exists but {MODEL_FILE} not in it: {os.listdir(snap)[:10]}")
    except Exception as e:
        print(f"[cache MISS] resolve_snapshot_path failed: {e.__class__.__name__}: {str(e)[:120]}")

    # Pass 2: legacy MODEL_DIR copy (only relevant if a network volume is attached).
    legacy_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.isfile(legacy_path):
        size_gb = os.path.getsize(legacy_path) / (1024 ** 3)
        print(f"[legacy] {legacy_path} ({size_gb:.1f} GB)")
        return legacy_path

    # Pass 3: download to container disk (slow path — only on cold cache).
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[download] {MODEL_REPO}/{MODEL_FILE} → {MODEL_DIR}")
    downloaded = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=MODEL_DIR)
    size_gb = os.path.getsize(downloaded) / (1024 ** 3)
    print(f"[download complete] {downloaded} ({size_gb:.1f} GB)")
    return downloaded


def start_llama_server(model_path: str):
    cmd = [
        "/app/llama-server",
        "-m", model_path,
        "--port", str(LLAMA_PORT),
        "-ngl", str(N_GPU_LAYERS),
        "-c", str(CTX_SIZE),
        "--parallel", str(PARALLEL),
        "--host", "127.0.0.1",
        # Enable jinja chat template evaluation so the model's embedded
        # chat_template.jinja is used. For SuperGemma4, this enables
        # the enable_thinking=false default (skip CoT for normal turns)
        # and unlocks chat_template_kwargs in request bodies.
        "--jinja",
    ]

    print(f"Starting llama-server: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + 300
    stdout_tail = []
    while time.time() < deadline:
        if proc.poll() is not None:
            try:
                stdout = proc.stdout.read().decode() if proc.stdout else ""
            except Exception:
                stdout = ""
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode}:\n{stdout[-3000:]}"
            )
        # Drain stdout non-blocking-ish so we can capture tail on hang
        try:
            line = proc.stdout.readline() if proc.stdout else b""
            if line:
                stdout_tail.append(line.decode(errors="replace"))
                if len(stdout_tail) > 200:
                    stdout_tail = stdout_tail[-200:]
        except Exception:
            pass

        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                body = r.json()
                if body.get("status") == "ok":
                    print("llama-server is ready")
                    return proc
        except (requests.ConnectionError, requests.Timeout):
            pass

        time.sleep(1)

    try:
        proc.kill()
    except Exception:
        pass
    tail = "".join(stdout_tail[-50:])
    raise RuntimeError(f"llama-server failed to become healthy within 300s. Tail:\n{tail}")


# ---- Init phase, wrapped to surface failures via the handler ----
INIT_ERROR = None
INIT_DIAG = None
model_path = None
server_proc = None
try:
    _phase("init_start")
    print("=== Init phase: capturing diagnostic snapshot ===")
    INIT_DIAG = diagnostic_snapshot()
    _phase("diag_done")
    print(json.dumps(INIT_DIAG, indent=2))
    print("Ensuring model is available...")
    model_path = ensure_model()
    _phase("model_ready")
    print("Initializing llama-server...")
    server_proc = start_llama_server(model_path)
    _phase("llama_server_healthy")
    print("=== Init complete ===")
except Exception:
    INIT_ERROR = traceback.format_exc()
    print("=" * 60)
    print("HANDLER INIT FAILED:")
    print(INIT_ERROR)
    print("=" * 60)


def _forward(job_input):
    endpoint = job_input.pop("endpoint", "/v1/chat/completions")
    stream = job_input.get("stream", False)
    resp = requests.post(
        f"{BASE_URL}{endpoint}",
        json=job_input,
        stream=stream,
        timeout=300,
    )
    resp.raise_for_status()
    return resp, stream


_FIRST_JOB_RECEIVED = None

def _build_meta():
    """Diagnostic metadata attached to every response."""
    meta = {
        "phase_times": dict(PHASE_TIMES),
        "first_job_received_s": _FIRST_JOB_RECEIVED,
        "uptime_at_response_s": round(time.monotonic() - _MODULE_START, 3),
        "gpu": (INIT_DIAG or {}).get("nvidia_smi", "?"),
    }
    return meta

def handler(job):
    global _FIRST_JOB_RECEIVED
    if _FIRST_JOB_RECEIVED is None:
        _FIRST_JOB_RECEIVED = round(time.monotonic() - _MODULE_START, 3)
        _phase("first_job_received")
    if INIT_ERROR:
        return {
            "error": "WORKER_INIT_FAILED",
            "init_traceback": INIT_ERROR,
            "diag": INIT_DIAG,
            "_meta": _build_meta(),
        }
    job_input = job["input"]
    # Strip _meta_request from input if present (debug flag opt-in; non-breaking)
    job_input.pop("_meta_request", None)
    try:
        resp, _ = _forward(job_input)
        body = resp.json()
        if isinstance(body, dict):
            body["_meta"] = _build_meta()
        return body
    except requests.RequestException as e:
        return {"error": str(e), "_meta": _build_meta()}


def stream_handler(job):
    if INIT_ERROR:
        yield {"error": "WORKER_INIT_FAILED", "init_traceback": INIT_ERROR, "diag": INIT_DIAG}
        return
    job_input = job["input"]
    job_input["stream"] = True
    try:
        resp, _ = _forward(job_input)
    except requests.RequestException as e:
        yield {"error": str(e)}
        return

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            yield chunk
        except json.JSONDecodeError:
            continue


_phase("runpod_start_called")
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
