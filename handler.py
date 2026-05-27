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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Allow xet downloads by default (override via env if needed)
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")

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
    return snap


def ensure_model() -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, MODEL_FILE)

    if os.path.isfile(local_path):
        size_gb = os.path.getsize(local_path) / (1024 ** 3)
        print(f"Model already cached: {local_path} ({size_gb:.1f} GB)")
        return local_path

    print(f"Downloading {MODEL_REPO}/{MODEL_FILE} to {MODEL_DIR}...")
    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
    )
    size_gb = os.path.getsize(downloaded) / (1024 ** 3)
    print(f"Download complete: {downloaded} ({size_gb:.1f} GB)")
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
    print("=== Init phase: capturing diagnostic snapshot ===")
    INIT_DIAG = diagnostic_snapshot()
    print(json.dumps(INIT_DIAG, indent=2))
    print("Ensuring model is available...")
    model_path = ensure_model()
    print("Initializing llama-server...")
    server_proc = start_llama_server(model_path)
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


def handler(job):
    if INIT_ERROR:
        return {
            "error": "WORKER_INIT_FAILED",
            "init_traceback": INIT_ERROR,
            "diag": INIT_DIAG,
        }
    job_input = job["input"]
    try:
        resp, _ = _forward(job_input)
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


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


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
