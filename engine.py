import os
import sys
import subprocess
import importlib.util
import hashlib
import urllib.request
import threading
import pty
from pathlib import Path
from config import PROJECT_ROOT, WEBUI_DIR, LOGS_DIR, SYSTEM_TOTAL_GB
from hardware import get_model_size_and_ctx

def tee_output(fd, log_file):
    with open(log_file, "ab") as f:
        while True:
            try:
                data = os.read(fd, 1024)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                f.write(data)
                f.flush()
            except OSError:
                break

def ensure_tiktoken_cache() -> str:
    cache_dir = PROJECT_ROOT / ".tiktoken_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    urls = [
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    ]
    
    for url in urls:
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()
        target_path = cache_dir / url_hash
        
        if not target_path.exists() or target_path.stat().st_size < 1000:
            print(f">> Applying ARM64 workaround: Downloading offline tiktoken cache ({url.split('/')[-1]})...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response, open(target_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Warning: Could not download tiktoken file: {e}")
                
    return str(cache_dir)

def get_patched_env(port: int = None) -> dict:
    env = os.environ.copy()
    
    torch_lib = ""
    try:
        spec = importlib.util.find_spec("torch")
        if spec and spec.submodule_search_locations:
            potential_lib = Path(spec.submodule_search_locations[0]) / "lib"
            if potential_lib.exists():
                torch_lib = str(potential_lib)
    except Exception:
        pass

    system_cuda_lib = "/usr/local/cuda/lib64"
    
    current_ld = env.get("LD_LIBRARY_PATH", "")
    ld_paths = [p for p in [torch_lib, system_cuda_lib, current_ld] if p]
    
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"
    
    env["TORCH_CUDA_ARCH_LIST"] = "12.1a"
    env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    env["VLLM_TARGET_DEVICE"] = "cuda"
    
    # Add NVFP4/JIT Compilation OOM protections
    env["MAX_JOBS"] = "4"
    env["NINJA_FLAGS"] = "-j 4"
    
    # Completely isolate the compiler caches so parallel vLLM instances do not race
    base_cache = PROJECT_ROOT / ".compiler_cache"
    if port:
        comp_cache = base_cache / f"port_{port}"
    else:
        comp_cache = base_cache / "shared"
        
    comp_cache.mkdir(parents=True, exist_ok=True)
    env["TORCH_CUDA_COMPILER_CACHE_DIR"] = str(comp_cache)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(comp_cache)
    env["TRITON_CACHE_DIR"] = str(comp_cache)
    
    # CRITICAL: Isolate NCCL and Ray so they don't see other instances booting
    env["VLLM_HOST_IP"] = "127.0.0.1"
    env["NCCL_IGNORE_DISABLED_P2P"] = "1"
    env["NCCL_P2P_DISABLE"] = "1"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # Prevent PyTorch multiprocessing from sharing CUDA contexts across VLLM boundaries
    env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    
    # Give Ray a unique namespace per instance so the profilers don't trip over each other
    if port:
        env["RAY_ADDRESS"] = "" 
        env["RAY_CHDIR_TO_WORK_DIR"] = "0"
        env["RAY_worker_register_timeout_seconds"] = "60"
        
    env.pop("VLLM_PYTHON_EXECUTABLE", None)
    
    cache_dir = ensure_tiktoken_cache()
    env["TIKTOKEN_CACHE_DIR"] = cache_dir
    env["TIKTOKEN_RS_CACHE_DIR"] = cache_dir
    
    return env

def launch_vllm(model_path: Path, port: int, gpu_mem_util: float, max_model_len: int, enforce_eager: bool = False, reasoning_parser: str | None = None, session_id: str = "") -> subprocess.Popen:
    name = model_path.name
    vllm_exec = PROJECT_ROOT / ".venv" / "bin" / "vllm"
    if not vllm_exec.exists():
        vllm_exec = "vllm"
        
    # Hardware overrides to fix UMA profiling bugs
    # Calculate explicit KV cache size to bypass UMA memory profile bug
    weight_gb, _ = get_model_size_and_ctx(model_path)
    allocated_limit = SYSTEM_TOTAL_GB * gpu_mem_util
    kv_cache_budget_gb = max(0.25, allocated_limit - weight_gb - 2.0)
    kv_cache_bytes = int(kv_cache_budget_gb * 1024**3)
    
    cmd = [
        str(vllm_exec), "serve", str(model_path),
        "--served-model-name", name,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--kv-cache-memory-bytes", str(kv_cache_bytes),
        "--trust-remote-code"
    ]
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    
    env = get_patched_env(port)
    print(f"[{port}] Launching vLLM Engine for {name}...")
    
    if session_id:
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT, stdout=slave_fd, stderr=subprocess.STDOUT)
        os.close(slave_fd)
        
        log_file = LOGS_DIR / f"engine_{port}_{name}_{session_id}.log"
        t = threading.Thread(target=tee_output, args=(master_fd, log_file))
        t.daemon = True
        t.start()
    else:
        process = subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT)
        
    return process

def launch_webui(active_ports: list[int], session_id: str = "") -> subprocess.Popen:
    webui_exec = PROJECT_ROOT / ".venv" / "bin" / "open-webui"
    if not webui_exec.exists():
        webui_exec = "open-webui"
            
    cmd = [str(webui_exec), "serve", "--host", "0.0.0.0", "--port", "3000"]
    
    urls = ";".join([f"http://127.0.0.1:{p}/v1" for p in active_ports])
    keys = ";".join(["sk-dummy" for _ in active_ports])
    
    env = get_patched_env()
    
    # CRITICAL: Force WebUI to broadcast securely on the network interface
    env["HOST"] = "0.0.0.0" 
    env["PORT"] = "3000"
    env["WEBUI_PORT"] = "3000"
    
    # Because this explicitly targets your old folder, ALL your previous chats/users will load!
    env["DATA_DIR"] = str(WEBUI_DIR) 
    env["OPENAI_API_BASE_URLS"] = urls
    env["OPENAI_API_KEYS"] = keys
    
    print(f"[3000] Launching Open WebUI connected to engines on ports: {active_ports}")
    
    if session_id:
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT, stdout=slave_fd, stderr=subprocess.STDOUT)
        os.close(slave_fd)
        
        log_file = LOGS_DIR / f"webui_3000_{session_id}.log"
        t = threading.Thread(target=tee_output, args=(master_fd, log_file))
        t.daemon = True
        t.start()
    else:
        process = subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT)
        
    return process