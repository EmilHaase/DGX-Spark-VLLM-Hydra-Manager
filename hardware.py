import json
from pathlib import Path
from config import SYSTEM_TOTAL_GB

def get_model_size_and_ctx(model_path: Path) -> tuple[float, int]:
    """
    Safely calculates weight without double-counting formats (prioritizes safetensors).
    Extracts native context length as a sane default.
    """
    if not model_path.exists() or not model_path.is_dir():
        return 0.0, 32768

    weight_bytes = 0
    st_files = list(model_path.rglob("*.safetensors"))
    if st_files:
        weight_bytes = sum(f.stat().st_size for f in st_files)
    else:
        gguf_files = list(model_path.rglob("*.gguf"))
        if gguf_files:
            weight_bytes = sum(f.stat().st_size for f in gguf_files)
        else:
            bin_files = list(model_path.rglob("*.bin"))
            weight_bytes = sum(f.stat().st_size for f in bin_files)
            
    weight_gb = weight_bytes / (1024 ** 3)
    if weight_gb == 0.0:
        return 0.0, 32768
            
    # Force 32768 context length as a standard default instead of reading config
    return weight_gb, 32768

def calculate_simple_vram(weight_gb: float, max_model_len: int, gpu_mem_util: float) -> tuple[float, float]:
    """
    Uses the pragmatic heuristic: 
    Allocated Limit = System VRAM * Util
    Est. Needed = Model Weight + Context Overhead (~4GB per 32k tokens)
    """
    allocated_limit = SYSTEM_TOTAL_GB * gpu_mem_util
    
    # Simple, intuitive context overhead: 4GB scales linearly per 32,768 context length.
    # We add 2.0GB baseline to represent PyTorch CUDA Context & KV-Cache base allocations.
    context_overhead = (max_model_len / 32768.0) * 4.0
    
    est_needed = weight_gb + context_overhead + 2.0
    
    return allocated_limit, est_needed