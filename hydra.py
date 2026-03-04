import os
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass

from hardware import get_model_size_and_ctx, calculate_simple_vram
from config import MODELS_DIR, GPU_MEM_UTIL_DEFAULT, SYSTEM_TOTAL_GB
from engine import launch_vllm, launch_webui

@dataclass
class ModelContext:
    path: Path
    selected: bool = False
    max_model_len: int = 32768
    gpu_mem_util: float = GPU_MEM_UTIL_DEFAULT
    weight_gb: float = 0.0
    enforce_eager: bool = False
    reasoning_parser: str | None = None

def format_len(val: int) -> str:
    """Formats 32768 as '32k', 262144 as '262k' for an uncluttered UI."""
    if val % 1024 == 0:
        return f"{val // 1024}k"
    return str(val)

class HydraMenu:
    def __init__(self):
        self.models: dict[int, ModelContext] = {}
        self.global_gpu_mem_util = GPU_MEM_UTIL_DEFAULT
        self.scan_models()
        self.active = True

    def scan_models(self):
        self.models.clear()
        if not hasattr(self, 'global_gpu_mem_util'):
            self.global_gpu_mem_util = GPU_MEM_UTIL_DEFAULT
            
        if not MODELS_DIR.exists():
            return
            
        dirs = []
        for d in MODELS_DIR.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                if d.name not in ["bash", "vllm_profiles"]:
                    dirs.append(d)
        
        dirs.sort()
        
        idx = 1
        for model_dir in dirs:
            weight_gb, native_ctx_len = get_model_size_and_ctx(model_dir)
            if weight_gb > 0.0:
                self.models[idx] = ModelContext(
                    path=model_dir,
                    selected=False,
                    max_model_len=native_ctx_len, 
                    gpu_mem_util=self.global_gpu_mem_util,
                    weight_gb=weight_gb
                )
                idx += 1

    def draw(self):
        os.system("clear")
        print("=" * 120)
        print(" " * 42 + "DGX Spark VLLM Hydra Manager")
        print("=" * 120)

        if not self.models:
            print(f"No valid models found in {MODELS_DIR}.")
            print("=" * 120)
            return

        total_allocated_gb = 0.0
        total_est_needed_gb = 0.0
        total_gpu_util = 0.0
        has_oom_error = False
        
        for idx in sorted(self.models.keys()):
            ctx = self.models[idx]
            mark = "[X]" if ctx.selected else "[ ]"
            
            allocated_limit, est_needed = calculate_simple_vram(
                ctx.weight_gb, ctx.max_model_len, ctx.gpu_mem_util
            )
            
            if ctx.selected:
                total_allocated_gb += allocated_limit
                total_est_needed_gb += est_needed
                total_gpu_util += ctx.gpu_mem_util

            fits = est_needed <= allocated_limit
            status = "\033[92m[ OK ]\033[0m" if fits else "\033[91m[OOM?]\033[0m"
            if ctx.selected and not fits:
                has_oom_error = True

            row_color = "\033[96m" if ctx.selected else "\033[0m"
            ctx_str = format_len(ctx.max_model_len)
            eager_str = "E:1" if ctx.enforce_eager else "E:0"
            r_str = "R:" + (ctx.reasoning_parser[:8] if ctx.reasoning_parser else "None")
            
            # Formats exactly as you requested: Est: 29.0 / 115.2GB [OK]
            print(f"{row_color}{idx:2d}\033[0m) {mark} {ctx.path.name[:27]:<27} | L: {ctx_str:<6} | {eager_str} | {r_str:<10} | U: {ctx.gpu_mem_util:.2f} | W: {ctx.weight_gb:>5.1f}GB | Est: {est_needed:>5.1f} / {allocated_limit:>5.1f}GB {status}")

        print("-" * 120)
        print(f"Global Limit Target: {self.global_gpu_mem_util:.2f}  |  System VRAM: {float(SYSTEM_TOTAL_GB):.1f} GB")
        
        if total_gpu_util > 0:
            print(f"\nSelected Allocations: {total_allocated_gb:.1f} GB (Sum of U: {total_gpu_util:.2f})")
            print(f"Selected Est. Needed: {total_est_needed_gb:.1f} GB")
            
            # Retain critical safety checks for multi-model loading
            if total_gpu_util > 1.0:
                print(f"\033[91m[DANGER] Sum of 'U' ({total_gpu_util:.2f}) > 1.0! Concurrent engines will Kernel Panic fighting for VRAM.\033[0m")
            elif has_oom_error:
                print(f"\033[93m[WARNING] A selected model's Est. Needed VRAM exceeds its Allocation Limit.\033[0m")
                
        print("-" * 120)
        print("Commands:")
        print(" <ID>            : Toggle selection (e.g., 1)")
        print(" a <ID>          : Toggle reasoning parser 'deepseek_r1' (e.g. A)")
        print(" b <ID>          : Toggle 'enforce-eager' mode (e.g. B)")
        print(" c <ID>          : Toggle reasoning parser 'qwen3' (e.g. C)")
        print(" c <ID> <LEN>    : Change context length for model ID (e.g., c 1 16384 or c 1 32k)")
        print(" u <ID> <0.X>    : Set INDIVIDUAL GPU limit (e.g., u 1 0.45)")
        print(" limit <0.X>     : Set GLOBAL GPU limit (e.g., limit 0.90)")
        print(" scan            : Rescan models directory")
        print(" r               : Run Orchestrator")
        print(" q               : Quit to Master Menu\n")

    def handle_input(self, cmd: str) -> bool:
        cmd = cmd.strip().lower()
        if not cmd:
            return False
            
        parts = cmd.split()
        
        if parts[0] == "q":
            self.active = False
            return False
        elif parts[0] == "r":
            self.active = False
            return True
        elif parts[0] == "scan":
            self.scan_models()
        elif parts[0] == "limit" and len(parts) == 2:
            try:
                val = float(parts[1])
                self.global_gpu_mem_util = val
                self._rebalance_allocations()
            except ValueError:
                pass
        elif parts[0] == "u" and len(parts) == 3:
            try:
                idx = int(parts[1])
                val = float(parts[2])
                if idx in self.models:
                    self.models[idx].gpu_mem_util = val
                    self.models[idx].has_custom_util = True # Needs new flag on ModelContext
                    self._rebalance_allocations()
            except ValueError:
                pass
        elif parts[0] == "c" and len(parts) == 3:
            try:
                idx = int(parts[1])
                length_str = parts[2].lower()
                if length_str.endswith("k"):
                    length = int(float(length_str[:-1]) * 1024)
                else:
                    length = int(length_str)
                if idx in self.models:
                    self.models[idx].max_model_len = length
            except ValueError:
                pass
        elif parts[0] == "a" and len(parts) == 2:
            try:
                idx = int(parts[1])
                if idx in self.models:
                    if self.models[idx].reasoning_parser == "deepseek_r1":
                        self.models[idx].reasoning_parser = None
                    else:
                        self.models[idx].reasoning_parser = "deepseek_r1"
            except ValueError:
                pass
        elif parts[0] == "b" and len(parts) == 2:
            try:
                idx = int(parts[1])
                if idx in self.models:
                    self.models[idx].enforce_eager = not self.models[idx].enforce_eager
            except ValueError:
                pass
        elif parts[0] == "c" and len(parts) == 2:
            try:
                idx = int(parts[1])
                if idx in self.models:
                    if self.models[idx].reasoning_parser == "qwen3":
                        self.models[idx].reasoning_parser = None
                    else:
                        self.models[idx].reasoning_parser = "qwen3"
            except ValueError:
                pass
        elif parts[0].isdigit():
            idx = int(parts[0])
            if idx in self.models:
                self.models[idx].selected = not self.models[idx].selected
                if not self.models[idx].selected:
                    self.models[idx].has_custom_util = False # Reset custom flag on deselect
                self._rebalance_allocations()
                
        return False
        
    def _rebalance_allocations(self):
        selected_models = [m for m in self.models.values() if m.selected]
        unselected_models = [m for m in self.models.values() if not m.selected]
        
        # Ensure unselected models revert to default display
        for m in unselected_models:
            m.gpu_mem_util = self.global_gpu_mem_util
            
        if not selected_models:
            return

        # 1. Identify models with custom limits and calculate remaining budget
        custom_models = [m for m in selected_models if getattr(m, 'has_custom_util', False)]
        auto_models = [m for m in selected_models if not getattr(m, 'has_custom_util', False)]
        
        custom_budget = sum(m.gpu_mem_util for m in custom_models)
        remaining_budget = max(0.01, self.global_gpu_mem_util - custom_budget)
        
        # 2. Distribute remaining budget proportionally to Estimated Needs
        if auto_models:
            from hardware import calculate_simple_vram
            
            est_needs = []
            for m in auto_models:
                _, est = calculate_simple_vram(m.weight_gb, m.max_model_len, 0)
                est_needs.append(est)
                
            total_est = sum(est_needs)
            
            if total_est > 0:
                for m, est in zip(auto_models, est_needs):
                    fraction = est / total_est
                    m.gpu_mem_util = remaining_budget * fraction
            else:
                even_split = remaining_budget / len(auto_models)
                for m in auto_models:
                    m.gpu_mem_util = even_split
                
        return False

    def run(self):
        launch = False
        
        while self.active:
            self.draw()
            try:
                cmd = input("hydra> ")
                launch = self.handle_input(cmd)
            except (KeyboardInterrupt, EOFError):
                self.active = False
                return

        if not launch:
            return

        selected_models = [ctx for ctx in self.models.values() if ctx.selected]
        
        if not selected_models:
            print("No models selected... Returning to Master Menu.")
            time.sleep(1)
            return
            
        print("\nPreparing to launch engines...")
        
        print("Clearing network ports and stopping interfering Docker containers...")
        os.system("sudo docker stop $(sudo docker ps -q) > /dev/null 2>&1 || true")
        os.system("sudo fuser -k 3000/tcp > /dev/null 2>&1 || true")
        for p in range(8000, 8008):
            os.system(f"sudo fuser -k {p}/tcp > /dev/null 2>&1 || true")
        os.system("pkill -f vllm > /dev/null 2>&1 || true")
        time.sleep(1) # Ensure ports and processes are fully released
        
        print("Dropping OS caches to free UMA memory...")
        os.system("sudo sysctl -w vm.drop_caches=3 > /dev/null 2>&1")
        
        processes: list[subprocess.Popen] = []
        active_ports = []
        base_port = 8000
        
        session_id = time.strftime("%Y%m%d_%H%M%S")
        print(f"Session ID created: {session_id}")
        
        try:
            for i, ctx in enumerate(selected_models):
                port = base_port + i
                proc = launch_vllm(
                    model_path=ctx.path,
                    port=port,
                    gpu_mem_util=ctx.gpu_mem_util,
                    max_model_len=ctx.max_model_len,
                    enforce_eager=ctx.enforce_eager,
                    reasoning_parser=ctx.reasoning_parser,
                    session_id=session_id
                )
                processes.append(proc)
                active_ports.append(port)
                
                if i < len(selected_models) - 1:
                    print(f"\nWaiting 15 seconds before launching next engine to prevent UMA compile cache races...")
                    time.sleep(15)
                    print()
                
            print(f"Waiting 10 seconds for {len(selected_models)} vLLM instances to initialize...")
            time.sleep(10)
            
            webui_proc = launch_webui(active_ports, session_id=session_id)
            processes.append(webui_proc)
            
            print("\nAll systems online. Tail logging... (Press Ctrl+C to terminate)")
            
            for proc in processes:
                proc.wait()
                
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, cleanly terminating all spawned processes...")
        finally:
            print("\nCleaning up processes gracefully...")
            for proc in processes:
                try:
                    proc.terminate()
                except Exception:
                    pass
            
            time.sleep(2)
            for proc in processes:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    
            print("Running fallback pkill just in case...")
            os.system("sudo fuser -k 3000/tcp > /dev/null 2>&1 || true")
            for p in range(8000, 8008):
                os.system(f"sudo fuser -k {p}/tcp > /dev/null 2>&1 || true")
            os.system("pkill -f vllm > /dev/null 2>&1 || true")
            os.system("pkill -f open-webui > /dev/null 2>&1 || true")
            print("Cleanup complete. Returning to Master Menu...")
            time.sleep(1)