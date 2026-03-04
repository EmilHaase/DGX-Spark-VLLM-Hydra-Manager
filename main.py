import subprocess
import sys
import os
import shutil
import importlib.util
from pathlib import Path
from hydra import HydraMenu
from downloader import download_model_interactive
from disk_manager import disk_manager_menu
from tester import TesterMenu
from config import PROJECT_ROOT, WEBUI_DIR

def handle_hf_token():
    env_file = PROJECT_ROOT / ".env"
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    
    if not hf_token:
        print("\n" + "=" * 45)
        print("=== Initial HF_TOKEN Setup ===")
        print("=" * 45)
        print("It seems you don't have a HuggingFace token configured yet.")
        print("This is required to download models like Llama 3 or Qwen.")
        token_input = input("Please paste your HF_TOKEN (or press Enter to skip): ").strip()
        
        if token_input:
            env_content = ""
            if env_file.exists():
                with open(env_file, "r") as f:
                    env_content = f.read()
                    
            if "HF_TOKEN" in env_content:
                # Replace existing empty or wrong token
                import re
                env_content = re.sub(r'HF_TOKEN=.*', f'HF_TOKEN={token_input}', env_content)
            else:
                env_content += f"\nHF_TOKEN={token_input}\n"
                
            with open(env_file, "w") as f:
                f.write(env_content.strip() + "\n")
                
            os.environ["HF_TOKEN"] = token_input
            print("✅ Token saved to .env file!")
            import time
            time.sleep(1)

def main():
    handle_hf_token()
    while True:
        try:
            print("\n" + "=" * 45)
            print("=== DGX SPARK VLLM HYDRA MANAGER ===")
            print("=" * 45)
            print("1) Launch vLLM Hydra")
            print("2) Update vLLM (Core Engines & Torch)")
            print("3) Download New Model (HuggingFace)")
            print("4) Disk Manager (Models & Storage)")
            print("5) Reset WebUI Database (Fix Corruptions)")
            print("6) Update Open-WebUI & Frontend")
            print("7) Show System Versions")
            print("8) Engine Tester & Benchmarks")
            print("9) Exit")
            
            choice = input("\nSelect an option (1-9): ").strip()
            
            if choice == "1":
                menu = HydraMenu()
                menu.run()
            elif choice == "2":
                confirm = input("\nAre you sure you want to rebuild? Type 'yes' to continue or 'no' to cancel: ").strip().lower()
                if confirm != 'yes':
                    print("Rebuild cancelled. Returning to menu...")
                    continue
                    
                print("\n================================================")
                print("--- Updating Core Engine Dependencies (GB10) ---")
                print("================================================")
                
                env = os.environ.copy()
                env["CUDA_HOME"] = "/usr/local/cuda"
                env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"
                env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
                env["TORCH_CUDA_ARCH_LIST"] = "12.1a"
                env["VLLM_PYTHON_EXECUTABLE"] = sys.executable
                env["VLLM_TARGET_DEVICE"] = "cuda"
                env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
                env["MAX_JOBS"] = "18"
                
                print("\n>> NUKING Corrupted Virtual Environment to Force Clean Rebuild...")
                venv_dir = PROJECT_ROOT / ".venv"
                if venv_dir.exists():
                    subprocess.run(["rm", "-rf", str(venv_dir)], check=True)
                    
                subprocess.run(["uv", "venv", "--python", "3.12", "--seed", str(venv_dir)], check=True)
                
                env["PATH"] = f"{venv_dir / 'bin'}:{env['PATH']}"
                env["VIRTUAL_ENV"] = str(venv_dir)

                try:
                    print("\n>> 1/4: Installing PyTorch Native Aarch64 CUDA 13.0 Wheels...")
                    subprocess.run(["python3", "-m", "pip", "install", "-U", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu130", "--default-timeout=1200"], env=env, check=True)
                    
                    print("\n>> 2/4: Installing Accelerated Kernels...")
                    subprocess.run([
                        "uv", "pip", "install", "-U",
                        "xgrammar", "triton", "flashinfer-python",
                        "--prerelease=allow"
                    ], env=env, check=True)

                    print("\n>> 3/4: Building vLLM from source (sm_121a native Blackwell kernels)...")
                    print("     Installing pre-requisite build tools...")
                    subprocess.run([
                        "uv", "pip", "install", "-U", "setuptools_scm", "ninja", "cmake", "packaging", "wheel"
                    ], env=env, check=True)
                    print("     This will take 45-60 minutes. Do not interrupt.")
                    subprocess.run([
                        "pip", "install", "--no-build-isolation",
                        "git+https://github.com/vllm-project/vllm.git"
                    ], env=env, check=True)

                    print("\n>> 4/4: Force-upgrading Transformers to latest master...")
                    subprocess.run([
                        "uv", "pip", "install", "--reinstall-package", "transformers",
                        "git+https://github.com/huggingface/transformers.git"
                    ], env=env, check=True)

                    print("\n>> 5/5: Installing Front-End Dashboard (Open-WebUI)...")
                    subprocess.run([
                        "uv", "pip", "install", "-U", "open-webui", "huggingface_hub"
                    ], env=env, check=True)

                    if not Path("/usr/local/cuda/lib64/libcudart.so.12").exists() and Path("/usr/local/cuda/lib64/libcudart.so.13").exists():
                        print("\n>> Applying CUDA runtime symlink patch for vLLM...")
                        os.system("sudo ln -sf /usr/local/cuda/lib64/libcudart.so.13 /usr/local/cuda/lib64/libcudart.so.12")

                    print("\n✅ System Update successful via native wheels. Engine ready.")
                    input("Press Enter to continue...")
                    continue
                except subprocess.CalledProcessError as e:
                    print(f"\n❌ Installation failed with exit code {e.returncode}.")
                
                input("Press Enter to continue...")
            elif choice == "3":
                download_model_interactive()
            elif choice == "4":
                disk_manager_menu()
            elif choice == "5":
                print("\n================================================")
                print("--- Repairing Open-WebUI Database Corruptions ---")
                print("================================================")
                print("Backing up potentially corrupted databases...")
                db_path = WEBUI_DIR / "webui.db"
                vec_path = WEBUI_DIR / "vector_db"
                chroma_path = WEBUI_DIR / "chroma"
                
                if db_path.exists():
                    bak_path = db_path.with_suffix('.db.bak')
                    if bak_path.exists():
                        bak_path.unlink()
                    db_path.rename(bak_path)
                    print(f"✅ {db_path.name} backed up.")
                if vec_path.exists():
                    bak_path = vec_path.with_name(vec_path.name + '.bak')
                    if bak_path.exists():
                        shutil.rmtree(bak_path)
                    vec_path.rename(bak_path)
                    print(f"✅ {vec_path.name} backed up.")
                if chroma_path.exists():
                    bak_path = chroma_path.with_name(chroma_path.name + '.bak')
                    if bak_path.exists():
                        shutil.rmtree(bak_path)
                    chroma_path.rename(bak_path)
                    print(f"✅ {chroma_path.name} backed up.")
                
                print("\nDatabases repaired. Open-WebUI will generate fresh ones on the next launch.")
                input("Press Enter to return to menu...")
                input("Press Enter to return to menu...")
            elif choice == "6":
                print("\n================================================")
                print("--- Updating Open-WebUI & Frontend ---")
                print("================================================")
                venv_dir = PROJECT_ROOT / ".venv"
                if not venv_dir.exists():
                    print("❌ .venv not found. Please run 'Update vLLM' first.")
                else:
                    env = os.environ.copy()
                    env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"
                    env["VIRTUAL_ENV"] = str(venv_dir)
                    try:
                        print(">> Updating Open-WebUI via uv pip...")
                        subprocess.run(["uv", "pip", "install", "-U", "open-webui", "huggingface_hub"], env=env, check=True)
                        print("\n✅ WebUI successfully updated!")
                    except subprocess.CalledProcessError as e:
                        print(f"\n❌ Update failed with exit code {e.returncode}.")
                input("Press Enter to return to menu...")
            elif choice == "7":
                print("\n================================================")
                print("--- System Engine Versions ---")
                print("================================================")
                env = os.environ.copy()
                venv_bin = str(PROJECT_ROOT / ".venv" / "bin")
                env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
                
                packages = ["vllm", "torch", "transformers", "open-webui", "xgrammar", "triton", "flashinfer"]
                for pkg in packages:
                    try:
                        result = subprocess.run(["uv", "pip", "show", pkg], env=env, capture_output=True, text=True)
                        version_line = next((line for line in result.stdout.split('\n') if line.startswith('Version:')), None)
                        if version_line:
                            version = version_line.split(':', 1)[1].strip()
                            print(f"{pkg.ljust(15)} : {version}")
                        else:
                            print(f"{pkg.ljust(15)} : Not installed or version not found")
                    except FileNotFoundError:
                        print(f"{pkg.ljust(15)} : uv missing or environment error")
                        break
                        
                input("\nPress Enter to return to menu...")
            elif choice == "8":
                tester = TesterMenu()
                tester.run()
            elif choice == "9":
                print("\nExiting DGX Spark Ops.")
                sys.exit(0)
            else:
                print("\nInvalid choice. Please select 1-9.")
        except KeyboardInterrupt:
            print("\nGracefully exiting to the terminal.")
            sys.exit(0)
        except EOFError:
            print("\nExiting.")
            sys.exit(0)

if __name__ == "__main__":
    main()