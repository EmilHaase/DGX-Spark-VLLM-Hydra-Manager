import subprocess
import os
import shutil
from pathlib import Path
from config import PROJECT_ROOT, MODELS_DIR
from hardware import get_model_size_and_ctx

def manage_local_models():
    """Menu to view and delete downloaded models to free up disk space."""
    while True:
        os.system("clear")
        print("=" * 65)
        print(" " * 20 + "DGX Spark Ops - Model Manager")
        print("=" * 65)
        
        if not MODELS_DIR.exists():
            print("Models directory not found.")
            input("\nPress Enter to return...")
            return

        # Scan for models and sort alphabetically
        dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
        dirs = [d for d in dirs if d.name not in ["bash", "vllm_profiles"]]
        dirs.sort()
        
        model_data = []
        for idx, model_dir in enumerate(dirs, start=1):
            weight_gb, _ = get_model_size_and_ctx(model_dir)
            model_data.append({
                "id": idx,
                "path": model_dir,
                "name": model_dir.name,
                "size": weight_gb
            })
            
        if not model_data:
            print("No valid models found on disk.")
            input("\nPress Enter to return...")
            return
            
        total_size = 0.0
        for data in model_data:
            total_size += data["size"]
            print(f"{data['id']:2d}) {data['name']:<35} | Size: {data['size']:>5.1f} GB")
            
        print("-" * 65)
        print(f"Total Local Disk Usage: {total_size:.1f} GB")
        print("-" * 65)
        print("Commands:")
        print(" <ID>            : Delete the selected model from disk (e.g., 1)")
        print(" q               : Quit to Master Menu")
        
        cmd = input("\nmanager> ").strip().lower()
        if not cmd:
            continue
            
        if cmd == "q":
            return
            
        if cmd.isdigit():
            target_id = int(cmd)
            target_model = next((m for m in model_data if m["id"] == target_id), None)
            
            if target_model:
                print(f"\n[WARNING] Are you sure you want to permanently delete:")
                print(f"          '{target_model['name']}' ({target_model['size']:.1f} GB)?")
                confirm = input("Type 'yes' to confirm: ").strip().lower()
                
                if confirm == "yes":
                    print(f"Deleting {target_model['path']}...")
                    try:
                        shutil.rmtree(target_model['path'])
                        print("✅ Deletion successful.")
                    except Exception as e:
                        print(f"❌ Error deleting model: {e}")
                    input("Press Enter to continue...")
                else:
                    print("Deletion cancelled.")
                    input("Press Enter to continue...")
            else:
                print(f"Invalid ID: {target_id}")
                input("Press Enter to continue...")

def clear_hf_cache():
    """Runs the huggingface-cli to clear cache."""
    print("\n" + "=" * 40)
    print("--- HuggingFace Disk Manager ---")
    print("=" * 40)
    cmd = ["uv", "run", "python3", "-m", "huggingface_hub.cli.hf", "delete-cache"]
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"\nError running disk manager: {e}")
    except FileNotFoundError:
        print("\nError: huggingface-cli or uv not found.")
        
def disk_manager_menu():
    while True:
        os.system("clear")
        print("=" * 45)
        print(" " * 12 + "DISK & CACHE MANAGER")
        print("=" * 45)
        print("1) Manage & Delete Local Models")
        print("2) Clear HuggingFace Hub Cache")
        print("3) Return to Master Menu")
        print("-" * 45)
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            manage_local_models()
        elif choice == "2":
            clear_hf_cache()
            input("Press Enter to continue...")
        elif choice == "3" or choice == "q":
            break