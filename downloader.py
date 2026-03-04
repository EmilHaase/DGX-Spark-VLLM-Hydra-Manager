import os
import subprocess
from pathlib import Path
from config import MODELS_DIR, PROJECT_ROOT

def download_model_interactive():
    print("\n" + "=" * 40)
    print("--- HuggingFace Model Downloader ---")
    print("=" * 40)
    model_id = input("Enter HuggingFace Model ID (e.g., mistralai/Mistral-7B-Instruct-v0.2): ").strip()
    
    if not model_id:
        print("\nModel ID cannot be empty.")
        return

    model_name = model_id.split("/")[-1]
    output_dir = MODELS_DIR / model_name
    
    cmd = [
        "uv", "run", "python3", "-m", "huggingface_hub.cli.hf", "download", 
        model_id, 
        "--local-dir", str(output_dir)
    ]
    
    env = os.environ.copy()
    
    print(f"\nDownloading '{model_id}' to '{output_dir}'...")
    try:
        subprocess.run(cmd, env=env, check=True, cwd=PROJECT_ROOT)
        print("\nDownload complete!")
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading model: {e}")
    except FileNotFoundError:
        print("\nError: uv or huggingface-cli not found. Ensure it is installed in the virtual environment or system.")