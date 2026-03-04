import os
from pathlib import Path

# Base paths for the application
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Important DGX WebUI override requested by constraints
# We place this in the user's home directory so it's not accidentally committed to the repo
WEBUI_DIR = Path.home() / ".Hydra_Manager_data"
WEBUI_DIR.mkdir(parents=True, exist_ok=True)

# Process manually basic dotenv to avoid unecessary package dependencies
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

# Architecture Constants
try:
    # Dynamically calculate system unified memory
    SYSTEM_TOTAL_GB = round((os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / (1024.**3))
except ValueError:
    SYSTEM_TOTAL_GB = 128

# Allow environment override for default GPU memory utilization, fallback to 0.85
GPU_MEM_UTIL_DEFAULT = float(os.environ.get("GPU_MEM_UTIL_DEFAULT", 0.85))