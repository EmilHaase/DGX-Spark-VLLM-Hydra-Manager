#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "==============================================================="
echo "1. Checking System Headers"
echo "==============================================================="
if ! dpkg -s python3.12-dev build-essential ffmpeg &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3.12-dev build-essential ffmpeg
fi

echo "==============================================================="
echo "2. Purging Corrupted Environment"
echo "==============================================================="
if [ -d ".venv" ]; then
    echo "Removing .venv to apply proper package bounds..."
    rm -rf .venv
fi

uv venv --python 3.12 --seed .venv
source .venv/bin/activate

if [ ! -f .env ]; then
cat <<EOF > .env
HF_TOKEN=
EOF
fi

echo "==============================================================="
echo "3. Installing Grace Blackwell NATIVE PyTorch (cu130)"
echo "==============================================================="
echo "Installing PyTorch (This will be extremely fast as it uses your local cache)..."
python3 -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --default-timeout=1200

echo "==============================================================="
echo "4. Installing Accelerated Kernels"
echo "==============================================================="
uv pip install -U xgrammar triton flashinfer-python --prerelease=allow

echo "==============================================================="
echo "5. Installing WebUI & HuggingFace Dependencies"
echo "==============================================================="
uv pip install -U open-webui huggingface_hub

echo "==============================================================="
echo "6. Building vLLM from source (sm_121a native Blackwell kernels)"
echo "==============================================================="
echo "Installing pre-requisite build tools..."
uv pip install -U setuptools_scm ninja cmake packaging wheel

echo "This will take 45-60 minutes. Do not interrupt."
export TORCH_CUDA_ARCH_LIST="12.1a"
export VLLM_TARGET_DEVICE="cuda"
export MAX_JOBS=18
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
pip install --no-build-isolation git+https://github.com/vllm-project/vllm.git

echo "==============================================================="
echo "6b. Force-upgrading Transformers to latest master"
echo "==============================================================="
uv pip install --reinstall-package transformers "git+https://github.com/huggingface/transformers.git"

echo "==============================================================="
echo "7. Applying CUDA 13.0 Runtime Patch"
echo "==============================================================="
if [ ! -f /usr/local/cuda/lib64/libcudart.so.12 ] && [ -f /usr/local/cuda/lib64/libcudart.so.13 ]; then
    sudo ln -sf /usr/local/cuda/lib64/libcudart.so.13 /usr/local/cuda/lib64/libcudart.so.12 || true
fi

echo "==============================================================="
echo "7. Resetting WebUI Database Cache (Fixing Schema Conflicts)"
echo "==============================================================="
DB_DIR="$HOME/.Hydra_Manager_data"
mkdir -p "$DB_DIR"

if [ -f "$DB_DIR/webui.db" ]; then
    echo "Backing up mismatched webui.db..."
    mv "$DB_DIR/webui.db" "$DB_DIR/webui.db.bak"
fi
if [ -d "$DB_DIR/vector_db" ]; then
    echo "Backing up mismatched vector_db..."
    mv "$DB_DIR/vector_db" "$DB_DIR/vector_db.bak"
fi
if [ -d "$DB_DIR/chroma" ]; then
    echo "Backing up mismatched chroma db..."
    mv "$DB_DIR/chroma" "$DB_DIR/chroma.bak"
fi

echo "==============================================================="
echo "Setup complete. Databases and Packages are synchronized!"
echo "==============================================================="