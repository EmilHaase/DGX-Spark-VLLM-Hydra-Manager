# DGX Spark VLLM Hydra Manager

An automated orchestration tool and launcher built exclusively for the DGX Spark environment. This manager downloads AI models, handles complex vLLM engine orchestration, dynamically provisions VRAM, and runs a connected Open-WebUI instance-all optimized for Blackwell Native (sm121a) and PyTorch processing on UMA hardware.

## 🚀 Features

- **Automated Dependency Build**: Custom pipeline to install PyTorch natively for Aarch64, compile accelerated kernels (xgrammar, triton, flashinfer), and build vLLM from source.
- **Hydra Launcher GUI**: Interactive command-line UI to intelligently assign models, monitor global GPU memory allocation, toggle deepseek reasoning parsers, and adjust context lengths on the fly.
- **Advanced UMA Concurrency Scaling**: Automatically calculates explicit KV Cache limits and injects strict Ray/NCCL node isolation. This eliminates the vLLM memory profiling bugs on UMA (Unified Memory Architecture) devices, safely enabling simultaneous multi-model bootstrapping!
- **Zero-Config WebUI Integration**: Open-WebUI automatically binds the selected vLLM endpoints seamlessly in the background. Private chats and data are safely stored in your home directory (`~/.Hydra_Manager_data`).
- **Benchmarking & Testing Dashboard**: Built-in metrics dashboard to automatically scan active model endpoints and execute Time-To-First-Token (TTFT) Latency benchmarks, Tokens-Per-Second (TPS) Speed tests, Parallel Multi-Threaded load tests, and interactive chat.
- **Disk & Cache Manager**: Clean out heavy, unused `safetensors` models directly from the launcher to easily reclaim storage.

---

## 🛠️ Prerequisites

- NVIDIA DGX Spark (Aarch64 / ARM64 with GB10 Unified Memory)
- Tested with **CUDA 13.0** and **cuDNN 9**
- Python 3.12 (`python3.12-dev`)
- OS level tools: `build-essential`, `ffmpeg`, `uv`, `git`

---

## 📥 Setup & Launch

The launcher is fully self-building and self-containing. Simply run the launcher script. If the dependencies and engine need to be built, it will automatically handle it for you.

```bash
git clone https://github.com/EmilHaase/dgx-spark-vllm-hydra-manager.git
cd dgx-spark-vllm-hydra-manager
chmod +x run.sh
./run.sh
```

### HuggingFace Token
This manager downloads models dynamically through HuggingFace (e.g. Llama 3 or Qwen3).
**You do not need to configure anything beforehand!** On the first launch, the script will interactively ask you to paste your HuggingFace Token, and it will securely save it to your local environment for all future sessions.

---

## 🎛️ Navigation

Once the main dashboard boots, you have the following options:

1. **Launch vLLM Hydra**: The main orchestrator. Select models with their `ID`, change custom limits with `u <ID> <LIMIT>`, tweak reasoning with `a <ID>` or `c <ID>`, and hit `r` to spin up the servers.
2. **Update vLLM...**: Perform a clean rebuild from the ground up if you encounter CUDA issues or need fresh features from the vLLM master branch. Contains a safety confirmation prompt.
3. **Download New Model...**: Type the hugging face repository ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) to fetch the `safetensors`.
4. **Disk Manager...**: Easily delete old models and clear the `huggingface-cli` caches.
5. **Add Local Model...**: Manually add a directory of downloaded safetensors to the registry.
6. **Set GPU Utlization Default...**: Adjust the baseline fraction of system VRAM to allocate to engines.
7. **Toggle Environment Details...**: Debug your python and compiler routes.
8. **Test & Benchmark Engines...**: Enter the high-performance testing dashboard to validate running vLLM endpoints.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information. Note that this project is not officially affiliated with or endorsed by NVIDIA or the vLLM team.
