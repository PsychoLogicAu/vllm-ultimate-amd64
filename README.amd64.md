# AEON vLLM Ultimate — AMD64 Port for Consumer Blackwell (RTX 5090 / 5080)

AMD64 port of `vllm-ultimate-dgx-spark`, adapted for **consumer Blackwell GPUs** (RTX 5090 / 5080 / RTX PRO 6000 Blackwell) on standard x86_64 systems.

## What this is

Same AEON vLLM stack as the DGX Spark build, but for amd64 hosts:

- **vLLM 0.26.0** compiled for **sm_120** (RTX 5090) instead of 12.1a (GB10)
- All AEON carries preserved:
  - PR #44389: Triton NVFP4 KV cache (~3× KV capacity)
  - PR #40898: DFlash sliding-window attention
  - PR #41703: Prefix-cache corruption immunity
  - High-concurrency DFlash fix, cudagraph alignment, UMA clamp, NVFP4_AWQ
- Same patches, same specs, different base

## What changed from the original

| Original (DGX Spark / ARM64) | Port (RTX 5090 / AMD64) |
|---|---|
| `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:latest` | `nvcr.io/nvidia/cuda:13.0.2-base-ubuntu24.04` |
| `TORCH_CUDA_ARCH_LIST=12.1a` | `TORCH_CUDA_ARCH_LIST=12.0` |
| CUDA path: `targets/sbsa-linux` | `targets/x86_64-linux` |
| PyTorch baked into base | Install `torch torchvision torchaudio` from NVIDIA cu130 index |
| Ubuntu 22.04 (arm64) | Ubuntu 24.04 (x86_64) |
| Python3 baked into base | `apt install python3 python3-pip python3-dev` |

Everything else (patches, vLLM build, deps) is identical.

## Requirements

- Host: x86_64 Linux with NVIDIA RTX 5090 / 5080 / RTX PRO 6000 Blackwell
- GPU compute capability: **sm_120**
- NVIDIA driver supporting Blackwell (≥ 570.x recommended, ≥ 600.x for CUDA 13)
- Docker ≥ 24 with `nvidia-container-toolkit`
- **~40 GB** free disk for the image build
- **~60–90 min** build wall clock

## Build

Clone this repo and the vLLM source tree:

```bash
git clone https://github.com/AEON-7/vllm-ultimate-dgx-spark.git
cd vllm-ultimate-dgx-spark

# Get the AEON vLLM source (the same tree as the DGX Spark build)
git clone --branch aeon-v0.26.0 https://github.com/AEON-7/vllm.git vllm-src
```

Then build the amd64 image:

```bash
docker build --platform linux/amd64 -f Dockerfile.amd64 -t aeon-vllm-ultimate-amd64:latest .
```

### Build knobs (same as original)

| Env var | Default | Notes |
|---|---|---|
| `MAX_JOBS` | `12` | Lower to 8/6 if the build OOMs |
| `NVCC_THREADS` | `2` | Per-nvcc threads |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `8` | CMake parallelism |

You can override during build:

```bash
docker build --platform linux/amd64 -f Dockerfile.amd64 \
  --build-arg MAX_JOBS=8 \
  -t aeon-vllm-ultimate-amd64:latest .
```

## Serve (example)

Same serve recipes as the main DGX Spark readme apply — you're running the same vLLM build. Example:

```bash
docker run -d --name aeon-vllm-amd64 \
    --gpus all --ipc=host --shm-size=16g --net=host \
    -v /models/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP:/model:ro \
    --entrypoint vllm \
    aeon-vllm-ultimate-amd64:latest \
    serve /model \
        --quantization modelopt \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 32768 --max-num-seqs 16 \
        --gpu-memory-utilization 0.80 \
        --enable-chunked-prefill --enable-prefix-caching \
        --trust-remote-code
```

## Notes

- **NVFP4 KV cache**: works on sm_120 via the Triton software path (PR #44389). Use `--kv-cache-dtype nvfp4` with causal speculators (`mtp`, `qwen3_5_mtp`, `eagle3`). For DFlash (non-causal), use `--kv-cache-dtype fp8_e4m3`.
- **Memory**: RTX 5090 has 32 GB dedicated VRAM (unlike DGX Spark's 121 GB unified). Be more conservative with `--max-model-len` and `--max-num-seqs`.
- **Unified memory issues**: less of a concern on dedicated-VRAM cards, but the UMA clamp carry still helps avoid negative estimates.

## Branch

This port lives on the `amd64-sm120-rtx5090` branch.
