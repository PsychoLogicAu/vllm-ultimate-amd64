#!/bin/bash
# Build script for the AMD64 port of vllm-ultimate-dgx-spark
# Uses upstream vLLM v0.26.0 (AEON carries are largely merged upstream now).
# Usage: ./build-amd64.sh [IMAGE_TAG] [--max-jobs N]
set -euo pipefail

IMAGE_TAG="${1:-aeon-vllm-ultimate-amd64:latest}"
MAX_JOBS=12

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-jobs) MAX_JOBS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "Building $IMAGE_TAG with MAX_JOBS=$MAX_JOBS"
echo ""

# The AEON-merged vllm source is private (AEON-7). Use upstream v0.26.0 instead.
# Most AEON carries (#44389, #40898, #41703) are merged into v0.26.0 upstream.
# Remaining patches (cudagraph_align, kv_cache_utils) are applied in the Dockerfile.

if [ ! -d "vllm-src" ]; then
    echo "Cloning vLLM v0.26.0 (upstream)..."
    git clone --branch v0.26.0 https://github.com/vllm-project/vllm.git vllm-src
fi

docker build --platform linux/amd64 \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    -f Dockerfile.amd64 \
    -t "$IMAGE_TAG" \
    .

echo ""
echo "Build complete: $IMAGE_TAG"
echo ""
echo "Verify:"
echo "  docker run --rm --gpus all $IMAGE_TAG python3 -c '"
echo "    import vllm, torch, flashinfer"
echo "    print(\"vllm:\", vllm.__version__)"
echo "    print(\"torch:\", torch.__version__, torch.version.cuda)"
echo "    print(\"sm:\", torch.cuda.get_device_capability())"
echo "    print(\"flashinfer:\", flashinfer.__version__)"
echo "  '"
