#!/bin/bash
# Build script for the AMD64 port of vllm-ultimate-dgx-spark
# Usage: ./build-amd64.sh [IMAGE_TAG] [--max-jobs N] [--clone-src]
set -euo pipefail

IMAGE_TAG="${1:-aeon-vllm-ultimate-amd64:latest}"
MAX_JOBS=12
CLONE_SRC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-jobs) MAX_JOBS="$2"; shift 2 ;;
        --clone-src) CLONE_SRC=1; shift ;;
        *) shift ;;
    esac
done

echo "Building $IMAGE_TAG with MAX_JOBS=$MAX_JOBS"
echo ""

# Note: the AEON-7/vllm repo referenced in SOURCE.md is not public.
# AEON's AEON-merged vLLM source tree is only available via their prebuilt GHCR images.
#
# This port currently assumes either:
#   - You already have a copy of the AEON-merged vllm source at vllm-src/, OR
#   - AEON-7 shares the aeon-v0.26.0 branch with you.
#
# If neither, you'll need to reconstruct the source by cloning upstream v0.26.0
# and manually integrating the AEON carries (#44389, #40898, #41703, etc.).

if [ "$CLONE_SRC" = "1" ]; then
    if [ -d "vllm-src" ]; then
        echo "vllm-src/ already exists. Remove it or don't use --clone-src."
        exit 1
    fi
    echo "NOTE: AEON-7/vllm is not public. Cloning upstream v0.26.0 instead."
    echo "      This will miss AEON-specific carries that are not yet merged upstream."
    echo ""
    read -rp "Continue with upstream v0.26.0? (y/N) " ans || ans="N"
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
        echo "Aborted."
        exit 1
    fi
    git clone --branch v0.26.0 https://github.com/vllm-project/vllm.git vllm-src
fi

if [ ! -d "vllm-src" ]; then
    echo "vllm-src/ directory not found."
    echo ""
    echo "The AEON-merged vLLM source is required for a complete build."
    echo "Options:"
    echo "  1) If you have the AEON-merged source:"
    echo "       mkdir vllm-src && cp -r <aeon-vllm-source>/* vllm-src/"
    echo "  2) Use upstream v0.26.0 (missing some AEON carries):"
    echo "       ./build-amd64.sh --clone-src"
    echo "  3) Ask AEON-7 for access to their aeon-v0.26.0 branch."
    echo ""
    exit 1
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
