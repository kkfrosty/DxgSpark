#!/usr/bin/env bash
set -euo pipefail

# Test NVFP4 Quantized DeepSeek-R1-Distill-Llama-8B

MODEL_PATH="/home/kfrost/assets/models/nvfp4/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR="$HOME/.cache/huggingface"

echo "=========================================="
echo "Testing NVFP4 DeepSeek-R1-Distill-Llama-8B"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo ""

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: Model not found at ${MODEL_PATH}" >&2
  echo "Run ./quantize.sh first to create the quantized model." >&2
  exit 1
fi

# Check if HF_TOKEN is set
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. This may cause issues."
fi

echo "Running inference test..."
echo "Prompt: 'Paris is great because'"
echo ""

docker run \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${CACHE_DIR}:/root/.cache/huggingface/" \
  -v "${MODEL_PATH}:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    python examples/llm-api/quickstart_advanced.py \
      --model_dir /workspace/model/ \
      --prompt "Paris is great because" \
      --max_tokens 64
    '

echo ""
echo "✅ Test complete!"
