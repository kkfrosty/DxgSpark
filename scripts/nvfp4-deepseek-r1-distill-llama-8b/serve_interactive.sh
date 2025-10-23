#!/usr/bin/env bash
set -euo pipefail

# Serve NVFP4 Quantized DeepSeek-R1-Distill-Llama-8B using TensorRT-LLM
# Uses the same container and approach as test.sh (which works!)

MODEL_PATH="/home/kfrost/assets/models/nvfp4/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR="$HOME/.cache/huggingface"
PORT="8000"

echo "=========================================="
echo "Starting NVFP4 DeepSeek-R1-Distill-Llama-8B Server"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
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

echo "Loading model and starting interactive mode..."
echo "The model will stay loaded in memory."
echo "Press Ctrl+C to stop the server"
echo ""

docker run \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${CACHE_DIR}:/root/.cache/huggingface/" \
  -v "${MODEL_PATH}:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    echo "Loading model..."
    python examples/llm-api/quickstart_advanced.py \
      --model_dir /workspace/model/ \
      --prompt "Model loaded successfully. Ready for inference." \
      --max_tokens 10
    
    echo ""
    echo "=========================================="
    echo "Model is loaded and ready!"
    echo "=========================================="
    echo "Keeping container alive..."
    echo "You can now exec into this container to run inference"
    echo "Container will stay running until you press Ctrl+C"
    echo ""
    
    # Keep container alive
    tail -f /dev/null
  '
