#!/usr/bin/env bash
set -euo pipefail

# Serve NVFP4 Quantized DeepSeek-R1-Distill-Llama-8B with OpenAI-compatible API
# Uses TensorRT-LLM Python API (same approach that works in test.sh)

MODEL_PATH="/home/kfrost/assets/models/nvfp4/DeepSeek-R1-Distill-Llama-8B"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$HOME/.cache/huggingface"
PORT="8000"

echo "=========================================="
echo "Starting NVFP4 DeepSeek-R1-Distill-Llama-8B Server"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Backend: TensorRT-LLM Python API"
echo "API: OpenAI-compatible endpoints"
echo ""

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: Model not found at ${MODEL_PATH}" >&2
  echo "Run ./quantize.sh first to create the quantized model." >&2
  exit 1
fi

# Check if HF_TOKEN is set
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. This may cause issues with tokenizer loading."
fi

echo "Starting TensorRT-LLM API server..."
echo "Server will be available at: http://localhost:${PORT}"
echo ""
echo "Endpoints:"
echo "  - GET  /v1/models"
echo "  - POST /v1/chat/completions"
echo "  - POST /v1/completions"
echo "  - GET  /health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

docker run \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  -v "${CACHE_DIR}:/root/.cache/huggingface/" \
  -v "${MODEL_PATH}:/workspace/model" \
  -v "${SCRIPT_DIR}/trtllm_api_server.py:/workspace/trtllm_api_server.py" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host \
  -p ${PORT}:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "pip install flask && python /workspace/trtllm_api_server.py"
