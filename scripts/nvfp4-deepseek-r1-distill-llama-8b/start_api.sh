#!/bin/bash

# Start DeepSeek-R1-Distill-Llama-8B NVFP4 API Server
# This script starts the model as an OpenAI-compatible API server

set -e

# Configuration
CONTAINER_NAME="deepseek-nvfp4-api"
MODEL_PATH="/home/kfrost/assets/models/nvfp4/DeepSeek-R1-Distill-Llama-8B"
PORT=8000

echo "======================================"
echo "Starting DeepSeek NVFP4 API Server"
echo "======================================"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# Start the server
echo "Starting TensorRT-LLM API server..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${MODEL_PATH}:/workspace/model" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host \
  -p ${PORT}:${PORT} \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "trtllm-serve /workspace/model --backend pytorch --port ${PORT} --host 0.0.0.0 --max_num_tokens 131072 --max_seq_len 131072 --max_batch_size 8 --kv_cache_free_gpu_memory_fraction 0.15"

echo ""
echo "Server is starting up (this takes ~60 seconds)..."
echo "Waiting for server to be ready..."

# Wait for server to start
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "Uvicorn running"; then
        echo ""
        echo "✅ API Server is ready!"
        echo ""
        echo "======================================"
        echo "Server Information"
        echo "======================================"
        echo "Container: ${CONTAINER_NAME}"
        echo "Base URL: http://localhost:${PORT}"
        echo "Model: DeepSeek-R1-Distill-Llama-8B (NVFP4)"
        echo ""
        echo "API Endpoints:"
        echo "  - Completions: http://localhost:${PORT}/v1/completions"
        echo "  - Chat: http://localhost:${PORT}/v1/chat/completions"
        echo "  - Models: http://localhost:${PORT}/v1/models"
        echo ""
        echo "Example curl command:"
        echo "curl -X POST http://localhost:${PORT}/v1/chat/completions \\"
        echo "  -H 'Content-Type: application/json' \\"
        echo "  -d '{"
        echo "    \"model\": \"/workspace/model\","
        echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
        echo "    \"max_tokens\": 100"
        echo "  }'"
        echo ""
        echo "To stop: docker stop ${CONTAINER_NAME}"
        echo "To view logs: docker logs -f ${CONTAINER_NAME}"
        echo "======================================"
        exit 0
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo -n "."
done

echo ""
echo "❌ Server failed to start within ${MAX_WAIT} seconds"
echo "Check logs with: docker logs ${CONTAINER_NAME}"
exit 1
