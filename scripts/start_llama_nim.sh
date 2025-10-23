#!/usr/bin/env bash
set -euo pipefail

# Script to start Llama 3.1 8B DGX Spark NIM

CONTAINER_NAME="llama-8b-nim"
IMAGE="nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest"
PORT="8000"
CACHE_DIR="${HOME}/nim-cache"

echo "Starting Llama 3.1 8B DGX Spark NIM..."

# Check if NGC_API_KEY is set
if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "ERROR: NGC_API_KEY environment variable must be set." >&2
  echo "Export it first: export NGC_API_KEY='your-key-here'" >&2
  exit 1
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container '${CONTAINER_NAME}' already exists."
  
  # Check if it's running
  if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container is already running."
    echo "API available at: http://localhost:${PORT}"
    exit 0
  else
    echo "Starting existing container..."
    docker start "${CONTAINER_NAME}"
  fi
else
  echo "Creating and starting new container..."
  docker run -d \
    --gpus all \
    --shm-size 16g \
    --restart no \
    --name "${CONTAINER_NAME}" \
    -p "${PORT}:8000" \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    -v "${CACHE_DIR}:/opt/nim/.cache" \
    "${IMAGE}"
fi

echo "Waiting for NIM to be ready..."
timeout=300  # 5 minutes
elapsed=0
while ! curl -s http://localhost:${PORT}/v1/health/ready > /dev/null 2>&1; do
  if [[ $elapsed -ge $timeout ]]; then
    echo "ERROR: NIM failed to start within ${timeout} seconds." >&2
    echo "Check logs with: docker logs ${CONTAINER_NAME}" >&2
    exit 1
  fi
  echo -n "."
  sleep 5
  ((elapsed += 5))
done

echo ""
echo "✅ Llama 3.1 8B DGX Spark NIM is ready!"
echo "API available at: http://localhost:${PORT}"
echo ""
echo "Test with:"
echo "curl -X POST http://localhost:${PORT}/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"meta/llama-3.1-8b-instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 50}'"
