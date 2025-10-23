#!/bin/bash

# Stop DeepSeek-R1-Distill-Llama-8B NVFP4 API Server

set -e

CONTAINER_NAME="deepseek-nvfp4-api"

echo "======================================"
echo "Stopping DeepSeek NVFP4 API Server"
echo "======================================"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping container..."
    docker stop "${CONTAINER_NAME}"
    echo "Removing container..."
    docker rm "${CONTAINER_NAME}"
    echo ""
    echo "✅ Server stopped and removed"
else
    echo "⚠️  Server is not running"
fi

echo ""
echo "Freeing GPU memory..."
docker system prune -f > /dev/null 2>&1
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true

echo "✅ GPU memory freed"
echo "======================================"
