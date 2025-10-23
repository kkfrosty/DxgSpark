#!/usr/bin/env bash
set -euo pipefail

# Script to stop and cleanup Llama 3.1 8B DGX Spark NIM

CONTAINER_NAME="llama-8b-nim"

echo "Stopping Llama 3.1 8B DGX Spark NIM..."

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container '${CONTAINER_NAME}' does not exist."
  exit 0
fi

# Stop the container
echo "Stopping container..."
docker stop "${CONTAINER_NAME}" 2>/dev/null || true

# Remove the container
echo "Removing container..."
docker rm "${CONTAINER_NAME}" 2>/dev/null || true

# Clean Docker resources
echo "Cleaning up Docker resources..."
docker system prune -f > /dev/null 2>&1

# Clear system cache to free memory
echo "Clearing system cache to free memory..."
if command -v sudo &> /dev/null; then
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || echo "Note: Could not clear cache (may need sudo password)"
fi

echo "✅ Llama 3.1 8B DGX Spark NIM stopped and memory freed."
echo ""
echo "Memory status:"
free -h | grep -E "^Mem:"
