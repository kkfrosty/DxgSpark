#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================"
echo "  Multi-Agent Embedding Stack Startup"
echo "============================================"

# Check if GGUF model files exist
if [ ! -f "/home/kfrost/assets/models/gpt-oss-120b-mxfp4-00001-of-00003.gguf" ]; then
    echo "[ERROR] GPT-OSS-120B GGUF files not found!"
    echo "[INFO] Please download the model files first:"
    echo "  cd /home/kfrost/assets/models"
    echo "  curl -C - -L -o gpt-oss-120b-mxfp4-00001-of-00003.gguf https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
    echo "  curl -C - -L -o gpt-oss-120b-mxfp4-00002-of-00003.gguf https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00002-of-00003.gguf"
    echo "  curl -C - -L -o gpt-oss-120b-mxfp4-00003-of-00003.gguf https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00003-of-00003.gguf"
    exit 1
fi

if [ ! -f "/home/kfrost/assets/models/Qwen3-Embedding-4B-Q8_0.gguf" ]; then
    echo "[ERROR] Qwen3 embedding model not found!"
    echo "[INFO] Please download: curl -L -o /home/kfrost/assets/models/Qwen3-Embedding-4B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf"
    exit 1
fi

echo "[INFO] Clearing host page cache for optimal performance..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "[WARN] Could not clear cache (needs sudo)"

echo "[INFO] Building llama.cpp image (if needed)..."
docker compose build --quiet

echo "[INFO] Starting services..."
docker compose up -d

echo ""
echo "[SUCCESS] Services are launching!"
echo ""
echo "  GPT-OSS-120B:      http://localhost:8000"
echo "  Qwen3 Embeddings:  http://localhost:8001"
echo ""
echo "Monitor startup:"
echo "  docker compose ps"
echo "  docker compose logs -f gpt-oss-120b"
echo "  docker compose logs -f qwen3-embedding"
echo ""
echo "Test the API:"
echo '  curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '"'"'{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'"'"
echo ""
