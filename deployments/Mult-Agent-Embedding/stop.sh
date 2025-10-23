#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[INFO] Stopping GPT-OSS-120B and Qwen3 embedding services …"
docker compose down
