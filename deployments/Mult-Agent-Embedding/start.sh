#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[INFO] Clearing host page cache …"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

echo "[INFO] Building qwen3-embedding image (if needed) …"
docker compose build qwen3-embedding

echo "[INFO] Starting GPT-OSS-120B and Qwen3 embedding services …"
docker compose up -d

echo "[INFO] Services are launching. Use 'docker ps' and 'docker logs -f gpt-oss-120b' to monitor startup."
