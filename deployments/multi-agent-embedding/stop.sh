#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================"
echo "  Stopping Multi-Agent Embedding Stack"
echo "============================================"

docker compose down

echo ""
echo "[SUCCESS] Services stopped and removed."
echo ""
