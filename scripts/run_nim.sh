#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for NVIDIA NIM LLM containers with a persistent cache.

usage() {
  cat <<'EOF'
Usage: run_nim.sh [options]

Options:
  -i IMAGE        Container image (default: nvcr.io/nim/openai/gpt-oss-120b:latest)
  -n NAME         Container name (default: gpt-oss-120b-nim)
  -p PORT         Host port to expose (default: 8100)
  -c CACHE_DIR    Host directory to persist /opt/nim/.cache (default: $HOME/nim-cache)
  -m FRACTION     NIM_MAX_GPU_MEMORY_UTILIZATION_STARTUP value (default: 0.5)
  -s SIZE         Shared memory size for container (default: 16g)
  -h              Show this help message

Environment:
  NGC_API_KEY must be set in the shell environment before running.
EOF
}

image="nvcr.io/nim/openai/gpt-oss-120b:latest"
name="gpt-oss-120b-nim"
port="8100"
cache_dir="${HOME}/nim-cache"
memory_fraction="0.5"
shm_size="16g"

while getopts ":i:n:p:c:m:s:h" opt; do
  case "$opt" in
    i) image="$OPTARG" ;;
    n) name="$OPTARG" ;;
    p) port="$OPTARG" ;;
    c) cache_dir="$OPTARG" ;;
    m) memory_fraction="$OPTARG" ;;
    s) shm_size="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing argument for -$OPTARG" >&2; usage; exit 1 ;;
    \?) echo "Unknown option -$OPTARG" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "Environment variable NGC_API_KEY must be set before running." >&2
  exit 1
fi

mkdir -p "$cache_dir"

if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container '$name' already exists. Remove it first or choose a different name." >&2
  exit 1
fi

exec docker run -d \
  --gpus all \
  --shm-size "$shm_size" \
  --restart no \
  --name "$name" \
  -p "${port}:8000" \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -e NIM_MAX_GPU_MEMORY_UTILIZATION_STARTUP="$memory_fraction" \
  -v "${cache_dir}:/opt/nim/.cache" \
  "$image"
