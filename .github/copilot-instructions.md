# Copilot Workspace Instructions

## Environment & Scope
- Assume all commands run on the NVIDIA DGX Spark host over SSH with `bash`; repository root is `/home/kfrost/DxgSparkDev`.
- Models and large assets live outside the repo under `/home/kfrost/assets`; do not suggest moving or duplicating them.
- Reuse guidance and scripts from `/home/kfrost/dgx-spark-playbooks/` before inventing new flows; mirror their conventions when extending tooling here.

## Architecture Overview
- `deployments/cognitive-agentics/` packages NVIDIA NIM containers (GPT-OSS-120B and BGE-M3) via `docker-compose.yml`; both mount `/home/kfrost/assets/models` for cached weights.
- `fine-tuning/quantization-tests/` contains NVFP4 quantization bash scripts that wrap TensorRT-LLM containers; they expect BF16 inputs under `assets/models/bf16` and emit NVFP4 artifacts under `assets/models/nvfp4`.
- `models/` and `data/` directories are stubs for future artifacts; keep generated outputs in the matching subtree to avoid polluting deployment folders.

## Core Workflows
- LLM services: `cd deployments/cognitive-agentics` then run `./start.sh` (requires `NGC_API_KEY`) and `./stop.sh`; logs via `docker compose logs -f`.
- Health checks rely on OpenAI-compatible endpoints: `http://localhost:8000/v1/health` (LLM) and `http://localhost:8001/v1/health`; external access uses `192.168.2.180`.
- Quantization: execute `./fine-tuning/quantization-tests/quantize-gpt-oss-120b-nvfp4.sh`; the script already validates safetensor counts and free disk space before launching `nvcr.io/nvidia/tensorrt-llm`.
- Always persist sensitive env vars (e.g., `NGC_API_KEY`, `HF_TOKEN`) via `.env` files ignored by Git; never hardcode keys in generated code.

## Patterns & Conventions
- Bash scripts in this repo enable `set -e` and perform explicit directory checks; follow the same pattern and emit user-facing status blocks with clear emoji markers.
- Docker services run with named containers (`cognitive-llm-supervisor`, `cognitive-embeddings`) and map GPU resources via `deploy.resources`; keep new services consistent to simplify monitoring.
- Treat `/home/kfrost/assets/models` as the single source for model weights; new tooling should mount it read/write instead of downloading elsewhere.
- Document deviations from the DGX Spark default (ports, GPUs, paths) directly beside the code change so future agents stay aligned.

## External References
- When unsure about a workflow, inspect the closest matching playbook under `/home/kfrost/dgx-spark-playbooks/nvidia/`; cite the specific README section you are following.
- Prefer NVIDIA-containerized tooling (`nvcr.io/nim/*`, `nvcr.io/nvidia/tensorrt-llm/*`) already in use here; suggest alternatives only if a playbook recommends them.
