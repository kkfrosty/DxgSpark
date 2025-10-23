# Multi-Agent Embedding Stack

GPU-accelerated LLM and embedding services using llama.cpp on DGX Spark.

## Services

- **GPT-OSS-120B** (MXFP4 quantized) - Port 8000
- **Qwen3-Embedding-4B** (Q8) - Port 8001

## Quick Start

```bash
cd /home/kfrost/DxgSparkDev/deployments/multi-agent-embedding

# Start both services
./start.sh

# Stop services
./stop.sh
```

## Model Files

Models are loaded from `/home/kfrost/assets/models/`:

- `gpt-oss-120b-mxfp4-00001-of-00003.gguf` (12MB)
- `gpt-oss-120b-mxfp4-00002-of-00003.gguf` (~29GB)
- `gpt-oss-120b-mxfp4-00003-of-00003.gguf` (~29GB)
- `Qwen3-Embedding-4B-Q8_0.gguf` (~4GB)

### Download Models

If models are missing, download them:

```bash
cd /home/kfrost/assets/models

# GPT-OSS-120B (3 parts, ~58GB total)
curl -C - -L -o gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00001-of-00003.gguf

curl -C - -L -o gpt-oss-120b-mxfp4-00002-of-00003.gguf \
  https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00002-of-00003.gguf

curl -C - -L -o gpt-oss-120b-mxfp4-00003-of-00003.gguf \
  https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00003-of-00003.gguf

# Qwen3 Embeddings (~4GB)
curl -L -o Qwen3-Embedding-4B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf
```

## API Usage

### Chat Completions (GPT-OSS-120B)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

### Embeddings (Qwen3)

```bash
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your text to embed",
    "model": "qwen3-embedding"
  }'
```

## Monitoring

```bash
# Check status
docker compose ps

# View logs
docker compose logs -f gpt-oss-120b
docker compose logs -f qwen3-embedding

# Check health
curl http://localhost:8000/health
curl http://localhost:8001/health
```

## System Requirements

- **GPU**: NVIDIA GB10 (Blackwell) with 128GB memory
- **RAM**: 16GB+ system RAM
- **Storage**: ~62GB for model files
- **GPU Memory Usage**:
  - GPT-OSS-120B: ~35-45GB (MXFP4 quantization)
  - Qwen3 Embeddings: ~5GB

## Configuration

Edit [docker-compose.yml](docker-compose.yml) to adjust:

- Context length: `-n` parameter (default: 65536)
- GPU layers: `--n-gpu-layers` parameter (default: 70)
- Port mappings

## Troubleshooting

### Services won't start
```bash
# Check if models exist
ls -lh /home/kfrost/assets/models/*.gguf

# Check GPU availability
nvidia-smi

# View detailed logs
docker compose logs --tail=100 gpt-oss-120b
```

### Out of memory
Reduce context length or GPU layers in docker-compose.yml:
```yaml
command:
  - "-n"
  - "32768"  # Reduce from 65536
  - "--n-gpu-layers"
  - "50"     # Reduce from 70
```

## Architecture

```
┌─────────────────────────────┐
│   Client Applications       │
└─────────┬───────────┬───────┘
          │           │
    :8000 │           │ :8001
          │           │
┌─────────▼──────┐  ┌─▼──────────────┐
│ GPT-OSS-120B   │  │ Qwen3 Embedding│
│ (llama.cpp)    │  │ (llama.cpp)    │
│ MXFP4 GGUF     │  │ Q8 GGUF        │
└────────┬───────┘  └─┬──────────────┘
         │            │
         └────────┬───┘
                  │
          ┌───────▼────────┐
          │  NVIDIA GB10   │
          │  128GB VRAM    │
          └────────────────┘
```

## License

Apache 2.0
