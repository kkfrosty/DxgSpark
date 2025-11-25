# Unified Model Deployment

## Overview

All models are now deployed using **NVIDIA NIM** (NVIDIA Inference Microservices) for:
- ✅ Consistent interface across all models
- ✅ Production-grade performance with NVIDIA optimizations
- ✅ Users can load/unload models via the benchMark UI
- ✅ Native safetensors support (no format conversion needed)
- ✅ Automatic GPU optimization

## Architecture

### Single Deployment Stack
All models are in one docker-compose file for unified management:
- **GPT-OSS-20B** → Port 8000 (NVIDIA NIM)
- **GPT-OSS-120B** → Port 8010 (NVIDIA NIM)
- **BGE Embeddings** → Port 8001 (NVIDIA NIM)

### Model Format
All models use **safetensors** (MXFP4 quantization) - no GGUF conversion needed.

## Setup

### 1. NGC API Key Required

```bash
# Get your key from: https://ngc.nvidia.com/setup/api-key
export NGC_API_KEY="your-key-here"

# Add to ~/.bashrc for persistence
echo 'export NGC_API_KEY="your-key-here"' >> ~/.bashrc
```

### 2. Start the Deployment

```bash
cd /home/kfrost/DxgSpark/deployments/working-models
./start.sh
```

This starts only the embedding service. Models are loaded on-demand via the UI.

## Usage

### Via BenchMark UI (Recommended)
1. Open http://localhost:8080
2. Click on a model chip (e.g., GPT-OSS-20B)
3. Click "🚀 Load Model"
4. Wait for NIM to download and start the container (first time: 2-5 minutes)
5. Start chatting!

### Manual Commands

```bash
cd /home/kfrost/DxgSpark/deployments/working-models

# Load models
docker compose up -d gpt-oss-20b
docker compose up -d gpt-oss-120b

# Unload models
docker compose stop gpt-oss-20b
docker compose stop gpt-oss-120b

# Check status
docker compose ps

# View logs
docker compose logs -f gpt-oss-20b
```

## Benefits of NIM

### vs llama.cpp
- ✅ Native safetensors support (no conversion)
- ✅ Better GPU utilization
- ✅ Production-grade stability
- ✅ Automatic optimization for your GPU
- ❌ Requires NGC API key
- ❌ Larger container images (first download takes time)

### vs vLLM
- ✅ Pre-optimized by NVIDIA
- ✅ Built-in monitoring
- ✅ Consistent API across all models
- ✅ Better memory management

## Port Assignments

| Service | Port | Status |
|---------|------|--------|
| BGE Embeddings | 8001 | Always running |
| GPT-OSS-20B | 8000 | Load on-demand |
| GPT-OSS-120B | 8010 | Load on-demand |

**Note**: 20B and 120B can run simultaneously now (different ports)!

## First-Time Setup

On first load, NIM will:
1. Download the container image (~5GB)
2. Download/optimize the model weights
3. Initialize the inference engine

This takes **2-5 minutes** per model. Subsequent loads are much faster (~30 seconds).

## Troubleshooting

### Model won't load
```bash
# Check if NGC_API_KEY is set
echo $NGC_API_KEY

# Check Docker can access NGC registry
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC_API_KEY>

# View detailed logs
docker compose logs gpt-oss-20b
```

### Out of memory
NIM automatically optimizes for your GPU. If you still run out of memory:
- Only load one large model at a time
- Close other GPU applications
- Check GPU memory: `nvidia-smi`

### Port conflicts
```bash
# Check what's using a port
sudo netstat -tlnp | grep 8000

# Stop conflicting services
docker ps
docker stop <container-name>
```

## Migration from llama.cpp

The old llama.cpp deployment has been shut down. All functionality is now in this unified NIM deployment.

**Old paths**: `/home/kfrost/DxgSpark/deployments/multi-agent-embedding`  
**New path**: `/home/kfrost/DxgSpark/deployments/working-models`

The benchMark UI has been updated to use the new deployment automatically.
