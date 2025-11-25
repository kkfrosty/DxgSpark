# Copilot Workspace Instructions

## Critical Rules
- **NEVER create README.md, INSTRUCTIONS.md, or guide files unless explicitly requested**
- Complete tasks directly without documentation
- Store HF token at `~/.huggingface/token` (chmod 600)
- Repository root: `/home/kfrost/DxgSpark` (not DxgSparkDev)
- Model assets: `/home/kfrost/assets/models/`

## Hugging Face Downloads
**Token location:** `~/.huggingface/token` (already stored)

**Download pattern:**
```bash
cd /home/kfrost/assets/models/{model-name}
HF_TOKEN=$(cat ~/.huggingface/token)
wget --header="Authorization: Bearer $HF_TOKEN" -c "https://huggingface.co/{org}/{model}/resolve/main/{file}"
```

**Common models:**
- GPT-OSS models: `openai/gpt-oss-{size}` (safetensors format)
- Embeddings: `CompendiumLabs/bge-large-en-v1.5-gguf` (public, no auth)
- Download ALL model parts (check with API first): `model-00000-of-00002.safetensors`, etc.
- Always get: `config.json`, `tokenizer.json`, `tokenizer_config.json`

## Docker & Containers
**llama.cpp container:**
- Running at `http://localhost:8080`
- Mount models: `-v /home/kfrost/assets/models:/models`
- Load model: `-m /models/{model-dir}/{model-file}`
- Supports safetensors and GGUF formats

**NIM containers (cognitive-agentics):**
```bash
cd /home/kfrost/DxgSpark/deployments/cognitive-agentics
./start.sh  # requires NGC_API_KEY
./stop.sh
docker compose logs -f
```

**Health endpoints:**
- LLM: `http://localhost:8000/v1/health`
- Embeddings: `http://localhost:8001/v1/health`
- External: `192.168.2.180`

## Model Organization
```
/home/kfrost/assets/models/
├── gpt-oss-120b-mxfp4/          # Current production model (60GB, 3 GGUF parts)
├── gpt-oss-20b-mxfp4/           # Downloaded (13GB, 3 safetensors)
├── bge-large-en-v1.5-gguf/      # Embedding model (342MB, Q8_0)
└── {new-models}/                # Create dirs as needed
```

## Benchmarking
**Location:** `/home/kfrost/DxgSpark/apps/benchMark/`
- Update benchmark results after model changes
- Include: model name, size, format, inference speed, memory usage
- Store results in structured format (JSON preferred)

## Quantization (NVFP4)
```bash
cd /home/kfrost/DxgSpark/scripts/fine-tuning/quantization-tests
./quantize-gpt-oss-120b-nvfp4.sh
```
- Input: BF16 under `assets/models/bf16`
- Output: NVFP4 under `assets/models/nvfp4`
- Uses: `nvcr.io/nvidia/tensorrt-llm`

## Security
- Store tokens/keys in `~/.huggingface/token`, `.env` files (git-ignored)
- Set permissions: `chmod 600` for token files, `chmod 700` for dirs
- Never hardcode credentials in code or commit them

## Environment
- Shell: `bash`
- Python: Available at `/usr/bin/python3` (3.12.3)
- wget, curl: Both available
- GPU: NVIDIA DGX Spark with CUDA support
