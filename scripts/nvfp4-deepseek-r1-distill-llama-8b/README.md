# NVFP4 Quantization Scripts for DeepSeek-R1-Distill-Llama-8B

Scripts for quantizing and serving the DeepSeek-R1-Distill-Llama-8B model in NVFP4 format on DGX Spark.

## Prerequisites

- NVIDIA DGX Spark with Blackwell GPU
- Docker with GPU support
- HuggingFace token (set in `~/.bashrc` as `HF_TOKEN`)
- 116+ GB free memory
- 20+ GB free disk space

## Scripts

### 1. `quantize.sh` - Quantize the model to NVFP4

Downloads the original DeepSeek-R1-Distill-Llama-8B model from HuggingFace and quantizes it to NVFP4 format.

**Usage:**
```bash
./quantize.sh
```

**What it does:**
- Downloads the 15 GB original model to `~/.cache/huggingface/`
- Quantizes to NVFP4 (reduces to ~5.7 GB)
- Saves to `/home/kfrost/assets/models/nvfp4/DeepSeek-R1-Distill-Llama-8B/`
- Fixes file ownership and organization

**Duration:** 45-90 minutes depending on network speed

### 2. `test.sh` - Test the quantized model

Runs a quick inference test to verify the quantized model works correctly.

**Usage:**
```bash
./test.sh
```

**What it does:**
- Loads the quantized model
- Runs inference with prompt "Paris is great because"
- Generates 64 tokens
- Displays output

**Duration:** ~30 seconds for initial load, then instant inference

### 3. `start_api.sh` - Start the model as an API server

Starts a persistent TensorRT-LLM server with OpenAI-compatible API endpoints.

**Usage:**
```bash
./start_api.sh
```

**What it does:**
- Starts server on http://localhost:8000
- Provides OpenAI-compatible API endpoints
- Uses PyTorch backend
- Supports batch inference
- Runs as persistent background service

**Available Endpoints:**
- `/v1/completions` - Text completion
- `/v1/chat/completions` - Chat completion
- `/v1/models` - List models

**Test the server:**
```bash
# Chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/model",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100
  }'

# Text completions
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/model",
    "prompt": "The capital of France is",
    "max_tokens": 50
  }'
```

### 4. `stop_api.sh` - Stop the API server

Stops the running API server and frees GPU memory.

**Usage:**
```bash
./stop_api.sh
```

## Workflow

1. **First time setup:**
   ```bash
   ./quantize.sh   # Downloads and quantizes the model (45-90 min)
   ```

2. **Test the model:**
   ```bash
   ./test.sh       # Verify it works (~30 seconds)
   ```

3. **Start the API server:**
   ```bash
   ./start_api.sh  # Start persistent API server (~60 seconds)
   ```

4. **Stop the server when done:**
   ```bash
   ./stop_api.sh   # Stop and free GPU memory
   ```

## Output Structure

```
/home/kfrost/assets/models/nvfp4/DeepSeek-R1-Distill-Llama-8B/
├── chat_template.jinja
├── config.json
├── generation_config.json
├── hf_quant_config.json
├── model-00001-of-00002.safetensors  (4.7 GB)
├── model-00002-of-00002.safetensors  (1.0 GB)
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

## Model Details

- **Original Size:** 15 GB (FP16/BF16)
- **Quantized Size:** 5.7 GB (NVFP4)
- **Compression:** ~2.6x smaller
- **GPU Memory Usage:** ~71 GB during inference
- **Format:** NVFP4 (4-bit floating point for Blackwell)

## Troubleshooting

**Memory errors:**
```bash
# Clear cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
free -h
```

**Permission errors:**
```bash
# Fix ownership
sudo chown -R $USER:$USER /home/kfrost/assets/models/nvfp4/DeepSeek-R1-Distill-Llama-8B/
```

**HuggingFace auth errors:**
```bash
# Check token
echo $HF_TOKEN
# Re-export if needed
export HF_TOKEN="your_token_here"
```

## Notes

- The original 15 GB model is cached in `~/.cache/huggingface/` and can be reused for fine-tuning
- Quantization is a one-time process; subsequent runs will use the cached original model
- For fine-tuning, use the original model first, then re-quantize the fine-tuned version
