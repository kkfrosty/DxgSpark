# Model Loading Feature

## Overview

The DxGChatBenchMark application supports **dynamic model loading and unloading** using **NVIDIA NIM** (NVIDIA Inference Microservices). Users can select any model from the UI and load it on-demand.

### Why NVIDIA NIM?

All models now use NVIDIA NIM for:
- **Unified experience** - Same interface for all models
- **Production-grade performance** - NVIDIA-optimized inference
- **Native safetensors support** - No format conversion needed
- **Easy management** - Load/unload via UI, no manual scripts

## How It Works

### Backend (API)

The application tracks three model states:

1. **Available**: The model files exist and can potentially be loaded
2. **Loaded**: The model server process is running and ready to accept requests
3. **Offline**: The model is not available or loaded

#### New API Endpoints

- **GET /api/models** - Returns model status including `loaded`, `available`, and `can_load` flags
- **POST /api/models/{model_id}/load** - Starts the model server process
- **POST /api/models/{model_id}/unload** - Stops the model server process

#### Model Configuration

Models in `MODELS` dictionary now include:

```python
{
    "model_path": "/path/to/model.gguf",  # Path to model file
    "load_command": "llama-server --model ... --port 8000 ..."  # Command to start server
}
```

### Frontend (UI)

Users can interact with models through the enhanced modal:

1. Click on any model chip in the footer
2. The modal shows model status: "Loaded & Ready", "Available (Not Loaded)", or "Offline"
3. If the model supports auto-loading (`can_load: true`):
   - Click **"🚀 Load Model"** to start the model server
   - Click **"⏹️ Unload Model"** to stop the model server
4. Once loaded, the model is ready to receive chat messages

### Model Status Indicators

- **Green dot** = Loaded & Ready
- **Red dot** = Not loaded or offline
- **"Available (Not Loaded)"** = Model files exist but server isn't running yet

## Current Model Configuration

### GPT-OSS-20B

- **Model Path**: `/home/kfrost/assets/models/gpt-oss-20b-mxfp4/` (safetensors)
- **Server Port**: 8000
- **Inference Engine**: NVIDIA NIM
- **Format**: Safetensors (MXFP4)
- **Docker Service**: `gpt-oss-20b` in `working-models` stack
- **Status**: ✅ Can be loaded via UI

### GPT-OSS-120B

- **Model Path**: `/home/kfrost/assets/models/gpt-oss-120b-mxfp4/` (safetensors)
- **Server Port**: 8010
- **Inference Engine**: NVIDIA NIM
- **Format**: Safetensors (MXFP4)
- **Docker Service**: `gpt-oss-120b` in `working-models` stack
- **Status**: ✅ Can be loaded via UI
- **Note**: Can run simultaneously with 20B (different ports)

### Other Models

DeepSeek-NVFP4 and Llama-NIM models are configured but don't support auto-loading through this interface yet. They need to be started manually via their respective deployment scripts.

## Usage Example

1. Open the web UI: `http://localhost:8080`
2. No models are loaded initially - you'll see "No models available" or models with red dots
3. Click on the **GPT-OSS-20B** chip
4. In the modal, click **"🚀 Load Model"**
5. Wait 5-10 seconds for the model to load
6. The status will change to **"Loaded & Ready"** with a green dot
7. Close the modal and start chatting!

## Process Management

- Models are loaded via Docker Compose for better isolation and GPU management
- Docker containers are tracked and managed by the application
- Logs are saved to:
  - Application logs: `logs/{model_id}.log` (startup logs)
  - Container logs: `docker compose logs {service_name}`
- When the application shuts down, all loaded Docker containers are automatically stopped
- If a model container crashes, it can be restarted by clicking Load again

## Multiple Models

✅ **Good news**: GPT-OSS-20B (port 8000) and GPT-OSS-120B (port 8010) use different ports and can run simultaneously! Load both if you have enough GPU memory.

## Troubleshooting

### Model fails to load

1. Check `logs/{model_id}.log` for startup logs
2. Check Docker container logs: `cd /home/kfrost/DxgSpark/deployments/multi-agent-embedding && docker compose logs gpt-oss-20b`
3. Verify the model file exists at the specified path: `ls -lh /home/kfrost/assets/models/gpt-oss-20b-mxfp4/`
4. Ensure Docker is running: `docker ps`
5. Check that no other process is using the model's port: `sudo netstat -tlnp | grep 8000`

### Model shows "Available (Not Loaded)" but won't load

- Verify the `load_command` is correct in `app.py`
- Check file permissions on the model file
- Ensure sufficient GPU memory is available

### Port already in use error

- Unload any other models using the same port first
- Check for running containers: `docker ps`
- Stop conflicting containers: `cd /home/kfrost/DxgSpark/deployments/multi-agent-embedding && docker compose stop`
- Check for stray processes: `sudo netstat -tlnp | grep 8000`

## Adding New Models

To add a new model with auto-loading support:

```python
MODELS = {
    "my-model": {
        "name": "My Model Name",
        "url": f"http://{HOST_IP}:9000",  # Unique port
        "type": "llama.cpp",
        "size": "7B",
        "format": "GGUF",
        "supports_tools": False,
        "model_path": "/path/to/my-model.gguf",
        "load_command": "llama-server --model /path/to/my-model.gguf --port 9000 --host 0.0.0.0 --n-gpu-layers 999"
    }
}
```

Key points:
- Use a unique port for each model
- Set `model_path` to the actual model file location
- Customize `load_command` with appropriate flags for your model
- The `can_load` flag is automatically set based on whether `load_command` exists
