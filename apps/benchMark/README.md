# DxGChatBenchMark

A beautiful web-based chat interface with real-time performance benchmarking for multiple LLM models running on DGX Spark.

## Features

- 🎯 **Multi-Model Support** - Switch between any deployed models
- 📊 **Real-Time Benchmarks** - See prompt processing and token generation speeds
- 💬 **Clean Chat Interface** - Modern, responsive UI
- ⚡ **Performance Metrics** - Detailed statistics for every response:
  - Prompt tokens & completion tokens
  - Total processing time
  - Prompt processing speed (tokens/sec)
  - Token generation speed (tokens/sec)
  - Overall throughput
- 🔄 **Auto-Discovery** - Automatically detects running models
- ⚙️ **Configurable** - Adjust temperature and max tokens per request

## Quick Start

1. **Start the web app:**
   ```bash
   cd /home/kfrost/DxgSparkDev/apps/benchMark
   chmod +x start.sh
   ./start.sh
   ```

2. **Open in browser:**
   ```
   http://localhost:8080
   ```

3. **Select a model** from the dropdown and start chatting!

## Architecture

```
DxGChatBenchMark/
├── app.py                  # FastAPI backend server
├── requirements.txt        # Python dependencies
├── start.sh               # Startup script
├── static/
│   └── index.html         # Web UI (HTML/CSS/JavaScript)
└── README.md              # This file
```

### How It Works

1. **Backend (FastAPI):**
   - Proxies requests to model APIs (TensorRT-LLM, NIM, vLLM, etc.)
   - Measures performance metrics (timing, token counts)
   - Provides REST API for frontend

2. **Frontend (Vanilla JS):**
   - Modern chat interface
   - Real-time benchmark visualization
   - Model selector with health status
   - Conversation history management

3. **Model Discovery:**
   - Automatically checks configured model endpoints
   - Shows online/offline status
   - Supports any OpenAI-compatible API

## Adding New Models

Edit `app.py` and add your model to the `MODELS` dictionary:

```python
MODELS = {
    "your-model-id": {
        "name": "Your Model Name",
        "url": "http://localhost:PORT",
        "type": "NIM/TensorRT-LLM/vLLM",
        "size": "8B/70B/etc",
        "format": "FP16/NVFP4/etc"
    },
}
```

The app will automatically detect if the model is online and add it to the dropdown.

## Performance Metrics Explained

- **Prompt Tokens:** Number of tokens in your input
- **Completion Tokens:** Number of tokens generated
- **Total Time:** End-to-end response time
- **Prefill Speed:** How fast the model processes your prompt (tokens/sec)
- **Generation Speed:** How fast the model generates new tokens (tokens/sec)
- **Throughput:** Overall tokens per second (prompt + completion)

## API Endpoints

- `GET /` - Web interface
- `GET /api/models` - List available models with status
- `POST /api/chat` - Send chat request and get benchmarked response
- `GET /api/health` - Health check

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- httpx
- At least one running LLM API endpoint

## Troubleshooting

**No models showing up:**
- Check that your model servers are running
- Verify the URLs in `app.py` match your deployments
- Check firewall/network settings

**Slow responses:**
- Normal! Benchmarking shows you exactly how fast your model is
- Check GPU memory usage
- Consider optimizing KV cache settings

**Connection errors:**
- Ensure model APIs are accessible from the server
- Check logs with `docker logs <container-name>`

## Example Usage

```bash
# Start the DeepSeek NVFP4 model
cd /home/kfrost/DxgSparkDev/scripts/nvfp4-deepseek-r1-distill-llama-8b
./start_api.sh

# Start the benchmark UI
cd /home/kfrost/DxgSparkDev/apps/benchMark
./start.sh

# Open browser to http://localhost:8080
# Select "DeepSeek-R1-Distill-Llama-8B (NVFP4)" from dropdown
# Start chatting and see real-time benchmarks!
```

## License

Part of the DxgSparkDev project.
