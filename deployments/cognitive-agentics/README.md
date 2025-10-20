# Cognitive Agentics

GPU-accelerated LLM inference services for .NET applications on DGX Spark.

## Overview

Provides OpenAI-compatible API endpoints for:
- **GPT-OSS-120B** - Main reasoning and financial analysis model
- **BGE-M3** - Document embeddings (1024 dimensions, 8192 token context)

Models are downloaded to and cached in `/home/kfrost/assets/models` for reuse.

## Quick Start

```bash
# Navigate to deployment directory
cd /home/kfrost/DxgSparkDev/deployments/cognitive-agentics

# Make scripts executable (first time only)
chmod +x start.sh stop.sh

# Start services
./start.sh

# Stop services
./stop.sh
```

## First Time Setup

If `NGC_API_KEY` is not in your environment:

```bash
# Option 1: Export temporarily
export NGC_API_KEY='your-key-here'
./start.sh

# Option 2: Create .env file (persists across sessions)
cp .env.example .env
nano .env  # Add your NGC_API_KEY
./start.sh
```

Get your NGC API key from: https://ngc.nvidia.com/setup/api-key

## API Endpoints

### From .NET Applications (External Access)

```csharp
// LLM Endpoint
var llmUrl = "http://192.168.2.180:8000/v1/chat/completions";

// Embeddings Endpoint
var embeddingUrl = "http://192.168.2.180:8001/v1/embeddings";
```

### From Local Spark

```bash
# LLM
curl http://localhost:8000/v1/chat/completions

# Embeddings
curl http://localhost:8001/v1/embeddings
```

## .NET Integration

### Installation

```bash
dotnet add package OpenAI
```

### Chat Completion Example

```csharp
using OpenAI;
using OpenAI.Chat;

var client = new OpenAIClient(
    new ApiKeyCredential("not-needed"),
    new OpenAIClientOptions 
    { 
        Endpoint = new Uri("http://192.168.2.180:8000/v1") 
    }
);

var chatClient = client.GetChatClient("gpt-oss-120b");

var response = await chatClient.CompleteChatAsync(
    new ChatMessage[]
    {
        new SystemChatMessage("You are a financial planning assistant."),
        new UserChatMessage("Analyze retirement scenario for age 35 with $50k saved")
    }
);

Console.WriteLine(response.Value.Content[0].Text);
```

### Embeddings Example

```csharp
using OpenAI.Embeddings;

var embeddingClient = new OpenAIClient(
    new ApiKeyCredential("not-needed"),
    new OpenAIClientOptions 
    { 
        Endpoint = new Uri("http://192.168.2.180:8001/v1") 
    }
).GetEmbeddingClient("bge-m3");

var embedding = await embeddingClient.GenerateEmbeddingAsync(
    "Financial document text to embed"
);

// Returns 1024-dimensional vector
float[] vector = embedding.Value.ToFloats().ToArray();
```

### Streaming Example

```csharp
var streamingResponse = chatClient.CompleteChatStreamingAsync(messages);

await foreach (var update in streamingResponse)
{
    foreach (var contentPart in update.ContentUpdate)
    {
        Console.Write(contentPart.Text);
    }
}
```

### Tool Calling (Function Calling) Example

```csharp
var tools = new List<ChatTool>
{
    ChatTool.CreateFunctionTool(
        functionName: "calculate_retirement",
        functionDescription: "Calculate retirement projections",
        functionParameters: BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "current_age": { "type": "integer" },
                "retirement_age": { "type": "integer" },
                "current_savings": { "type": "number" },
                "annual_contribution": { "type": "number" }
            },
            "required": ["current_age", "retirement_age"]
        }
        """)
    )
};

var options = new ChatCompletionOptions();
tools.ForEach(tool => options.Tools.Add(tool));

var response = await chatClient.CompleteChatAsync(messages, options);

// Handle tool calls in response
foreach (var toolCall in response.Value.ToolCalls)
{
    // Execute your C# function
    // Return result to model
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              .NET Applications                           │
│        (Your financial planning apps)                    │
└────────────────────┬───────────────┬────────────────────┘
                     │               │
            http://192.168.2.180:8000│:8001
                     │               │
         ┌───────────▼──────────┐   ┌▼─────────────────┐
         │   GPT-OSS-120B NIM   │   │   BGE-M3 NIM     │
         │   Port 8000          │   │   Port 8001      │
         │   120B params        │   │   1024 dims      │
         │   65K context        │   │   8192 context   │
         └──────────────────────┘   └──────────────────┘
                     │                      │
                     └──────────┬───────────┘
                                │
                        ┌───────▼────────┐
                        │  DGX Spark GPU  │
                        │  Blackwell Arch │
                        │  128GB Memory   │
                        └─────────────────┘
                                │
                        ┌───────▼────────┐
                        │ Model Storage   │
                        │ /home/kfrost/  │
                        │ assets/models   │
                        └─────────────────┘
```

## Model Storage

All models are stored in: `/home/kfrost/assets/models`

First run downloads:
- **GPT-OSS-120B**: ~60GB (10-30 minutes)
- **BGE-M3**: ~3GB (2-5 minutes)

Subsequent runs use cached models (start in 2-5 minutes).

## System Requirements

- **Memory**: ~78GB GPU memory usage
  - GPT-OSS-120B: ~75GB
  - BGE-M3: ~3GB
  - **Free**: ~50GB for request processing

- **Storage**: ~65GB for cached models

- **Network**: NGC access for initial model download

## Management Commands

```bash
# Start services
./start.sh

# Stop services (unloads from GPU memory)
./stop.sh

# Restart
./stop.sh && ./start.sh

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f llm-supervisor
docker compose logs -f embedding-service

# Check service status
docker compose ps

# Check GPU usage
nvidia-smi
```

## Performance

| Metric | Value |
|--------|-------|
| **LLM Latency** | ~3-5 tokens/sec |
| **Embedding Latency** | ~50-100ms |
| **Max Context** | 65,536 tokens (LLM), 8,192 (embeddings) |
| **Embedding Dimensions** | 1024 |
| **Concurrent Requests** | 2-4 (with batching) |

## Troubleshooting

### Services won't start

```bash
# Check NGC API key
echo $NGC_API_KEY

# If empty, set it:
export NGC_API_KEY='your-key-here'

# Or create .env file
cp .env.example .env
nano .env
```

### First run is slow

This is normal! Models download on first run:
- Monitor progress: `docker compose logs -f`
- GPT-OSS-120B: ~60GB download
- Can take 10-30 minutes depending on network

### Out of memory

```bash
# Clear GPU cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Restart services
./stop.sh && ./start.sh
```

### Can't connect from .NET app

Check firewall allows connections to ports 8000 and 8001:
```bash
sudo ufw allow 8000
sudo ufw allow 8001
```

Test connectivity:
```bash
# From .NET machine
curl http://192.168.2.180:8000/v1/health
curl http://192.168.2.180:8001/v1/health
```

### Models taking too much space

Models are in `/home/kfrost/assets/models`. To free space:
```bash
# Stop services first
./stop.sh

# Remove cached models (will re-download on next start)
rm -rf /home/kfrost/assets/models/*
```

## Advanced Configuration

### Change GPU

Edit `docker-compose.yml`:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Change to 1 for second GPU
```

### Adjust Memory Usage

To reduce context length (saves memory):

Edit `docker-compose.yml` and add to LLM service:
```yaml
environment:
  - MAX_MODEL_LEN=32768  # Reduces from 65K to 32K
```

### Enable Debug Logging

```bash
docker compose logs -f --tail=100
```

## API Documentation

- **OpenAI API Format**: https://platform.openai.com/docs/api-reference
- **NVIDIA NIM Docs**: https://docs.nvidia.com/nim/
- **OpenAI .NET SDK**: https://github.com/openai/openai-dotnet

## Support

- **DGX Spark**: https://build.nvidia.com/spark
- **NGC Catalog**: https://catalog.ngc.nvidia.com/
- **Issues**: Check `docker compose logs` for errors

## License

Apache 2.0
