#!/bin/bash
#
# Cognitive Agentics - Start LLM Services
#

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠 Cognitive Agentics - Starting LLM Services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for NGC API key
if [ -z "$NGC_API_KEY" ]; then
    echo "❌ Error: NGC_API_KEY not found in environment"
    echo ""
    echo "The key should be set in ~/.bashrc"
    echo "If you need to set it, run:"
    echo "  echo 'export NGC_API_KEY=\"your-key\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo ""
    echo "Get your key from: https://ngc.nvidia.com/setup/api-key"
    exit 1
fi

echo "✅ NGC_API_KEY configured"
echo ""

# Verify model storage directory
if [ ! -d "/home/kfrost/assets/models" ]; then
    echo "❌ Error: Model directory not found at /home/kfrost/assets/models"
    exit 1
fi
echo "✅ Model storage: /home/kfrost/assets/models"
echo ""

# Check if services are already running
if docker ps | grep -q "cognitive-llm-supervisor\|cognitive-embeddings"; then
    echo "⚠️  Services are already running!"
    echo ""
    echo "To restart, first run: ./stop.sh"
    echo "Then run: ./start.sh"
    exit 1
fi

# Start services
echo "🚀 Starting Docker services..."
echo ""
echo "⏳ First run will download models:"
echo "   • GPT-OSS-120B: ~60GB (10-30 minutes)"
echo "   • BGE-M3: ~3GB (2-5 minutes)"
echo ""
echo "   Models will be cached in /home/kfrost/assets/models"
echo "   Subsequent runs will start in 2-5 minutes"
echo ""

docker compose up -d

echo ""
echo "⏳ Waiting for services to initialize..."
echo "   This may take 15-20 minutes on first run"
echo ""

# Monitor startup
timeout=1200  # 20 minutes
elapsed=0
check_interval=10

while [ $elapsed -lt $timeout ]; do
    # Check if containers exist
    llm_exists=$(docker ps -a --format '{{.Names}}' | grep -c "cognitive-llm-supervisor" || echo "0")
    embed_exists=$(docker ps -a --format '{{.Names}}' | grep -c "cognitive-embeddings" || echo "0")
    
    if [ "$llm_exists" == "0" ] || [ "$embed_exists" == "0" ]; then
        echo "⚠️  Containers not found, check docker compose logs"
        break
    fi
    
    # Check health status
    llm_health=$(docker inspect --format='{{.State.Health.Status}}' cognitive-llm-supervisor 2>/dev/null || echo "starting")
    embed_health=$(docker inspect --format='{{.State.Health.Status}}' cognitive-embeddings 2>/dev/null || echo "starting")
    
    # Success condition
    if [ "$llm_health" == "healthy" ] && [ "$embed_health" == "healthy" ]; then
        echo ""
        echo "✅ All services are healthy!"
        break
    fi
    
    # Show progress
    echo -ne "\r⏳ LLM: $llm_health | Embeddings: $embed_health | Elapsed: ${elapsed}s    "
    
    sleep $check_interval
    elapsed=$((elapsed + check_interval))
done

if [ $elapsed -ge $timeout ]; then
    echo ""
    echo "⚠️  Services did not become healthy within ${timeout}s"
    echo ""
    echo "Check logs with:"
    echo "   docker compose logs -f llm-supervisor"
    echo "   docker compose logs -f embedding-service"
    echo ""
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Cognitive Agentics is Ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📡 API Endpoints:"
echo "   Internal: http://localhost:8000 (LLM)"
echo "   Internal: http://localhost:8001 (Embeddings)"
echo ""
echo "   External: http://192.168.2.180:8000/v1/chat/completions"
echo "   External: http://192.168.2.180:8001/v1/embeddings"
echo ""
echo "💾 Models stored in: /home/kfrost/assets/models"
echo ""
echo "🔧 Management:"
echo "   View logs:  docker compose logs -f"
echo "   Stop:       ./stop.sh"
echo "   Restart:    ./stop.sh && ./start.sh"
echo ""
echo "🧪 Quick Test:"
echo '   curl -X POST http://192.168.2.180:8000/v1/chat/completions \'
echo '     -H "Content-Type: application/json" \'
echo '     -d '"'"'{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'"'"
echo ""
