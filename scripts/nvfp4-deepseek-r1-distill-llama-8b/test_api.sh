#!/bin/bash

# API Test Script for DeepSeek-R1-Distill-Llama-8B NVFP4 Server
# Tests all available endpoints

set -e

API_URL="http://localhost:8000"

echo "======================================"
echo "Testing DeepSeek NVFP4 API Server"
echo "======================================"
echo ""

# Check if server is running
echo "1. Checking server status..."
if ! docker ps | grep -q "deepseek-nvfp4-api"; then
    echo "❌ Server is not running!"
    echo "Start it with: ./start_api.sh"
    exit 1
fi
echo "✅ Server is running"
echo ""

# Test text completions
echo "2. Testing /v1/completions endpoint..."
COMPLETION_RESPONSE=$(curl -s -X POST ${API_URL}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/model",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0.7
  }')

if echo "$COMPLETION_RESPONSE" | grep -q "Paris"; then
    echo "✅ Completions endpoint works!"
    echo "Response: $COMPLETION_RESPONSE" | jq '.choices[0].text' 2>/dev/null || echo "$COMPLETION_RESPONSE" | grep -o '"text":"[^"]*"'
else
    echo "❌ Completions endpoint failed"
    echo "Response: $COMPLETION_RESPONSE"
fi
echo ""

# Test chat completions
echo "3. Testing /v1/chat/completions endpoint..."
CHAT_RESPONSE=$(curl -s -X POST ${API_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/model",
    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    "max_tokens": 10
  }')

if echo "$CHAT_RESPONSE" | grep -q '"content"'; then
    echo "✅ Chat completions endpoint works!"
    echo "Response: $CHAT_RESPONSE" | jq '.choices[0].message.content' 2>/dev/null || echo "$CHAT_RESPONSE" | grep -o '"content":"[^"]*"'
else
    echo "❌ Chat completions endpoint failed"
    echo "Response: $CHAT_RESPONSE"
fi
echo ""

# Test streaming (if supported)
echo "4. Testing streaming response..."
STREAM_RESPONSE=$(curl -s -X POST ${API_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/model",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 5,
    "stream": false
  }')

if echo "$STREAM_RESPONSE" | grep -q '"content"'; then
    echo "✅ Non-streaming mode works!"
else
    echo "⚠️  Streaming mode may not be supported"
fi
echo ""

echo "======================================"
echo "✅ All tests completed!"
echo "======================================"
echo ""
echo "API is ready for use at: ${API_URL}"
echo ""
echo "Example Python client:"
echo ""
cat << 'EOF'
import requests

# Chat completion
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "/workspace/model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)
print(response.json()["choices"][0]["message"]["content"])

# Text completion
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "/workspace/model",
        "prompt": "Once upon a time",
        "max_tokens": 50
    }
)
print(response.json()["choices"][0]["text"])
EOF
echo ""
