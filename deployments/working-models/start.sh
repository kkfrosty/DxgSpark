#!/usr/bin/env bash
#
# Unified Model Deployment Setup
# Sets up NVIDIA NIM-based model serving with consistent interface
#

set -euo pipefail

cd "$(dirname "$0")"

echo "============================================"
echo "  Unified Model Deployment - NVIDIA NIM"
echo "============================================"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "📝 IMPORTANT: You must add your NGC API key to .env file"
    echo ""
    echo "Steps:"
    echo "  1. Get your NGC API key from: https://ngc.nvidia.com/setup/api-key"
    echo "  2. Edit .env file: nano .env"
    echo "  3. Replace 'NGC_API_KEY=' with your actual key"
    echo "  4. Run this script again: ./start.sh"
    echo ""
    exit 1
fi

# Load environment variables from .env
export $(grep -v '^#' .env | xargs)

# Check if NGC API key is actually set
if [ -z "${NGC_API_KEY:-}" ] || [ "${NGC_API_KEY}" = "your_ngc_api_key_here" ]; then
    echo "⚠️  NGC_API_KEY not configured in .env file!"
    echo ""
    echo "Please edit .env and add your NGC API key:"
    echo "  nano .env"
    echo ""
    echo "Get your key from: https://ngc.nvidia.com/setup/api-key"
    echo ""
    exit 1
fi

echo "✅ NGC API key loaded from .env file"
echo ""

# Check if model directories exist
echo "Checking model files..."
if [ ! -d "/home/kfrost/assets/models/gpt-oss-20b-mxfp4" ]; then
    echo "⚠️  GPT-OSS-20B directory not found"
fi

if [ ! -d "/home/kfrost/assets/models/gpt-oss-120b-mxfp4" ]; then
    echo "⚠️  GPT-OSS-120B directory not found"
fi

# Start only the embedding service by default
echo ""
echo "Starting shared embedding service..."
docker compose up -d bge-embedding

echo ""
echo "✅ Deployment ready!"
echo ""
echo "📋 Available Services:"
echo "  • BGE Embeddings:  http://localhost:8001 (RUNNING)"
echo "  • GPT-OSS-20B:     http://localhost:8000 (load via UI)"
echo "  • GPT-OSS-120B:    http://localhost:8010 (load via UI)"
echo ""
echo "🎮 Model Management:"
echo "  • Use the benchMark UI to load/unload models on-demand"
echo "  • UI: http://localhost:8080"
echo ""
echo "🔧 Manual Commands:"
echo "  Load 20B:    docker compose up -d gpt-oss-20b"
echo "  Load 120B:   docker compose up -d gpt-oss-120b"
echo "  Unload 20B:  docker compose stop gpt-oss-20b"
echo "  Unload 120B: docker compose stop gpt-oss-120b"
echo "  View logs:   docker compose logs -f [service-name]"
echo ""
echo "📊 Check Status:"
echo "  docker compose ps"
echo ""
