#!/bin/bash

# Start DxGChatBenchMark Web Application

set -e

cd "$(dirname "$0")"

echo "========================================"
echo "🚀 DxGChatBenchMark Startup"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "✅ Dependencies installed"
echo ""
echo "Starting DxGChatBenchMark server..."
echo ""
echo "========================================"
echo "📊 Server Information"
echo "========================================"
echo "Web Interface: http://localhost:8080"
echo "API Docs:      http://localhost:8080/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start the application
python3 app.py
