#!/bin/bash
# Stop the DxGChatBenchMark FastAPI server and clean up the venv if needed

cd "$(dirname "$0")"

PIDFILE=".dxgchatbenchmark.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping DxGChatBenchMark server (PID $PID)..."
        kill $PID
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            echo "Force killing..."
            kill -9 $PID
        fi
        echo "✅ Server stopped."
    else
        echo "No running server found (PID $PID not active)."
    fi
    rm -f "$PIDFILE"
else
    # Fallback: try to kill uvicorn on port 8080
    PID=$(lsof -ti:8080)
    if [ -n "$PID" ]; then
        echo "Killing process on port 8080 (PID $PID)..."
        kill $PID
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID
        fi
        echo "✅ Server stopped."
    else
        echo "No running server found on port 8080."
    fi
fi

# Optionally clean up venv (uncomment if you want to remove it)
# rm -rf venv
