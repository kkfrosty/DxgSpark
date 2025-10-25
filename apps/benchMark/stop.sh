#!/bin/bash
# Stop the DxGChatBenchMark FastAPI server and clean up the venv if needed

cd "$(dirname "$0")"

PIDFILE=".dxgchatbenchmark.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if ps -p $PID > /dev/null 2>&1; then
        #!/bin/bash
# Stop all DxG Benchmark containers and free memory

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${RED}🛑 Stopping DxG Benchmark Services${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Change to script directory
cd "$(dirname "$0")"

# Show current container status
echo -e "${YELLOW}📊 Current container status:${NC}"
docker compose ps
echo

# Get memory usage before stopping
echo -e "${YELLOW}💾 Memory usage before shutdown:${NC}"
free -h | grep -E "Mem:|Swap:"
echo

# Stop all containers
echo -e "${YELLOW}🛑 Stopping containers...${NC}"
docker compose down

echo -e "${GREEN}✅ Containers stopped${NC}"
echo

# Optional: Remove stopped containers and clean up
echo -e "${YELLOW}🧹 Cleaning up stopped containers...${NC}"
docker container prune -f > /dev/null 2>&1 || true

# Clear system cache to free memory (requires sudo)
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}🧹 Clearing system cache...${NC}"
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo -e "${GREEN}✅ System cache cleared${NC}"
else
    echo -e "${YELLOW}ℹ️  Run with sudo to clear system cache: sudo ./stop.sh${NC}"
fi

echo

# Show memory after cleanup
echo -e "${YELLOW}💾 Memory usage after shutdown:${NC}"
free -h | grep -E "Mem:|Swap:"
echo

# Show docker memory usage
echo -e "${YELLOW}🐳 Docker resource usage:${NC}"
docker system df
echo

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Shutdown Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo -e "${YELLOW}📖 Additional cleanup commands:${NC}"
echo "   🗑️  Remove volumes:        docker compose down -v"
echo "   🗑️  Remove images:         docker compose down --rmi all"
echo "   🧹 Clean Docker system:   docker system prune -a"
echo "   🚀 Restart services:      ./setup-searxng.sh"
echo

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
