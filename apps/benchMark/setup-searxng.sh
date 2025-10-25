#!/bin/bash
# Complete setup and startup script for DxG Benchmark + SearxNG

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🔍 DxG Benchmark + SearxNG Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Change to script directory
cd "$(dirname "$0")"

# Check if secret key needs to be generated
if grep -q "replace-with-random-key-in-production" searxng/settings.yml 2>/dev/null; then
    echo -e "${YELLOW}📝 Generating SearxNG secret key...${NC}"
    SECRET_KEY=$(openssl rand -hex 32)
    sed -i "s/replace-with-random-key-in-production/$SECRET_KEY/" searxng/settings.yml
    echo -e "${GREEN}✅ Secret key generated and updated${NC}"
else
    echo -e "${GREEN}✅ Secret key already configured${NC}"
fi
echo

# Stop any existing containers
echo -e "${YELLOW}🛑 Stopping any existing containers...${NC}"
docker compose down 2>/dev/null || true
echo

# Build the benchmark container
echo -e "${YELLOW}🔨 Building benchmark application...${NC}"
docker compose build benchmark
echo

# Start all services
echo -e "${YELLOW}🐳 Starting Docker services...${NC}"
echo "   📦 SearxNG (metasearch engine) - Port 8888"
echo "   📦 Redis (cache) - Port 6379"
echo "   📦 Benchmark App (FastAPI) - Port 8080"
echo

docker compose up -d

echo
echo -e "${YELLOW}⏳ Waiting for services to initialize...${NC}"
sleep 8

# Check Redis
echo
echo -e "${YELLOW}🔍 Checking Redis...${NC}"
if docker exec dgx-searxng-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is ready!${NC}"
else
    echo -e "${RED}❌ Redis is not responding${NC}"
fi

# Check SearxNG
echo
echo -e "${YELLOW}🔍 Checking SearxNG...${NC}"
SEARXNG_READY=false
for i in {1..15}; do
    if curl -s http://localhost:8888/healthz > /dev/null 2>&1; then
        echo -e "${GREEN}✅ SearxNG is ready!${NC}"
        SEARXNG_READY=true
        break
    fi
    echo "   Waiting... ($i/15)"
    sleep 2
done

if [ "$SEARXNG_READY" = false ]; then
    echo -e "${RED}❌ SearxNG failed to start. Check logs: docker compose logs searxng${NC}"
fi

# Check Benchmark App
echo
echo -e "${YELLOW}🔍 Checking Benchmark App...${NC}"
sleep 3
if curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Benchmark App is ready!${NC}"
else
    echo -e "${YELLOW}⚠️  Benchmark App may still be starting...${NC}"
fi

# Show container status
echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo -e "${YELLOW}📊 Service Status:${NC}"
docker compose ps
echo
echo -e "${YELLOW}🌐 Access URLs:${NC}"
echo "   🖥️  Benchmark App:  http://localhost:8080"
echo "   🔍 SearxNG Web UI: http://localhost:8888"
echo "   📡 SearxNG API:    http://localhost:8888/search?q=test&format=json"
echo "   💾 Redis:          localhost:6379"
echo
echo -e "${YELLOW}📖 Quick Commands:${NC}"
echo "   📜 View logs:       docker compose logs -f"
echo "   📜 View app logs:   docker compose logs -f benchmark"
echo "   🔄 Restart:         docker compose restart"
echo "   🛑 Stop all:        ./stop.sh  (or docker compose down)"
echo "   🗑️  Clean up:        docker compose down -v"
echo
echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo "   1. Visit http://localhost:8080 to use the chat interface"
echo "   2. Enable 'RAG' toggle to activate web search"
echo "   3. Ask questions requiring real-time data (weather, news, etc.)"
echo
echo -e "${YELLOW}📚 Documentation:${NC}"
echo "   • SEARXNG_SETUP.md - SearxNG configuration details"
echo "   • SEARXNG_INTEGRATION.md - Integration overview"
echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
