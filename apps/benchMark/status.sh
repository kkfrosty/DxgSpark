#!/bin/bash
# Check status of all DxG Benchmark services

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd "$(dirname "$0")"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 DxG Benchmark Service Status${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Check if docker compose is running
echo -e "${YELLOW}🐳 Docker Container Status:${NC}"
docker compose ps
echo

# Check individual services
echo -e "${YELLOW}🔍 Service Health Checks:${NC}"
echo

# Check Redis
echo -n "   Redis (Cache):         "
if docker exec dgx-searxng-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Running${NC}"
else
    echo -e "${RED}❌ Not responding${NC}"
fi

# Check SearxNG
echo -n "   SearxNG (Search):      "
if curl -s http://localhost:8888/healthz > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Running${NC}"
else
    echo -e "${RED}❌ Not responding${NC}"
fi

# Check Benchmark App
echo -n "   Benchmark App (API):   "
if curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Running${NC}"
else
    echo -e "${RED}❌ Not responding${NC}"
fi

echo

# Show resource usage
echo -e "${YELLOW}💾 System Resources:${NC}"
echo
free -h | grep -E "Mem:|Swap:"
echo

echo -e "${YELLOW}🐳 Docker Resource Usage:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" dgx-chat-benchmark dgx-searxng dgx-searxng-redis 2>/dev/null || echo "   (Containers not running)"
echo

# Show recent logs summary
echo -e "${YELLOW}📜 Recent Activity:${NC}"
echo -e "${BLUE}━━ Benchmark App (last 5 lines) ━━${NC}"
docker compose logs --tail=5 benchmark 2>/dev/null || echo "   (No logs available)"
echo
echo -e "${BLUE}━━ SearxNG (last 5 lines) ━━${NC}"
docker compose logs --tail=5 searxng 2>/dev/null || echo "   (No logs available)"
echo

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🌐 Access URLs:${NC}"
echo "   🖥️  Benchmark App:  http://localhost:8080"
echo "   🔍 SearxNG Web UI: http://localhost:8888"
echo "   📡 SearxNG API:    http://localhost:8888/search?q=test&format=json"
echo
echo -e "${YELLOW}📖 Quick Commands:${NC}"
echo "   📜 View all logs:   docker compose logs -f"
echo "   🔄 Restart:         docker compose restart"
echo "   🛑 Stop all:        ./stop.sh"
echo "   🚀 Start:           ./setup-searxng.sh"
echo
