#!/bin/bash
#
# Cognitive Agentics - Stop LLM Services
#

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛑 Cognitive Agentics - Stopping LLM Services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if services are running
if ! docker ps | grep -q "cognitive-llm-supervisor\|cognitive-embeddings"; then
    echo "ℹ️  Services are not currently running"
    echo ""
    exit 0
fi

# Stop and remove containers
echo "🔄 Stopping Docker services..."
docker compose down

echo ""
echo "✅ Services stopped and removed from memory"
echo ""
echo "💾 Data preserved:"
echo "   Models remain in: /home/kfrost/assets/models"
echo "   No data was lost"
echo ""
echo "To restart services, run: ./start.sh"
echo ""
