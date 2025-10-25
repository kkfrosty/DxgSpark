# DxGChatBenchMark - Docker Setup

## 🚀 Quick Start

### Prerequisites
- Docker installed
- Docker Compose installed
- Your LLM model running on the host (e.g., llama.cpp on port 8000)

### Start the Application

```bash
# Navigate to the app directory
cd /home/kfrost/DxgSparkDev/apps/benchMark

# Build and start the container
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

The application will be available at: **http://localhost:8080**

### Stop the Application

```bash
docker-compose down
```

## 📦 What's Included

- **FastAPI Server**: Chat & benchmark interface
- **Playwright**: Real browser (Chromium) for web searching
- **PostgreSQL**: Optional database for chat logs (disabled by default)
- **Hot Reload**: Code changes auto-reload in development

## 🔧 Configuration

### Using Host Network Mode (Default)

The container uses `network_mode: host` to access your local LLM models:
- Your models on `localhost:8000` are accessible from the container
- Port mapping is handled by host network

### Using PostgreSQL (Optional)

To enable PostgreSQL for chat history:

```bash
# Start with PostgreSQL
docker-compose --profile with-postgres up -d
```

Edit `docker-compose.yml` and set:
```yaml
environment:
  - USE_POSTGRES=true
```

## 📊 Features

### Browser-Based Web Search
- Uses real Chromium browser via Playwright
- Bypasses bot detection (202 rate limits)
- Works like a human browsing DuckDuckGo
- Automatic caching (5-minute TTL)

### Performance Monitoring
- Real-time token generation speed
- Time to first token (TTFT)
- System resource usage
- WebSocket-based live updates

### Multi-Model Support
Add your models in `app.py`:
```python
MODELS = {
    "gpt-oss-120b": {
        "name": "GPT-OSS-120B (MXFP4)",
        "url": "http://localhost:8000"
    },
    # Add more models here
}
```

## 🛠️ Development

### Hot Reload
Code changes are automatically detected and reload the server:
```bash
# Edit files locally, changes apply immediately
vim app.py
# Server reloads automatically in container
```

### View Logs
```bash
# Follow logs
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100
```

### Rebuild After Dependency Changes
```bash
# If you modify requirements.txt
docker-compose down
docker-compose up --build
```

## 📁 Data Persistence

### SQLite Database
Located at `./data/` (persisted across restarts)

### Playwright Browsers
Cached in Docker volume (no re-download needed)

## 🐛 Troubleshooting

### Browser Launch Fails
If Playwright can't launch browser:
```bash
# Rebuild with fresh browsers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Can't Access Local Model
If container can't reach your model on port 8000:
1. Check model is running: `curl http://localhost:8000/v1/models`
2. Verify `network_mode: host` is set in `docker-compose.yml`

### Port 8080 Already in Use
```bash
# Find what's using port 8080
sudo lsof -i :8080

# Kill the existing server
pkill -f uvicorn

# Or change port in docker-compose.yml
```

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | `1` | Show logs immediately |
| `USE_POSTGRES` | `false` | Use PostgreSQL instead of SQLite |
| `DATABASE_URL` | SQLite | PostgreSQL connection string |
| `PLAYWRIGHT_BROWSERS_PATH` | `/ms-playwright` | Browser cache location |

## 🔒 Security Notes

- Container runs with minimal privileges
- Browser runs in sandbox mode
- No root access required
- All traffic stays on host network

## 📈 Performance

**First Build**: ~5 minutes (downloads Chromium ~170MB)  
**Subsequent Builds**: ~30 seconds (uses cache)  
**Startup Time**: ~2 seconds  
**Search Latency**: 2-4 seconds (real browser)

## 🆘 Support

Issues? Check:
1. Docker version: `docker --version` (need 20.10+)
2. Container logs: `docker-compose logs`
3. Browser status: Check for Playwright errors in logs
