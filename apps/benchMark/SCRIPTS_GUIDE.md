# 🚀 Quick Start Guide - DxG Benchmark Scripts

## Available Scripts

All scripts are located in `/home/kfrost/DxgSparkDev/apps/benchMark/`

### 🟢 Start Services

```bash
./start.sh
# or
./setup-searxng.sh
```

**What it does:**
- ✅ Generates secure SearxNG secret key (first run only)
- ✅ Stops any existing containers
- ✅ Builds the benchmark Docker image
- ✅ Starts all 3 containers (App, SearxNG, Redis)
- ✅ Waits for services to be ready
- ✅ Shows health status and access URLs

**Services started:**
- `dgx-chat-benchmark` - FastAPI app on port 8080
- `dgx-searxng` - SearxNG search engine on port 8888
- `dgx-searxng-redis` - Redis cache on port 6379

---

### 🔴 Stop Services

```bash
./stop.sh
```

**What it does:**
- 🛑 Stops all running containers
- 🧹 Cleans up stopped containers
- 💾 Shows memory usage before/after
- 📊 Displays Docker resource usage

**For deep cleanup (also clears cache):**
```bash
sudo ./stop.sh
```

---

### 📊 Check Status

```bash
./status.sh
```

**What it shows:**
- ✅ Health status of each service
- 💾 Memory and CPU usage
- 🐳 Docker container stats
- 📜 Recent log activity
- 🌐 Access URLs

---

### 🗑️ Complete Cleanup

```bash
# Stop containers and remove volumes (⚠️ deletes data!)
docker-compose down -v

# Stop containers and remove images
docker-compose down --rmi all

# Nuclear option - clean entire Docker system
docker system prune -a --volumes
```

---

## Typical Workflow

### First Time Setup
```bash
cd /home/kfrost/DxgSparkDev/apps/benchMark
./setup-searxng.sh
```

### Daily Use
```bash
# Start
./start.sh

# Check if running
./status.sh

# View logs
docker-compose logs -f

# Stop when done
./stop.sh
```

### Troubleshooting
```bash
# Check what's wrong
./status.sh

# View specific service logs
docker-compose logs -f benchmark
docker-compose logs -f searxng
docker-compose logs -f redis

# Restart a specific service
docker-compose restart benchmark
docker-compose restart searxng

# Full restart
./stop.sh
./start.sh
```

---

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Benchmark App** | http://localhost:8080 | Main chat interface |
| **API Docs** | http://localhost:8080/docs | FastAPI Swagger UI |
| **Health Check** | http://localhost:8080/api/health | App health status |
| **SearxNG Web** | http://localhost:8888 | Search engine UI |
| **SearxNG API** | http://localhost:8888/search?q=test&format=json | JSON search API |
| **Redis** | localhost:6379 | Cache (internal) |

---

## Useful Docker Commands

```bash
# View all containers
docker-compose ps

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f benchmark
docker-compose logs -f searxng

# Restart specific service
docker-compose restart benchmark

# Restart all services
docker-compose restart

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build

# Execute command in container
docker exec -it dgx-chat-benchmark bash
docker exec -it dgx-searxng sh

# Check Redis
docker exec dgx-searxng-redis redis-cli ping

# Monitor resource usage
docker stats
```

---

## Environment Variables

You can customize behavior by setting these in `docker-compose.yml`:

```yaml
environment:
  - SEARXNG_URL=http://localhost:8888  # SearxNG API URL
  - DATABASE_URL=sqlite:///./data/benchmark.db  # Database location
  - PYTHONUNBUFFERED=1  # Python logging
```

---

## File Locations

```
/home/kfrost/DxgSparkDev/apps/benchMark/
├── setup-searxng.sh     # 🟢 Main setup/start script
├── start.sh             # 🟢 Quick start (calls setup-searxng.sh)
├── stop.sh              # 🔴 Stop all services
├── status.sh            # 📊 Check service health
├── docker-compose.yml   # 🐳 Docker configuration
├── Dockerfile           # 🐳 App container build
├── app.py               # 🐍 Main FastAPI application
├── requirements.txt     # 📦 Python dependencies
├── searxng/
│   ├── settings.yml     # ⚙️  SearxNG configuration
│   └── limiter.toml     # ⚙️  Rate limit settings
├── data/                # 💾 Database and logs
└── static/              # 🌐 Web UI files
```

---

## Quick Tips

### 💡 RAG Search Not Working?
1. Check SearxNG: `curl http://localhost:8888/healthz`
2. View logs: `docker-compose logs -f searxng`
3. Try manual search: http://localhost:8888
4. Restart: `docker-compose restart searxng`

### 💡 Port Already in Use?
```bash
# Find what's using port 8080
sudo lsof -i :8080

# Kill the process
sudo kill -9 <PID>

# Or change port in docker-compose.yml
```

### 💡 Out of Memory?
```bash
# Stop services
./stop.sh

# Check Docker usage
docker system df

# Clean up
docker system prune -a

# Clear system cache (requires sudo)
sudo ./stop.sh
```

### 💡 Slow Performance?
1. Check system resources: `./status.sh`
2. Check if LLM models are running (they use lots of RAM)
3. Restart services: `docker-compose restart`
4. Check Redis cache: `docker exec dgx-searxng-redis redis-cli INFO stats`

---

## Support & Documentation

- 📚 **SearxNG Setup**: `SEARXNG_SETUP.md`
- 📚 **Integration Guide**: `SEARXNG_INTEGRATION.md`
- 📚 **Docker Guide**: `DOCKER_README.md`
- 🌐 **SearxNG Docs**: https://docs.searxng.org/

---

## Emergency Commands

```bash
# Everything is broken, start fresh
./stop.sh
docker-compose down -v
docker system prune -f
./setup-searxng.sh

# Can't stop containers
docker kill dgx-chat-benchmark dgx-searxng dgx-searxng-redis
docker rm -f dgx-chat-benchmark dgx-searxng dgx-searxng-redis

# Port conflicts
docker-compose down
sudo lsof -ti:8080 | xargs kill -9
sudo lsof -ti:8888 | xargs kill -9
```
