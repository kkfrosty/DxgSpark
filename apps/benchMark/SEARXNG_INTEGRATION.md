# SearxNG Integration Summary

## What We've Done

We've integrated **SearxNG** - a self-hosted metasearch engine - into your DxG Benchmark application to solve the web search/RAG issues you were experiencing with DuckDuckGo and Playwright.

## Architecture

**3 Docker Containers** (all managed via docker-compose):

1. **dgx-chat-benchmark** - Your main FastAPI application (port 8080)
2. **dgx-searxng** - SearxNG metasearch engine (port 8888)
3. **dgx-searxng-redis** - Redis cache for SearxNG performance

## Files Created/Modified

### New Files:
- `searxng/settings.yml` - SearxNG configuration (engines, timeouts, etc.)
- `searxng/limiter.toml` - Rate limiting configuration
- `SEARXNG_SETUP.md` - Complete documentation
- `setup-searxng.sh` - Quick setup script

### Modified Files:
- `docker-compose.yml` - Added SearxNG and Redis services
- `app.py` - Replaced Playwright/DuckDuckGo with SearxNG API calls
- `requirements.txt` - Removed playwright and duckduckgo-search dependencies
- `Dockerfile` - Removed Playwright browser installation (much smaller image!)

## Key Changes in app.py

**Before:**
- Used Playwright to scrape Google (slow, blocked by bot detection)
- Used DuckDuckGo API (rate-limited, often returns no results)

**After:**
- Uses SearxNG JSON API at `http://localhost:8888/search`
- SearxNG aggregates results from multiple engines (Google, Bing, DDG, Brave)
- Results are cached for 5 minutes to avoid redundant searches
- Much more reliable and faster

## How to Start

### Option 1: Quick Setup (Recommended)
```bash
cd /home/kfrost/DxgSparkDev/apps/benchMark
./setup-searxng.sh
```

### Option 2: Manual Setup
```bash
cd /home/kfrost/DxgSparkDev/apps/benchMark

# Generate a secure secret key
SECRET_KEY=$(openssl rand -hex 32)
sed -i "s/replace-with-random-key-in-production/$SECRET_KEY/" searxng/settings.yml

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

## Testing the Integration

### 1. Test SearxNG directly:
```bash
curl "http://localhost:8888/search?q=test&format=json" | jq .
```

### 2. Test in the web UI:
1. Open http://localhost:8080
2. Toggle "Enable RAG" on
3. Ask a question like "What is the weather in New York?"
4. You should see web search results being used

### 3. Check logs for search activity:
```bash
docker-compose logs -f benchmark
# Look for messages like:
# "Searching web via SearxNG for: ..."
# "SearxNG returned 5 results"
```

## Benefits Over Previous Approach

| Feature | Playwright/Google | DuckDuckGo API | **SearxNG** |
|---------|------------------|----------------|-------------|
| Reliability | ❌ Often blocked | ❌ Rate limited | ✅ Stable |
| Speed | ❌ Slow (browser) | ⚠️ Moderate | ✅ Fast (API) |
| Multiple Sources | ❌ Google only | ❌ DDG only | ✅ 4+ engines |
| Caching | ❌ No | ❌ No | ✅ Redis |
| Maintenance | ❌ High | ⚠️ Medium | ✅ Low |
| Docker Image Size | ❌ Large (~1GB) | ✅ Small | ✅ Small |

## Enabled Search Engines

In `searxng/settings.yml`, these engines are enabled:
- **Google** (shortcut: `!go`)
- **Bing** (shortcut: `!bi`)
- **DuckDuckGo** (shortcut: `!ddg`)
- **Brave** (shortcut: `!br`)
- **Wikipedia** (shortcut: `!wp`)
- **Stack Overflow** (shortcut: `!so`)
- **GitHub** (shortcut: `!gh`)

You can add more or disable some by editing the settings file.

## Network Configuration

Since your benchmark app uses **host networking** to access local LLM models:
- The app accesses SearxNG at `http://localhost:8888`
- SearxNG and Redis run on the bridge network
- All services can communicate properly

## Monitoring

### View all service status:
```bash
docker-compose ps
```

### Check SearxNG health:
```bash
curl http://localhost:8888/healthz
```

### View search statistics in web UI:
Open http://localhost:8888/stats/errors

## Troubleshooting

### SearxNG not responding:
```bash
docker-compose restart searxng
docker-compose logs searxng
```

### No search results:
1. Check SearxNG web UI: http://localhost:8888
2. Try a manual search there
3. Check engine status: http://localhost:8888/preferences

### App can't connect to SearxNG:
```bash
# From inside the benchmark container:
docker exec dgx-chat-benchmark curl http://localhost:8888/healthz
```

## Customization

### Add more search engines:
Edit `searxng/settings.yml` and add to the `engines:` section. See available engines at:
https://docs.searxng.org/admin/engines/index.html

### Adjust cache duration:
In `app.py`, change `_search_cache_ttl = 300` (currently 5 minutes)

### Enable authentication:
Edit `searxng/settings.yml`:
```yaml
server:
  limiter: true
  # Then configure HTTP basic auth or use a reverse proxy
```

## Next Steps

1. **Run the setup**: `./setup-searxng.sh`
2. **Test it**: Open http://localhost:8080 and enable RAG
3. **Monitor it**: `docker-compose logs -f`
4. **Customize it**: Edit `searxng/settings.yml` as needed

## Resources

- SearxNG documentation: https://docs.searxng.org/
- Your detailed setup guide: `SEARXNG_SETUP.md`
- Docker compose reference: `docker-compose.yml`
