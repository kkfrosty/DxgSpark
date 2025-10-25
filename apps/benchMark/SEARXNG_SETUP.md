# SearxNG Setup for DxG Benchmark

This project uses a self-hosted SearxNG instance for web search functionality in the RAG (Retrieval-Augmented Generation) system.

## What is SearxNG?

SearxNG is a privacy-respecting metasearch engine that aggregates results from multiple search engines (Google, Bing, DuckDuckGo, Brave, etc.) and provides them through a clean API. By self-hosting it, we avoid rate limits and bot detection from individual search engines.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌──────────────┐
│  Benchmark App  │────────▶│    SearxNG      │────────▶│ Google/Bing  │
│  (Port 8080)    │         │  (Port 8888)    │         │ DuckDuckGo   │
│                 │         │                 │         │ Brave, etc.  │
└─────────────────┘         └─────────────────┘         └──────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  Redis Cache    │
                            │  (Port 6379)    │
                            └─────────────────┘
```

## Configuration Files

### `searxng/settings.yml`
Main configuration file for SearxNG:
- **Enabled engines**: Google, Bing, DuckDuckGo, Brave, Wikipedia, Stack Overflow, GitHub
- **Disabled engines**: Videos, images, news, social media (reduces noise for text-based RAG)
- **Timeouts**: 10s request timeout, 15s max
- **Rate limiting**: Disabled for local use
- **Redis cache**: Enabled for performance

### `searxng/limiter.toml`
Rate limiting configuration:
- Set to very high limits (200 requests/hour per IP) for local use
- Bot detection checks disabled for API usage

## Environment Variables

The app automatically detects SearxNG at:
- `SEARXNG_URL=http://localhost:8888` (when using host networking)
- `SEARXNG_URL=http://searxng:8888` (when using bridge networking)

## Starting the Services

```bash
# Start all services (SearxNG + Redis + Benchmark App)
docker-compose up -d

# View logs
docker-compose logs -f searxng

# Check SearxNG health
curl http://localhost:8888/healthz

# Test a search
curl "http://localhost:8888/search?q=test&format=json"
```

## Accessing SearxNG

- **Web UI**: http://localhost:8888
- **JSON API**: http://localhost:8888/search?q=YOUR_QUERY&format=json
- **Health Check**: http://localhost:8888/healthz

## Customizing Search Engines

Edit `searxng/settings.yml` to enable/disable engines:

```yaml
engines:
  - name: google
    engine: google
    disabled: false  # Set to true to disable

  - name: custom_api
    engine: json_engine
    search_url: https://api.example.com/search?q={query}
    # Add your custom search engines
```

## Performance

- **Redis caching**: Results are cached for faster subsequent searches
- **Concurrent requests**: SearxNG handles multiple engines in parallel
- **Connection pooling**: 100 connections, 20 max per host

## Security Notes

1. **Change the secret key** in `settings.yml` before production:
   ```yaml
   server:
     secret_key: "your-random-secure-key-here"
   ```

2. SearxNG is exposed on port 8888. If deploying to production:
   - Add authentication (basic auth or OAuth)
   - Enable rate limiting
   - Use HTTPS with a reverse proxy

3. The current setup is optimized for **local/internal use only**.

## Troubleshooting

### SearxNG returns no results
- Check logs: `docker-compose logs searxng`
- Verify network connectivity: `docker exec dgx-searxng ping google.com`
- Check engine status in web UI: http://localhost:8888/preferences

### Connection refused
- Ensure SearxNG is running: `docker ps | grep searxng`
- Check port binding: `netstat -tlnp | grep 8888`
- Verify SEARXNG_URL environment variable matches your network mode

### Redis connection errors
- Ensure Redis is running: `docker ps | grep redis`
- Check Redis logs: `docker-compose logs redis`

## API Response Format

SearxNG returns JSON in this format:

```json
{
  "query": "your search query",
  "results": [
    {
      "title": "Result Title",
      "url": "https://example.com",
      "content": "Snippet text...",
      "engine": "google",
      "score": 1.0
    }
  ],
  "number_of_results": 10,
  "suggestions": ["related", "queries"]
}
```

## Resources

- [SearxNG Documentation](https://docs.searxng.org/)
- [SearxNG GitHub](https://github.com/searxng/searxng)
- [Available Engines](https://docs.searxng.org/admin/engines/index.html)
