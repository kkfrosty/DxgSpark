# Brave Search API Pricing (as of Oct 2024)

## FREE TIER
- **2,000 queries per month**
- **1 query per second rate limit**
- No credit card required
- Sign up at: https://brave.com/search/api/

## Is 1 query/second a problem?

For a RAG chatbot: **NO!**

### Typical Usage Pattern:
1. User asks: "What's the weather in Charlotte?"
2. Agent decides it needs web search
3. Makes 1 search call → "weather Charlotte NC"
4. Gets results in ~500ms
5. Processes results and responds

**Time between searches:** Usually 5-30 seconds (while LLM processes)
**Burst rate needed:** Rarely more than 1/second

### When 1 query/second IS a problem:
- High-traffic public API (100+ concurrent users)
- Batch processing (scraping/indexing)
- Real-time monitoring systems

### For your use case (benchmark testing):
- You'll make maybe 10-20 searches in a testing session
- **1 query/second is totally fine**

## Other Free Options:

### 1. Wikipedia API (Completely Free)
- Unlimited requests
- Great for factual information
- No search engine, but has search endpoint

### 2. wttr.in (Free Weather)
- curl wttr.in/Charlotte
- No API key needed
- Works perfectly in your browser

### 3. Free News APIs
- NewsAPI.org - 100 requests/day free
- GNews.io - 100 requests/day free

### 4. SearXNG (Self-hosted)
- **Completely free**
- Run your own meta-search engine
- Aggregates results from multiple sources
- 10-minute Docker setup

## Real Cost Comparison:

| Service | Free Tier | Rate Limit | Notes |
|---------|-----------|------------|-------|
| Brave Search | 2000/month | 1/sec | Good for testing |
| Tavily AI | $0 trial | 1000 queries | Optimized for RAG |
| SerpAPI | 100/month | 1/sec | Google results |
| SearXNG | Unlimited | Your server | Self-hosted |
| Direct APIs | Varies | Generous | Weather, news, wiki |

## My Recommendation:

**For your RAG benchmark app:**
1. **Brave Search API** for general queries (free tier is fine)
2. **wttr.in** for weather (no key needed)
3. **Wikipedia API** for facts (no key needed)

**1 query/second is NOT a limitation for interactive chat!**

Want me to implement this multi-source approach? Or would you prefer to self-host SearXNG?
