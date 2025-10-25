# Search API Alternatives for RAG

DuckDuckGo is actively rate-limiting our requests. Here are production-ready alternatives:

## 1. **Brave Search API** (Recommended - Best Value)
- **Free Tier**: 2,000 queries/month
- **Pricing**: $3/1,000 queries after free tier
- **Setup**: Get API key at https://brave.com/search/api/
- **Install**: `pip install brave-search`

```python
from brave import Brave
brave = Brave(api_key="YOUR_API_KEY")
results = brave.search(query, count=5)
```

## 2. **SerpAPI** (Google Search Results)
- **Free Tier**: 100 queries/month
- **Pricing**: $50/month for 5,000 searches
- **Setup**: Get API key at https://serpapi.com/
- **Install**: `pip install google-search-results`

```python
from serpapi import GoogleSearch
search = GoogleSearch({"q": query, "api_key": "YOUR_API_KEY"})
results = search.get_dict()
```

## 3. **Tavily AI** (AI-Optimized Search)
- **Free Tier**: 1,000 queries/month
- **Pricing**: $100/month for 50,000 queries
- **Setup**: Get API key at https://tavily.com/
- **Install**: `pip install tavily-python`

```python
from tavily import TavilyClient
tavily = TavilyClient(api_key="YOUR_API_KEY")
results = tavily.search(query, max_results=5)
```

## 4. **Google Custom Search API**
- **Free Tier**: 100 queries/day
- **Pricing**: $5 per 1,000 queries (max 10,000/day)
- **Setup**: Enable at https://console.cloud.google.com/
- **Install**: `pip install google-api-python-client`

## 5. **Fallback: Use Selenium with Real Browser**
If you want to stick with free solutions, use Selenium to actually open a browser:

```bash
pip install selenium webdriver-manager
```

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(f"https://duckduckgo.com/?q={query}")
# Parse results from actual browser session
```

## Recommendation:

For your benchmark app, I recommend **Brave Search API** because:
- Free tier of 2,000 queries/month is generous for testing
- Very affordable paid tier ($3/1,000 queries)
- Fast and reliable
- Returns clean, structured JSON
- No rate limiting issues

Would you like me to implement any of these alternatives?
