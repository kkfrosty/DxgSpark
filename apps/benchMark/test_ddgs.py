from duckduckgo_search import DDGS

query = "What is the weather forecast for Charlotte NC"
print(f"Searching for: {query}")

with DDGS() as ddgs:
    results = list(ddgs.text(query, max_results=5))
    
print(f"\nFound {len(results)} results:\n")

for idx, result in enumerate(results, 1):
    print(f"{idx}. {result.get('title', 'No title')}")
    print(f"   URL: {result.get('href', 'No URL')}")
    print(f"   Snippet: {result.get('body', 'No snippet')[:100]}...")
    print()
