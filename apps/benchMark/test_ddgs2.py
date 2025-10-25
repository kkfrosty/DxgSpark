from duckduckgo_search import DDGS
import time

query = "weather Charlotte NC"
print(f"Searching for: {query}")

try:
    # Try with different settings
    ddgs = DDGS(timeout=20)
    results = ddgs.text(query, max_results=3)
    
    print(f"\nResults:")
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. {result.get('title', 'No title')}")
        print(f"   URL: {result.get('href', 'No URL')}")
        print(f"   Snippet: {result.get('body', 'No snippet')[:150]}...")
        
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
