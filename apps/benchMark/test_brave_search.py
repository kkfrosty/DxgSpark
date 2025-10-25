#!/usr/bin/env python3
"""
Test Brave Search API

Get a free API key from: https://brave.com/search/api/
Free tier: 2000 queries/month

Usage:
  export BRAVE_API_KEY="your-key-here"
  python test_brave_search.py
"""

import os
import requests

def test_brave_search():
    api_key = os.getenv('BRAVE_API_KEY')
    
    if not api_key:
        print("❌ BRAVE_API_KEY environment variable not set")
        print("\nTo use Brave Search API:")
        print("1. Sign up at: https://brave.com/search/api/")
        print("2. Get your free API key")
        print("3. Set it: export BRAVE_API_KEY='your-key-here'")
        return
    
    query = "weather Charlotte NC"
    print(f"Searching for: {query}\n")
    
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": 5
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('web', {}).get('results', [])
        
        print(f"✅ Got {len(results)} results:\n")
        print("="*80)
        
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. {result.get('title', 'No title')}")
            print(f"   URL: {result.get('url', 'N/A')}")
            description = result.get('description', '')
            if description:
                print(f"   {description[:200]}...")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == '__main__':
    test_brave_search()
