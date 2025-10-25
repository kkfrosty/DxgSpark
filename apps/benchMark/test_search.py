#!/usr/bin/env python3
"""
Test script to debug DuckDuckGo search with httpx
"""
import asyncio
import httpx
from bs4 import BeautifulSoup

async def test_search():
    query = "What is the weather forecast for Charlotte NC"
    search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
    
    print(f"🔍 Testing search for: {query}")
    print(f"📍 URL: {search_url}")
    print("-" * 80)
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        print("🌐 Fetching results with httpx...")
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            response = await client.get(search_url, headers=headers)
            
        print(f"� Response status: {response.status_code}")
        print(f"📏 Content length: {len(response.text)} bytes")
        
        # Save HTML
        with open('/app/search_page_html.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 Saved HTML to /app/search_page_html.html")
        
        # Parse
        soup = BeautifulSoup(response.text, 'html.parser')
        print(f"📄 Page title: {soup.title.string if soup.title else 'No title'}")
        
        # Find result links
        print("\n" + "=" * 80)
        print("🔍 PARSING RESULTS")
        print("=" * 80)
        
        result_links = soup.find_all('a', class_='result__a')
        print(f"\n📋 Found {len(result_links)} links with class='result__a'")
        
        results = []
        for idx, link in enumerate(result_links[:5]):
            title = link.get_text(strip=True)
            url = link.get('href', '')
            
            # Get snippet
            parent = link.find_parent('div', class_='result__body')
            snippet = ""
            if parent:
                snippet_elem = parent.find('a', class_='result__snippet')
                if snippet_elem:
                    snippet = snippet_elem.get_text(strip=True)
            
            if title and url:
                results.append({'title': title, 'url': url, 'snippet': snippet})
                print(f"\n{idx+1}. {title}")
                print(f"   URL: {url}")
                print(f"   Snippet: {snippet[:100]}...")
        
        if results:
            print(f"\n✅ Successfully extracted {len(results)} results!")
        else:
            print("\n❌ NO RESULTS FOUND")
            print("Check /app/search_page_html.html for the actual response")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_search())
