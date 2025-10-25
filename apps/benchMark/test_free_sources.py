#!/usr/bin/env python3
"""
Test FREE search/info sources that need NO API keys
"""

import asyncio
import os
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import json

os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/root/.cache/ms-playwright'

async def test_free_sources():
    """Test completely free information sources"""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        page = await browser.new_page()
        
        print("Testing FREE sources (no API keys needed)\n")
        print("="*80)
        
        # 1. WTTR.IN - Free Weather (ASCII art!)
        print("\n1. WTTR.IN - Weather API")
        print("   URL: https://wttr.in/Charlotte?format=j1")
        try:
            await page.goto("https://wttr.in/Charlotte?format=j1", timeout=10000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            pre = soup.find('pre')
            if pre:
                data = json.loads(pre.get_text())
                current = data['current_condition'][0]
                print(f"   ✅ SUCCESS")
                print(f"   Temperature: {current['temp_F']}°F")
                print(f"   Conditions: {current['weatherDesc'][0]['value']}")
                print(f"   Humidity: {current['humidity']}%")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        # 2. Wikipedia API
        print("\n2. Wikipedia API")
        print("   URL: https://en.wikipedia.org/api/rest_v1/page/summary/Charlotte,_North_Carolina")
        try:
            await page.goto("https://en.wikipedia.org/api/rest_v1/page/summary/Charlotte,_North_Carolina", timeout=10000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            pre = soup.find('pre')
            if pre:
                data = json.loads(pre.get_text())
                print(f"   ✅ SUCCESS")
                print(f"   Title: {data.get('title', 'N/A')}")
                print(f"   Extract: {data.get('extract', 'N/A')[:150]}...")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        # 3. HTML DuckDuckGo (sometimes works)
        print("\n3. DuckDuckGo HTML (hit or miss)")
        print("   URL: https://html.duckduckgo.com/html/?q=Charlotte+NC+weather")
        try:
            await page.goto("https://html.duckduckgo.com/html/?q=Charlotte+NC+weather", timeout=10000)
            content = await page.content()
            
            if "202" in content or "ratelimit" in content.lower():
                print(f"   ❌ BLOCKED - Rate limited")
            else:
                soup = BeautifulSoup(content, 'html.parser')
                results = soup.find_all('a', class_='result__a')
                if results:
                    print(f"   ✅ SUCCESS - Found {len(results)} results")
                    for r in results[:2]:
                        print(f"      - {r.get_text()[:60]}")
                else:
                    print(f"   ⚠️  Loaded but no results found")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        await browser.close()
        
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        print("✅ wttr.in - Weather (works reliably, no key)")
        print("✅ Wikipedia API - Facts (works reliably, no key)")  
        print("⚠️  DuckDuckGo HTML - Search (unreliable, gets blocked)")
        print()
        print("💡 For reliable search: Use Brave API (2000 free/month, 1/sec is fine)")
        print("   Or self-host SearXNG for unlimited local search")

if __name__ == '__main__':
    asyncio.run(test_free_sources())
