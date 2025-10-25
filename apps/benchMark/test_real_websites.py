#!/usr/bin/env python3
"""
Test that Playwright CAN visit real websites (just not search engines)
"""

import asyncio
import os
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Set Playwright browser path
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/root/.cache/ms-playwright'

async def test_real_sites():
    """Test visiting actual websites (not search engines)"""
    
    sites_to_test = [
        ("Wikipedia", "https://en.wikipedia.org/wiki/Charlotte,_North_Carolina"),
        ("Weather.com", "https://weather.com/weather/today/l/Charlotte+NC"),
        ("BBC News", "https://www.bbc.com/news"),
        ("Example.com", "https://example.com"),
    ]
    
    async with async_playwright() as p:
        print("Launching browser...\n")
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox', 
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled'
            ]
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await context.new_page()
        
        results = []
        
        for name, url in sites_to_test:
            print(f"Testing: {name}")
            print(f"  URL: {url}")
            
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                await asyncio.sleep(1)
                
                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check if we got blocked
                text_lower = html.lower()
                if any(keyword in text_lower for keyword in ['captcha', 'unusual traffic', 'blocked', 'access denied']):
                    status = "❌ BLOCKED"
                    print(f"  Status: {status}")
                else:
                    # Get page title and some text
                    title = soup.find('title')
                    title_text = title.get_text() if title else "No title"
                    
                    body = soup.find('body')
                    body_text = body.get_text()[:200] if body else ""
                    
                    status = "✅ SUCCESS"
                    print(f"  Status: {status}")
                    print(f"  Title: {title_text}")
                    print(f"  Preview: {body_text.strip()[:100]}...")
                
                results.append((name, status))
                
            except Exception as e:
                status = f"❌ ERROR: {str(e)[:50]}"
                print(f"  Status: {status}")
                results.append((name, status))
            
            print()
        
        await browser.close()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        for name, status in results:
            print(f"{name:20s} {status}")
        
        print("\n💡 The point: Your browser can visit MOST websites just fine.")
        print("   Only SEARCH ENGINES specifically block automation.")

if __name__ == '__main__':
    asyncio.run(test_real_sites())
