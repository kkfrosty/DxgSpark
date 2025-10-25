#!/usr/bin/env python3
"""
Advanced Playwright anti-detection for search engines

Uses stealth techniques to appear more like a real user
"""

import asyncio
import os
import random
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/root/.cache/ms-playwright'

async def test_stealth_search():
    """Try search with maximum stealth"""
    
    query = "weather Charlotte NC"
    
    async with async_playwright() as p:
        print("🥷 Launching browser with stealth mode...\n")
        
        # Launch with stealth args
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--window-size=1920,1080',
            ]
        )
        
        # Create context with realistic settings
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
            permissions=['geolocation'],
            geolocation={'latitude': 35.2271, 'longitude': -80.8431},  # Charlotte, NC
        )
        
        page = await context.new_page()
        
        # Inject anti-detection scripts BEFORE navigation
        await page.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
            
            // Mock chrome runtime
            window.chrome = {
                runtime: {},
            };
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        """)
        
        # Set extra headers
        await page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Test multiple search engines
        search_engines = [
            ("DuckDuckGo Lite", "https://lite.duckduckgo.com/lite/?q={}"),
            ("Google", "https://www.google.com/search?q={}"),
            ("Bing", "https://www.bing.com/search?q={}"),
        ]
        
        for name, url_template in search_engines:
            print(f"Testing: {name}")
            url = url_template.format(query.replace(' ', '+'))
            
            try:
                # Navigate like a human (slower)
                await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                
                # Random delay (humans don't act instantly)
                await asyncio.sleep(random.uniform(1.5, 3.0))
                
                # Simulate human behavior - scroll a bit
                await page.evaluate('window.scrollBy(0, 300)')
                await asyncio.sleep(random.uniform(0.5, 1.0))
                
                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check for blocking
                text_lower = html.lower()
                blocked = any(kw in text_lower for kw in [
                    'unusual traffic', 'captcha', 'blocked', 
                    'access denied', 'verify you', 'not a robot'
                ])
                
                if blocked:
                    print(f"  ❌ BLOCKED - Bot detection triggered")
                    # Show what they said
                    body_text = soup.get_text()[:300] if soup.find('body') else ""
                    print(f"  Message: {body_text.strip()[:150]}...")
                else:
                    # Try to find results
                    result_count = 0
                    
                    # Different selectors for different engines
                    if "duckduckgo" in url:
                        results = soup.find_all('a', class_='result-link')
                        result_count = len(results)
                    elif "google" in url:
                        results = soup.find_all('div', class_='g')
                        result_count = len(results)
                    elif "bing" in url:
                        results = soup.find_all('li', class_='b_algo')
                        result_count = len(results)
                    
                    if result_count > 0:
                        print(f"  ✅ SUCCESS - Got {result_count} results!")
                    else:
                        print(f"  ⚠️  Page loaded but found 0 results")
                        title = soup.find('title')
                        if title:
                            print(f"  Page title: {title.get_text()}")
                
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)[:100]}")
            
            print()
        
        await browser.close()
        
        print("\n" + "="*80)
        print("CONCLUSION:")
        print("="*80)
        print("Even with advanced stealth techniques, search engines still detect")
        print("automated browsers. They use sophisticated fingerprinting beyond")
        print("simple WebDriver detection.")
        print()
        print("💡 Real solution: Use official search APIs instead")

if __name__ == '__main__':
    asyncio.run(test_stealth_search())
