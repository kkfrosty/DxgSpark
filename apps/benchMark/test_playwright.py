import asyncio
import os
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Set Playwright browser path
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/root/.cache/ms-playwright'

async def test_browser_search():
    query = "weather Charlotte NC"
    print(f"Testing browser search for: {query}\n")
    
    # Try Bing instead of Google (less aggressive bot detection)
    async with async_playwright() as p:
        print("Launching Chromium browser...")
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox', 
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        )
        
        print("Creating new page...")
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await context.new_page()
        
        # Add extra headers
        await page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        })
        
        search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        print(f"Navigating to: {search_url}")
        
        await page.goto(search_url, wait_until='networkidle', timeout=30000)
        
        print("Waiting for content to load...")
        await asyncio.sleep(1)
        
        print("Getting page content...")
        html = await page.content()
        
        # Save for debugging
        with open('/app/google_debug.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Saved HTML to /app/google_debug.html")
        
        await browser.close()
        
        print(f"Got {len(html)} bytes of HTML\n")
        
        # Parse
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try multiple selector strategies
        print("Trying different selectors:")
        
        # Bing uses li.b_algo for results
        search_results = soup.find_all('li', class_='b_algo')
        print(f"  li.b_algo: {len(search_results)} results")
        
        # Also try div.g (Google) just in case
        search_divs = soup.find_all('div', class_='g')
        print(f"  div.g: {len(search_divs)} results")
        
        # Any h3 tags
        h3_tags = soup.find_all('h3')
        print(f"  h3 tags: {len(h3_tags)} results")
        
        # Check if we got blocked
        if "unusual traffic" in html.lower() or "captcha" in html.lower() or "blocked" in html.lower():
            print("  ⚠️  Search engine returned bot detection page!")
        
        # Show first 500 chars of body text
        body = soup.find('body')
        if body:
            text = body.get_text()[:500]
            print(f"\nFirst 500 chars of page:\n{text}\n")
        
        print("\nResults:\n" + "="*80)
        
        # Use Bing results if available
        if search_results:
            for idx, li in enumerate(search_results[:5], 1):
                title_elem = li.find('h2')
                link_elem = li.find('a', href=True)
                
                if title_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    url = link_elem.get('href', '')
                    
                    if url and url.startswith('http'):
                        print(f"\n{idx}. {title}")
                        print(f"   {url}")
        # Fall back to generic h3 search
        elif h3_tags:
            for idx, h3 in enumerate(h3_tags[:5], 1):
                parent_link = h3.find_parent('a')
                if parent_link and parent_link.get('href', '').startswith('http'):
                    print(f"\n{idx}. {h3.get_text(strip=True)}")
                    print(f"   {parent_link.get('href')}")

asyncio.run(test_browser_search())
