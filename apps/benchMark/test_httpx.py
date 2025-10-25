import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        resp = await client.get('https://html.duckduckgo.com/html/?q=weather')
        print('Status:', resp.status_code)
        print('Content-Encoding:', resp.headers.get('content-encoding'))
        print('Length:', len(resp.text))
        print('First 500 chars:')
        print(resp.text[:500])

asyncio.run(test())
