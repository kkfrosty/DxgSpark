#!/usr/bin/env python3
"""
DxGChatBenchMark - Multi-Model Chat & Benchmark Interface
Provides a web UI for chatting with any running LLM and measuring performance
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any, Optional as TypingOptional
import httpx
import time
import json
import asyncio
from datetime import datetime
from collections import deque
import psutil
import subprocess
import os
from pathlib import Path
import re

from sqlalchemy import Column, DateTime, Integer, String, Text, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import SQLAlchemyError

try:
    from bs4 import BeautifulSoup
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG dependencies not installed. Install with: pip install beautifulsoup4")

app = FastAPI(title="DxGChatBenchMark", version="1.0.0")

# Global monitoring storage
system_metrics_history = deque(maxlen=100)  # Last 100 system snapshots
model_performance_history = deque(maxlen=50)  # Last 50 chat interactions
active_websockets = []  # For real-time monitoring updates

# Search cache to avoid rate limits (cache for 5 minutes)
_search_cache: Dict[str, tuple[List[Dict[str, str]], float]] = {}
_search_cache_ttl = 300  # 5 minutes

# SearxNG configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")

# Get host IP for accessing LLM models from inside Docker container
HOST_IP = os.getenv("HOST_IP", "host.docker.internal")

# Model registry - add your models here
MODELS = {
    "gpt-oss-120b": {
        "name": "GPT-OSS-120B (MXFP4)",
        "url": f"http://{HOST_IP}:8000",
        "type": "llama.cpp",
        "size": "120B",
        "format": "MXFP4",
        "supports_tools": False  # llama.cpp doesn't support native tool calling
    },
    "deepseek-nvfp4": {
        "name": "DeepSeek-R1-Distill-Llama-8B (NVFP4)",
        "url": f"http://{HOST_IP}:8002",
        "type": "TensorRT-LLM",
        "size": "8B",
        "format": "NVFP4",
        "supports_tools": False
    },
    "llama-nim": {
        "name": "Llama 3.1 8B Instruct (NIM)",
        "url": f"http://{HOST_IP}:8003",
        "type": "NIM",
        "size": "8B",
        "format": "FP16",
        "supports_tools": True
    },
    # Add more models as you deploy them
}

# Tool definition for web search
SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for current information, news, stock prices, weather, or any real-time data. Use this when you need up-to-date information that you don't have in your training data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and concise."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPTS_FILE = BASE_DIR / "system_prompts.json"
DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()
engine = None
AsyncSessionLocal: TypingOptional[async_sessionmaker[AsyncSession]] = None

if DATABASE_URL:
    engine = create_async_engine(DATABASE_URL, future=True, pool_pre_ping=True)
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class SystemPromptRecord(Base):
    __tablename__ = "system_prompts"

    model_id = Column(String(255), primary_key=True)
    prompt = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ChatLogRecord(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    request_messages = Column(JSONB, nullable=False)
    response_payload = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


_system_prompts_lock = asyncio.Lock()
system_prompts: Dict[str, str] = {}


def _load_system_prompts_from_file() -> Dict[str, str]:
    if not SYSTEM_PROMPTS_FILE.exists():
        return {}

    try:
        with SYSTEM_PROMPTS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:
        print(f"Failed to load system prompts from file: {exc}")

    return {}


def _save_system_prompts_to_file(snapshot: Dict[str, str]) -> None:
    try:
        SYSTEM_PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SYSTEM_PROMPTS_FILE.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
    except Exception as exc:
        print(f"Failed to save system prompts to file: {exc}")


def _format_sse(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _load_system_prompts_from_db() -> Dict[str, str]:
    if not AsyncSessionLocal:
        return {}

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(SystemPromptRecord))
            prompts = {record.model_id: record.prompt for record in result.scalars()}
            return prompts
    except SQLAlchemyError as exc:
        print(f"Failed to load system prompts from database: {exc}")
    return {}


async def _write_system_prompt_to_db(model_id: str, prompt_text: str) -> None:
    if not AsyncSessionLocal:
        return

    try:
        async with AsyncSessionLocal() as session:
            existing = await session.get(SystemPromptRecord, model_id)
            if prompt_text.strip():
                if existing:
                    existing.prompt = prompt_text
                else:
                    session.add(SystemPromptRecord(model_id=model_id, prompt=prompt_text))
            else:
                if existing:
                    await session.delete(existing)
            await session.commit()
    except SQLAlchemyError as exc:
        print(f"Failed to store system prompt in database: {exc}")


async def _log_chat_interaction(model_id: str, request_messages: List[Dict[str, Any]], response_payload: Dict[str, Any]) -> None:
    if not AsyncSessionLocal:
        return

    try:
        request_copy = json.loads(json.dumps(request_messages))
        response_copy = json.loads(json.dumps(response_payload))
    except (TypeError, ValueError) as exc:
        print(f"Skipping chat logging due to serialization error: {exc}")
        return

    record = ChatLogRecord(
        model_id=model_id,
        request_messages=request_copy,
        response_payload=response_copy,
    )

    try:
        async with AsyncSessionLocal() as session:
            session.add(record)
            await session.commit()
    except SQLAlchemyError as exc:
        print(f"Failed to persist chat interaction: {exc}")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    use_rag: Optional[bool] = False


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_time: float
    prompt_processing_speed: float
    token_generation_speed: float
    throughput: float
    timestamp: str
    response_text: str
    prompt_tokens_per_sec: Optional[float] = None
    generation_tokens_per_sec: Optional[float] = None
    time_to_first_token: Optional[float] = None
    prefill_duration: Optional[float] = None
    generation_duration: Optional[float] = None


class SystemPromptUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str = ""


async def _search_web(query: str, num_results: int = 5) -> tuple[List[Dict[str, str]], List[str]]:
    """Search the web using self-hosted SearxNG metasearch engine"""
    results = []
    step_logs = []
    
    # Check cache first
    cache_key = f"{query}:{num_results}"
    if cache_key in _search_cache:
        cached_results, timestamp = _search_cache[cache_key]
        if time.time() - timestamp < _search_cache_ttl:
            step_logs.append(f"✓ Using cached results for: {query}")
            return cached_results, step_logs
    
    step_logs.append(f"Searching web via SearxNG for: {query}...")
    
    try:
        # SearxNG HTML endpoint (JSON is blocked by default settings)
        search_url = f"{SEARXNG_URL}/search"
        params = {
            "q": query,
            # Remove "format": "json" - use HTML and parse it
            "pageno": 1,
            "language": "en",
            "categories": "general"
        }
        
        step_logs.append(f"Querying SearxNG at {SEARXNG_URL}...")
        
        # Headers to bypass bot detection
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        async with httpx.AsyncClient(timeout=30.0, http2=False, follow_redirects=True) as client:
            response = await client.get(search_url, params=params, headers=headers)
            response.raise_for_status()
            
            # Parse HTML response
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all result articles (SearxNG uses <article> tags for results)
            articles = soup.find_all('article', class_='result')
            
            step_logs.append(f"SearxNG returned {len(articles)} results")
            
            # Parse and format results
            for article in articles[:num_results]:
                # Extract title and URL from h3 > a
                title_elem = article.find('h3')
                if title_elem:
                    link = title_elem.find('a')
                    if link:
                        title = link.get_text(strip=True)
                        url = link.get('href', '')
                        
                        # Extract snippet from p.content
                        snippet_elem = article.find('p', class_='content')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        if title and url:
                            results.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet[:300] if snippet else ""
                            })
                            step_logs.append(f"  ✓ {title[:60]}...")
            
            if results:
                step_logs.append(f"✓ Successfully retrieved {len(results)} results")
                # Cache the results
                _search_cache[cache_key] = (results, time.time())
            else:
                step_logs.append("⚠ SearxNG returned no usable results")
                
    except httpx.HTTPError as e:
        step_logs.append(f"✗ HTTP error connecting to SearxNG: {str(e)[:100]}")
        step_logs.append(f"⚠ Is SearxNG running at {SEARXNG_URL}?")
        print(f"SearxNG HTTP error: {e}")
    except Exception as e:
        step_logs.append(f"✗ Error: {str(e)[:100]}")
        print(f"SearxNG search error: {e}")
        import traceback
        print(traceback.format_exc())
    
    if not results:
        step_logs.append("No web search results available")
    
    return results, step_logs


def _format_rag_context(search_results: List[Dict[str, str]]) -> str:
    """Format search results into a context string for the LLM."""
    if not search_results:
        return ""
    
    current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p %Z")
    
    context_parts = [
        f"CURRENT DATE/TIME: {current_datetime}",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🌐 WEB SEARCH COMPLETED - USE THIS INFORMATION",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "The web search has ALREADY been performed for you. Below are the current results from the internet.",
        "You DO NOT need to call any search function or tool.",
        "You MUST use this information to answer the question.",
        ""
    ]
    
    for idx, result in enumerate(search_results, 1):
        context_parts.append(f"━━━ SOURCE {idx} ━━━")
        context_parts.append(f"📰 Title: {result['title']}")
        context_parts.append(f"🔗 URL: {result['url']}")
        context_parts.append(f"📄 Content: {result['snippet']}")
        context_parts.append("")
    
    context_parts.extend([
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "✅ INSTRUCTIONS:",
        "1. The search has ALREADY been completed - DO NOT output search function calls",
        "2. The information above is CURRENT as of " + current_datetime,
        "3. You HAVE access to real-time data through the search results above",
        "4. Provide a detailed answer using the information from these sources",
        "5. Cite sources by number (e.g., 'According to Source 1...')",
        "",
        "Now answer the user's question using the search results above:",
        ""
    ])
    
    return "\n".join(context_parts)


def _extract_stream_segments(delta: Dict[str, Any]) -> Dict[str, str]:
    """Normalize streamed delta payloads into content and reasoning strings."""
    text_buffer: List[str] = []
    reasoning_buffer: List[str] = []

    content = delta.get("content")
    if isinstance(content, str):
        text_buffer.append(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "output_text"}:
                text_buffer.append(item.get("text", ""))
            elif item_type in {"reasoning", "thinking"}:
                reasoning_buffer.append(item.get("text", ""))

    reasoning = delta.get("reasoning")
    if isinstance(reasoning, str):
        reasoning_buffer.append(reasoning)
    elif isinstance(reasoning, list):
        for item in reasoning:
            if isinstance(item, dict):
                reasoning_buffer.append(item.get("text", ""))

    reasoning_content = delta.get("reasoning_content")
    if isinstance(reasoning_content, str):
        reasoning_buffer.append(reasoning_content)
    elif isinstance(reasoning_content, list):
        for item in reasoning_content:
            if isinstance(item, dict):
                reasoning_buffer.append(item.get("text", ""))

    thinking = delta.get("thinking")
    if isinstance(thinking, str):
        reasoning_buffer.append(thinking)
    elif isinstance(thinking, list):
        for item in thinking:
            if isinstance(item, dict):
                reasoning_buffer.append(item.get("text", ""))

    return {
        "content": "".join(text_buffer),
        "reasoning": "".join(reasoning_buffer),
    }


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize persistent storage and load cached prompts."""
    prompts: Dict[str, str] = {}

    if AsyncSessionLocal and engine:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            prompts = await _load_system_prompts_from_db()
        except SQLAlchemyError as exc:
            print(f"Database initialization failed, falling back to file storage: {exc}")
            prompts = _load_system_prompts_from_file()
    else:
        prompts = _load_system_prompts_from_file()

    async with _system_prompts_lock:
        system_prompts.clear()
        system_prompts.update(prompts)


@app.get("/")
async def root():
    """Serve the main chat interface"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/models")
async def get_models():
    """Return available models with health status"""
    models_status = []
    async with _system_prompts_lock:
        prompts_snapshot = dict(system_prompts)

    async with httpx.AsyncClient(timeout=2.0) as client:
        for model_id, config in MODELS.items():
            try:
                # Try to ping the model's health endpoint
                response = await client.get(f"{config['url']}/health", timeout=1.0)
                is_available = response.status_code == 200
            except:
                # If health check fails, try models endpoint
                try:
                    response = await client.get(f"{config['url']}/v1/models", timeout=1.0)
                    is_available = response.status_code == 200
                except:
                    is_available = False
            
            models_status.append({
                "id": model_id,
                "name": config["name"],
                "type": config["type"],
                "size": config["size"],
                "format": config["format"],
                "available": is_available,
                "url": config["url"],
                "system_prompt": prompts_snapshot.get(model_id, "")
            })
    
    return {"models": models_status}


@app.post("/api/chat", response_model=BenchmarkResult)
async def chat(request: ChatRequest):
    """Send chat request to model and return response with benchmarks"""
    
    if request.model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    
    model_config = MODELS[request.model_id]
    api_url = f"{model_config['url']}/v1/chat/completions"
    async with _system_prompts_lock:
        system_prompt = system_prompts.get(request.model_id, "")

    # Prepare the request payload
    messages_payload = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    if system_prompt.strip():
        messages_payload.insert(0, {"role": "system", "content": system_prompt})

    payload = {
        "model": "default",  # Most endpoints accept any model name
        "messages": messages_payload,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": False  # Non-streaming for benchmarking
    }
    
    # Time the request
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Model API error: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Parse response
    result = response.json()
    
    # Extract metrics
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    
    # Extract generated text
    generated_text = ""
    if "choices" in result and len(result["choices"]) > 0:
        choice = result["choices"][0]
        if "message" in choice:
            generated_text = choice["message"].get("content", "")
        elif "text" in choice:
            generated_text = choice.get("text", "")
    
    # Calculate performance metrics
    # Estimate: prefill is ~10-20% of total time (depends on prompt size)
    prefill_ratio = min(0.2, prompt_tokens / max(total_tokens, 1) * 0.5)
    estimated_prefill_time = total_time * prefill_ratio
    estimated_generation_time = total_time - estimated_prefill_time
    
    prompt_processing_speed = prompt_tokens / estimated_prefill_time if estimated_prefill_time > 0 else 0
    token_generation_speed = completion_tokens / estimated_generation_time if estimated_generation_time > 0 else 0
    throughput = total_tokens / total_time if total_time > 0 else 0
    
    benchmark_result = BenchmarkResult(
        model_id=request.model_id,
        model_name=model_config["name"],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        total_time=round(total_time, 2),
        prompt_processing_speed=round(prompt_processing_speed, 2),
        token_generation_speed=round(token_generation_speed, 2),
        throughput=round(throughput, 2),
        timestamp=datetime.now().isoformat(),
        response_text=generated_text,
        prompt_tokens_per_sec=round(prompt_processing_speed, 2),
        generation_tokens_per_sec=round(token_generation_speed, 2)
    )
    
    # Store in performance history
    model_performance_history.append({
        "model": model_config["name"],
        "timestamp": benchmark_result.timestamp,
        "tokens_per_sec": token_generation_speed,
        "total_time": total_time,
        "tokens": total_tokens
    })
    
    # Notify websocket clients of new performance data
    for ws in active_websockets:
        try:
            await ws.send_json({
                "type": "performance_update",
                "data": {
                    "model": model_config["name"],
                    "tokens_per_sec": token_generation_speed,
                    "timestamp": benchmark_result.timestamp
                }
            })
        except:
            pass
    
    await _log_chat_interaction(request.model_id, payload["messages"], result)

    return benchmark_result


@app.post("/api/rag/clear-cache")
async def clear_rag_cache():
    """Clear the RAG search cache"""
    global _search_cache
    _search_cache.clear()
    return {"status": "cache_cleared", "message": "RAG search cache has been cleared"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat completions and metrics to the client via SSE."""

    if request.model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    model_config = MODELS[request.model_id]
    api_url = f"{model_config['url']}/v1/chat/completions"

    async with _system_prompts_lock:
        system_prompt = system_prompts.get(request.model_id, "")

    messages_payload = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    print(f"DEBUG: use_rag = {request.use_rag}, RAG_AVAILABLE = {RAG_AVAILABLE}")
    
    # RAG: If enabled, search the web and add context
    search_results = []
    search_step_logs = []
    if request.use_rag and RAG_AVAILABLE and len(messages_payload) > 0:
        # Get the last user message as the search query
        last_user_message = None
        for msg in reversed(messages_payload):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if last_user_message:
            print(f"RAG SEARCH STARTING for query: {last_user_message[:100]}...")
            search_results, search_step_logs = await _search_web(last_user_message, num_results=5)
            print(f"RAG SEARCH COMPLETED: Got {len(search_results)} results")
            if search_results:
                print(f"RAG SEARCH RESULTS: {[r['title'] for r in search_results]}")
                rag_context = _format_rag_context(search_results)
                # Prepend RAG context directly to the user's message
                for msg in reversed(messages_payload):
                    if msg["role"] == "user":
                        msg["content"] = f"{rag_context}\n\n---\n\nUser Question: {msg['content']}\n\nPlease answer using the web search results provided above."
                        print(f"RAG CONTEXT ADDED to user message")
                        break
            else:
                print("RAG SEARCH RETURNED NO RESULTS")
        else:
            print("RAG: No user message found")
    else:
        print(f"RAG SKIPPED: use_rag={request.use_rag}, RAG_AVAILABLE={RAG_AVAILABLE}, messages={len(messages_payload)}")
    
    if system_prompt.strip():
        messages_payload.insert(0, {"role": "system", "content": system_prompt})
    
    # Add RAG capability instruction if enabled
    if request.use_rag and RAG_AVAILABLE:
        rag_system_prompt = (
            "You are an AI assistant with access to real-time web search results. "
            "When search results are provided in the conversation, you MUST use them to answer the user's questions. "
            "The search results contain current, up-to-date information. "
            "Do NOT say you cannot access real-time data - you have it in the search results. "
            "Always cite the sources when using information from search results."
        )
        messages_payload.insert(0, {"role": "system", "content": rag_system_prompt})

    payload = {
        "model": "default",
        "messages": messages_payload,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": True
    }

    async def event_generator():
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        response_text: str = ""
        reasoning_text: str = ""
        usage_payload: Dict[str, Any] = {}
        timings_payload: Dict[str, Any] = {}
        content_token_count: int = 0
        reasoning_token_count: int = 0
        
        # Emit initial step
        yield _format_sse("step", {"message": f"Preparing request to {model_config['name']}", "type": "info"})
        
        # Emit search step logs
        for log in search_step_logs:
            # Determine step type based on log content
            step_type = "info"
            if "✓" in log or "Found" in log or "cached" in log.lower():
                step_type = "success"
            elif "✗" in log or "error" in log.lower() or "failed" in log.lower():
                step_type = "error"
            elif "⚠" in log or "warning" in log.lower() or "No results" in log:
                step_type = "warning"
            
            yield _format_sse("step", {"message": log, "type": step_type})
        
        # Notify client if RAG was used
        if search_results:
            yield _format_sse("step", {"message": f"Using {len(search_results)} web sources for context", "type": "success"})
            yield _format_sse("rag_context", {"enabled": True, "sources": len(search_results), "results": search_results})
        else:
            if request.use_rag and RAG_AVAILABLE:
                yield _format_sse("step", {"message": "No web search results available", "type": "warning"})

        yield _format_sse("status", {"state": "started"})
        yield _format_sse("step", {"message": "Sending request to model API", "type": "info"})

        try:
            timeout = httpx.Timeout(120.0, read=None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                yield _format_sse("step", {"message": "Waiting for model response...", "type": "info"})
                async with client.stream("POST", api_url, json=payload) as upstream:
                    upstream.raise_for_status()
                    yield _format_sse("step", {"message": "Model connection established, streaming response", "type": "success"})
                    async for raw_line in upstream.aiter_lines():
                        if raw_line is None:
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Capture usage data if provided
                        if chunk.get("usage"):
                            usage_payload = chunk["usage"]
                        
                        # Capture llama.cpp timings data if provided (GPT-OSS-120B)
                        if chunk.get("timings"):
                            timings_payload = chunk["timings"]

                        choices = chunk.get("choices", [])
                        for choice in choices:
                            delta = choice.get("delta", {})
                            segments = _extract_stream_segments(delta)

                            if segments["content"]:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                response_text += segments["content"]
                                content_token_count += 1
                                yield _format_sse("token", {"text": segments["content"]})

                            if segments["reasoning"]:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                reasoning_text += segments["reasoning"]
                                reasoning_token_count += 1
                                yield _format_sse("reasoning", {"text": segments["reasoning"]})

        except httpx.HTTPError as exc:
            yield _format_sse("step", {"message": f"HTTP error: {str(exc)}", "type": "error"})
            yield _format_sse("error", {"message": f"Model API error: {str(exc)}"})
            return
        except Exception as exc:
            yield _format_sse("step", {"message": f"Unexpected error: {str(exc)}", "type": "error"})
            yield _format_sse("error", {"message": str(exc)})
            return

        yield _format_sse("step", {"message": "Processing performance metrics", "type": "info"})

        end_time = time.perf_counter()
        if first_token_time is None:
            first_token_time = end_time

        # Use llama.cpp timings if available (GPT-OSS-120B provides this)
        if timings_payload:
            prompt_tokens = timings_payload.get("prompt_n", 0)
            completion_tokens = timings_payload.get("predicted_n", 0)
            total_tokens = prompt_tokens + completion_tokens
            
            # Use actual speeds from the model
            prompt_speed = timings_payload.get("prompt_per_second", 0.0)
            generation_speed = timings_payload.get("predicted_per_second", 0.0)
            
            # Use actual timings from the model (convert ms to seconds)
            prefill_duration = timings_payload.get("prompt_ms", 0.0) / 1000.0
            generation_duration = timings_payload.get("predicted_ms", 0.0) / 1000.0
            total_duration = prefill_duration + generation_duration
            
            throughput = total_tokens / total_duration if total_duration > 0 else 0.0
            
        else:
            # Fallback to usage payload or calculated values
            prompt_tokens = usage_payload.get("prompt_tokens", 0)
            completion_tokens = usage_payload.get("completion_tokens", 0)
            
            # If model didn't provide usage, use our token counts (less accurate)
            if completion_tokens == 0 and (content_token_count > 0 or reasoning_token_count > 0):
                completion_tokens = content_token_count + reasoning_token_count
            
            total_tokens = usage_payload.get("total_tokens", prompt_tokens + completion_tokens)
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            prefill_duration = max(first_token_time - start_time, 0.0)
            generation_duration = max(end_time - first_token_time, 0.0)
            total_duration = max(end_time - start_time, 0.0)

            prompt_speed = prompt_tokens / prefill_duration if prefill_duration > 0 else 0.0
            generation_speed = completion_tokens / generation_duration if generation_duration > 0 else 0.0
            throughput = total_tokens / total_duration if total_duration > 0 else 0.0

        benchmark_result = BenchmarkResult(
            model_id=request.model_id,
            model_name=model_config["name"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_time=round(total_duration, 2),
            prompt_processing_speed=round(prompt_speed, 2),
            token_generation_speed=round(generation_speed, 2),
            throughput=round(throughput, 2),
            timestamp=datetime.now().isoformat(),
            response_text=response_text,
            prompt_tokens_per_sec=round(prompt_speed, 2),
            generation_tokens_per_sec=round(generation_speed, 2),
            time_to_first_token=round(prefill_duration, 4),
            prefill_duration=round(prefill_duration, 4),
            generation_duration=round(generation_duration, 4)
        )

        await _log_chat_interaction(request.model_id, messages_payload, {
            "response": response_text,
            "reasoning": reasoning_text,
            "usage": usage_payload
        })

        model_performance_history.append({
            "model": model_config["name"],
            "timestamp": benchmark_result.timestamp,
            "tokens_per_sec": benchmark_result.token_generation_speed,
            "total_time": total_duration,
            "tokens": total_tokens
        })

        for ws in list(active_websockets):
            try:
                await ws.send_json({
                    "type": "performance_update",
                    "data": {
                        "model": model_config["name"],
                        "tokens_per_sec": benchmark_result.token_generation_speed,
                        "timestamp": benchmark_result.timestamp
                    }
                })
            except Exception:
                if ws in active_websockets:
                    active_websockets.remove(ws)

        summary_payload = {
            "result": benchmark_result.model_dump(),
            "reasoning": reasoning_text
        }

        yield _format_sse("summary", summary_payload)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/api/models/{model_id}/system-prompt")
async def get_model_system_prompt(model_id: str):
    """Return the saved system prompt for a model"""
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    async with _system_prompts_lock:
        prompt = system_prompts.get(model_id, "")

    return {"model_id": model_id, "system_prompt": prompt}


@app.put("/api/models/{model_id}/system-prompt")
async def update_model_system_prompt(model_id: str, update: SystemPromptUpdate):
    """Update and persist the system prompt for a model"""
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    prompt_text = update.prompt or ""

    has_prompt = bool(prompt_text.strip())

    async with _system_prompts_lock:
        if has_prompt:
            system_prompts[model_id] = prompt_text
        else:
            system_prompts.pop(model_id, None)

        snapshot = dict(system_prompts)

    if AsyncSessionLocal:
        await _write_system_prompt_to_db(model_id, prompt_text if has_prompt else "")
    else:
        _save_system_prompts_to_file(snapshot)

    return {"model_id": model_id, "system_prompt": snapshot.get(model_id, "")}


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "DxGChatBenchMark"}


@app.get("/api/system/metrics")
async def get_system_metrics():
    """Get current system metrics (CPU, RAM, GPU)"""
    try:
        # CPU and RAM metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        
        # GPU metrics using nvidia-smi (if available)
        gpu_metrics = []
        try:
            # Check if nvidia-smi exists before calling it
            if os.path.exists('/usr/bin/nvidia-smi') or os.path.exists('/usr/local/bin/nvidia-smi'):
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 7:
                                gpu_metrics.append({
                                    "index": int(parts[0]),
                                    "name": parts[1],
                                    "utilization": float(parts[2]) if parts[2] != '[N/A]' else 0,
                                    "memory_used_mb": float(parts[3]) if parts[3] != '[N/A]' else 0,
                                    "memory_total_mb": float(parts[4]) if parts[4] != '[N/A]' else 0,
                                    "temperature": float(parts[5]) if parts[5] != '[N/A]' else 0,
                                    "power_draw": float(parts[6]) if parts[6] != '[N/A]' else 0
                                })
        except Exception:
            # Silently skip GPU metrics if not available
            pass
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "ram_used_gb": ram.used / (1024**3),
            "ram_total_gb": ram.total / (1024**3),
            "ram_percent": ram.percent,
            "gpus": gpu_metrics
        }
        
        # Store in history
        system_metrics_history.append(metrics)
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@app.get("/api/system/history")
async def get_system_history():
    """Get historical system metrics"""
    return {"metrics": list(system_metrics_history)}


@app.get("/api/performance/history")
async def get_performance_history():
    """Get historical model performance data"""
    return {"history": list(model_performance_history)}


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket for real-time system monitoring"""
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        while True:
            # Send system metrics every 2 seconds
            metrics = await get_system_metrics()
            await websocket.send_json({
                "type": "system_metrics",
                "data": metrics
            })
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
