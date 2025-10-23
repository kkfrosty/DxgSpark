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

from sqlalchemy import Column, DateTime, Integer, String, Text, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import SQLAlchemyError

app = FastAPI(title="DxGChatBenchMark", version="1.0.0")

# Global monitoring storage
system_metrics_history = deque(maxlen=100)  # Last 100 system snapshots
model_performance_history = deque(maxlen=50)  # Last 50 chat interactions
active_websockets = []  # For real-time monitoring updates

# Model registry - add your models here
MODELS = {
    "gpt-oss-120b": {
        "name": "GPT-OSS-120B (MXFP4)",
        "url": "http://localhost:8000",
        "type": "llama.cpp",
        "size": "120B",
        "format": "MXFP4"
    },
    "deepseek-nvfp4": {
        "name": "DeepSeek-R1-Distill-Llama-8B (NVFP4)",
        "url": "http://localhost:8002",
        "type": "TensorRT-LLM",
        "size": "8B",
        "format": "NVFP4"
    },
    "llama-nim": {
        "name": "Llama 3.1 8B Instruct (NIM)",
        "url": "http://localhost:8003",
        "type": "NIM",
        "size": "8B",
        "format": "FP16"
    },
    # Add more models as you deploy them
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


class SystemPromptUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str = ""


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
        prompt_processing_speed=round(prompt_processing_speed, 1),
        token_generation_speed=round(token_generation_speed, 1),
        throughput=round(throughput, 1),
        timestamp=datetime.now().isoformat(),
        response_text=generated_text
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
        
        # GPU metrics using nvidia-smi
        gpu_metrics = []
        try:
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
        except Exception as e:
            print(f"GPU metrics error: {e}")
        
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
