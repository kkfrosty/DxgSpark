#!/usr/bin/env python3
"""
Simple OpenAI-compatible API server for NVFP4 quantized models using TensorRT-LLM Python API
This works around the Blackwell GPU compatibility issues in trtllm-serve and vLLM
"""

import os
import json
from flask import Flask, request, jsonify
from datetime import datetime
import uuid

app = Flask(__name__)

# Model will be loaded on first request to save memory
model = None
tokenizer = None
MODEL_PATH = "/workspace/model"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

def load_model():
    global model, tokenizer
    if model is not None:
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded successfully!")

@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models"""
    return jsonify({
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "user"
        }]
    })

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    load_model()
    
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    
    # Convert messages to prompt
    if messages:
        prompt = messages[-1].get("content", "")
    else:
        prompt = data.get("prompt", "")
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the output
    response_text = generated_text[len(prompt):].strip()
    
    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
            "total_tokens": len(outputs[0])
        }
    })

@app.route("/v1/completions", methods=["POST"])
def completions():
    """OpenAI-compatible completions endpoint"""
    load_model()
    
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(datetime.now().timestamp()),
        "model": MODEL_NAME,
        "choices": [{
            "text": generated_text,
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
            "total_tokens": len(outputs[0])
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    import torch
    print("Starting NVFP4 DeepSeek API Server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Server will run on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, threaded=False)
