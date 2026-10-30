"""
RunPod Serverless GPU Handler for vLLM Inference.

Alternative to Modal deployment with RunPod's serverless GPU infrastructure.

Usage:
    # Build and push Docker image
    docker build -t your-registry/rag-vllm-runpod -f Dockerfile.runpod .
    docker push your-registry/rag-vllm-runpod

    # Deploy on RunPod
    # Create a serverless endpoint with the Docker image
"""

import os
import json
import time
import subprocess
from typing import Optional

# =============================================================================
# RunPod Handler Configuration
# =============================================================================

# Environment variables set in RunPod dashboard:
# - RUNPOD_API_KEY: Your RunPod API key
# - MODEL_NAME: Model to serve (default: meta-llama/Llama-3.2-3B-Instruct)
# - HF_TOKEN: Hugging Face token for gated models

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
CACHE_DIR = "/workspace/cache"
PORT = 8000

# =============================================================================
# Global Server State
# =============================================================================

server_process = None
server_ready = False


def initialize_server():
    """Initialize vLLM server with model caching."""
    global server_process, server_ready

    if server_ready:
        return

    print(f"Initializing vLLM server for {MODEL_NAME}...")
    start_time = time.time()

    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Start vLLM server
    server_process = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.9",
            "--dtype", "auto",
            "--download-dir", CACHE_DIR,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    import requests
    for i in range(120):  # 2 minute timeout
        try:
            response = requests.get(f"http://localhost:{PORT}/health", timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"vLLM server ready in {elapsed:.2f}s")
                server_ready = True
                return
        except:
            pass
        time.sleep(1)

    raise RuntimeError("vLLM server failed to start within 2 minutes")


def generate_completion(messages, max_tokens=1024, temperature=0.7):
    """Generate completion using vLLM."""
    import requests

    response = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    return response.json()


# =============================================================================
# RunPod Handler Functions
# =============================================================================

def handler(job):
    """
    Main RunPod handler function.

    Called for each incoming request.
    """
    try:
        # Initialize server on first request (cold start)
        initialize_server()

        # Parse job input
        job_input = job.get("input", {})
        job_type = job_input.get("type", "chat")

        if job_type == "chat":
            # Chat completion
            messages = job_input.get("messages", [])
            max_tokens = job_input.get("max_tokens", 1024)
            temperature = job_input.get("temperature", 0.7)

            result = generate_completion(messages, max_tokens, temperature)
            return result

        elif job_type == "rag":
            # RAG-specific completion
            question = job_input.get("question", "")
            context = job_input.get("context", "")
            chat_history = job_input.get("chat_history", [])

            system_prompt = (
                "You are a helpful RAG assistant. Answer based ONLY on the provided context. "
                "If the context doesn't contain enough information, say so."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]

            if chat_history:
                messages = [messages[0]] + chat_history + [messages[1]]

            result = generate_completion(messages)
            return {
                "generation": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
            }

        elif job_type == "health":
            # Health check
            return {"status": "healthy", "model": MODEL_NAME}

        else:
            return {"error": f"Unknown job type: {job_type}"}

    except Exception as e:
        return {"error": str(e)}


def modelfile_handler(job):
    """
    Alternative handler using RunPod's modelfile approach.
    """
    return handler(job)


# =============================================================================
# RunPod Serverless Entry Point
# =============================================================================

# For RunPod serverless, this file is used as the handler
# Set handler: "handler.handler" in RunPod dashboard

if __name__ == "__main__":
    # Local testing
    print("Testing RunPod handler locally...")
    test_job = {
        "input": {
            "type": "rag",
            "question": "What is RAG?",
            "context": "RAG combines retrieval-augmented generation with vector search.",
            "chat_history": []
        }
    }
    result = handler(test_job)
    print(json.dumps(result, indent=2))
