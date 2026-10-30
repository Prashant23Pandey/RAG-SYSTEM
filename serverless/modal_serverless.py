"""
Modal Serverless GPU Deployment for vLLM Inference.

Deploys Llama-3.2-3B-Instruct on a single NVIDIA T4/A10G GPU with:
- Cold-start mitigation via model weight caching
- Secure API webhook endpoint for LangGraph backend
- Dynamic scaling based on request load

Usage:
    modal deploy modal_serverless.py
    modal serve modal_serverless.py
"""

import modal
import os
from typing import Optional

# =============================================================================
# Modal App & Image Configuration
# =============================================================================

app = modal.App("rag-vllm-serverless")

# Custom image with vLLM and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
    )
    .apt_install("curl", "git")
)

# =============================================================================
# Network Storage for Model Cache (Cold-Start Mitigation)
# =============================================================================

model_cache = modal.Volume.from_name("rag-model-cache", create_if_missing=True)
CACHE_PATH = "/cache/models"

# =============================================================================
# vLLM Inference Class
# =============================================================================

@app.cls(
    image=image,
    gpu="T4",  # Options: "T4", "A10G", "A100", "H100"
    timeout=600,
    container_idle_timeout=120,
    allow_concurrent_inputs=100,
    volumes={CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class VLLMInference:
    """Serverless vLLM inference with model caching."""

    @modal.enter()
    def setup(self):
        """Initialize vLLM server with cached model weights."""
        import subprocess
        import time

        print("Starting vLLM server initialization...")
        start_time = time.time()

        # Check if model is cached
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        cached_model_path = f"{CACHE_PATH}/{model_name.replace('/', '_')}"

        if os.path.exists(cached_model_path):
            print(f"Loading cached model from {cached_model_path}")
            model_name = cached_model_path
        else:
            print(f"Downloading model {model_name}...")
            os.makedirs(cached_model_path, exist_ok=True)

        # Start vLLM server in background
        self.server_process = subprocess.Popen(
            [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_name,
                "--host", "0.0.0.0",
                "--port", "8000",
                "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.9",
                "--dtype", "auto",
                "--download-dir", CACHE_PATH,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        import requests
        for i in range(60):
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"vLLM server ready in {elapsed:.2f}s")
                    break
            except:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("vLLM server failed to start")

        # Cache model weights for future cold starts
        if not os.path.exists(cached_model_path):
            print("Caching model weights for faster cold starts...")
            model_cache.commit()

    @modal.exit()
    def cleanup(self):
        """Cleanup on container exit."""
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()

    @modal.web_endpoint(method="GET")
    def health(self):
        """Health check endpoint."""
        return {"status": "healthy", "model": "meta-llama/Llama-3.2-3B-Instruct"}

    @modal.web_endpoint(method="POST")
    def generate(self, request_data: dict):
        """
        Generate completion for RAG query.

        Request format:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }
        """
        import requests
        import json

        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={
                    "model": "meta-llama/Llama-3.2-3B-Instruct",
                    "messages": request_data.get("messages", []),
                    "max_tokens": request_data.get("max_tokens", 1024),
                    "temperature": request_data.get("temperature", 0.7),
                    "top_p": request_data.get("top_p", 0.9),
                },
                timeout=120,
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    @modal.web_endpoint(method="POST")
    def rag_completion(self, request_data: dict):
        """
        RAG-specific completion endpoint for LangGraph backend.

        Request format:
        {
            "question": "What is...",
            "context": "Relevant documents...",
            "chat_history": []
        }
        """
        import requests

        system_prompt = (
            "You are a helpful RAG assistant. Answer based ONLY on the provided context. "
            "If the context doesn't contain enough information, say so."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{request_data.get('context', '')}\n\nQuestion: {request_data.get('question', '')}"},
        ]

        # Add chat history if provided
        chat_history = request_data.get("chat_history", [])
        if chat_history:
            messages = [messages[0]] + chat_history + [messages[1]]

        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={
                    "model": "meta-llama/Llama-3.2-3B-Instruct",
                    "messages": messages,
                    "max_tokens": request_data.get("max_tokens", 1024),
                    "temperature": request_data.get("temperature", 0.7),
                },
                timeout=120,
            )
            result = response.json()

            # Format for LangGraph compatibility
            return {
                "generation": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "model": result.get("model", ""),
            }
        except Exception as e:
            return {"error": str(e)}

    @modal.web_endpoint(method="POST")
    def embed(self, request_data: dict):
        """
        Embedding endpoint for vector operations.

        Request format:
        {
            "texts": ["text1", "text2", ...]
        }
        """
        import requests

        try:
            # Use vLLM's embedding endpoint if available
            response = requests.post(
                "http://localhost:8000/v1/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "input": request_data.get("texts", []),
                },
                timeout=60,
            )
            return response.json()
        except Exception as e:
            # Fallback: return empty embeddings
            return {"error": str(e), "embeddings": []}


# =============================================================================
# Local Entry Point for Testing
# =============================================================================

@app.local_entrypoint()
def main():
    """Test the serverless deployment locally."""
    import requests
    import json

    # Deploy and get the endpoint URL
    print("Deploying to Modal...")
    print("Use 'modal serve modal_serverless.py' for local testing")
    print("Use 'modal deploy modal_serverless.py' for production")

    # Example usage (after deployment):
    # inference = VLLMInference()
    # result = inference.rag_completion.remote({
    #     "question": "What is RAG?",
    #     "context": "RAG combines retrieval and generation...",
    #     "chat_history": []
    # })
    # print(result)


# =============================================================================
# Configuration for LangGraph Backend Integration
# =============================================================================

# Add to your .env file:
# MODAL_WEBHOOK_URL=https://your-username--rag-vllm-serverless-rag-endpoint.modal.run
# LLM_BACKEND=modal

# Example integration in llm_factory.py:
"""
import os
import requests

MODAL_WEBHOOK_URL = os.getenv("MODAL_WEBHOOK_URL")

def get_modal_llm():
    def _call(messages, **kwargs):
        response = requests.post(
            f"{MODAL_WEBHOOK_URL}/rag_completion",
            json={
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.7),
            },
            timeout=120,
        )
        return response.json()
    return _call
"""
