"""
vLLM High-Throughput Inference Server.

Deployment script and Docker configuration for serving open-source LLMs
with PagedAttention and continuous batching.

Usage:
    # Start vLLM server
    python vllm_server.py --model meta-llama/Llama-3.2-3B-Instruct

    # Or via docker compose
    docker compose up vllm
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("rag.vllm_server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
DEFAULT_TENSOR_PARALLEL = 1
DEFAULT_GPU_MEMORY_UTIL = 0.9
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_BATCH_SIZE = 64


def build_vllm_command(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tensor_parallel: int = DEFAULT_TENSOR_PARALLEL,
    gpu_memory_util: float = DEFAULT_GPU_MEMORY_UTIL,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_num_seqs: int = DEFAULT_BATCH_SIZE,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    quantization: Optional[str] = None,
) -> list[str]:
    """Build the vLLM server command."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--dtype", dtype,
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if quantization:
        cmd.extend(["--quantization", quantization])

    return cmd


def start_vllm_server(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    **kwargs,
):
    """Start the vLLM server as a subprocess."""
    cmd = build_vllm_command(model=model, host=host, port=port, **kwargs)
    logger.info("Starting vLLM server: %s", " ".join(cmd))
    print(f"\n{'='*60}")
    print(f"  vLLM Server")
    print(f"  Model: {model}")
    print(f"  Endpoint: http://{host}:{port}")
    print(f"  OpenAI-compatible: http://{host}:{port}/v1")
    print(f"{'='*60}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nvLLM server stopped.")
    except FileNotFoundError:
        logger.error(
            "vllm not installed. Install with: pip install vllm"
        )
        sys.exit(1)


def wait_for_vllm(url: str, timeout: int = 120) -> bool:
    """Wait for vLLM server to be ready."""
    import requests

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                logger.info("vLLM server ready at %s", url)
                return True
        except Exception:
            pass
        time.sleep(2)

    logger.error("vLLM server not ready after %ds", timeout)
    return False


# ---------------------------------------------------------------------------
# Docker Compose snippet generator
# ---------------------------------------------------------------------------
def generate_docker_compose_snippet(
    model: str = DEFAULT_MODEL,
    port: int = DEFAULT_PORT,
    gpu_count: int = 1,
    memory: str = "8g",
) -> str:
    """Generate a docker-compose service snippet for vLLM."""
    return f"""
  # ------------------------------------------------------------------
  # vLLM — High-throughput inference server (Phase 9)
  # ------------------------------------------------------------------
  vllm:
    image: vllm/vllm-openai:latest
    container_name: rag-vllm
    ports:
      - "{port}:{port}"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${{HF_TOKEN:-}}
    volumes:
      - vllm_cache:/root/.cache/huggingface
    command: >
      --model {model}
      --host 0.0.0.0
      --port {port}
      --tensor-parallel-size {gpu_count}
      --gpu-memory-utilization 0.9
      --max-model-len 4096
      --max-num-seqs 64
      --dtype auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {gpu_count}
              capabilities: [gpu]
        limits:
          memory: {memory}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 15s
      timeout: 10s
      retries: 10
      start_period: 60s
    restart: unless-stopped
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="vLLM Server Manager")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--tensor-parallel", type=int, default=DEFAULT_TENSOR_PARALLEL,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-util", type=float, default=DEFAULT_GPU_MEMORY_UTIL,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--quantization", type=str, default=None,
        help="Quantization method (awq, gptq, etc.)",
    )
    parser.add_argument(
        "--generate-docker", action="store_true",
        help="Print docker-compose snippet and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.generate_docker:
        snippet = generate_docker_compose_snippet(
            model=args.model, port=args.port
        )
        print(snippet)
        return

    start_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel=args.tensor_parallel,
        gpu_memory_util=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_batch_size,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
