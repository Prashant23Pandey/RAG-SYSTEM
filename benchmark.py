"""
Concurrency Benchmark for the Agentic RAG system.

Uses asyncio to test requests-per-second under multi-user load
against the FastAPI backend or directly against vLLM.

Usage:
    python benchmark.py --url http://localhost:8000/api/chat --concurrent 10 --requests 100
    python benchmark.py --url http://localhost:8000/api/chat --mode locust
"""

import os
import sys
import time
import asyncio
import argparse
import json
import statistics
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latencies: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def rps(self) -> float:
        return self.total_requests / self.duration if self.duration > 0 else 0

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "  BENCHMARK RESULTS",
            "=" * 60,
            f"  Total Requests:    {self.total_requests}",
            f"  Successful:        {self.successful}",
            f"  Failed:            {self.failed}",
            f"  Duration:          {self.duration:.2f}s",
            f"  Throughput:        {self.rps:.1f} req/s",
            f"",
            f"  Latency:",
            f"    Average:         {self.avg_latency*1000:.0f}ms",
            f"    P50:             {self.p50*1000:.0f}ms",
            f"    P95:             {self.p95*1000:.0f}ms",
            f"    P99:             {self.p99*1000:.0f}ms",
            "=" * 60,
        ]
        if self.errors:
            lines.append(f"\n  Top errors:")
            error_counts = {}
            for e in self.errors:
                error_counts[e] = error_counts.get(e, 0) + 1
            for err, count in sorted(
                error_counts.items(), key=lambda x: -x[1]
            )[:5]:
                lines.append(f"    [{count}x] {err[:80]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async benchmark
# ---------------------------------------------------------------------------
async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    result: BenchmarkResult,
):
    """Send a single request and record timing."""
    try:
        t0 = time.perf_counter()
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            latency = time.perf_counter() - t0
            result.latencies.append(latency)
            if resp.status == 200:
                result.successful += 1
            else:
                result.failed += 1
                result.errors.append(f"HTTP {resp.status}")
    except asyncio.TimeoutError:
        result.failed += 1
        result.errors.append("Timeout")
    except Exception as e:
        result.failed += 1
        result.errors.append(str(e)[:100])


async def run_async_benchmark(
    url: str,
    total_requests: int = 100,
    concurrent: int = 10,
    questions: Optional[list[str]] = None,
) -> BenchmarkResult:
    """Run an async benchmark with controlled concurrency."""
    if questions is None:
        questions = [
            "What is LangGraph?",
            "How does ChromaDB work?",
            "Explain hybrid search.",
            "What are document guardrails?",
            "How does RAG work?",
        ]

    result = BenchmarkResult(total_requests=total_requests)
    semaphore = asyncio.Semaphore(concurrent)

    async def bounded_request(session, question):
        async with semaphore:
            payload = {
                "prompt": question,
                "session_id": f"bench-{time.time_ns()}",
            }
            await _send_request(session, url, payload, result)

    result.start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(total_requests):
            q = questions[i % len(questions)]
            tasks.append(bounded_request(session, q))
        await asyncio.gather(*tasks)

    result.end_time = time.perf_counter()
    return result


# ---------------------------------------------------------------------------
# Direct vLLM benchmark (bypasses FastAPI)
# ---------------------------------------------------------------------------
async def run_vllm_benchmark(
    vllm_url: str,
    total_requests: int = 100,
    concurrent: int = 10,
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
) -> BenchmarkResult:
    """Benchmark vLLM directly via OpenAI-compatible API."""
    url = f"{vllm_url.rstrip('/')}/v1/chat/completions"
    questions = [
        "What is LangGraph?",
        "How does ChromaDB work?",
        "Explain hybrid search.",
    ]

    result = BenchmarkResult(total_requests=total_requests)
    semaphore = asyncio.Semaphore(concurrent)

    async def bounded_request(session, question):
        async with semaphore:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                "max_tokens": 256,
                "temperature": 0,
            }
            try:
                t0 = time.perf_counter()
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    latency = time.perf_counter() - t0
                    result.latencies.append(latency)
                    if resp.status == 200:
                        result.successful += 1
                    else:
                        result.failed += 1
                        result.errors.append(f"HTTP {resp.status}")
            except asyncio.TimeoutError:
                result.failed += 1
                result.errors.append("Timeout")
            except Exception as e:
                result.failed += 1
                result.errors.append(str(e)[:100])

    result.start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(total_requests):
            q = questions[i % len(questions)]
            tasks.append(bounded_request(session, q))
        await asyncio.gather(*tasks)

    result.end_time = time.perf_counter()
    return result


# ---------------------------------------------------------------------------
# Locust benchmark generator
# ---------------------------------------------------------------------------
def generate_locustfile(url: str, questions: list[str]) -> str:
    """Generate a Locust load testing file."""
    questions_json = json.dumps(questions, indent=4)
    return f'''
"""
Locust benchmark for Agentic RAG.
Run with: locust -f locustfile.py --host http://localhost:8000
"""
import json
from locust import HttpUser, task, between


QUESTIONS = {questions_json}


class RAGUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat(self):
        import random
        question = random.choice(QUESTIONS)
        self.client.post(
            "/api/chat",
            json={{"prompt": question, "session_id": "locust-test"}},
        )
'''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark Tool")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/chat",
        help="API endpoint to benchmark",
    )
    parser.add_argument(
        "--concurrent", type=int, default=10, help="Concurrent requests"
    )
    parser.add_argument(
        "--requests", type=int, default=100, help="Total requests to send"
    )
    parser.add_argument(
        "--mode",
        choices=["async", "vllm", "locust"],
        default="async",
        help="Benchmark mode",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL for vllm mode",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name for vllm mode",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    print(f"\nBenchmarking: {args.url}")
    print(f"Mode: {args.mode} | Concurrent: {args.concurrent} | Total: {args.requests}")

    if args.mode == "locust":
        questions = [
            "What is LangGraph?",
            "How does ChromaDB work?",
            "Explain hybrid search.",
            "What are document guardrails?",
            "How does RAG work?",
        ]
        content = generate_locustfile(args.url, questions)
        with open("locustfile.py", "w") as f:
            f.write(content)
        print("\nGenerated locustfile.py")
        print("Run with: locust -f locustfile.py --host http://localhost:8000")
        return

    if args.mode == "vllm":
        result = asyncio.run(
            run_vllm_benchmark(
                args.vllm_url, args.requests, args.concurrent, args.model
            )
        )
    else:
        result = asyncio.run(
            run_async_benchmark(args.url, args.requests, args.concurrent)
        )

    print(result.summary())

    if args.save:
        data = {
            "total_requests": result.total_requests,
            "successful": result.successful,
            "failed": result.failed,
            "duration_s": round(result.duration, 2),
            "rps": round(result.rps, 1),
            "avg_latency_ms": round(result.avg_latency * 1000, 0),
            "p50_ms": round(result.p50 * 1000, 0),
            "p95_ms": round(result.p95 * 1000, 0),
            "p99_ms": round(result.p99 * 1000, 0),
        }
        with open(args.save, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
