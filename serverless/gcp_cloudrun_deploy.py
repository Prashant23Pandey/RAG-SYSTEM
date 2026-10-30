"""
GCP Cloud Run Deployment for vLLM Inference.

Deploys vLLM to Google Cloud Run with GPU support.

Usage:
    # Install gcloud CLI
    # Authenticate: gcloud auth login
    # Deploy: python gcp_cloudrun_deploy.py

Requirements:
    - Google Cloud SDK (gcloud)
    - Cloud Run API enabled
    - GPU quota allocated
"""

import os
import subprocess
import json
from typing import Optional

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
REGION = os.getenv("GCP_REGION", "us-central1")
SERVICE_NAME = "rag-vllm-server"
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/{SERVICE_NAME}"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# =============================================================================
# Deployment Functions
# =============================================================================

def build_and_push_image():
    """Build and push Docker image to Google Container Registry."""
    print("Building Docker image...")

    # Create Dockerfile
    dockerfile_content = """
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3-pip \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install vLLM and dependencies
RUN pip3 install --no-cache-dir \\
    vllm>=0.6.0 \\
    torch>=2.1.0 \\
    transformers>=4.40.0 \\
    accelerate>=0.30.0

# Set environment variables
ENV MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
ENV CACHE_DIR=/workspace/cache
ENV PORT=8080

# Create cache directory
RUN mkdir -p /workspace/cache

# Set working directory
WORKDIR /workspace

# Expose port
EXPOSE 8080

# Start vLLM server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \\
     "--model", "meta-llama/Llama-3.2-3B-Instruct", \\
     "--host", "0.0.0.0", \\
     "--port", "8080", \\
     "--max-model-len", "4096", \\
     "--gpu-memory-utilization", "0.9"]
"""

    with open("Dockerfile.cloudrun", "w") as f:
        f.write(dockerfile_content)

    # Build image
    subprocess.run([
        "gcloud", "builds", "submit",
        "--tag", IMAGE_NAME,
        "--project", PROJECT_ID,
        ".",
    ], check=True)

    print(f"Image pushed to {IMAGE_NAME}")


def deploy_to_cloud_run():
    """Deploy to Cloud Run with GPU."""
    print("Deploying to Cloud Run...")

    subprocess.run([
        "gcloud", "run", "deploy", SERVICE_NAME,
        "--image", IMAGE_NAME,
        "--region", REGION,
        "--project", PROJECT_ID,
        "--platform", "managed",
        "--memory", "16Gi",
        "--cpu", "4",
        "--gpu", "1",
        "--gpu-type", "nvidia-l4",
        "--min-instances", "0",
        "--max-instances", "3",
        "--concurrency", "10",
        "--timeout", "600",
        "--allow-unauthenticated",
        "--set-env-vars", f"MODEL_NAME={MODEL_NAME}",
    ], check=True)

    # Get service URL
    result = subprocess.run([
        "gcloud", "run", "services", "describe", SERVICE_NAME,
        "--region", REGION,
        "--project", PROJECT_ID,
        "--format", "value(status.url)",
    ], capture_output=True, text=True)

    service_url = result.stdout.strip()
    print(f"Service deployed at: {service_url}")
    return service_url


def create_domain_mapping(domain: str):
    """Map custom domain to Cloud Run service."""
    subprocess.run([
        "gcloud", "run", "domain-mappings", "create",
        "--service", SERVICE_NAME,
        "--domain", domain,
        "--region", REGION,
        "--project", PROJECT_ID,
    ], check=True)

    print(f"Domain mapping created for {domain}")
    print("Update your DNS records as instructed by gcloud")


def setup_vpc_connector():
    """Setup VPC connector for private networking (optional)."""
    connector_name = "rag-vpc-connector"

    subprocess.run([
        "gcloud", "compute", "networks", "vpc-access", "connectors", "create",
        connector_name,
        "--region", REGION,
        "--project", PROJECT_ID,
        "--network", "default",
        "--range", "10.8.0.0/28",
    ], check=True)

    print(f"VPC connector {connector_name} created")


# =============================================================================
# Main Deployment
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GCP Cloud Run Deployment for vLLM")
    print("=" * 60)

    # Build and push image
    build_and_push_image()

    # Deploy to Cloud Run
    service_url = deploy_to_cloud_run()

    # Save deployment info
    deployment_info = {
        "project_id": PROJECT_ID,
        "region": REGION,
        "service_name": SERVICE_NAME,
        "service_url": service_url,
        "model": MODEL_NAME,
    }

    with open("cloudrun_deployment.json", "w") as f:
        json.dump(deployment_info, f, indent=2)

    print("\nDeployment complete!")
    print(f"Service URL: {service_url}")
    print("Deployment info saved to cloudrun_deployment.json")

    # Optional: Setup domain mapping
    # create_domain_mapping("vllm.yourdomain.com")
