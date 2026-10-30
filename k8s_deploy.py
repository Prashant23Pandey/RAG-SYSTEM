"""
Kubernetes & Helm Deployment Script for Agentic RAG.

Automates the deployment of the RAG system to a Kubernetes cluster.

Usage:
    # Deploy to local Minikube
    python k8s_deploy.py --env local

    # Deploy to GKE
    python k8s_deploy.py --env gke --project your-project-id

    # Deploy with custom values
    python k8s_deploy.py --env production --values custom-values.yaml
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger("rag.k8s_deploy")

# =============================================================================
# Configuration
# =============================================================================

ENVIRONMENTS = {
    "local": {
        "context": "minikube",
        "namespace": "rag-local",
        "values_file": "values-local.yaml",
    },
    "staging": {
        "context": "gke_your-project_us-central1-staging",
        "namespace": "rag-staging",
        "values_file": "values-staging.yaml",
    },
    "production": {
        "context": "gke_your-project_us-central1",
        "namespace": "rag-production",
        "values_file": "values-production.yaml",
    },
}

HELM_CHART_PATH = "./helm/rag-system"
NAMESPACE = "rag-system"


# =============================================================================
# Deployment Functions
# =============================================================================

def run_command(cmd: list, capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        check=check,
    )
    if capture_output and result.stdout:
        logger.debug(result.stdout)
    return result


def setup_kubernetes_context(env: str, project_id: Optional[str] = None) -> Dict:
    """Setup Kubernetes context for the target environment."""
    config = ENVIRONMENTS.get(env, ENVIRONMENTS["local"])

    print("\n" + "=" * 60)
    print(f"  Setting up Kubernetes context for: {env}")
    print("=" * 60 + "\n")

    if env == "local":
        # Start Minikube if not running
        try:
            run_command(["minikube", "status"], check=False)
        except:
            print("Starting Minikube...")
            run_command(["minikube", "start", "--memory=8192", "--cpus=4", "--gpus=all"])

    elif env in ["staging", "production"]:
        # Setup GKE context
        if not project_id:
            project_id = os.getenv("GCP_PROJECT_ID")

        cluster_name = config["context"].split("_")[1]
        region = config["context"].split("_")[2]

        print(f"Getting GKE credentials for {cluster_name}...")
        run_command([
            "gcloud", "container", "clusters", "get-credentials",
            cluster_name,
            "--region", region,
            "--project", project_id,
        ])

    # Create namespace
    print(f"Creating namespace: {config['namespace']}")
    run_command([
        "kubectl", "create", "namespace", config["namespace"],
        "--dry-run=client", "-o=yaml", "|", "kubectl", "apply", "-f", "-",
    ], check=False)

    return config


def install_cert_manager():
    """Install Cert-Manager for automatic SSL certificates."""
    print("\nInstalling Cert-Manager...")

    # Add Jetstack Helm repo
    run_command([
        "helm", "repo", "add", "jetstack", "https://charts.jetstack.io",
    ], check=False)

    run_command(["helm", "repo", "update"])

    # Install Cert-Manager
    run_command([
        "helm", "install", "cert-manager", "jetstack/cert-manager",
        "--namespace", "cert-manager",
        "--create-namespace",
        "--set", "installCRDs=true",
        "--wait",
    ], check=False)

    print("Cert-Manager installed successfully!")


def install_ingress_controller():
    """Install NGINX Ingress Controller."""
    print("\nInstalling NGINX Ingress Controller...")

    # Add ingress-nginx Helm repo
    run_command([
        "helm", "repo", "add", "ingress-nginx",
        "https://kubernetes.github.io/ingress-nginx",
    ], check=False)

    run_command(["helm", "repo", "update"])

    # Install ingress-nginx
    run_command([
        "helm", "install", "ingress-nginx", "ingress-nginx/ingress-nginx",
        "--namespace", "ingress-nginx",
        "--create-namespace",
        "--set", "controller.replicaCount=2",
        "--set", "controller.resources.limits.memory=512Mi",
        "--wait",
    ], check=False)

    print("NGINX Ingress Controller installed successfully!")


def install_gpu_operator():
    """Install NVIDIA GPU Operator for GPU support."""
    print("\nInstalling NVIDIA GPU Operator...")

    # Add NVIDIA Helm repo
    run_command([
        "helm", "repo", "add", "nvidia",
        "https://nvidia.github.io/gpu-operator",
    ], check=False)

    run_command(["helm", "repo", "update"])

    # Install GPU Operator
    run_command([
        "helm", "install", "gpu-operator", "nvidia/gpu-operator",
        "--namespace", "gpu-operator",
        "--create-namespace",
        "--wait",
    ], check=False)

    print("NVIDIA GPU Operator installed successfully!")


def deploy_rag(env: str, config: Dict, values_file: Optional[str] = None):
    """Deploy the RAG system using Helm."""
    print("\n" + "=" * 60)
    print("  Deploying Agentic RAG System")
    print("=" * 60 + "\n")

    namespace = config["namespace"]
    release_name = f"rag-{env}"

    # Select values file
    if values_file:
        values_path = values_file
    else:
        values_path = os.path.join(HELM_CHART_PATH, config["values_file"])

    # Check if values file exists
    if not os.path.exists(values_path):
        print(f"Values file not found: {values_path}")
        print("Using default values...")

    # Build Helm command
    cmd = [
        "helm", "upgrade", "--install", release_name,
        HELM_CHART_PATH,
        "--namespace", namespace,
        "--create-namespace",
    ]

    if os.path.exists(values_path):
        cmd.extend(["-f", values_path])

    # Add environment-specific values
    if env == "production":
        cmd.extend([
            "--set", "backend.replicaCount=3",
            "--set", "vllm.autoscaling.enabled=true",
            "--set", "ingress.enabled=true",
            "--set", "certManager.enabled=true",
        ])

    # Deploy
    print(f"Deploying Helm release: {release_name}")
    run_command(cmd)

    print("\nDeployment complete!")
    print(f"Namespace: {namespace}")
    print(f"Release: {release_name}")


def verify_deployment(env: str, config: Dict):
    """Verify the deployment is running correctly."""
    print("\n" + "=" * 60)
    print("  Verifying Deployment")
    print("=" * 60 + "\n")

    namespace = config["namespace"]

    # Check pods
    print("Checking pods...")
    result = run_command(
        ["kubectl", "get", "pods", "-n", namespace, "-o", "wide"],
        capture_output=True,
    )
    print(result.stdout)

    # Check services
    print("\nChecking services...")
    result = run_command(
        ["kubectl", "get", "services", "-n", namespace],
        capture_output=True,
    )
    print(result.stdout)

    # Check ingress
    print("\nChecking ingress...")
    result = run_command(
        ["kubectl", "get", "ingress", "-n", namespace],
        capture_output=True,
    )
    print(result.stdout)

    # Check HPA
    print("\nChecking horizontal pod autoscalers...")
    result = run_command(
        ["kubectl", "get", "hpa", "-n", namespace],
        capture_output=True,
    )
    print(result.stdout)

    # Get endpoints
    print("\nEndpoints:")
    result = run_command(
        ["kubectl", "get", "endpoints", "-n", namespace],
        capture_output=True,
    )
    print(result.stdout)


def get_service_urls(env: str, config: Dict):
    """Get the URLs for deployed services."""
    namespace = config["namespace"]

    print("\n" + "=" * 60)
    print("  Service URLs")
    print("=" * 60 + "\n")

    if env == "local":
        # Minikube
        print("For Minikube, use these commands to access services:")
        print(f"  Frontend: minikube service rag-{env}-frontend -n {namespace} --url")
        print(f"  Backend: minikube service rag-{env}-backend -n {namespace} --url")
        print(f"  vLLM: minikube service rag-{env}-vllm -n {namespace} --url")
    else:
        # GKE - get external IPs
        result = run_command(
            ["kubectl", "get", "ingress", "-n", namespace, "-o", "json"],
            capture_output=True,
        )

        try:
            ingress = json.loads(result.stdout)
            for item in ingress.get("items", []):
                hostname = item.get("status", {}).get("loadBalancer", {}).get("ingress", [{}])[0].get("hostname")
                if hostname:
                    print(f"  Ingress: http://{hostname}")
        except:
            print("  Could not retrieve ingress URLs")


# =============================================================================
# Main Deployment
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deploy Agentic RAG to Kubernetes")
    parser.add_argument(
        "--env",
        choices=["local", "staging", "production"],
        default="local",
        help="Target environment",
    )
    parser.add_argument("--project-id", help="GCP project ID (for GKE)")
    parser.add_argument("--values", help="Custom Helm values file")
    parser.add_argument("--skip-infra", action="store_true", help="Skip infrastructure setup")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing deployment")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without deploying")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Setup Kubernetes context
    config = setup_kubernetes_context(args.env, args.project_id)

    if args.verify_only:
        verify_deployment(args.env, config)
        get_service_urls(args.env, config)
        return

    # Install infrastructure components
    if not args.skip_infra:
        try:
            install_cert_manager()
        except Exception as e:
            print(f"Cert-Manager installation failed (may already be installed): {e}")

        try:
            install_ingress_controller()
        except Exception as e:
            print(f"Ingress controller installation failed (may already be installed): {e}")

        # Install GPU operator for GPU nodes
        if args.env == "production":
            try:
                install_gpu_operator()
            except Exception as e:
                print(f"GPU Operator installation failed: {e}")

    # Deploy RAG system
    if not args.dry_run:
        deploy_rag(args.env, config, args.values)
        verify_deployment(args.env, config)
        get_service_urls(args.env, config)
    else:
        print("\nDry run - would deploy with:")
        print(f"  Environment: {args.env}")
        print(f"  Namespace: {config['namespace']}")
        print(f"  Values file: {args.values or config['values_file']}")


if __name__ == "__main__":
    main()
