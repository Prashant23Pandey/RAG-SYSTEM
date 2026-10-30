# Phase 10: Cloud Deployment, Kubernetes Scaling, & Edge AI

Production-grade deployment solutions for the Agentic RAG system.

## Overview

Phase 10 provides three deployment strategies:

1. **Kubernetes & Helm** - Production-grade orchestration with auto-scaling
2. **Serverless GPU** - Dynamic GPU instances on Modal/RunPod/GCP Cloud Run
3. **Edge AI** - Offline-first RAG in browsers and desktop apps

## Directory Structure

```
├── helm/                          # Helm chart for Kubernetes
│   └── rag-system/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── vllm-deployment.yaml
│           ├── vllm-service.yaml
│           ├── vllm-hpa.yaml
│           ├── chromadb-statefulset.yaml
│           ├── backend-deployment.yaml
│           ├── frontend-deployment.yaml
│           ├── ingress.yaml
│           └── cert-manager-issuer.yaml
│
├── k8s/                           # Raw Kubernetes manifests
│
├── serverless/                    # Serverless GPU deployments
│   ├── modal_serverless.py        # Modal deployment
│   ├── runpod_handler.py          # RunPod handler
│   ├── Dockerfile.runpod          # RunPod Docker image
│   └── gcp_cloudrun_deploy.py     # GCP Cloud Run
│
├── edge/                          # Edge/On-Device RAG
│   ├── export_onnx.py             # ONNX model exporter
│   ├── edge-rag.js                # Browser RAG implementation
│   ├── edge-rag.d.ts              # TypeScript definitions
│   ├── package.json               # NPM package config
│   └── README.md                  # Edge RAG documentation
│
└── k8s_deploy.py                  # Kubernetes deployment script
```

## Quick Start

### Option 1: Local Kubernetes (Minikube)

```bash
# Start Minikube with GPU support
minikube start --memory=8192 --cpus=4 --gpus=all

# Deploy using the script
python k8s_deploy.py --env local

# Access services
minikube service rag-local-frontend -n rag-local --url
```

### Option 2: Serverless GPU (Modal)

```bash
# Install Modal CLI
pip install modal

# Setup Modal token
modal token set

# Deploy to Modal
modal deploy serverless/modal_serverless.py

# Get endpoint URL
# Output: https://your-username--rag-vllm-serverless-rag-endpoint.modal.run
```

### Option 3: Edge RAG (Browser)

```bash
# Export models to ONNX
cd edge
pip install optimum[onnxruntime] transformers
python export_onnx.py embeddings
python export_onnx.py slm
python export_onnx.py config

# Install JavaScript dependencies
npm install

# Start local server
npx http-server .
# Open http://localhost:8080
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, or Minikube)
- Helm 3.x
- kubectl configured
- NVIDIA GPU Operator (for GPU nodes)

### Infrastructure Setup

```bash
# Install Cert-Manager for SSL
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager --namespace cert-manager --create-namespace --set installCRDs=true

# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --create-namespace

# Install GPU Operator (for GPU nodes)
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm install gpu-operator nvidia/gpu-operator --namespace gpu-operator --create-namespace
```

### Deploy RAG System

```bash
# Deploy to local
python k8s_deploy.py --env local

# Deploy to GKE
python k8s_deploy.py --env gke --project your-project-id

# Deploy to production
python k8s_deploy.py --env production --values values-production.yaml
```

### Configuration

Edit `helm/rag-system/values.yaml` to customize:

- **vLLM**: GPU resources, model configuration, autoscaling
- **ChromaDB**: Storage size, replication
- **Backend**: Replicas, environment variables
- **Ingress**: Domain, SSL certificates

### Auto-Scaling

The Helm chart includes Horizontal Pod Autoscalers (HPA) for:

- **vLLM**: Scales based on GPU memory usage and request queue depth
- **Backend**: Scales based on CPU and memory utilization

```yaml
# Example HPA configuration
vllm:
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 4
    targetGPUUtilization: 80
    targetRequestQueueDepth: 10
```

## Serverless GPU Deployment

### Modal

```python
# Deploy to Modal
import modal

app = modal.App("rag-vllm")

@app.cls(gpu="T4")
class VLLMInference:
    @modal.web_endpoint(method="POST")
    def rag_completion(self, request_data):
        # Your inference code
        pass

# Deploy
modal deploy modal_serverless.py
```

**Features:**
- Cold-start mitigation via model weight caching
- Automatic scaling based on request load
- Secure webhook endpoints
- GPU options: T4, A10G, A100, H100

### RunPod

```bash
# Build Docker image
docker build -t your-registry/rag-vllm-runpod -f Dockerfile.runpod .
docker push your-registry/rag-vllm-runpod

# Deploy on RunPod dashboard
# Set handler: "handler.handler"
```

### GCP Cloud Run

```bash
# Deploy to Cloud Run
python gcp_cloudrun_deploy.py

# Or manually
gcloud run deploy rag-vllm \
  --image gcr.io/your-project/rag-vllm \
  --region us-central1 \
  --memory 16Gi \
  --gpu 1 \
  --gpu-type nvidia-l4
```

## Edge AI Deployment

### ONNX Model Export

```bash
# Export embedding model
python edge/export_onnx.py embeddings --model nomic-ai/nomic-embed-text-v1

# Export small language model
python edge/export_onnx.py slm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Generate transformers.js config
python edge/export_onnx.py config
```

### Browser Integration

```html
<!DOCTYPE html>
<html>
<head>
  <title>Edge RAG</title>
</head>
<body>
  <div id="rag-container"></div>

  <script type="module">
    import { EdgeRAGUI } from './edge-rag.js';

    const ragUI = new EdgeRAGUI('rag-container', {
      cloudEndpoint: 'https://api.example.com/rag',
      enableWebGPU: true,
    });

    await ragUI.initialize();
  </script>
</body>
</html>
```

### Desktop Apps (Tauri/Electron)

```javascript
import { DesktopRAG } from './edge-rag.js';

const rag = new DesktopRAG({
  cacheDir: await getDesktopCacheDir(),
  enableWebGPU: true,
});

await rag.initialize();

// Sync with cloud for offline access
await rag.syncWithCloud();

// Export for fully offline use
await rag.exportForOffline('./offline-data.json');
```

### Cloud Fallback Logic

The edge RAG automatically falls back to cloud when:

1. **Complex Keywords**: Query contains "search", "web", "current", etc.
2. **Insufficient Context**: No relevant documents found locally
3. **Long Queries**: Query longer than 200 characters

## Environment Variables

```bash
# Kubernetes
K8S_NAMESPACE=rag-system
HELM_RELEASE=rag-production

# Serverless GPU
MODAL_WEBHOOK_URL=https://your-endpoint.modal.run
RUNPOD_API_KEY=your-runpod-key
GCP_PROJECT_ID=your-project-id

# Edge RAG
CLOUD_API_ENDPOINT=https://api.example.com/rag
EDGE_MODELS_DIR=./onnx
ENABLE_WEBGPU=true
```

## Monitoring

### Prometheus Metrics

The Helm chart includes ServiceMonitor for Prometheus scraping:

```yaml
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
```

### Grafana Dashboard

Import the RAG system dashboard from `dashboards/rag-dashboard.json`.

## Troubleshooting

### vLLM Pod Not Starting

```bash
# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check pod events
kubectl describe pod <pod-name> -n rag-system
```

### Cold Start Issues (Modal)

```bash
# Check model cache
modal volume ls rag-model-cache

# Force cache rebuild
modal volume delete rag-model-cache
```

### Edge RAG Not Loading

```bash
# Check ONNX model files
ls -la edge/models/

# Verify WebGPU support
# Open Chrome DevTools → Console → navigator.gpu
```

## Cost Optimization

### Kubernetes

- Use spot instances for non-production workloads
- Configure HPA to scale down during off-peak hours
- Use Preemptible VMs for batch processing

### Serverless

- Modal: ~$0.50/hour for T4 GPU
- RunPod: ~$0.40/hour for RTX 3090
- GCP Cloud Run: ~$0.05/GPU-second

### Edge

- Zero inference costs (runs on user's device)
- Reduced latency for local queries
- Offline capability reduces API calls

## Security

- SSL certificates via Cert-Manager (automatic renewal)
- Network policies for pod-to-pod communication
- PodDisruptionBudgets for high availability
- Secrets management via Kubernetes Secrets

## License

MIT
