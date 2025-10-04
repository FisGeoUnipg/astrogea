#!/bin/bash

# Astrogea Kubernetes Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="astrogea"
IMAGE_NAME="astrogea:latest"
REGISTRY=""

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi
    
    # Check docker (if building image)
    if ! command -v docker &> /dev/null; then
        log_warn "Docker not found. Skipping image build."
    fi
    
    log_info "Prerequisites check passed."
}

build_image() {
    if command -v docker &> /dev/null; then
        log_info "Building Docker image..."
        
        # Create Dockerfile if not exists
        if [ ! -f "Dockerfile" ]; then
            create_dockerfile
        fi
        
        docker build -t $IMAGE_NAME .
        
        if [ ! -z "$REGISTRY" ]; then
            log_info "Pushing image to registry..."
            docker tag $IMAGE_NAME $REGISTRY/$IMAGE_NAME
            docker push $REGISTRY/$IMAGE_NAME
            IMAGE_NAME="$REGISTRY/$IMAGE_NAME"
        fi
        
        log_info "Image built successfully: $IMAGE_NAME"
    else
        log_warn "Docker not available. Using existing image: $IMAGE_NAME"
    fi
}

create_dockerfile() {
    log_info "Creating Dockerfile..."
    cat > Dockerfile << 'EOF'
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/astrogea

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for distributed processing
RUN pip install --no-cache-dir \
    dask[complete] \
    dask-kubernetes \
    kubernetes \
    boto3 \
    google-cloud-storage \
    azure-storage-blob \
    pyyaml

# Copy astrogea source code
COPY astrogea/ ./astrogea/
COPY pyproject.toml .

# Install astrogea
RUN pip install -e .

# Create data directory
RUN mkdir -p /data /cache /var/log/astrogea

# Set environment variables
ENV ASTROGEA_MODE=kubernetes
ENV PYTHONPATH=/opt/astrogea

# Expose ports
EXPOSE 8786 8787

# Default command
CMD ["python", "-m", "astrogea.worker"]
EOF
    log_info "Dockerfile created."
}

deploy_namespace() {
    log_info "Deploying namespace and storage..."
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/storage.yaml
    log_info "Namespace and storage deployed."
}

deploy_config() {
    log_info "Deploying configuration..."
    kubectl apply -f k8s/configmap.yaml
    log_info "Configuration deployed."
}

deploy_dask_scheduler() {
    log_info "Deploying Dask scheduler..."
    
    # Update image in scheduler deployment
    sed "s|astrogea:latest|$IMAGE_NAME|g" k8s/dask-scheduler.yaml | kubectl apply -f -
    
    # Wait for scheduler to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/dask-scheduler -n $NAMESPACE
    log_info "Dask scheduler deployed and ready."
}

deploy_dask_workers() {
    log_info "Deploying Dask workers..."
    
    # Update image in workers deployment
    sed "s|astrogea:latest|$IMAGE_NAME|g" k8s/dask-workers.yaml | kubectl apply -f -
    
    # Wait for workers to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/dask-workers -n $NAMESPACE
    log_info "Dask workers deployed and ready."
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pods
    kubectl get pods -n $NAMESPACE
    
    # Check services
    kubectl get svc -n $NAMESPACE
    
    # Check scheduler logs
    log_info "Checking scheduler logs..."
    kubectl logs -l app=dask-scheduler -n $NAMESPACE --tail=10
    
    # Check worker logs
    log_info "Checking worker logs..."
    kubectl logs -l app=dask-workers -n $NAMESPACE --tail=10
    
    log_info "Deployment verification completed."
}

setup_port_forward() {
    log_info "Setting up port forwarding for dashboard..."
    log_info "Dashboard will be available at: http://localhost:8787"
    kubectl port-forward svc/dask-scheduler 8787:8787 -n $NAMESPACE &
    PORT_FORWARD_PID=$!
    echo $PORT_FORWARD_PID > .port-forward.pid
    log_info "Port forwarding started (PID: $PORT_FORWARD_PID)"
}

cleanup() {
    log_info "Cleaning up..."
    
    # Stop port forwarding
    if [ -f ".port-forward.pid" ]; then
        PID=$(cat .port-forward.pid)
        kill $PID 2>/dev/null || true
        rm .port-forward.pid
    fi
    
    log_info "Cleanup completed."
}

show_status() {
    log_info "Current deployment status:"
    echo ""
    kubectl get pods -n $NAMESPACE
    echo ""
    kubectl get svc -n $NAMESPACE
    echo ""
    kubectl get pvc -n $NAMESPACE
}

# Main deployment function
deploy() {
    log_info "Starting Astrogea Kubernetes deployment..."
    
    check_prerequisites
    build_image
    deploy_namespace
    deploy_config
    deploy_dask_scheduler
    deploy_dask_workers
    verify_deployment
    
    log_info "Deployment completed successfully!"
    log_info "You can now submit jobs to the cluster."
    
    # Ask if user wants port forwarding
    read -p "Start port forwarding for dashboard? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_port_forward
    fi
}

# Command line options
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        kubectl logs -l app=dask-scheduler -n $NAMESPACE -f
        ;;
    "worker-logs")
        kubectl logs -l app=dask-workers -n $NAMESPACE -f
        ;;
    "dashboard")
        setup_port_forward
        ;;
    *)
        echo "Usage: $0 {deploy|status|cleanup|logs|worker-logs|dashboard}"
        echo ""
        echo "Commands:"
        echo "  deploy      - Deploy astrogea to Kubernetes"
        echo "  status      - Show deployment status"
        echo "  cleanup     - Clean up port forwarding"
        echo "  logs        - Show scheduler logs"
        echo "  worker-logs - Show worker logs"
        echo "  dashboard   - Start port forwarding for dashboard"
        exit 1
        ;;
esac

