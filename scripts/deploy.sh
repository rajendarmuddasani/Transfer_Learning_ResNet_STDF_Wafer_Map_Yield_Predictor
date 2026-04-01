#!/bin/bash

# Local stack build script for Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-development}
NAMESPACE="stdf-wafer-map-yield-predictor"

echo "======================================"
echo "Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor Build"
echo "Environment: $ENVIRONMENT"
echo "======================================"

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed"
    exit 1
fi

echo "Building Docker images..."
docker build -t ${NAMESPACE}-api:latest -f docker/Dockerfile.api .
docker build -t ${NAMESPACE}-frontend:latest -f docker/Dockerfile.frontend .

echo "Images built successfully!"

if [ "$ENVIRONMENT" == "production" ]; then
    echo "Tagging images for production..."
    docker tag ${NAMESPACE}-api:latest ${NAMESPACE}-api:prod-$(date +%Y%m%d-%H%M%S)
    docker tag ${NAMESPACE}-frontend:latest ${NAMESPACE}-frontend:prod-$(date +%Y%m%d-%H%M%S)
    
    # Push to registry (configure your registry)
    # docker push your-registry.com/${NAMESPACE}-api:prod-$(date +%Y%m%d-%H%M%S)
    # docker push your-registry.com/${NAMESPACE}-frontend:prod-$(date +%Y%m%d-%H%M%S)
fi

echo "Build step complete."
echo ""
echo "To access the application:"
echo "  API: http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  MLflow: http://localhost:5000"
echo "  Prometheus: http://localhost:9090"
echo ""
echo "This script does not prove production deployment readiness by itself."
echo "Run 'docker-compose logs -f' to view logs"
