#!/bin/bash

# P02 Yield Predictor Deployment Script
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-development}
NAMESPACE="p02-yield-predictor"

echo "======================================"
echo "P02 Yield Predictor Deployment"
echo "Environment: $ENVIRONMENT"
echo "======================================"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed"
    exit 1
fi

echo "Building Docker images..."
docker build -t p02-api:latest -f docker/Dockerfile.api .
docker build -t p02-frontend:latest -f docker/Dockerfile.frontend .

echo "Images built successfully!"

if [ "$ENVIRONMENT" == "production" ]; then
    echo "Tagging images for production..."
    docker tag p02-api:latest p02-api:prod-$(date +%Y%m%d-%H%M%S)
    docker tag p02-frontend:latest p02-frontend:prod-$(date +%Y%m%d-%H%M%S)
    
    # Push to registry (configure your registry)
    # docker push your-registry.com/p02-api:prod-$(date +%Y%m%d-%H%M%S)
    # docker push your-registry.com/p02-frontend:prod-$(date +%Y%m%d-%H%M%S)
fi

echo "Deployment complete!"
echo ""
echo "To access the application:"
echo "  API: http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  MLflow: http://localhost:5000"
echo "  Grafana: http://localhost:3001"
echo ""
echo "Run 'docker-compose logs -f' to view logs"
