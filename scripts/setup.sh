#!/bin/bash

# Setup script for Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor
# Run this after cloning the repository

set -e

echo "======================================"
echo "Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor Setup"
echo "======================================"

# Check Python version
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw,processed,wafer_maps/{train,val,test}}
mkdir -p models/{checkpoints,onnx,tensorrt}
mkdir -p logs/{tensorboard,training}
mkdir -p tmp/uploads

# Setup Git LFS (if available)
if command -v git-lfs &> /dev/null; then
    echo "Initializing Git LFS..."
    git lfs install
fi

# Setup DVC (if needed)
if command -v dvc &> /dev/null; then
    echo "Initializing DVC..."
    dvc init --no-scm || true
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start services: docker-compose up -d"
echo "3. Initialize database: docker-compose exec postgres psql -U waferuser -d stdf_wafer_map_yield_predictor -f /docker-entrypoint-initdb.d/init.sql"
echo "4. Start API: python -m src.api.main"
echo "5. Start frontend: cd frontend && npm install && npm run dev"
echo ""
echo "Training artifacts and a verified training entrypoint are not currently included in this repo."
echo "If you are targeting an NVIDIA runtime, install optional GPU extras with: pip install -r requirements-gpu.txt"
echo "See the workspace pending assessment before claiming local training readiness."
echo ""
echo "Documentation: README.md"
