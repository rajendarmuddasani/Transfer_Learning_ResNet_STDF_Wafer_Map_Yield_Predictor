#!/bin/bash

# Setup script for P02 Yield Predictor
# Run this after cloning the repository

set -e

echo "======================================"
echo "P02 Yield Predictor Setup"
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
echo "3. Initialize database: docker-compose exec postgres psql -U p02user -d p02_yield_predictor -f /scripts/init_db.sql"
echo "4. Start API: python -m src.api.main"
echo "5. Start frontend: cd frontend && npm install && npm run dev"
echo ""
echo "For training:"
echo "  ./scripts/train.sh --config config/train_config.yaml --phase 1"
echo ""
echo "Documentation: README.md"
