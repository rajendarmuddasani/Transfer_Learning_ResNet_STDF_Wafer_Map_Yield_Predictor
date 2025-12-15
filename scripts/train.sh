#!/bin/bash

# Training Script for P02 Yield Predictor
# Usage: ./train.sh --config config/train_config.yaml --phase 1

set -e

CONFIG_FILE="config/train_config.yaml"
PHASE=1
EPOCHS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "P02 Model Training"
echo "Config: $CONFIG_FILE"
echo "Phase: $PHASE"
echo "Epochs: $EPOCHS"
echo "======================================"

# Check if Python environment is set up
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No GPU detected, training will use CPU"
fi

# Create necessary directories
mkdir -p logs/tensorboard
mkdir -p models/checkpoints
mkdir -p data/wafer_maps/{train,val,test}

echo ""
echo "Starting training..."
echo ""

# Run training (placeholder - actual implementation would be in a Python script)
# python scripts/train_model.py --config $CONFIG_FILE --phase $PHASE --epochs $EPOCHS

echo "Training script placeholder - implement train_model.py for actual training"
echo ""
echo "Next steps:"
echo "1. Prepare your wafer map dataset in data/wafer_maps/"
echo "2. Implement train_model.py with your training loop"
echo "3. Monitor training with: tensorboard --logdir logs/tensorboard"
echo "4. View experiments with: mlflow ui"
