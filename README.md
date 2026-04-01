# Transfer Learning ResNet STDF Wafer Map Yield Predictor

End-to-end wafer yield prediction pipeline that converts raw STDF (Standard Test Data Format) binary files into wafer-map images and classifies defect patterns using transfer-learned ResNet models. Includes a FastAPI backend and React dashboard for real-time inference.

## Problem

Semiconductor test floors generate STDF binary files containing millions of per-die test results. Converting this raw data into actionable defect intelligence requires parsing binary records, reconstructing spatial wafer maps, and recognizing defect patterns — a workflow that is manual, slow, and error-prone. This project automates the full pipeline from binary STDF input to classified defect output.

## Pipeline

```
STDF Binary File
      │
      ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ STDF Parser  │────▶│ Wafer Map Gen    │────▶│ ResNet Classifier │
│ (binary→die  │     │ (die coords →    │     │ (ImageNet → wafer │
│  records)    │     │  300×300 RGB)    │     │  fine-tuned)     │
└──────────────┘     └──────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
                                              Defect Class + Yield
                                              Prediction + Confidence
```

## Defect Pattern Classes

| Class | Description |
|-------|-------------|
| Normal | No systematic defect pattern |
| EdgeEffect | Die failures concentrated at wafer edge |
| CenterCluster | Defect cluster near wafer center |
| RingPattern | Concentric ring-shaped failure band |
| QuadrantFailure | Failures localized to one quadrant |
| Scratch | Linear scratch damage across wafer |
| RandomFailure | Spatially random die failures |
| MixedMode | Multiple overlapping defect signatures |

## Technical Approach

- **STDF parsing**: Custom binary parser extracts wafer ID, lot ID, die coordinates, hard bins, and soft bins from STDF v4 records
- **Wafer map generation**: Die-level results mapped onto 300×300 RGB images with bin-to-color encoding
- **Transfer learning**: ResNet-18/50 backbone pretrained on ImageNet, fine-tuned for 8-class wafer pattern classification
- **Serving**: FastAPI with async prediction endpoints, model versioning, and health monitoring

## Repository Structure

```
├── src/
│   ├── api/              # FastAPI app, inference engine, routes
│   ├── data/             # STDF parser, wafer map generator
│   ├── models/           # ResNet transfer learning, dataset class
│   └── utils/            # Config loader, logging
├── config/               # API and training configuration
├── docker/               # Dockerfiles for API and frontend
├── frontend/             # React dashboard (Vite + TypeScript)
├── scripts/              # Deployment and setup scripts
├── requirements.txt
└── docker-compose.yml
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Start the API:**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Start with Docker Compose (optional):**
```bash
docker-compose up -d
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | Classify a single wafer (STDF file or image) |
| POST | `/api/v1/predict/batch` | Submit batch prediction job |
| GET | `/api/v1/models` | List available model versions |
| GET | `/api/v1/health` | Service and model health check |

## Results (WM-811K Real Data)

Trained on 35,519 labeled wafers from the [WM-811K dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) using 3-phase progressive fine-tuning on a ResNet-18 backbone pretrained on ImageNet.

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **89.0%** |
| **Best Validation Accuracy** | 89.7% |
| Dataset Split | 24,863 train / 5,327 val / 5,329 test |
| Training Phases | Frozen head (5 ep) → Layer4 (5 ep) → Full (15 ep) |
| ONNX Export | Validated, production-ready |

### Per-Class Performance

| Defect Type | Precision | Recall | F1 Score | Support |
|------------|-----------|--------|----------|---------|
| Center | 96.6% | 94.8% | 95.7% | 652 |
| Donut | 77.9% | 93.1% | 84.8% | 72 |
| Edge-Loc | 90.6% | 88.2% | 89.4% | 821 |
| Edge-Ring | 99.7% | 91.4% | 95.4% | 1,455 |
| Loc | 83.9% | 79.2% | 81.5% | 528 |
| Near-full | 62.5% | 55.6% | 58.8% | 18 |
| Normal | 84.2% | 96.1% | 89.8% | 1,500 |
| Random | 82.3% | 69.0% | 75.0% | 126 |
| Scratch | 90.5% | 62.0% | 73.6% | 157 |

> Near-full and Scratch classes have fewer training samples (149 and 1,193 respectively), which limits per-class performance. GPU-scale training with advanced augmentation (Mixup, CutMix) would improve these.

## Requirements

- Python 3.10+
- PyTorch 2.x, ONNX Runtime
- PostgreSQL 14+ (optional, for model metadata)
- See `requirements.txt` for full list

## License

MIT
