# PDF and Repository Claim Audit

## Current Verdict

The source resume's Project 4 wording references ResNet-18, eight wafer-pattern
classes, FastAPI, and React. Those architecture elements exist, but the prior
repository did not support its public 89.0% WM-811K result: the exact source
pickle, result JSON, checkpoint, ONNX model, split manifest, and environment were
absent. The historical row split also had no duplicate or lot isolation.

Project 4 is therefore represented by a new independently generated synthetic
benchmark and a matching hash-verified image-classification runtime. The old
WM-811K metric is unsupported lineage and is not eligible for the resume.

## Claim Reconciliation

| Claim | Prior repository evidence | Audit result | Safe wording |
|---|---|---|---|
| ResNet-18 classifies wafer patterns | Model wrapper and unverified scripts | Implemented and reproduced on a new grouped synthetic benchmark | ResNet-18 synthetic wafer-pattern classifier |
| Eight defect classes | Synthetic taxonomy in model/API | Confirmed; one canonical class order now binds generator, evidence, ONNX, API, and UI | Eight independently generated synthetic pattern classes |
| 89.0% WM-811K test accuracy | README table only | Unsupported; required source and artifacts absent | Do not publish as a current metric |
| End-to-end STDF classification | Hand-written partial parser | Unsupported; parser fields and multi-wafer behavior are not specification-validated | STDF integration remains a future gate |
| Wafer yield prediction | Untrained regression classes and nullable API placeholder | Not implemented | Do not claim yield prediction |
| Production-ready ONNX | No prior model artifact | Replaced by a hash-verified confirmed synthetic ONNX reference | Deployment reference; not production-proven |

## Confirmed Public Evidence

- 1,920 training, 480 validation, and 800 confirmation synthetic wafer maps.
- Train, validation, and confirmation pattern families are disjoint.
- Frozen ImageNet ResNet-18 feature extractor with a validation-selected linear
  head (`C=0.1`).
- Validation-only temperature scaling (`T=0.71947`).
- 93.63% confirmation accuracy (95% Wilson CI 91.71-95.12%).
- 0.9361 macro F1 (95% stratified bootstrap CI 0.9200-0.9514).
- 0.9273 MCC (95% stratified bootstrap CI 0.9089-0.9446).
- 0.9966 macro ROC-AUC and 0.9801 macro PR-AUC.
- 0.0270 top-label expected calibration error.
- 83.0% minimum class recall for QuadrantFailure.
- Exact PyTorch/ONNX parity gate and SHA-256-bound runtime metadata.
- Local ONNX CPU latency: 3.76 ms p50 and 10.72 ms p95 over 100 iterations.

## Resume Gate

Cumulative resume v54 may include Project 4 only when the wording:

1. says independently generated synthetic wafer maps;
2. uses 93.63% accuracy and 0.936 macro F1 from grouped confirmation;
3. mentions 83% minimum class recall or the QuadrantFailure tail;
4. describes the ONNX/FastAPI/React implementation as a reference deployment;
5. excludes WM-811K, production silicon, STDF parsing, and yield-prediction outcomes.
