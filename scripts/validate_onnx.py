"""
ONNX Validation — compare PyTorch vs ONNX outputs

Loads both the PyTorch (.pth) and ONNX (.onnx) models, feeds
the same input through both, and verifies they produce identical
predictions.

Usage:
    cd Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor
    python scripts/validate_onnx.py
"""

import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
NUM_CLASSES = 8
IMG_SIZE = 224


def validate():
    from src.models.resnet_model import ResNetTransferLearning

    device = 'cpu'
    pth_path = os.path.join(ARTIFACTS, 'resnet18_wafer_best.pth')
    onnx_path = os.path.join(ARTIFACTS, 'resnet18_wafer.onnx')

    if not os.path.exists(pth_path):
        print(f'ERROR: PyTorch model not found at {pth_path}'); return False
    if not os.path.exists(onnx_path):
        print(f'ERROR: ONNX model not found at {onnx_path}'); return False

    pytorch_model = ResNetTransferLearning(
        architecture='resnet18', num_classes=NUM_CLASSES,
        pretrained=False, freeze_backbone=False)
    pytorch_model.load_state_dict(
        torch.load(pth_path, map_location=device, weights_only=False))
    pytorch_model.eval()
    print(f'PyTorch model loaded: {os.path.getsize(pth_path)/1e6:.1f} MB')

    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    onnx_size = os.path.getsize(onnx_path)
    onnx_data = onnx_path + '.data'
    if os.path.exists(onnx_data):
        onnx_size += os.path.getsize(onnx_data)
    print(f'ONNX model loaded:   {onnx_size/1e6:.1f} MB')

    np.random.seed(42)
    n_test = 5
    results = []

    for i in range(n_test):
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            pt_out = pytorch_model(torch.from_numpy(dummy)).numpy()
        pt_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        onnx_out = ort_session.run(None, {input_name: dummy})[0]
        onnx_time = (time.perf_counter() - t0) * 1000

        max_diff = np.abs(pt_out - onnx_out).max()
        preds_match = np.array_equal(pt_out.argmax(1), onnx_out.argmax(1))

        results.append({
            'sample': i+1, 'max_diff': float(max_diff),
            'predictions_match': bool(preds_match),
            'pytorch_ms': round(pt_time, 2), 'onnx_ms': round(onnx_time, 2),
        })
        status = 'PASS' if preds_match and max_diff < 1e-4 else 'WARN' if preds_match else 'FAIL'
        print(f'  Sample {i+1}: {status}  max_diff={max_diff:.2e}  PT={pt_time:.1f}ms  ONNX={onnx_time:.1f}ms')

    all_match = all(r['predictions_match'] for r in results)
    max_of_max = max(r['max_diff'] for r in results)
    avg_pt = np.mean([r['pytorch_ms'] for r in results])
    avg_onnx = np.mean([r['onnx_ms'] for r in results])

    print(f'\n  All match: {all_match}  Max diff: {max_of_max:.2e}')
    print(f'  Avg PT: {avg_pt:.1f}ms  Avg ONNX: {avg_onnx:.1f}ms  Speedup: {avg_pt/max(avg_onnx,0.01):.2f}x')
    print('RESULT:', 'ONNX validated.' if all_match else 'Differences detected.')

    summary = {'all_match': all_match, 'max_diff': max_of_max,
               'avg_pytorch_ms': round(avg_pt,2), 'avg_onnx_ms': round(avg_onnx,2),
               'speedup': round(avg_pt/max(avg_onnx,0.01),2), 'samples': results}
    out = os.path.join(ARTIFACTS, 'onnx_validation.json')
    with open(out, 'w') as f: json.dump(summary, f, indent=2)
    print(f'Saved → {out}')
    return all_match

if __name__ == '__main__':
    sys.exit(0 if validate() else 1)
