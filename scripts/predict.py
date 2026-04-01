"""
Predict on Unseen Wafer Maps — ResNet-18 Yield Predictor

Loads the trained ResNet-18 model and classifies new wafer map
images into 8 defect categories.

Usage:
    cd Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor
    python scripts/predict.py                          # synthetic samples
    python scripts/predict.py --image path/to/img.png  # single image
    python scripts/predict.py --dir   path/to/folder   # folder of images
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.resnet_model import ResNetTransferLearning

# ── Constants ────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Normal', 'EdgeEffect', 'CenterCluster', 'RingPattern',
    'QuadrantFailure', 'Scratch', 'RandomFailure', 'MixedMode',
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
DEVICE = 'mps' if torch.backends.mps.is_available() else (
         'cuda' if torch.cuda.is_available() else 'cpu')

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Helpers ──────────────────────────────────────────────────────────

def load_model():
    model = ResNetTransferLearning(
        architecture='resnet18', num_classes=NUM_CLASSES,
        pretrained=False, freeze_backbone=False)
    ckpt = os.path.join(ARTIFACTS, 'resnet18_wafer_best.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f'Model loaded on {DEVICE}')
    return model


def preprocess(img: Image.Image) -> torch.Tensor:
    return INFERENCE_TRANSFORM(img.convert('RGB')).unsqueeze(0).to(DEVICE)


def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return int(probs.argmax()), probs


def generate_synthetic_images(n=8):
    """Create n synthetic wafer maps with distinct defect patterns."""
    images, labels = [], []
    size = 300
    for i in range(n):
        rng = np.random.RandomState(100 + i)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        Y, X = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        wafer = dist < (size * 0.45)
        img[wafer] = [30, 30, 30]

        cls = i % NUM_CLASSES
        if cls == 0:    # Normal
            noise = rng.randint(0, 15, img.shape, dtype=np.uint8)
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        elif cls == 1:  # EdgeEffect
            ring = wafer & (dist > size * 0.37)
            img[ring] = [180, 40, 40]
        elif cls == 2:  # CenterCluster
            centre = dist < (size * 0.10)
            img[centre] = [40, 180, 40]
        elif cls == 3:  # RingPattern
            ring = wafer & (dist > size * 0.20) & (dist < size * 0.28)
            img[ring] = [40, 40, 180]
        elif cls == 4:  # QuadrantFailure
            quad = wafer & (X > cx) & (Y < cy)
            img[quad] = [180, 180, 40]
        elif cls == 5:  # Scratch
            for y in range(size):
                x = int(cx + 0.3 * (y - cy) + rng.randn() * 2)
                if 0 <= x < size and wafer[y, x]:
                    img[max(0,y-1):y+2, max(0,x-1):x+2] = [180, 80, 40]
        elif cls == 6:  # RandomFailure
            pts = rng.randint(0, size, size=(80, 2))
            for py, px in pts:
                if wafer[py, px]:
                    img[max(0,py-2):py+3, max(0,px-2):px+3] = [180, 40, 180]
        elif cls == 7:  # MixedMode
            ring = wafer & (dist > size * 0.35)
            img[ring] = [140, 40, 40]
            centre = dist < size * 0.08
            img[centre] = [40, 140, 40]

        images.append(Image.fromarray(img))
        labels.append(CLASS_NAMES[cls])
    return images, labels


def visualise(images, preds, probs_list, true_labels, save_path):
    n = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            ax.imshow(images[i])
            pred_name = CLASS_NAMES[preds[i]]
            conf = probs_list[i][preds[i]]
            title = f'Pred: {pred_name}\nConf: {conf:.1%}'
            if true_labels:
                title += f'\nTrue: {true_labels[i]}'
            colour = 'green' if (not true_labels or pred_name == true_labels[i]) else 'red'
            ax.set_title(title, fontsize=9, color=colour)
        ax.axis('off')

    plt.suptitle('Unseen Wafer Map Predictions', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved visualisation → {save_path}')


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Predict wafer defect class')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--dir',   type=str, help='Folder of images')
    args = parser.parse_args()

    model = load_model()
    os.makedirs(os.path.join(ARTIFACTS, '..', 'figures'), exist_ok=True)

    true_labels = None
    if args.image:
        imgs = [Image.open(args.image)]
    elif args.dir:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        paths = sorted(p for p in os.listdir(args.dir)
                       if os.path.splitext(p)[1].lower() in exts)
        imgs = [Image.open(os.path.join(args.dir, p)) for p in paths[:16]]
    else:
        print('No input supplied; generating 8 synthetic wafer maps …')
        imgs, true_labels = generate_synthetic_images(8)

    preds, probs_list = [], []
    for i, img in enumerate(imgs):
        tensor = preprocess(img)
        cls, probs = predict(model, tensor)
        preds.append(cls)
        probs_list.append(probs)
        top3 = np.argsort(probs)[::-1][:3]
        top3_str = ', '.join(f'{CLASS_NAMES[c]} ({probs[c]:.1%})' for c in top3)
        print(f'  Image {i+1}: {CLASS_NAMES[cls]}  [{top3_str}]')

    save_path = os.path.join(ARTIFACTS, '..', 'figures', 'unseen_predictions.png')
    visualise(imgs, preds, probs_list, true_labels, save_path)

    summary = [{'image': i, 'predicted': CLASS_NAMES[p],
                'confidence': round(float(probs_list[i][p]), 4),
                'true_label': true_labels[i] if true_labels else None}
               for i, p in enumerate(preds)]
    summary_path = os.path.join(ARTIFACTS, 'unseen_predictions.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary → {summary_path}')


if __name__ == '__main__':
    main()
