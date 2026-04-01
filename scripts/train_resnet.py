"""
Transfer Learning ResNet — Wafer Map Defect Classification
==========================================================

Pipeline:
1. Generate synthetic wafer map images (8 defect classes)
2. Train ResNet18 with ImageNet transfer learning (progressive fine-tuning)
3. Evaluate on held-out test set
4. Export model artifacts (PyTorch + ONNX)

Run: python scripts/train_resnet.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, time, warnings, random
warnings.filterwarnings("ignore")

from src.models.resnet_model import ResNetTransferLearning
from src.models.dataset import WaferMapDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Directories
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'wafer_maps')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
FIG_DIR = os.path.join(ARTIFACT_DIR, 'figures')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DEFECT_CLASSES = [
    'Normal', 'EdgeEffect', 'CenterCluster', 'RingPattern',
    'QuadrantFailure', 'Scratch', 'RandomFailure', 'MixedMode'
]

# ═══════════════════════════════════════════════════════════════════
# 1  Generate Synthetic Wafer Map Images
# ═══════════════════════════════════════════════════════════════════

IMG_SIZE = 300          # wafer map canvas
DIE_SIZE = 3            # pixels per die
N_PER_CLASS_TRAIN = 200
N_PER_CLASS_VAL = 40
N_PER_CLASS_TEST = 40

def _wafer_mask(size):
    """Circular boolean mask for the wafer."""
    yy, xx = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    r = size // 2 - 5
    return ((xx - cx)**2 + (yy - cy)**2) <= r**2

def _die_positions(size, die_size, mask):
    """Return (row, col) positions inside circular mask."""
    positions = []
    for r in range(0, size - die_size, die_size + 1):
        for c in range(0, size - die_size, die_size + 1):
            cr, cc = r + die_size // 2, c + die_size // 2
            if 0 <= cr < size and 0 <= cc < size and mask[cr, cc]:
                positions.append((r, c))
    return positions

def _generate_wafer(defect_class, size=IMG_SIZE, die_size=DIE_SIZE):
    """Generate a single synthetic wafer map image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    mask = _wafer_mask(size)
    positions = _die_positions(size, die_size, mask)
    cx, cy = size // 2, size // 2
    radius = size // 2 - 5

    # Assign pass/fail per die based on defect class
    for r, c in positions:
        dr, dc = r + die_size // 2, c + die_size // 2
        dist = np.sqrt((dr - cy)**2 + (dc - cx)**2)

        if defect_class == 'Normal':
            fail = random.random() < 0.02
        elif defect_class == 'EdgeEffect':
            fail = random.random() < (0.7 if dist > radius * 0.78 else 0.02)
        elif defect_class == 'CenterCluster':
            fail = random.random() < (0.65 if dist < radius * 0.28 else 0.02)
        elif defect_class == 'RingPattern':
            ring_dist = abs(dist - radius * (0.45 + random.gauss(0, 0.03)))
            fail = random.random() < (0.6 if ring_dist < radius * 0.08 else 0.02)
        elif defect_class == 'QuadrantFailure':
            quadrant = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            in_quad = ((dr - cy) * quadrant[0] > 0) and ((dc - cx) * quadrant[1] > 0)
            fail = random.random() < (0.55 if in_quad else 0.02)
        elif defect_class == 'Scratch':
            angle = random.uniform(0, np.pi)
            line_dist = abs((dr - cy) * np.cos(angle) - (dc - cx) * np.sin(angle))
            fail = random.random() < (0.7 if line_dist < radius * 0.04 else 0.02)
        elif defect_class == 'RandomFailure':
            fail = random.random() < 0.18
        elif defect_class == 'MixedMode':
            edge_fail = dist > radius * 0.82
            center_fail = dist < radius * 0.2
            fail = random.random() < (0.5 if (edge_fail or center_fail) else 0.04)
        else:
            fail = False

        color = (200, 40, 40) if fail else (40, 180, 40)
        img[r:r+die_size, c:c+die_size] = color

    # Draw wafer boundary
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.ellipse([5, 5, size - 5, size - 5], outline='white', width=2)
    return np.array(pil)

def generate_dataset():
    """Generate full train/val/test dataset."""
    splits = {'train': N_PER_CLASS_TRAIN, 'val': N_PER_CLASS_VAL, 'test': N_PER_CLASS_TEST}
    total = 0
    for split, n_per_class in splits.items():
        for cls_name in DEFECT_CLASSES:
            out_dir = os.path.join(DATA_DIR, split, cls_name)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n_per_class):
                img = _generate_wafer(cls_name)
                Image.fromarray(img).save(os.path.join(out_dir, f'{cls_name}_{i:04d}.png'))
                total += 1
    print(f"Generated {total} wafer map images ({len(DEFECT_CLASSES)} classes × 3 splits)")
    return total

print("Generating synthetic wafer map images …")
t0 = time.time()
generate_dataset()
print(f"  Done in {time.time()-t0:.1f}s\n")

# ═══════════════════════════════════════════════════════════════════
# 2  Data Loaders with Augmentation
# ═══════════════════════════════════════════════════════════════════

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = WaferMapDataset(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_ds = WaferMapDataset(os.path.join(DATA_DIR, 'val'), transform=val_transform)
test_ds = WaferMapDataset(os.path.join(DATA_DIR, 'test'), transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
print(f"Classes: {DEFECT_CLASSES}\n")

# ═══════════════════════════════════════════════════════════════════
# 3  Model — ResNet18 Transfer Learning
# ═══════════════════════════════════════════════════════════════════

model = ResNetTransferLearning(
    num_classes=len(DEFECT_CLASSES),
    architecture='resnet18',
    pretrained=True,
    freeze_backbone=True,
)
model = model.to(device)

class_weights = torch.FloatTensor([1.0, 2.5, 1.8, 2.0, 3.0, 2.2, 1.5, 1.2]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

def evaluate(model, loader, name=''):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            running_loss += criterion(outputs, labels).item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    loss = running_loss / total
    return acc, loss, np.array(all_preds), np.array(all_labels)

# ═══════════════════════════════════════════════════════════════════
# 4  Progressive Fine-Tuning
# ═══════════════════════════════════════════════════════════════════
print("="*60)
print("PHASE 1: Freeze backbone — train classifier head only")
print("="*60)

# Backbone already frozen by constructor (freeze_backbone=True)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=0.01
)

train_losses, val_losses, val_accs = [], [], []

for epoch in range(1, 6):
    model.train()
    running = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    t_loss = running / len(train_ds)
    v_acc, v_loss, _, _ = evaluate(model, val_loader)
    train_losses.append(t_loss)
    val_losses.append(v_loss)
    val_accs.append(v_acc)
    print(f"  Epoch {epoch}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.2%}")

print("\n" + "="*60)
print("PHASE 2: Unfreeze layer4 — fine-tune deeper features")
print("="*60)

# Unfreeze layer4
model.unfreeze_last_block()

optimizer = optim.AdamW([
    {'params': model.model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.model.fc.parameters(), 'lr': 5e-4},
], weight_decay=0.01)

for epoch in range(1, 6):
    model.train()
    running = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    t_loss = running / len(train_ds)
    v_acc, v_loss, _, _ = evaluate(model, val_loader)
    train_losses.append(t_loss)
    val_losses.append(v_loss)
    val_accs.append(v_acc)
    print(f"  Epoch {epoch}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.2%}")

print("\n" + "="*60)
print("PHASE 3: Full fine-tuning — discriminative learning rates")
print("="*60)

# Unfreeze everything
model.unfreeze_all()

optimizer = optim.AdamW([
    {'params': model.model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.model.layer3.parameters(), 'lr': 5e-5},
    {'params': model.model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.model.fc.parameters(), 'lr': 1e-3},
], weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
best_val_acc = 0.0

for epoch in range(1, 16):
    model.train()
    running = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    t_loss = running / len(train_ds)
    v_acc, v_loss, _, _ = evaluate(model, val_loader)
    scheduler.step()

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    val_accs.append(v_acc)

    marker = ''
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'resnet18_wafer_best.pth'))
        marker = ' *'
    print(f"  Epoch {epoch:2d}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.2%}{marker}")

# Reload best model
model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, 'resnet18_wafer_best.pth'), weights_only=True))

# ═══════════════════════════════════════════════════════════════════
# 5  Final Evaluation on Test Set
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

test_acc, test_loss, y_pred, y_true = evaluate(model, test_loader, 'Test')
print(f"\nTest Accuracy: {test_acc:.2%}  Test Loss: {test_loss:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=DEFECT_CLASSES))

# ═══════════════════════════════════════════════════════════════════
# 6  Export ONNX
# ═══════════════════════════════════════════════════════════════════
print("Exporting ONNX model …")
model.eval()
dummy = torch.randn(1, 3, 224, 224).to(device)
onnx_path = os.path.join(ARTIFACT_DIR, 'resnet18_wafer.onnx')
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=['image'], output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
)
print(f"  Saved {onnx_path}")

# ═══════════════════════════════════════════════════════════════════
# 7  Save Artifacts & Figures
# ═══════════════════════════════════════════════════════════════════

# Evaluation results
eval_results = {
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'best_val_accuracy': float(best_val_acc),
    'num_classes': len(DEFECT_CLASSES),
    'classes': DEFECT_CLASSES,
    'model': 'resnet18',
    'pretrained': 'imagenet',
    'device': device,
    'train_samples': len(train_ds),
    'val_samples': len(val_ds),
    'test_samples': len(test_ds),
}
with open(os.path.join(ARTIFACT_DIR, 'evaluation_results.json'), 'w') as f:
    json.dump(eval_results, f, indent=2)

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, len(train_losses) + 1)
# Phase annotations
axes[0].axvline(5, color='gray', linestyle='--', alpha=0.5, label='Phase 2')
axes[0].axvline(10, color='gray', linestyle=':', alpha=0.5, label='Phase 3')
axes[0].plot(epochs, train_losses, label='Train Loss')
axes[0].plot(epochs, val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()

axes[1].axvline(5, color='gray', linestyle='--', alpha=0.5, label='Phase 2')
axes[1].axvline(10, color='gray', linestyle=':', alpha=0.5, label='Phase 3')
axes[1].plot(epochs, val_accs, label='Val Accuracy', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'training_curves.png'), dpi=150)
plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 7))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=DEFECT_CLASSES, yticklabels=DEFECT_CLASSES, ax=ax)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title(f'Test Confusion Matrix (Acc: {test_acc:.1%})')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# Per-class accuracy bar chart
per_class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(DEFECT_CLASSES, per_class_acc, color=sns.color_palette('muted', 8))
ax.set_ylabel('Accuracy')
ax.set_title('Per-Class Test Accuracy')
ax.set_ylim(0, 1.05)
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.0%}', ha='center', fontsize=9)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'per_class_accuracy.png'), dpi=150)
plt.close()

# Sample wafer maps grid
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for idx, cls in enumerate(DEFECT_CLASSES):
    ax = axes[idx // 4, idx % 4]
    sample_dir = os.path.join(DATA_DIR, 'test', cls)
    sample_img = Image.open(os.path.join(sample_dir, os.listdir(sample_dir)[0]))
    ax.imshow(sample_img)
    ax.set_title(cls, fontsize=10)
    ax.axis('off')
plt.suptitle('Sample Wafer Maps per Defect Class', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sample_wafer_maps.png'), dpi=150)
plt.close()

print(f"\nArtifacts saved to {ARTIFACT_DIR}/")
print(f"Figures saved to {FIG_DIR}/")
print(f"\n{'='*60}")
print("DONE — ResNet18 trained, evaluated, and exported.")
print(f"{'='*60}")
