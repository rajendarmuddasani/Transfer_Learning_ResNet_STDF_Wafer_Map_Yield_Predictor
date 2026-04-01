"""
WM-811K Real Data Training — ResNet-18 Wafer Map Classification
================================================================

Trains on the real WM-811K dataset (811K wafer maps from semiconductor fabs).
Replaces synthetic data with actual production wafer maps.

9 classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Normal, Random, Scratch

Usage:
    cd Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor
    python scripts/train_wm811k.py
"""

import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.resnet_model import ResNetTransferLearning

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
FIG_DIR = os.path.join(ARTIFACT_DIR, 'figures')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PICKLE_PATH = os.path.join(BASE_DIR, '..', 'data_shared', 'LSWMD.pkl')
IMG_SIZE = 224
BATCH_SIZE = 64
MAX_NONE_SAMPLES = 10000  # downsample "none" to avoid overwhelming defect classes
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

# WM-811K defect classes (sorted for consistency)
CLASS_NAMES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
               'Loc', 'Near-full', 'Normal', 'Random', 'Scratch']
NUM_CLASSES = len(CLASS_NAMES)

print("=" * 70)
print("  WM-811K Real Data Training — ResNet-18 Classification")
print("=" * 70)
print(f"  Device       : {DEVICE}")
print(f"  Image size   : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size   : {BATCH_SIZE}")
print(f"  Classes      : {NUM_CLASSES} → {CLASS_NAMES}")
print(f"  Pickle path  : {PICKLE_PATH}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPROCESS WM-811K
# ═══════════════════════════════════════════════════════════════════

def load_wm811k():
    """Load WM-811K pickle and extract labeled wafer maps."""
    import pandas as pd

    print("\n[STEP 1/7] Loading WM-811K pickle file (2GB, takes ~30-60 seconds)...")
    t0 = time.time()
    df = pd.read_pickle(PICKLE_PATH)
    print(f"  ✓ Loaded in {time.time()-t0:.1f}s — {len(df):,} total wafer maps")

    # Parse failureType labels
    print("  Parsing failure type labels...")
    def parse_label(ft):
        if isinstance(ft, (list, np.ndarray)):
            if len(ft) == 0:
                return None
            ft = ft[0]
            if isinstance(ft, (list, np.ndarray)):
                if len(ft) == 0:
                    return None
                ft = ft[0]
        return str(ft) if ft and str(ft) != 'none' else 'Normal' if ft and str(ft) == 'none' else None

    df['label'] = df['failureType'].apply(parse_label)

    # Drop unlabeled (empty failureType → None)
    labeled = df.dropna(subset=['label']).copy()
    print(f"  ✓ Labeled wafers: {len(labeled):,} out of {len(df):,}")

    # Show class distribution BEFORE downsampling
    dist = Counter(labeled['label'])
    print("\n  Class distribution (before balancing):")
    for cls in sorted(dist.keys()):
        bar = "█" * min(50, dist[cls] // 200)
        print(f"    {cls:12s}: {dist[cls]:>6,}  {bar}")

    # Downsample Normal to MAX_NONE_SAMPLES
    normal_idx = labeled[labeled['label'] == 'Normal'].index
    if len(normal_idx) > MAX_NONE_SAMPLES:
        drop_idx = np.random.choice(normal_idx, len(normal_idx) - MAX_NONE_SAMPLES, replace=False)
        labeled = labeled.drop(drop_idx)
        print(f"\n  ✓ Downsampled Normal: {len(normal_idx):,} → {MAX_NONE_SAMPLES:,}")

    # Map labels to indices
    label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    labeled['class_idx'] = labeled['label'].map(label_to_idx)
    labeled = labeled.dropna(subset=['class_idx'])
    labeled['class_idx'] = labeled['class_idx'].astype(int)

    print(f"\n  Final dataset: {len(labeled):,} labeled wafer maps")
    dist2 = Counter(labeled['class_idx'])
    for idx in sorted(dist2.keys()):
        print(f"    {CLASS_NAMES[idx]:12s} (class {idx}): {dist2[idx]:>6,}")

    return labeled


def wafer_map_to_tensor(wafer_map, size=IMG_SIZE):
    """Convert WM-811K wafer map (0/1/2 array) directly to normalized tensor.
    Done on-the-fly per sample to avoid storing 5GB of pre-converted images."""
    h, w = wafer_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[wafer_map == 1] = [0, 180, 0]    # green = pass die
    rgb[wafer_map == 2] = [200, 0, 0]    # red = fail die
    img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


class WM811KDataset(Dataset):
    """PyTorch dataset that converts wafer maps on-the-fly.
    Stores only the compact wafer maps (~100MB) not full images (~5GB)."""
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, wafer_maps, labels, augment=False, img_size=IMG_SIZE):
        self.wafer_maps = wafer_maps  # list of small numpy arrays (26x26 to 53x58)
        self.labels = labels          # numpy int64 array
        self.augment = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert tiny wafer map → 224x224 RGB on-the-fly (~0.5ms)
        img = wafer_map_to_tensor(self.wafer_maps[idx], self.img_size)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k).copy()

        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        img = (img - self.MEAN) / self.STD
        return img, self.labels[idx]


# ═══════════════════════════════════════════════════════════════════
# 2. PREPARE DATA — ON-THE-FLY APPROACH (fits in 8GB RAM)
# ═══════════════════════════════════════════════════════════════════
# Strategy: Store original wafer maps (~3KB each, ~100MB total) in memory.
# Convert to 224x224 RGB on-the-fly in __getitem__. This keeps memory < 1GB
# for data, vs 5.3GB if we pre-convert all images.
# ═══════════════════════════════════════════════════════════════════

import gc

print("\n" + "=" * 70)
df = load_wm811k()

print(f"\n[STEP 2/7] Extracting {len(df):,} compact wafer maps + labels...")
print("  (On-the-fly conversion — NO pre-allocation of 5GB image array)")
t0 = time.time()

total = len(df)
all_wafer_maps = []  # list of small numpy arrays (26x26 to 53x58, ~3KB each)
all_labels = np.zeros(total, dtype=np.int64)

for i, (_, row) in enumerate(df.iterrows()):
    all_wafer_maps.append(row['waferMap'])
    all_labels[i] = row['class_idx']
    if (i + 1) % 10000 == 0 or i == total - 1:
        elapsed = time.time() - t0
        print(f"  Extracted {i+1:>6,}/{total:,} ({(i+1)/total*100:.1f}%) — {elapsed:.1f}s")

# Free the DataFrame to reclaim ~2GB
del df
gc.collect()
mem_est = sum(wm.nbytes for wm in all_wafer_maps) / 1e6
print(f"  ✓ Extracted in {time.time()-t0:.1f}s — wafer maps use ~{mem_est:.0f} MB (vs 5.3 GB if pre-converted)")

# Train / Val / Test split (70% / 15% / 15%) — just split indices, no image copies
print("\n  Splitting into train/val/test (70/15/15)...")
indices = np.random.permutation(total)
n_train = int(0.70 * total)
n_val = int(0.15 * total)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

train_maps = [all_wafer_maps[i] for i in train_idx]
val_maps = [all_wafer_maps[i] for i in val_idx]
test_maps = [all_wafer_maps[i] for i in test_idx]
train_labels = all_labels[train_idx]
val_labels = all_labels[val_idx]
test_labels = all_labels[test_idx]

# Free combined list
del all_wafer_maps, all_labels
gc.collect()

n_total = len(train_maps) + len(val_maps) + len(test_maps)
print(f"  Train: {len(train_maps):,}  |  Val: {len(val_maps):,}  |  Test: {len(test_maps):,}")

train_ds = WM811KDataset(train_maps, train_labels, augment=True)
val_ds = WM811KDataset(val_maps, val_labels, augment=False)
test_ds = WM811KDataset(test_maps, test_labels, augment=False)

# Weighted sampler for class imbalance
class_counts = Counter(train_labels.tolist())
total_train = len(train_labels)
weights_per_class = {c: total_train / cnt for c, cnt in class_counts.items()}
sample_weights = [weights_per_class[int(l)] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"  Train batches/epoch: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# ═══════════════════════════════════════════════════════════════════
# 3. MODEL
# ═══════════════════════════════════════════════════════════════════

print(f"\n[STEP 3/7] Initialising ResNet-18 on {DEVICE}...")
model = ResNetTransferLearning(
    num_classes=NUM_CLASSES,
    architecture='resnet18',
    pretrained=True,
    freeze_backbone=True,
)
model = model.to(DEVICE)
params_total = sum(p.numel() for p in model.parameters())
params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters : {params_total:,}")
print(f"  Trainable (head) : {params_train:,}")
print(f"  ImageNet weights : ✓ loaded")

# Class weights for loss
class_weight_list = [weights_per_class.get(i, 1.0) for i in range(NUM_CLASSES)]
max_w = max(class_weight_list)
class_weight_list = [w / max_w * 3.0 for w in class_weight_list]  # normalize
class_weights = torch.FloatTensor(class_weight_list).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

print(f"  Loss: CrossEntropyLoss (weighted, label_smoothing=0.1)")
print(f"  Class weights: {['%.2f' % w for w in class_weight_list]}")


def evaluate(model, loader, name=''):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
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


def train_epoch(model, loader, optimizer, epoch, phase_name, num_batches):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()
    log_every = max(1, num_batches // 5)  # log ~5 times per epoch

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % log_every == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            remaining = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
            batch_acc = correct / total
            print(f"    [{phase_name}] Epoch {epoch} | "
                  f"Batch {batch_idx+1}/{num_batches} | "
                  f"Loss {running_loss/total:.4f} | "
                  f"Acc {batch_acc:.1%} | "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s left")

    return running_loss / total


# ═══════════════════════════════════════════════════════════════════
# 4. PHASE 1 — Train classifier head only (backbone frozen)
# ═══════════════════════════════════════════════════════════════════
train_losses, val_losses, val_accs = [], [], []
num_batches = len(train_loader)

print("\n" + "=" * 70)
print("[STEP 4/7] PHASE 1: Frozen backbone — train classifier head (5 epochs)")
print(f"  Estimated time: ~{5 * num_batches * 0.02:.0f}-{5 * num_batches * 0.06:.0f} seconds")
print("=" * 70)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=0.01
)

for epoch in range(1, 6):
    epoch_t0 = time.time()
    t_loss = train_epoch(model, train_loader, optimizer, epoch, "Phase1", num_batches)
    v_acc, v_loss, _, _ = evaluate(model, val_loader)
    epoch_time = time.time() - epoch_t0

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    val_accs.append(v_acc)
    print(f"  ► Epoch {epoch} DONE | train_loss={t_loss:.4f} | "
          f"val_loss={v_loss:.4f} | val_acc={v_acc:.2%} | {epoch_time:.0f}s")

# ═══════════════════════════════════════════════════════════════════
# 5. PHASE 2 — Unfreeze layer4 (5 epochs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[STEP 5/7] PHASE 2: Unfreeze layer4 — fine-tune deeper features (5 epochs)")
print("=" * 70)

model.unfreeze_last_block()
params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {params_train:,}")

optimizer = optim.AdamW([
    {'params': model.model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.model.fc.parameters(), 'lr': 5e-4},
], weight_decay=0.01)

for epoch in range(1, 6):
    epoch_t0 = time.time()
    t_loss = train_epoch(model, train_loader, optimizer, epoch, "Phase2", num_batches)
    v_acc, v_loss, _, _ = evaluate(model, val_loader)
    epoch_time = time.time() - epoch_t0

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    val_accs.append(v_acc)
    print(f"  ► Epoch {epoch} DONE | train_loss={t_loss:.4f} | "
          f"val_loss={v_loss:.4f} | val_acc={v_acc:.2%} | {epoch_time:.0f}s")

# ═══════════════════════════════════════════════════════════════════
# 6. PHASE 3 — Full fine-tuning with discriminative LRs (15 epochs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[STEP 6/7] PHASE 3: Full fine-tuning — discriminative learning rates (15 epochs)")
print("  This is the longest phase. Each epoch takes ~1-3 minutes on MPS.")
print("=" * 70)

model.unfreeze_all()
params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {params_train:,} (all)")

optimizer = optim.AdamW([
    {'params': model.model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.model.layer3.parameters(), 'lr': 5e-5},
    {'params': model.model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.model.fc.parameters(), 'lr': 1e-3},
], weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
best_val_acc = max(val_accs) if val_accs else 0.0

for epoch in range(1, 16):
    epoch_t0 = time.time()
    t_loss = train_epoch(model, train_loader, optimizer, epoch, "Phase3", num_batches)
    v_acc, v_loss, _, _ = evaluate(model, val_loader)
    scheduler.step()
    epoch_time = time.time() - epoch_t0

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    val_accs.append(v_acc)

    marker = ''
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'resnet18_wm811k_best.pth'))
        marker = ' ★ NEW BEST'
    print(f"  ► Epoch {epoch:2d} DONE | train_loss={t_loss:.4f} | "
          f"val_loss={v_loss:.4f} | val_acc={v_acc:.2%} | {epoch_time:.0f}s{marker}")

    # Estimate remaining
    remaining_epochs = 15 - epoch
    if remaining_epochs > 0:
        print(f"    → {remaining_epochs} epochs remaining, ~{remaining_epochs * epoch_time:.0f}s to go")

# ═══════════════════════════════════════════════════════════════════
# 7. FINAL EVALUATION + EXPORT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[STEP 7/7] Final Test Set Evaluation")
print("=" * 70)

# Reload best model
best_path = os.path.join(ARTIFACT_DIR, 'resnet18_wm811k_best.pth')
if os.path.exists(best_path):
    model.load_state_dict(torch.load(best_path, weights_only=True, map_location=DEVICE))
    print(f"  ✓ Loaded best model (val_acc={best_val_acc:.2%})")

test_acc, test_loss, y_pred, y_true = evaluate(model, test_loader, 'Test')
print(f"\n  Test Accuracy : {test_acc:.2%}")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ONNX Export
print("Exporting ONNX model...")
model.eval()
model_cpu = model.cpu()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
onnx_path = os.path.join(ARTIFACT_DIR, 'resnet18_wm811k.onnx')
torch.onnx.export(
    model_cpu, dummy, onnx_path,
    input_names=['image'], output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
)
onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"  ✓ ONNX exported: {onnx_size:.1f} MB")
model.to(DEVICE)

# Save results JSON
results = {
    'dataset': 'WM-811K (real production data)',
    'total_wafers': n_total,
    'train_size': len(train_maps),
    'val_size': len(val_maps),
    'test_size': len(test_maps),
    'num_classes': NUM_CLASSES,
    'class_names': CLASS_NAMES,
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'best_val_accuracy': float(best_val_acc),
    'model': 'ResNet-18 (ImageNet pretrained)',
    'training_phases': {
        'phase1_frozen': 5,
        'phase2_layer4': 5,
        'phase3_full': 15,
    },
    'per_class_report': classification_report(y_true, y_pred,
                                               target_names=CLASS_NAMES,
                                               output_dict=True),
}
with open(os.path.join(ARTIFACT_DIR, 'wm811k_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Results saved to artifacts/wm811k_results.json")

# ═══════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

# 1. Training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
epochs_x = list(range(1, len(train_losses) + 1))
axes[0].plot(epochs_x, train_losses, label='Train')
axes[0].plot(epochs_x, val_losses, label='Val')
axes[0].axvline(5, color='gray', ls='--', alpha=0.5, label='Phase 2')
axes[0].axvline(10, color='gray', ls=':', alpha=0.5, label='Phase 3')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()

axes[1].plot(epochs_x, val_accs, 'g-o', markersize=3)
axes[1].axvline(5, color='gray', ls='--', alpha=0.5)
axes[1].axvline(10, color='gray', ls=':', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy')

# Per-class accuracy
per_class = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
class_accs = [per_class[c]['recall'] for c in CLASS_NAMES]
colors = sns.color_palette('husl', NUM_CLASSES)
axes[2].barh(CLASS_NAMES, class_accs, color=colors)
axes[2].set_xlabel('Recall')
axes[2].set_title('Per-Class Recall')
axes[2].set_xlim(0, 1)
for i, v in enumerate(class_accs):
    axes[2].text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'wm811k_training_curves.png'), dpi=150)
plt.close()
print(f"  ✓ Training curves → figures/wm811k_training_curves.png")

# 2. Confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'WM-811K Confusion Matrix (Test Acc: {test_acc:.2%})')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'wm811k_confusion_matrix.png'), dpi=150)
plt.close()
print(f"  ✓ Confusion matrix → figures/wm811k_confusion_matrix.png")

# 3. Sample predictions grid
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
sample_indices = np.random.choice(len(test_maps), 18, replace=False)
for i, idx in enumerate(sample_indices):
    ax = axes[i // 6, i % 6]
    img = wafer_map_to_tensor(test_maps[idx])  # convert on-the-fly
    ax.imshow(img)
    true_cls = CLASS_NAMES[int(test_labels[idx])]
    pred_cls = CLASS_NAMES[y_pred[idx]]
    color = 'green' if true_cls == pred_cls else 'red'
    ax.set_title(f'T:{true_cls}\nP:{pred_cls}', fontsize=7, color=color)
    ax.axis('off')
plt.suptitle('WM-811K Sample Predictions (Green=Correct, Red=Wrong)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'wm811k_sample_predictions.png'), dpi=150)
plt.close()
print(f"  ✓ Sample predictions → figures/wm811k_sample_predictions.png")

total_time = time.time() - t0
print(f"\n{'=' * 70}")
print(f"  TRAINING COMPLETE")
print(f"  Total time       : {total_time/60:.1f} minutes")
print(f"  Best val accuracy: {best_val_acc:.2%}")
print(f"  Test accuracy    : {test_acc:.2%}")
print(f"  Artifacts saved  : {ARTIFACT_DIR}")
print(f"{'=' * 70}")
