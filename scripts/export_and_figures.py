"""Post-training: ONNX export + figure generation."""
import os, sys, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.resnet_model import ResNetTransferLearning
from src.models.dataset import WaferMapDataset
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
FIG_DIR = os.path.join(ARTIFACT_DIR, 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'wafer_maps')
os.makedirs(FIG_DIR, exist_ok=True)

DEFECT_CLASSES = ['Normal','EdgeEffect','CenterCluster','RingPattern',
                  'QuadrantFailure','Scratch','RandomFailure','MixedMode']

# Load best model
model = ResNetTransferLearning(
    architecture='resnet18', num_classes=8, pretrained=False, freeze_backbone=False
)
model.load_state_dict(
    torch.load(os.path.join(ARTIFACT_DIR, 'resnet18_wafer_best.pth'), weights_only=True)
)
model.eval()

# ONNX export
dummy = torch.randn(1, 3, 224, 224)
onnx_path = os.path.join(ARTIFACT_DIR, 'resnet18_wafer.onnx')
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=['image'], output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
)
print(f'ONNX exported: {os.path.getsize(onnx_path)/(1024*1024):.1f} MB')

# Evaluate on test set
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
test_ds = WaferMapDataset(os.path.join(DATA_DIR, 'test'), transform=val_transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        _, preds = model(imgs).max(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
y_pred, y_true = np.array(all_preds), np.array(all_labels)
test_acc = (y_pred == y_true).mean()
print(f'Test accuracy: {test_acc:.2%}')

# Update eval results
results_path = os.path.join(ARTIFACT_DIR, 'evaluation_results.json')
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
else:
    results = {}
results['test_accuracy'] = float(test_acc)
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
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

# Per-class accuracy
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
    img = Image.open(os.path.join(sample_dir, sorted(os.listdir(sample_dir))[0]))
    ax.imshow(img)
    ax.set_title(cls, fontsize=10)
    ax.axis('off')
plt.suptitle('Sample Wafer Maps per Defect Class', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sample_wafer_maps.png'), dpi=150)
plt.close()

print(f'Figures saved to {FIG_DIR}/')
print('DONE')
