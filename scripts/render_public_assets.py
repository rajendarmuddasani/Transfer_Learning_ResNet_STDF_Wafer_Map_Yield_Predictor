"""Render Project 4 public evidence assets from canonical artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import DEFECT_CLASSES  # noqa: E402
from src.data.synthetic_benchmark import (  # noqa: E402
    build_sample_specs,
    generate_wafer_image,
)


ASSET_DIR = ROOT / "docs" / "assets"
SAMPLE_DIR = ROOT / "data" / "sample_wafer_maps"
NAVY = "#17324D"
GREEN = "#18745A"
MINT = "#DCECE6"
CORAL = "#D86452"
AMBER = "#E0AD3B"
INK = "#1E2927"
MUTED = "#60716D"
GRID = "#D2DDDA"


def load_evidence() -> dict:
    return json.loads(
        (ROOT / "evidence" / "public_synthetic_evaluation.json").read_text(
            encoding="utf-8"
        )
    )


def style_axis(axis, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", color=NAVY, pad=18)
    axis.text(0.0, 1.01, subtitle, transform=axis.transAxes, fontsize=9, color=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def render_confirmation(evidence: dict) -> None:
    metrics = evidence["confirmation_metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")
    summary_labels = ["Accuracy", "Macro F1", "MCC", "Macro PR-AUC"]
    summary_values = [
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["mcc"],
        metrics["macro_pr_auc"],
    ]
    axes[0].bar(summary_labels, summary_values, color=[GREEN, GREEN, NAVY, AMBER], width=0.62)
    axes[0].set_ylim(0.75, 1.01)
    axes[0].tick_params(axis="x", rotation=18)
    style_axis(axes[0], "Grouped confirmation", "800 wafers from unseen pattern families")
    for index, value in enumerate(summary_values):
        axes[0].text(index, value + 0.008, f"{value:.3f}", ha="center", color=NAVY, fontweight="bold")

    recalls = [metrics["per_class"][class_name]["recall"] for class_name in DEFECT_CLASSES]
    colors = [CORAL if value == min(recalls) else GREEN for value in recalls]
    axes[1].barh(list(DEFECT_CLASSES), recalls, color=colors)
    axes[1].set_xlim(0.75, 1.01)
    axes[1].invert_yaxis()
    style_axis(axes[1], "Class recall", "QuadrantFailure is the confirmed tail")
    for index, value in enumerate(recalls):
        axes[1].text(value + 0.004, index, f"{value:.0%}", va="center", color=NAVY, fontweight="bold")

    figure.suptitle("Wafer Pattern Classifier | Confirmation Evidence", x=0.055, ha="left", fontsize=22, fontweight="bold", color=INK)
    figure.text(0.055, 0.025, "Independent synthetic benchmark only. Not WM-811K or production silicon evidence.", fontsize=10, color=MUTED)
    figure.tight_layout(rect=(0.04, 0.06, 0.98, 0.90), w_pad=4)
    figure.savefig(ASSET_DIR / "confirmation_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_confusion_matrix(evidence: dict) -> None:
    matrix = np.asarray(evidence["confirmation_metrics"]["confusion_matrix"], dtype=np.float64)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    image = axis.imshow(normalized, cmap="BuGn", vmin=0.0, vmax=1.0)
    for row in range(len(DEFECT_CLASSES)):
        for column in range(len(DEFECT_CLASSES)):
            value = normalized[row, column]
            if value >= 0.01:
                axis.text(column, row, f"{value:.0%}", ha="center", va="center", color="white" if value > 0.55 else INK, fontsize=9)
    axis.set_xticks(range(len(DEFECT_CLASSES)), DEFECT_CLASSES, rotation=35, ha="right")
    axis.set_yticks(range(len(DEFECT_CLASSES)), DEFECT_CLASSES)
    axis.set_xlabel("Predicted pattern")
    axis.set_ylabel("Generated pattern")
    axis.set_title("Normalized Confirmation Confusion Matrix", loc="left", fontsize=18, fontweight="bold", color=NAVY, pad=16)
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04, label="Row fraction")
    figure.tight_layout()
    figure.savefig(ASSET_DIR / "confirmation_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_design(evidence: dict) -> None:
    figure, axis = plt.subplots(figsize=(15, 5.8), facecolor="white")
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 6)
    axis.axis("off")
    boxes = [
        (0.4, "Train", "1,920 wafers\n24 families / class", MINT),
        (4.1, "Validation", "480 wafers\nselect C + temperature", "#F6EAC8"),
        (7.8, "Freeze", "C = 0.1\nT = 0.71947", "#E7EAF0"),
        (11.5, "Confirmation", "800 wafers\n10 unseen families / class", "#F4DCD7"),
    ]
    for x_position, title, body, color in boxes:
        axis.add_patch(plt.Rectangle((x_position, 2.0), 3.0, 2.1, facecolor=color, edgecolor=NAVY, linewidth=1.3))
        axis.text(x_position + 1.5, 3.45, title, ha="center", va="center", fontsize=13, color=NAVY, fontweight="bold")
        axis.text(x_position + 1.5, 2.65, body, ha="center", va="center", fontsize=10, color=INK, linespacing=1.4)
    for start in (3.4, 7.1, 10.8):
        axis.annotate("", xy=(start + 0.65, 3.05), xytext=(start + 0.05, 3.05), arrowprops={"arrowstyle": "->", "color": GREEN, "linewidth": 2})
    axis.text(0.4, 5.35, "Selection Discipline", fontsize=22, fontweight="bold", color=INK)
    axis.text(0.4, 4.85, "Pattern families and sample seeds are disjoint across all three partitions.", fontsize=11, color=MUTED)
    axis.text(0.4, 0.75, f"Manifest SHA-256 {evidence['dataset']['manifest_sha256'][:24]}...  |  Confirmation opened after selection freeze", fontsize=10, color=NAVY)
    figure.tight_layout()
    figure.savefig(ASSET_DIR / "benchmark_design.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_samples() -> None:
    specs = build_sample_specs(variants_per_family=1)
    confirmation_specs = [
        next(spec for spec in specs if spec.split == "confirmation" and spec.class_name == class_name)
        for class_name in DEFECT_CLASSES
    ]
    figure, axes = plt.subplots(2, 4, figsize=(12, 7.2), facecolor="white")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for axis, spec in zip(axes.flat, confirmation_specs):
        image = generate_wafer_image(spec)
        axis.imshow(image)
        axis.set_title(spec.class_name, color=NAVY, fontsize=11, fontweight="bold")
        axis.axis("off")
        from PIL import Image

        Image.fromarray(image).save(SAMPLE_DIR / f"{spec.class_name}.png")
    figure.suptitle("Unseen Synthetic Confirmation Families", fontsize=20, fontweight="bold", color=INK)
    figure.text(0.5, 0.02, "One deterministic example per class; green = pass die, coral = fail die.", ha="center", fontsize=10, color=MUTED)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.09, hspace=0.34, wspace=0.16)
    figure.savefig(ASSET_DIR / "confirmation_samples.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    render_confirmation(evidence)
    render_confusion_matrix(evidence)
    render_design(evidence)
    render_samples()
    for path in sorted(ASSET_DIR.glob("*.png")):
        print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
