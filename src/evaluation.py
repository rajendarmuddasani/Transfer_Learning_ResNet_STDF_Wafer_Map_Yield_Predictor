"""Classification metrics and calibration helpers for public evidence."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.geomspace(0.25, 4.0, 161)
    losses = [log_loss(labels, softmax(logits, candidate)) for candidate in candidates]
    best_index = int(np.argmin(losses))
    return float(candidates[best_index]), float(losses[best_index])


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if np.any(in_bin):
            error += float(np.mean(in_bin)) * abs(
                float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin]))
            )
    return error


def wilson_interval(successes: int, total: int, z_score: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z_score**2 / total
    center = (proportion + z_score**2 / (2.0 * total)) / denominator
    spread = (
        z_score
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_score**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - spread), min(1.0, center + spread)]


def bootstrap_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    iterations: int = 1_000,
    seed: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == class_index) for class_index in np.unique(labels)]
    values = {"macro_f1": [], "mcc": []}
    for _ in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        sampled_labels = labels[sampled]
        sampled_predictions = predictions[sampled]
        values["macro_f1"].append(
            f1_score(sampled_labels, sampled_predictions, average="macro")
        )
        values["mcc"].append(matthews_corrcoef(sampled_labels, sampled_predictions))
    return {
        name: [float(np.percentile(metric_values, 2.5)), float(np.percentile(metric_values, 97.5))]
        for name, metric_values in values.items()
    }


def summarize_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...] | list[str],
    include_bootstrap: bool = False,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.argmax(probabilities, axis=1)
    class_indices = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_indices,
        zero_division=0,
    )
    binary_labels = np.eye(len(class_names), dtype=np.float64)[labels]
    if len(class_names) == 2:
        macro_roc_auc = roc_auc_score(labels, probabilities[:, 1])
        macro_pr_auc = average_precision_score(labels, probabilities[:, 1])
    else:
        macro_roc_auc = roc_auc_score(
            label_binarize(labels, classes=class_indices),
            probabilities,
            average="macro",
            multi_class="ovr",
        )
        macro_pr_auc = average_precision_score(
            binary_labels, probabilities, average="macro"
        )
    accuracy = accuracy_score(labels, predictions)
    matrix = confusion_matrix(labels, predictions, labels=class_indices)
    result = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy),
        "accuracy_95ci": wilson_interval(int(np.sum(labels == predictions)), len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "macro_roc_auc_ovr": float(macro_roc_auc),
        "macro_pr_auc": float(macro_pr_auc),
        "log_loss": float(log_loss(labels, probabilities, labels=class_indices)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - binary_labels) ** 2, axis=1))),
        "top_label_ece": expected_calibration_error(probabilities, labels),
        "minimum_class_recall": float(np.min(recall)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
    }
    if include_bootstrap:
        result["bootstrap_95ci"] = bootstrap_intervals(labels, predictions)
    return result
