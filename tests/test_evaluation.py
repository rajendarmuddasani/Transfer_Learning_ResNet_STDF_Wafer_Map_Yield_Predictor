"""Metric and calibration contracts."""

import numpy as np

from src.evaluation import fit_temperature, softmax, summarize_predictions


def test_softmax_is_stable_and_normalized():
    probabilities = softmax(np.array([[1_000.0, 999.0], [-1_000.0, -999.0]]))

    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_temperature_fit_does_not_worsen_validation_log_loss():
    logits = np.array([[8.0, 0.0], [7.0, 0.0], [0.0, 6.0], [0.0, 5.0]])
    labels = np.array([0, 1, 1, 1])
    uncalibrated = summarize_predictions(labels, softmax(logits), ("a", "b"))

    temperature, calibrated_loss = fit_temperature(logits, labels)

    assert 0.25 <= temperature <= 4.0
    assert calibrated_loss <= uncalibrated["log_loss"]


def test_summary_reports_class_tail_and_calibration_metrics():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]], dtype=np.float64
    )

    metrics = summarize_predictions(labels, probabilities, ("normal", "defect"))

    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["defect"]["recall"] == 0.5
    assert 0.0 <= metrics["top_label_ece"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
