"""Cross-platform contracts for evidence recomputation comparisons."""

import pytest

from scripts.validate_evidence import assert_nested_close


def test_nested_evidence_comparison_allows_only_last_bit_float_variation():
    expected = {
        "score": 0.9360728678551831,
        "confusion_matrix": [[58, 2], [1, 59]],
        "gate": True,
        "label": "QuadrantFailure",
    }
    actual = {
        "score": expected["score"] + 5e-10,
        "confusion_matrix": [[58, 2], [1, 59]],
        "gate": True,
        "label": "QuadrantFailure",
    }

    assert_nested_close(actual, expected, "metrics")


@pytest.mark.parametrize(
    "actual",
    (
        {"score": 0.9361, "confusion_matrix": [[58, 2], [1, 59]]},
        {"score": 0.9360728678551831, "confusion_matrix": [[58, 3], [1, 59]]},
        {"score": 0.9360728678551831},
    ),
)
def test_nested_evidence_comparison_rejects_meaningful_or_structural_drift(actual):
    expected = {
        "score": 0.9360728678551831,
        "confusion_matrix": [[58, 2], [1, 59]],
    }

    with pytest.raises(AssertionError):
        assert_nested_close(actual, expected, "metrics")
