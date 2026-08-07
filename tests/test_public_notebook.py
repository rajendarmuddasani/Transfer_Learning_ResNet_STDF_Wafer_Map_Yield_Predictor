"""Execution, output, and privacy contracts for the public evidence notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_wafer_defect_classifier_walkthrough.ipynb"


def test_public_notebook_is_fully_executed_without_errors():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert len(code_cells) == 7
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert all(cell["outputs"] for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell["outputs"]
    )


def test_public_notebook_contains_canonical_evidence_and_no_private_paths():
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    required = (
        "public_synthetic_evaluation.json",
        "public_synthetic_resnet18_v1",
        "Accuracy",
        "0.93625",
        "QuadrantFailure",
        "83.0% recall",
        "Group overlap counts",
        "All promotion gates passed: True",
        "predicted_class",
        "WM-811K test accuracy",
    )
    restricted = (
        "internal-reference",
        "intra.infineon.com",
        "Rajendar.Muddasani@Infineon.com",
        "pdocs/",
        "pdocs\\",
    )

    assert all(value in text for value in required)
    assert all(value not in text for value in restricted)
