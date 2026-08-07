"""Wafer data contract and fail-closed STDF integration boundary."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WaferData:
    """Die-level wafer data accepted by the wafer-map renderer."""

    wafer_id: str
    lot_id: str
    wafer_num: int
    die_count: int
    pass_count: int
    fail_count: int
    coordinates: np.ndarray
    bins: np.ndarray
    test_results: dict[str, np.ndarray] | None = None
    metadata: dict[str, Any] | None = None


class STDFParser:
    """Fail-closed placeholder until a specification-tested parser is integrated."""

    def __init__(self, stdf_path: str):
        self.stdf_path = Path(stdf_path)
        if not self.stdf_path.is_file():
            raise FileNotFoundError(f"STDF file not found: {stdf_path}")

    def parse(self) -> WaferData:
        raise NotImplementedError(
            "STDF parsing is outside the confirmed public model boundary. "
            "Integrate a licensed, specification-tested parser before use."
        )


def parse_stdf_file(stdf_path: str) -> WaferData:
    """Reject STDF input until the parser promotion gate is satisfied."""
    return STDFParser(stdf_path).parse()
