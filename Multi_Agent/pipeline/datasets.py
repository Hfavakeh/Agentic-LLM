"""Per-dataset specs (per the project README).

Splits, sampling rate, and window-seconds for each preprocessed CSV. Splits
are non-contiguous (e.g. radar val sits AFTER test in the file) and must be
honored exactly to keep results comparable to the thesis baselines.
"""

from pathlib import Path
from typing import Any, Dict, Optional

DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "preprocessed-RadarEXP1(in).csv": {
        "hz": 4,
        "window_seconds": 3,
        "splits": {"train": (0, 1200), "val": (1600, 2000), "test": (1200, 1600)},
    },
    "preprocessed-RadarEXP2(in).csv": {
        "hz": 4,
        "window_seconds": 3,
        "splits": {"train": (0, 1200), "val": (1600, 2000), "test": (1200, 1600)},
    },
    "preprocessed-RadarEXP3(in).csv": {
        "hz": 4,
        "window_seconds": 3,
        "splits": {"train": (0, 1200), "val": (1600, 2000), "test": (1200, 1600)},
    },
    "preprocessed-CapEXP3(in).csv": {
        "hz": 3,
        "window_seconds": 5,
        "splits": {"train": (326, 1301), "val": (0, 325), "test": (1301, 1626)},
    },
    "preprocessed-IR-EXP2(in).csv": {
        "hz": 5,
        "window_seconds": 1,
        "splits": {"train": (1200, 3000), "val": (0, 600), "test": (600, 1200)},
    },
}


def get_dataset_spec(csv_path: str) -> Optional[Dict[str, Any]]:
    """Return the spec for a CSV by its basename, or None if unknown."""
    return DATASET_SPECS.get(Path(csv_path).name)
