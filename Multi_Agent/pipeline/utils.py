"""Small numeric / reproducibility helpers shared across the project."""

import random
from typing import Any, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_trapz(y: List[float], dx: float = 1.0) -> float:
    """Numpy-version-safe trapezoidal integration."""
    if hasattr(np, "trapz"):
        return float(np.trapz(y, dx=dx))
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    y_arr = np.asarray(y, dtype=float)
    if len(y_arr) < 2:
        return 0.0
    return float(np.sum((y_arr[1:] + y_arr[:-1]) * 0.5 * dx))


def is_finite_number(value: Any) -> bool:
    """Return True only for finite numeric scalars (explicitly excludes bool)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return bool(np.isfinite(float(value)))


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with None for safe JSON serialisation."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
