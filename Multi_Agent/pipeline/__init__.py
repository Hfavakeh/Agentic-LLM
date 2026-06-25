"""Self-contained training stack: configuration, data, model, training, eval.

Importing this package configures the shared run logger (see
``logging_setup``). The public API mirrors what the old monolithic
``model_pipeline.py`` exported.
"""

from .logging_setup import logger
from .utils import is_finite_number, safe_trapz, sanitize_for_json, set_seed
from .search_space import (
    HP_BOUNDS, HP_GRID, LOSS_SHAPING_KEYS, NUMERIC_HP_KEYS, OPTIMIZER_CHOICES,
)
from .datasets import DATASET_SPECS, get_dataset_spec
from .config import Config
from .data import DataProcessor, TimeSeriesDataset, compute_speed_bin_edges
from .model import LSTM_Localizer
from .trainer import DEFAULT_LOSS_SHAPING, Trainer
from .evaluator import Evaluator

__all__ = [
    "logger",
    "is_finite_number", "safe_trapz", "sanitize_for_json", "set_seed",
    "HP_BOUNDS", "HP_GRID", "LOSS_SHAPING_KEYS", "NUMERIC_HP_KEYS", "OPTIMIZER_CHOICES",
    "DATASET_SPECS", "get_dataset_spec",
    "Config",
    "DataProcessor", "TimeSeriesDataset", "compute_speed_bin_edges",
    "LSTM_Localizer",
    "DEFAULT_LOSS_SHAPING", "Trainer",
    "Evaluator",
]
