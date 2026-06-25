"""Back-compat shim — the training stack now lives in the ``pipeline`` package.

Import from ``pipeline`` (or its submodules) in new code:

    from pipeline import Config, Trainer, HP_GRID, logger

This module only re-exports the old public names so historical imports
(`from model_pipeline import ...`) keep working. Safe to delete once nothing
imports `model_pipeline` anymore.
"""

from pipeline import (  # noqa: F401
    Config,
    DATASET_SPECS,
    DEFAULT_LOSS_SHAPING,
    DataProcessor,
    Evaluator,
    HP_BOUNDS,
    HP_GRID,
    LOSS_SHAPING_KEYS,
    LSTM_Localizer,
    NUMERIC_HP_KEYS,
    OPTIMIZER_CHOICES,
    TimeSeriesDataset,
    Trainer,
    compute_speed_bin_edges,
    get_dataset_spec,
    is_finite_number,
    logger,
    safe_trapz,
    sanitize_for_json,
    set_seed,
)
