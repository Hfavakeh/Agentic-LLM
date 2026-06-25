"""Hyperparameter search space — single source of truth.

HP_GRID is consumed by every arm (LLM validator, random sampler, Optuna
sampler, rule-based controller) so the only difference between arms is *how*
a setting is chosen, never *what* is reachable.
"""

from typing import Dict, Set

# HP_GRID is the single source of truth for the search space — the exact
# discrete value lists mandated by the professor's protocol (see memory:
# professor-protocol). Every arm (LLM, random, Optuna, rule-based) draws from
# these identical sets so the only difference between arms is *how* a setting
# is chosen, never *what* is reachable. Order matters only for readability;
# membership is what the validator and samplers consume.
#
# NOTE: this dict is mutated in place by ``Config.__post_init__`` (the
# baseline window_size is injected so the search can reach the baseline
# config). All modules must therefore import THIS object, never copy it.
HP_GRID = {
    "learning_rate":    [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay":     [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
    "dropout":          [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "batch_size":       [32, 64, 128],
    "lstm_hidden":      [64, 128, 256],
    "lstm_layers":      [1, 2, 3],
    "window_size":      [10, 20, 30, 40, 50],
    "optimizer_choice": ["adam", "adamw"],
    "patience":         [8, 12, 16],
}

# Numeric (non-categorical) grid params, for samplers / validators that need
# to reason about ordering or nearest-value snapping.
NUMERIC_HP_KEYS: Set[str] = {
    "learning_rate", "weight_decay", "dropout", "batch_size",
    "lstm_hidden", "lstm_layers", "window_size", "patience",
}

# Back-compat continuous bounds, DERIVED from the discrete grid (min/max of
# each numeric param). Kept so range-normalising helpers (e.g. the proposal
# change-magnitude metric) keep working; the grid — not these bounds — is the
# authoritative membership test.
HP_BOUNDS = {k: (min(v), max(v)) for k, v in HP_GRID.items() if k in NUMERIC_HP_KEYS}

# Motion-aware loss-shaping levers (Phase-3 scientific core). DEFERRED from the
# search space for the main protocol run — the loss-shaping machinery in the
# Trainer stays intact for a later motion experiment, but no arm tunes these
# for now (motion still feeds the LLM prompt qualitatively, per step 6g).
LOSS_SHAPING_KEYS: Set[str] = {
    "v_max", "lambda_vel", "lambda_smooth",
    "bin_weight_slow", "bin_weight_medium", "bin_weight_fast",
}

OPTIMIZER_CHOICES = list(HP_GRID["optimizer_choice"])
