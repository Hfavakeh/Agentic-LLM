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

# Motion-aware loss-shaping levers (Phase-3 scientific core / motion thesis).
# These reshape the TRAINING OBJECTIVE, not the optimiser. They are DEFERRED
# from the main protocol run (the 9-HP bake-off does not tune them), but the
# motion experiment (--motion-experiment) uses LOSS_SHAPING_GRID below as its
# search space with the 9 conventional HPs frozen at baseline.
LOSS_SHAPING_KEYS: Set[str] = {
    "v_max", "lambda_vel", "lambda_smooth",
    "bin_weight_slow", "bin_weight_medium", "bin_weight_fast",
}

# Discrete search grid for the loss-shaping levers, in the same style as
# HP_GRID (single source of truth for every motion arm — LLM validator, random,
# and the motion rule-based controller). The NEUTRAL value is included for each
# lever so "no shaping" (plain MSE) is reachable: lambda_vel=0 / lambda_smooth=0
# switch a penalty off, bin weights of 1.0 are neutral. `v_max` only bites when
# lambda_vel > 0 (it is the plausible top human speed in m/s).
#
# v_max RANGE FIX (2026-07-16): the original grid floor was 1.0 m/s, but every
# supported dataset is slow indoor walking — p95 speed is only 0.41 (cap) / 0.48
# (IR) / 0.60 (radar) m/s. So the motion heuristic's `v_max = 1.1 x p95` (0.45 -
# 0.65) always snapped to the 1.0 floor: the knob was a CONSTANT across datasets
# and the velocity penalty almost never fired (it needs a predicted step > 1.0
# m/s, i.e. >3x the mean speed). The LLM hit the same wall — it correctly
# reasoned "set v_max to match p95 (0.6)" but could not express it. The range now
# spans the data (0.5 - 2.0) so v_max can actually track the observed speed.
# Kept at 5 values so the search-space size — and hence the random control's
# difficulty — is unchanged.
LOSS_SHAPING_GRID = {
    "v_max":            [0.5, 0.75, 1.0, 1.5, 2.0],
    "lambda_vel":       [0.0, 0.05, 0.1, 0.2, 0.3],
    "lambda_smooth":    [0.0, 0.05, 0.1, 0.2, 0.3],
    "bin_weight_slow":  [0.5, 1.0, 1.5, 2.0, 3.0],
    "bin_weight_medium":[0.5, 1.0, 1.5, 2.0, 3.0],
    "bin_weight_fast":  [0.5, 1.0, 1.5, 2.0, 3.0],
}

# The neutral (plain-MSE) loss-shaping vector — the motion experiment's ANCHOR
# start point, so the LLM/controller propose moves AWAY from "no shaping".
LOSS_SHAPING_NEUTRAL = {
    "v_max": 2.0, "lambda_vel": 0.0, "lambda_smooth": 0.0,
    "bin_weight_slow": 1.0, "bin_weight_medium": 1.0, "bin_weight_fast": 1.0,
}

# Continuous bounds DERIVED from the discrete lever grid (min/max per lever),
# mirroring HP_BOUNDS. Used by range-normalising helpers and the motion
# rule-based controller's v_max clamp.
LOSS_SHAPING_BOUNDS = {k: (min(v), max(v)) for k, v in LOSS_SHAPING_GRID.items()}

OPTIMIZER_CHOICES = list(HP_GRID["optimizer_choice"])
