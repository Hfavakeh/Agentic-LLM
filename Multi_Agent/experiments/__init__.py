"""Experiment orchestration: the per-run bake-off (baseline + 4 search arms),
the final evaluation on fresh seeds, and the multi-seed replication runner."""

from .final_eval import FINAL_EVAL_SEEDS_POOL, run_final_evaluation
from .runner import (
    EXPERIMENT_SEEDS, run_full_experiment, run_multi_seed_experiment, run_one_seed,
)
from .single_run import run_experiment

__all__ = [
    "FINAL_EVAL_SEEDS_POOL", "run_final_evaluation",
    "EXPERIMENT_SEEDS", "run_full_experiment", "run_multi_seed_experiment",
    "run_one_seed",
    "run_experiment",
]
