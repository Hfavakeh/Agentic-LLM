"""Reporting utilities: per-run plots, protocol report writers, and the
cross-run metric aggregation. Decoupled from core execution — nothing here
trains a model or runs a search."""

from .cross_run import compute_cross_run_metrics
from .plots import (
    plot_action_impact_scatter, plot_cross_run_summary,
    plot_hp_search_dynamics, plot_hyperparameter_trajectory,
    plot_optimization_trajectories, plot_training_dynamics,
)
from .reports import (
    write_budget_table, write_final_eval_report, write_per_setting_report,
)

__all__ = [
    "compute_cross_run_metrics",
    "plot_action_impact_scatter", "plot_cross_run_summary",
    "plot_hp_search_dynamics", "plot_hyperparameter_trajectory",
    "plot_optimization_trajectories", "plot_training_dynamics",
    "write_budget_table", "write_final_eval_report", "write_per_setting_report",
]
