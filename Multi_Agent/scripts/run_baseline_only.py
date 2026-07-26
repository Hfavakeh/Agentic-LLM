"""Re-run ONLY the fixed-reference baseline, with full logging.

Produces a clean, reproducible record of the baseline configuration:
  1. one representative training (logged per-epoch curves + plots) via run_baseline
  2. the protocol headline: the baseline setting evaluated on the 30 fresh
     final-eval seeds (201-230), trained from scratch each seed, test RMSE in
     metres — identical code path to the full experiment's final-eval stage.

Run: ../venv/Scripts/python.exe run_baseline_only.py
"""

import sys as _sys, pathlib as _pathlib
_REPO_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
import json
from pathlib import Path

import numpy as np

from arms.baseline import run_baseline
from experiments.final_eval import run_final_evaluation
from pipeline import Config, logger


def main():
    cfg = Config()
    cfg.output_dir = Path("outputs-baseline-rerun")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BASELINE-ONLY RERUN")
    logger.info("  csv=%s hz=%s window_size=%s", cfg.csv_path, cfg.hz, cfg.window_size)
    logger.info("  lstm_hidden=%s lstm_layers=%s dropout=%s", cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout)
    logger.info("  lr=%s weight_decay=%s batch=%s optimizer=%s patience=%s",
                cfg.learning_rate, cfg.weight_decay, cfg.batch_size,
                cfg.optimizer_choice, cfg.early_stopping_patience)
    logger.info("  max_epochs=%s early_stopping=%s", cfg.epochs_per_round, cfg.enable_early_stopping)
    logger.info("=" * 60)

    # 1) representative single training (writes baseline_history/predictions/errors plots)
    base = run_baseline(cfg, run_id=1)
    logger.info("Representative baseline single-run: test RMSE=%.4f m  R2=%.4f",
                base["metrics"]["rmse"], base["metrics"]["r2"])

    # 2) protocol headline: 30 fresh seeds (201-230), same final-eval machinery
    baseline_setting = {
        "learning_rate":    cfg.learning_rate,
        "weight_decay":     cfg.weight_decay,
        "dropout":          cfg.dropout,
        "batch_size":       cfg.batch_size,
        "lstm_hidden":      cfg.lstm_hidden,
        "lstm_layers":      cfg.lstm_layers,
        "window_size":      cfg.window_size,
        "optimizer_choice": cfg.optimizer_choice,
        "patience":         cfg.early_stopping_patience,
    }
    seeds = list(range(201, 231))   # 30 fresh seeds
    fe = run_final_evaluation(cfg, {"baseline": baseline_setting}, seeds=seeds)
    arm = fe["per_arm"]["baseline"]

    summary = {
        "baseline_setting": baseline_setting,
        "representative_single_run": {
            "test_rmse_m": base["metrics"]["rmse"],
            "r2": base["metrics"]["r2"],
        },
        "final_eval_30_seeds": {
            "mean_rmse_m": arm["mean_rmse"],
            "std_rmse_m":  arm["std_rmse"],
            "min_rmse_m":  arm["best_seed"]["rmse"],
            "min_seed":    arm["best_seed"]["seed"],
            "max_rmse_m":  arm["worst_seed"]["rmse"],
            "max_seed":    arm["worst_seed"]["seed"],
            "mean_r2":     arm["mean_r2"],
            "mean_best_epoch": arm["mean_best_epoch"],
            "n_params":    arm["n_params"],
            "n_seeds":     arm["n_seeds"],
            "seeds":       seeds,
            "per_seed":    arm["per_seed"],
        },
    }
    out = cfg.output_dir / "baseline_rerun_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("BASELINE RERUN COMPLETE")
    print(f"  representative single run: test RMSE = {base['metrics']['rmse']:.4f} m")
    print(f"  30-seed final eval (201-230): "
          f"mean = {arm['mean_rmse']:.4f} m | std = {arm['std_rmse']:.4f} | "
          f"min = {arm['best_seed']['rmse']:.4f} (seed {arm['best_seed']['seed']}) | "
          f"max = {arm['worst_seed']['rmse']:.4f} (seed {arm['worst_seed']['seed']})")
    print(f"  params = {arm['n_params']} | mean best epoch = {arm['mean_best_epoch']:.1f}")
    print(f"  summary -> {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
