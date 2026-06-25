"""Experiment runners: full experiment (N runs), per-seed isolation, and the
multi-seed replication runner with cross-seed aggregation."""

import copy
import json
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline import Config, logger, set_seed
from reporting import compute_cross_run_metrics, plot_cross_run_summary

from .single_run import run_experiment

# 30 fixed, distinct, non-sequential seeds. Hard-coded (not randomly sampled)
# so every run of the experiment is reproducible.
EXPERIMENT_SEEDS: List[int] = [
    17, 42, 73, 128, 256, 314, 451, 512, 666, 777]


# ---------------------------------------------------------------------------
# Full experiment (N runs → aggregate → report)
# ---------------------------------------------------------------------------

async def run_full_experiment(
    config: Config,
    llm_model: str,
    *,
    seed: Optional[int] = None,
):
    """Run the full experiment (all runs) for a single seed.

    If *seed* is provided it overrides config.seed so that each seed
    replication is truly independent.
    """
    if seed is not None:
        config.seed = seed
    set_seed(config.seed)

    all_results = []

    for run_id in range(1, config.num_experiment_runs + 1):
        set_seed(config.seed + run_id)
        logger.info(
            "\n%s\nRUN %d / %d\n%s",
            "=" * 60, run_id, config.num_experiment_runs, "=" * 60,
        )
        result = await run_experiment(
            config=config,
            run_id=run_id,
            llm_model=llm_model,
        )
        all_results.append(result)

    cross = compute_cross_run_metrics(all_results)
    with open(config.output_dir / "cross_run_metrics.json", "w") as f:
        json.dump(cross, f, indent=2, default=str)

    # Cross-run summary: all runs overlaid, one panel per optimiser.
    plot_cross_run_summary(config, all_results, filename="cross_run_summary.png")

    q = cross["optimization_quality"]
    e = cross["optimization_efficiency"]

    logger.info('\\n%s\\nEXPERIMENT COMPLETE\\n%s', '=' * 60, '=' * 60)
    logger.info('Results saved to: %s', config.output_dir)
    logger.info(
        'RMSE | LLM %.4f (+-%.4f)  Baseline %.4f (+-%.4f)  Random %.4f (+-%.4f)  Rule-Based %.4f (+-%.4f)',
        q['rmse']['llm']['mean']        or 0, q['rmse']['llm']['std']        or 0,
        q['rmse']['baseline']['mean']   or 0, q['rmse']['baseline']['std']   or 0,
        q['rmse']['random']['mean']     or 0, q['rmse']['random']['std']     or 0,
        q['rmse']['rule_based']['mean'] or 0, q['rmse']['rule_based']['std'] or 0,
    )
    logger.info(
        'R2   | LLM %.4f (+-%.4f)  Baseline %.4f (+-%.4f)  Random %.4f (+-%.4f)  Rule-Based %.4f (+-%.4f)',
        q['r2']['llm']['mean']        or 0, q['r2']['llm']['std']        or 0,
        q['r2']['baseline']['mean']   or 0, q['r2']['baseline']['std']   or 0,
        q['r2']['random']['mean']     or 0, q['r2']['random']['std']     or 0,
        q['r2']['rule_based']['mean'] or 0, q['r2']['rule_based']['std'] or 0,
    )
    logger.info('LLM improvement over baseline:        %.1f%%', q['pct_improvement_over_baseline'] or 0)
    logger.info('Rule-Based improvement over baseline: %.1f%%', q['pct_improvement_rule_based_over_baseline'] or 0)
    if q.get('pct_improvement_optuna_over_baseline') is not None:
        logger.info('Optuna improvement over baseline:     %.1f%%', q['pct_improvement_optuna_over_baseline'])
    logger.info('LLM win rate across runs:        %.0f%%', (q['win_rate'] or 0) * 100)
    logger.info('Rule-Based win rate across runs: %.0f%%', (q['win_rate_rule_based'] or 0) * 100)
    if q.get('win_rate_optuna') is not None:
        logger.info('Optuna win rate across runs:     %.0f%%', q['win_rate_optuna'] * 100)
    logger.info(
        'Avg trials to threshold | LLM %.1f  Random %.1f  Rule-Based %.1f  Optuna %s',
        e['trials_to_threshold']['llm']        or 0,
        e['trials_to_threshold']['random']     or 0,
        e['trials_to_threshold']['rule_based'] or 0,
        (f"{e['trials_to_threshold']['optuna']:.1f}"
         if e['trials_to_threshold'].get('optuna') is not None else "n/a"),
    )


# ---------------------------------------------------------------------------
# Per-seed isolation
# ---------------------------------------------------------------------------

async def run_one_seed(
    base_config: Config,
    seed: int,
    llm_model: str,
) -> Dict[str, Any]:
    """
    Run the complete experiment pipeline for a single *seed*.

    Each seed gets its own sub-directory (``<output_dir>/seed_<seed>``)
    so that plots, checkpoints and logs from different seeds never
    collide, making every replication fully isolated and reproducible.

    Returns a dict that mirrors the structure of a single-seed
    ``all_results`` list, tagged with the seed value.
    """

    cfg = copy.deepcopy(base_config)
    cfg.output_dir = base_config.output_dir / f"seed_{seed}"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n%s\nSEED %d  →  %s\n%s", "#" * 60, seed, cfg.output_dir, "#" * 60)

    await run_full_experiment(
        config=cfg,
        llm_model=llm_model,
        seed=seed,
    )

    # Collect the final cross-run metrics produced by run_full_experiment.
    metrics_path = cfg.output_dir / "cross_run_metrics.json"
    cross_metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as fh:
            cross_metrics = json.load(fh)

    return {"seed": seed, "output_dir": str(cfg.output_dir), "cross_run_metrics": cross_metrics}


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

async def run_multi_seed_experiment(
    base_config: Config,
    seeds: List[int],
    llm_model: str,
) -> None:
    """
    Run the full experiment pipeline for every seed in *seeds* and
    aggregate the results into a single ``multi_seed_summary.json``
    written to ``base_config.output_dir``.

    Aggregated statistics include mean ± std across seeds for:
    - RMSE (baseline, LLM, random)
    - LLM % improvement over baseline
    - LLM win rate
    - Trials-to-threshold (LLM, random)
    """
    logger.info("\n%s\nMULTI-SEED EXPERIMENT\nSeeds: %s\n%s",
                "=" * 60, seeds, "=" * 60)

    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        result = await run_one_seed(
            base_config=base_config,
            seed=seed,
            llm_model=llm_model,
        )
        seed_results.append(result)

    # ── Aggregate cross-seed metrics ──────────────────────────────────────
    def _collect(key_path: List[str]) -> List[Optional[float]]:
        """Extract a nested value from each seed's cross_run_metrics."""
        vals = []
        for r in seed_results:
            obj = r.get("cross_run_metrics", {})
            for k in key_path:
                if isinstance(obj, dict):
                    obj = obj.get(k)
                else:
                    obj = None
                    break
            vals.append(float(obj) if obj is not None and np.isfinite(float(obj)) else None)
        return vals

    def _stats(vals: List[Optional[float]]) -> Dict[str, Optional[float]]:
        finite = [v for v in vals if v is not None]
        if not finite:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        arr = np.array(finite)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "n":    len(arr),
        }

    summary: Dict[str, Any] = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "per_seed": seed_results,
        "aggregated": {
            "rmse": {
                "baseline":   _stats(_collect(["optimization_quality", "mean_rmse", "baseline"])),
                "llm":        _stats(_collect(["optimization_quality", "mean_rmse", "llm"])),
                "random":     _stats(_collect(["optimization_quality", "mean_rmse", "random"])),
                "rule_based": _stats(_collect(["optimization_quality", "mean_rmse", "rule_based"])),
                "optuna":     _stats(_collect(["optimization_quality", "mean_rmse", "optuna"])),
            },
            "r2": {
                "baseline":   _stats(_collect(["optimization_quality", "mean_r2", "baseline"])),
                "llm":        _stats(_collect(["optimization_quality", "mean_r2", "llm"])),
                "random":     _stats(_collect(["optimization_quality", "mean_r2", "random"])),
                "rule_based": _stats(_collect(["optimization_quality", "mean_r2", "rule_based"])),
                "optuna":     _stats(_collect(["optimization_quality", "mean_r2", "optuna"])),
            },
            "pct_improvement_over_baseline": _stats(
                _collect(["optimization_quality", "pct_improvement_over_baseline"])
            ),
            "pct_improvement_rule_based_over_baseline": _stats(
                _collect(["optimization_quality", "pct_improvement_rule_based_over_baseline"])
            ),
            "pct_improvement_optuna_over_baseline": _stats(
                _collect(["optimization_quality", "pct_improvement_optuna_over_baseline"])
            ),
            "win_rate":            _stats(_collect(["optimization_quality", "win_rate"])),
            "win_rate_rule_based": _stats(_collect(["optimization_quality", "win_rate_rule_based"])),
            "win_rate_optuna":     _stats(_collect(["optimization_quality", "win_rate_optuna"])),
            "trials_to_threshold": {
                "llm":        _stats(_collect(["optimization_efficiency", "trials_to_threshold", "llm"])),
                "random":     _stats(_collect(["optimization_efficiency", "trials_to_threshold", "random"])),
                "rule_based": _stats(_collect(["optimization_efficiency", "trials_to_threshold", "rule_based"])),
                "optuna":     _stats(_collect(["optimization_efficiency", "trials_to_threshold", "optuna"])),
            },
            "auc_val_loss_curve": {
                "baseline":   _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "baseline"])),
                "llm":        _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "llm"])),
                "random":     _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "random"])),
                "rule_based": _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "rule_based"])),
                "optuna":     _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "optuna"])),
            },
        },
    }

    out_path = base_config.output_dir / "multi_seed_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Multi-seed summary saved → %s", out_path)

    # ── Console report ────────────────────────────────────────────────────
    agg = summary["aggregated"]
    logger.info("\n%s\nMULTI-SEED SUMMARY  (%d seeds)\n%s", "=" * 60, len(seeds), "=" * 60)
    for arm in ["baseline", "llm", "random", "rule_based"]:
        rs = agg["rmse"][arm]
        r2s = agg["r2"][arm]
        logger.info(
            "RMSE %-8s | mean=%.4f  std=%.4f  [%.4f, %.4f]",
            arm, rs["mean"] or 0, rs["std"] or 0, rs["min"] or 0, rs["max"] or 0,
        )
        logger.info(
            "R²   %-8s | mean=%.4f  std=%.4f  [%.4f, %.4f]",
            arm, r2s["mean"] or 0, r2s["std"] or 0, r2s["min"] or 0, r2s["max"] or 0,
        )
    imp = agg["pct_improvement_over_baseline"]
    logger.info(
        "LLM improvement over baseline | mean=%.1f%%  std=%.1f%%",
        imp["mean"] or 0, imp["std"] or 0,
    )
    wr = agg["win_rate"]
    logger.info(
        "LLM win rate across seeds     | mean=%.0f%%  std=%.0f%%",
        (wr["mean"] or 0) * 100, (wr["std"] or 0) * 100,
    )
