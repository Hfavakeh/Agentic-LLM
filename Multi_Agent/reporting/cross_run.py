"""Cross-run quality and efficiency summaries."""

from typing import Any, Dict, List, Optional

import numpy as np

from pipeline import is_finite_number, safe_trapz


def compute_cross_run_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute compact cross-run quality and efficiency summaries.

    Per-arm statistics reported
    ---------------------------
    rmse : mean ± std across runs
    r2   : mean ± std across runs
    """
    def _arm_metric(run: Dict[str, Any], arm: str, key: str) -> float:
        """Pull a metric from an arm safely. Returns NaN when the arm is
        absent (e.g. Optuna skipped because the package wasn't installed) so
        downstream `np.nanmean` / `_safe_stats` simply ignore it.
        """
        slot = run.get(arm)
        if not isinstance(slot, dict):
            return np.nan
        return slot.get("metrics", {}).get(key, np.nan)

    b_rmses  = [_arm_metric(r, "baseline",   "rmse") for r in all_results]
    l_rmses  = [_arm_metric(r, "llm",        "rmse") for r in all_results]
    r_rmses  = [_arm_metric(r, "random",     "rmse") for r in all_results]
    rb_rmses = [_arm_metric(r, "rule_based", "rmse") for r in all_results]
    o_rmses  = [_arm_metric(r, "optuna",     "rmse") for r in all_results]

    b_r2s  = [_arm_metric(r, "baseline",   "r2") for r in all_results]
    l_r2s  = [_arm_metric(r, "llm",        "r2") for r in all_results]
    r_r2s  = [_arm_metric(r, "random",     "r2") for r in all_results]
    rb_r2s = [_arm_metric(r, "rule_based", "r2") for r in all_results]
    o_r2s  = [_arm_metric(r, "optuna",     "r2") for r in all_results]

    baseline_best_losses = []
    aucs: Dict[str, List[float]] = {
        "baseline": [], "llm": [], "random": [], "rule_based": [], "optuna": [],
    }

    for run in all_results:
        bv  = run["baseline"]["history"].get("val_loss", [])
        lv  = run["llm"]["history"].get("val_loss", [])
        rv  = run["random"]["history"].get("val_loss", [])
        rbv = run["rule_based"]["history"].get("val_loss", [])
        ov  = (run.get("optuna") or {}).get("history", {}).get("val_loss", []) or []
        if bv:
            baseline_best_losses.append(min(bv))
            aucs["baseline"].append(safe_trapz(bv))
        if lv:
            aucs["llm"].append(safe_trapz(lv))
        if rv:
            aucs["random"].append(safe_trapz(rv))
        if rbv:
            aucs["rule_based"].append(safe_trapz(rbv))
        if ov:
            aucs["optuna"].append(safe_trapz(ov))

    threshold = float(np.nanmedian(baseline_best_losses)) if baseline_best_losses else None
    llm_trials, random_trials, rb_trials, optuna_trials = [], [], [], []
    if threshold is not None:
        for run in all_results:
            opt_log = run["llm"]["optimization_log"]
            lv = list(
                float(x.get("val_loss"))
                for x in opt_log.get("rounds", [])
                if is_finite_number(x.get("val_loss"))
            )
            rv = [
                float(x.get("val_loss"))
                for x in run["random"].get("random_log", [])
                if is_finite_number(x.get("val_loss"))
            ]
            rbv = list(
                float(x.get("val_loss"))
                for x in run["rule_based"].get("optimization_log", {}).get("rounds", [])
                if is_finite_number(x.get("val_loss"))
            )
            optuna_slot = run.get("optuna") or {}
            ov = [
                float(x.get("val_loss"))
                for x in optuna_slot.get("optuna_log", [])
                if is_finite_number(x.get("val_loss"))
            ]
            llm_trials.append(
                next((i + 1 for i, v in enumerate(lv) if v is not None and v <= threshold), None)
            )
            random_trials.append(
                next((i + 1 for i, v in enumerate(rv) if v is not None and v <= threshold), None)
            )
            rb_trials.append(
                next((i + 1 for i, v in enumerate(rbv) if v is not None and v <= threshold), None)
            )
            optuna_trials.append(
                next((i + 1 for i, v in enumerate(ov) if v is not None and v <= threshold), None)
            )

    def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
        filtered = [v for v in values if v is not None]
        return float(np.nanmean(filtered)) if filtered else None

    def _safe_stats(vals: List[float]) -> Dict[str, Optional[float]]:
        """mean + std of finite values in *vals*."""
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return {"mean": None, "std": None}
        arr = np.array(finite, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    mean_b  = float(np.nanmean(b_rmses))
    mean_l  = float(np.nanmean(l_rmses))
    mean_r  = float(np.nanmean(r_rmses))
    mean_rb = float(np.nanmean(rb_rmses))
    mean_o  = float(np.nanmean(o_rmses)) if any(np.isfinite(o_rmses)) else float("nan")
    has_optuna = bool(np.any(np.isfinite(o_rmses)))
    return {
        "optimization_quality": {
            # ── RMSE ─────────────────────────────────────────────────────
            "mean_rmse": {
                "baseline":   mean_b,
                "llm":        mean_l,
                "random":     mean_r,
                "rule_based": mean_rb,
                "optuna":     mean_o if has_optuna else None,
            },
            "rmse": {
                "baseline":   _safe_stats(b_rmses),
                "llm":        _safe_stats(l_rmses),
                "random":     _safe_stats(r_rmses),
                "rule_based": _safe_stats(rb_rmses),
                "optuna":     _safe_stats(o_rmses),
            },
            # ── R² ───────────────────────────────────────────────────────
            "mean_r2": {
                "baseline":   float(np.nanmean(b_r2s)),
                "llm":        float(np.nanmean(l_r2s)),
                "random":     float(np.nanmean(r_r2s)),
                "rule_based": float(np.nanmean(rb_r2s)),
                "optuna":     float(np.nanmean(o_r2s)) if has_optuna else None,
            },
            "r2": {
                "baseline":   _safe_stats(b_r2s),
                "llm":        _safe_stats(l_r2s),
                "random":     _safe_stats(r_r2s),
                "rule_based": _safe_stats(rb_r2s),
                "optuna":     _safe_stats(o_r2s),
            },
            # ── Comparative ──────────────────────────────────────────────
            "pct_improvement_over_baseline": (
                (mean_b - mean_l) / mean_b * 100
                if mean_b and np.isfinite(mean_b) else None
            ),
            "pct_improvement_rule_based_over_baseline": (
                (mean_b - mean_rb) / mean_b * 100
                if mean_b and np.isfinite(mean_b) else None
            ),
            "pct_improvement_optuna_over_baseline": (
                (mean_b - mean_o) / mean_b * 100
                if (mean_b and np.isfinite(mean_b) and has_optuna and np.isfinite(mean_o)) else None
            ),
            "win_rate": (
                float(np.nanmean([l < b for l, b in zip(l_rmses, b_rmses)]))
                if l_rmses else None
            ),
            "win_rate_rule_based": (
                float(np.nanmean([rb < b for rb, b in zip(rb_rmses, b_rmses)]))
                if rb_rmses else None
            ),
            "win_rate_optuna": (
                float(np.nanmean([o < b for o, b in zip(o_rmses, b_rmses) if np.isfinite(o)]))
                if has_optuna else None
            ),
        },
        "optimization_efficiency": {
            "trials_to_threshold": {
                "llm":        _safe_mean(llm_trials),
                "random":     _safe_mean(random_trials),
                "rule_based": _safe_mean(rb_trials),
                "optuna":     _safe_mean(optuna_trials) if has_optuna else None,
                "threshold":  threshold,
            },
            "auc_val_loss_curve": {
                "baseline":   _safe_mean(aucs["baseline"]),
                "llm":        _safe_mean(aucs["llm"]),
                "random":     _safe_mean(aucs["random"]),
                "rule_based": _safe_mean(aucs["rule_based"]),
                "optuna":     _safe_mean(aucs["optuna"]) if aucs["optuna"] else None,
            },
        },
    }
