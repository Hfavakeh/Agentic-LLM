"""Motion-thesis experiment: can an LLM use human-motion knowledge to improve
the localization network by reshaping its LOSS?

The 9 conventional hyperparameters are FROZEN at the baseline setting; the only
thing that varies is the six-lever loss-shaping vector (see
``pipeline.search_space.LOSS_SHAPING_GRID``). Same protocol structure as the HP
bake-off: each searched vector is trained 3× from scratch (seeds 101/102/103),
scored by mean validation RMSE in metres; the winner is evaluated on fresh
final-eval seeds.

Arms:
  * baseline   — plain MSE (neutral levers). The floor.
  * C2 motion  — fixed motion heuristic (v_max = p95×1.1, gentle λ's, fast-bin
                 up-weight). "Does motion loss-shaping help at all?" (vs baseline)
  * random     — 25 random lever vectors from the grid. Undirected reference.
  * C3 LLM     — the LLM sets levers from motion summaries + per-regime error.
                 "Does the LLM's motion interpretation beat a fixed heuristic?"
                 (C3 vs C2)
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline import (
    Config, LOSS_SHAPING_GRID, LOSS_SHAPING_NEUTRAL, is_finite_number, logger,
)

from .engine import (
    TRAIN_SEEDS, _coerce_setting, _default_setting, evaluate_setting,
    train_and_test_setting,
)

_LEVER_ORDER = [
    "v_max", "lambda_vel", "lambda_smooth",
    "bin_weight_slow", "bin_weight_medium", "bin_weight_fast",
]


# ---------------------------------------------------------------------------
# Lever helpers
# ---------------------------------------------------------------------------

def lever_signature(levers: Dict[str, Any]) -> tuple:
    return tuple(round(float(levers.get(k, LOSS_SHAPING_NEUTRAL[k])), 6) for k in _LEVER_ORDER)


def _snap(key: str, val: float) -> float:
    return min(LOSS_SHAPING_GRID[key], key=lambda g: abs(float(g) - float(val)))


def heuristic_levers(motion_profile: Dict[str, Any]) -> Dict[str, Any]:
    """C2's fixed motion heuristic (mirrors MotionAwareRuleBasedOptimizer):
    v_max just above the p95 speed, gentle velocity/smoothness penalties, and a
    fast-regime up-weight. All snapped to the grid."""
    p95 = motion_profile.get("speed_p95_mps")
    v_max = _snap("v_max", float(p95) * 1.1) if is_finite_number(p95) else 2.0
    return {
        "v_max":             v_max,
        "lambda_vel":        _snap("lambda_vel", 0.1),
        # 0.05, not 0.1 — mirrors MotionAwareRuleBasedOptimizer._LAMBDA_SMOOTH
        # after the 2026-07-26 unit fix (see that constant).
        "lambda_smooth":     _snap("lambda_smooth", 0.05),
        "bin_weight_slow":   1.0,
        "bin_weight_medium": 1.0,
        "bin_weight_fast":   _snap("bin_weight_fast", 1.5),
    }


def random_levers(rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(LOSS_SHAPING_GRID[k]) for k in _LEVER_ORDER}


# ---------------------------------------------------------------------------
# Scoring one lever vector on the frozen-baseline model
# ---------------------------------------------------------------------------

def _score(base_cfg: Config, dataset: Dict, base_setting: Dict, levers: Optional[Dict]) -> Dict[str, Any]:
    """Train the frozen baseline HPs with ``levers`` applied to the loss; return
    the mean-val-RMSE result. ``levers=None`` → plain MSE (baseline)."""
    return evaluate_setting(base_cfg, dataset, base_setting, loss_shaping=levers)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _run_deterministic_arm(base_cfg, dataset, base_setting, levers, name):
    """Baseline / C2: score a single lever vector."""
    logger.info("--- MOTION arm %s: %s ---", name, levers if levers else "plain MSE")
    res = _score(base_cfg, dataset, base_setting, levers)
    logger.info("%s | val_rmse=%.4f ± %.4f", name, res["score"], res["val_rmse_std"])
    return {
        "best_levers": levers,
        "best_score": res["score"],
        "attempts": [{"attempt": 1, "levers": levers, "score": res["score"],
                      "motion_error_summary": res.get("motion_error_summary", {})}],
        "best_regime_error": res.get("motion_error_summary", {}),
    }


def _run_random_arm(base_cfg, dataset, base_setting, n_attempts, seed=2000):
    rng = random.Random(seed)
    tried, history = set(), []
    best_score, best_levers, best_reg = float("inf"), None, {}
    a = 0
    guard = 0
    while a < n_attempts and guard < n_attempts * 20:
        guard += 1
        lv = random_levers(rng)
        sig = lever_signature(lv)
        if sig in tried:
            continue
        tried.add(sig)
        a += 1
        res = _score(base_cfg, dataset, base_setting, lv)
        history.append({"attempt": a, "levers": lv, "score": res["score"]})
        if is_finite_number(res["score"]) and res["score"] < best_score:
            best_score, best_levers = res["score"], lv
            best_reg = res.get("motion_error_summary", {})
        logger.info("random(motion) %d/%d | val_rmse=%.4f | best=%.4f",
                    a, n_attempts, res["score"], best_score)
    return {"best_levers": best_levers, "best_score": best_score,
            "attempts": history, "best_regime_error": best_reg}


async def _run_llm_arm(base_cfg, dataset, base_setting, agent, n_attempts):
    """C3: the LLM proposes lever vectors from motion summaries + per-regime error."""
    neutral = {k: LOSS_SHAPING_NEUTRAL[k] for k in _LEVER_ORDER}
    tried, history = set(), []
    best_score, best_levers, best_reg = float("inf"), None, {}
    best_curve: List[float] = []

    def is_tried(lv: Dict[str, Any]) -> bool:
        return lever_signature(lv) in tried

    for a in range(1, n_attempts + 1):
        anchor = best_levers if best_levers is not None else neutral
        context = {
            "history":            history,
            "anchor_levers":      anchor,
            "anchor_regime_error": best_reg,
            "is_tried":           is_tried,
            "motion_profile":     dataset.get("motion_profile", {}),
        }
        proposal = await agent.propose_loss_shaping(context)
        if not proposal.get("valid"):
            history.append({"attempt": a, "setting": dict(anchor), "score": float("inf"),
                            "output_status": "rejected",
                            "reason": proposal.get("failure_reason", "")})
            best_curve.append(best_score if is_finite_number(best_score) else float("nan"))
            logger.info("C3-LLM(motion) %d/%d | REJECTED (%s) | best=%.4f",
                        a, n_attempts, proposal.get("failure_reason", "?"),
                        best_score if is_finite_number(best_score) else float("nan"))
            continue
        levers = {k: proposal["resolved_setting"][k] for k in _LEVER_ORDER}
        res = _score(base_cfg, dataset, base_setting, levers)
        tried.add(lever_signature(levers))
        history.append({"attempt": a, "setting": levers, "score": res["score"],
                        "strategy": proposal.get("strategy"), "reason": proposal.get("reason"),
                        "output_status": proposal.get("output_status", "clean")})
        if is_finite_number(res["score"]) and res["score"] < best_score:
            best_score, best_levers = res["score"], levers
            best_reg = res.get("motion_error_summary", {})
        best_curve.append(best_score if is_finite_number(best_score) else float("nan"))
        logger.info("C3-LLM(motion) %d/%d | val_rmse=%.4f | best=%.4f | %s",
                    a, n_attempts, res["score"], best_score,
                    proposal.get("output_status", "clean"))
    return {"best_levers": best_levers, "best_score": best_score,
            "attempts": history, "best_curve": best_curve, "best_regime_error": best_reg}


# ---------------------------------------------------------------------------
# Final evaluation (test set) of each arm's chosen lever vector
# ---------------------------------------------------------------------------

def _final_eval(base_cfg, dataset, base_setting, levers, seeds) -> Dict[str, Any]:
    rmses = []
    for s in seeds:
        out = train_and_test_setting(base_cfg, dataset, base_setting, seed=int(s), loss_shaping=levers)
        rmse = out["metrics"].get("rmse")
        if is_finite_number(rmse):
            rmses.append(float(rmse))
    return {
        "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "std_rmse":  float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "n_seeds":   len(rmses),
        "per_seed":  rmses,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

ALL_MOTION_ARMS = ("baseline", "motion_rule", "random", "llm")


async def run_motion_experiment(
    base_cfg: Config,
    dataset: Dict[str, Any],
    llm_agent,
    n_attempts: int,
    final_eval_seeds: List[int],
    run_id: int = 1,
    arms_to_run=("llm",),
    search_seed: int = 2000,
) -> Dict[str, Any]:
    """Run the motion loss-shaping experiment for the selected arms, then
    final-evaluate each arm's chosen levers on fresh seeds.

    ``arms_to_run`` selects which of baseline / motion_rule (C2) / random / llm
    (C3) to run. Default is LLM-only: the baseline (plain MSE, HPs frozen at
    baseline) is IDENTICAL to the HP bake-off's baseline arm, so it never needs
    recomputing — reuse that number. Add "motion_rule" when you need the C2
    comparator (a single deterministic vector — cheap)."""
    allow_arch = bool(getattr(base_cfg, "allow_arch_changes", True))
    base_setting = _coerce_setting(_default_setting(base_cfg, allow_arch), allow_arch)
    motion_profile = dataset.get("motion_profile", {})
    logger.info("=== MOTION EXPERIMENT (run %d) — arms=%s — HPs frozen at baseline: %s ===",
                run_id, list(arms_to_run), base_setting)

    arms: Dict[str, Dict[str, Any]] = {}
    if "baseline" in arms_to_run:
        arms["baseline"] = _run_deterministic_arm(base_cfg, dataset, base_setting, None, "baseline (plain MSE)")
    if "motion_rule" in arms_to_run:
        arms["motion_rule"] = _run_deterministic_arm(
            base_cfg, dataset, base_setting, heuristic_levers(motion_profile), "C2 (motion heuristic)")
    if "random" in arms_to_run:
        # Seed the random search off the outer search seed so multiple seeds give
        # DIFFERENT random searches (a distribution to paired-test), not a repeat.
        arms["random"] = _run_random_arm(base_cfg, dataset, base_setting, n_attempts,
                                         seed=int(search_seed))
    if "llm" in arms_to_run:
        arms["llm"] = await _run_llm_arm(base_cfg, dataset, base_setting, llm_agent, n_attempts)

    # Final eval each run arm's chosen levers on fresh seeds.
    final: Dict[str, Any] = {}
    for name, arm in arms.items():
        fe = _final_eval(base_cfg, dataset, base_setting, arm.get("best_levers"), final_eval_seeds)
        final[name] = fe
        logger.info("MOTION final eval | %-14s test RMSE = %.4f ± %.4f (n=%d) | levers=%s",
                    name, fe["mean_rmse"], fe["std_rmse"], fe["n_seeds"], arm.get("best_levers"))

    return {
        "base_setting": base_setting,
        "motion_profile": motion_profile,
        "n_attempts": n_attempts,
        "arms_run": [a for a in ALL_MOTION_ARMS if a in arms],
        "arms": arms,
        "final_eval": final,
        "final_eval_seeds": list(final_eval_seeds),
    }


def save_motion_results(results: Dict[str, Any], out_dir: Path, run_id: int = 1) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"motion_experiment_run{run_id}.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Compact markdown table: baseline / C2 / random / C3 test RMSE.
    fe = results["final_eval"]
    lines = ["# Motion loss-shaping experiment\n",
             f"HPs frozen at baseline: `{results['base_setting']}`\n",
             "| Arm | test RMSE | ± | n | best levers |",
             "|---|--:|--:|--:|---|"]
    labels = {"baseline": "baseline (plain MSE)", "motion_rule": "C2 motion heuristic",
              "random": "random levers", "llm": "C3 LLM motion"}
    for name in results.get("arms_run", ALL_MOTION_ARMS):
        f = fe.get(name, {})
        lv = results["arms"].get(name, {}).get("best_levers")
        lv_s = "plain MSE" if not lv else ", ".join(f"{k}={lv[k]}" for k in _LEVER_ORDER)
        lines.append(f"| {labels[name]} | {f.get('mean_rmse', float('nan')):.4f} | "
                     f"{f.get('std_rmse', 0):.4f} | {f.get('n_seeds', 0)} | {lv_s} |")
    (out_dir / f"motion_experiment_run{run_id}.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Motion results saved -> %s", out_dir)
