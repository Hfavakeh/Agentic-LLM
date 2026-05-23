"""Pareto scalarizer for the LLM multi-objective optimisation loop.

Three cost axes are measured per round in `Trainer.pareto_metrics()`:
  * latency_ms       — measured forward-pass inference time per window.
  * stability_std_m  — std of per-sample Euclidean error (metres).
  * params_trainable — model parameter count (resource proxy).

The LLM proposes a weight vector (w_lat, w_stab, w_res) summing to 1, based on
the motion profile of the dataset. We collapse the three axes into a single
score the optimiser can target:

    score = val_loss × (1 + w_lat·lat_norm + w_stab·stab_norm + w_res·res_norm)

Each axis is min-max-normalised across the rounds seen so far so the three
disparate units (ms / metres / parameters) become comparable. val_loss stays
the primary signal — the weighted cost term acts as a multiplicative penalty.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


PARETO_AXES: Tuple[str, ...] = ("latency_ms", "stability_std_m", "params_trainable")
WEIGHT_KEYS: Tuple[str, ...] = ("w_lat", "w_stab", "w_res")
DEFAULT_WEIGHTS: Dict[str, float] = {"w_lat": 1 / 3, "w_stab": 1 / 3, "w_res": 1 / 3}

# How close to sum=1 the LLM's weights must be before we auto-normalise.
WEIGHTS_SUM_TOLERANCE = 0.10

# An axis whose observed spread is below this fraction of its magnitude is
# treated as NOT meaningfully varying. Min-max-normalising a noise-level
# range (e.g. sub-millisecond CPU forward-pass jitter, or a few-cm wobble in
# stability) would otherwise stretch pure measurement noise into a full-
# weight cost penalty and make the optimiser chase noise. Architecture
# changes — the case where deployment cost genuinely matters — move
# params/latency far past this floor, so real trade-offs are preserved.
MIN_RELATIVE_RANGE = 0.20


def _is_finite(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _axis_range(history: List[Dict[str, Any]], axis: str) -> Tuple[float, float]:
    """Min/max of one axis across rounds that have it. (lo, hi); hi==lo possible."""
    vals: List[float] = []
    for entry in history:
        pareto = entry.get("pareto") or {}
        v = pareto.get(axis)
        if _is_finite(v):
            vals.append(float(v))
    if not vals:
        return 0.0, 0.0
    return min(vals), max(vals)


def normalize(round_metrics: Dict[str, Any],
              history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Min-max-normalise the three Pareto axes against the history.

    Returns 0.0 when:
      * the value is missing / non-finite, or
      * the axis range is degenerate (hi == lo), or
      * the axis spread is below `MIN_RELATIVE_RANGE` of its magnitude — a
        noise-level range that must not be amplified into a real penalty.
    """
    out: Dict[str, float] = {}
    for axis in PARETO_AXES:
        v = round_metrics.get(axis) if round_metrics else None
        if not _is_finite(v):
            out[axis] = 0.0
            continue
        lo, hi = _axis_range(history, axis)
        spread = hi - lo
        scale  = max(abs(hi), abs(lo))
        if spread <= 0 or (scale > 0 and spread / scale < MIN_RELATIVE_RANGE):
            # Degenerate or noise-level range — nothing meaningful to penalise.
            out[axis] = 0.0
        else:
            out[axis] = (float(v) - lo) / spread
    return out


def normalise_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Coerce/sanitise an LLM-proposed weight dict.

    Missing or non-finite entries default to a uniform 1/3 share, then the
    full vector is renormalised to sum to exactly 1. Negative weights are
    clipped to 0 (a cost axis the LLM thinks doesn't matter).
    """
    if not isinstance(weights, dict):
        weights = {}
    cleaned: Dict[str, float] = {}
    for k in WEIGHT_KEYS:
        v = weights.get(k)
        if not _is_finite(v):
            cleaned[k] = DEFAULT_WEIGHTS[k]
        else:
            cleaned[k] = max(0.0, float(v))
    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in cleaned.items()}


def score(val_loss: Optional[float],
          normalized: Dict[str, float],
          weights: Dict[str, float]) -> Optional[float]:
    """Scalarized objective. Lower is better.

    score = val_loss × (1 + w_lat·lat_norm + w_stab·stab_norm + w_res·res_norm)
    """
    if not _is_finite(val_loss):
        return None
    penalty = (
        weights["w_lat"]  * normalized.get("latency_ms",       0.0)
        + weights["w_stab"] * normalized.get("stability_std_m",  0.0)
        + weights["w_res"]  * normalized.get("params_trainable", 0.0)
    )
    return float(val_loss) * (1.0 + penalty)


def score_history(history: List[Dict[str, Any]],
                  weights: Dict[str, float]) -> List[Dict[str, Any]]:
    """Re-score every past round under the *current* weight vector.

    Each entry gains a `_scored` dict: {normalized, score}. Returned list is a
    shallow copy — caller decides whether to mutate originals.
    """
    out: List[Dict[str, Any]] = []
    for entry in history:
        pareto = entry.get("pareto") or {}
        norm = normalize(pareto, history)
        s = score(entry.get("val_loss"), norm, weights)
        scored = dict(entry)
        scored["_scored"] = {"normalized": norm, "score": s, "weights": dict(weights)}
        out.append(scored)
    return out


def pick_best(history: List[Dict[str, Any]],
              weights: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Best round (lowest score) under the given weights. None if no scores."""
    scored = score_history(history, weights)
    candidates = [
        e for e in scored
        if _is_finite(e.get("_scored", {}).get("score"))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e["_scored"]["score"])
