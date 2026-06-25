"""Qualitative label helpers (numbers -> words for the LLM prompt).

The protocol mandates that the LLM sees only qualitative labels — raw numbers
stay in the logs/report. The rule-based controller consumes the SAME labels
so the only difference between the arms is how a label is acted on.
"""

from typing import Any, List, Optional

from pipeline import is_finite_number

# Soft diagnosis labels (email protocol). Underscored form is canonical; the
# prompt shows the spaced form and `_normalize_diagnosis` maps either way.
PROTOCOL_DIAGNOSES = {
    "healthy",
    "possible_overfitting_tendency",
    "possible_underfitting_tendency",
    "plateau",
    "unstable",
    "inconclusive",
}


def _normalize_diagnosis(text: Any) -> str:
    """Map free-text / legacy diagnosis wording onto a canonical soft label.

    Small models phrase these loosely ('overfitting', 'over-fit tendency',
    'looks healthy'), so match on keywords rather than exact membership.
    """
    s = str(text or "").strip().lower().replace("-", " ")
    if not s:
        return "inconclusive"
    if "unstable" in s or "diverg" in s or "nan" in s:
        return "unstable"
    if "overfit" in s or "over fit" in s:
        return "possible_overfitting_tendency"
    if "underfit" in s or "under fit" in s:
        return "possible_underfitting_tendency"
    if "plateau" in s or "stagnant" in s or "stuck" in s:
        return "plateau"
    if "healthy" in s or "good" in s or "on track" in s:
        return "healthy"
    collapsed = s.replace(" ", "_")
    return collapsed if collapsed in PROTOCOL_DIAGNOSES else "inconclusive"


def _qual_variation(score: Any, std: Any) -> str:
    """Reliability across the 3 trainings, from the coefficient of variation."""
    if not (is_finite_number(score) and is_finite_number(std)) or float(score) <= 0:
        return "unknown"
    cv = float(std) / float(score)
    return "low" if cv < 0.05 else "medium" if cv < 0.15 else "high"


def _qual_gap(mean_val_loss: Any, gap: Any) -> str:
    """Train/val gap size, normalised by the validation-loss scale."""
    if not (is_finite_number(mean_val_loss) and is_finite_number(gap)) or float(mean_val_loss) <= 0:
        return "unknown"
    ratio = float(gap) / float(mean_val_loss)
    return "small" if ratio < 0.10 else "medium" if ratio < 0.30 else "large"


def _qual_epoch_timing(mean_best_epoch: Any, max_epochs: int) -> str:
    """Where the best epoch landed within the training budget."""
    if not is_finite_number(mean_best_epoch) or not max_epochs:
        return "unknown"
    frac = float(mean_best_epoch) / float(max_epochs)
    return "early" if frac < 0.33 else "middle" if frac < 0.66 else "late"


def _qual_quality(score: Any, all_scores: List[float]) -> str:
    """Validation quality of one setting relative to all attempts so far."""
    finite = sorted(s for s in all_scores if is_finite_number(s))
    if not finite or not is_finite_number(score):
        return "unknown"
    best = finite[0]
    if float(score) <= best * 1.02:
        return "best"
    # Position among finite scores (0 = best, 1 = worst).
    worse = sum(1 for s in finite if s < float(score))
    frac = worse / max(len(finite) - 1, 1)
    return "good" if frac < 0.34 else "average" if frac < 0.67 else "poor"


def _qual_level_10(score: Any, all_scores: List[float]) -> Optional[int]:
    """Map a setting's score onto a 1-10 quality level (1 = best, 10 = worst),
    scaled between the best and worst finite scores seen so far.

    The 3-bucket `_qual_quality` (best/good/average/poor) is too coarse for the
    LLM to read a *trend* off a list of attempts; this finer ordinal lets it see
    how much better/worse one setting is than another and which direction the
    search is moving. Raw numbers still stay out of the prompt."""
    finite = sorted(s for s in all_scores if is_finite_number(s))
    if not finite or not is_finite_number(score):
        return None
    best, worst = finite[0], finite[-1]
    if worst <= best:           # only one distinct value seen
        return 1
    frac = (float(score) - best) / (worst - best)   # 0 = best, 1 = worst
    return int(round(1 + frac * 9))                  # 1..10


def _behavior_label(variation: str, gap: str, quality: str) -> str:
    """Soft behavior label from the qualitative signals (email's label set)."""
    if variation == "high":
        return "unstable"
    if gap == "large":
        return "possible_overfitting_tendency"
    if gap == "small" and quality == "poor":
        return "possible_underfitting_tendency"
    if quality in ("best", "good", "average"):
        return "healthy"
    return "inconclusive"


def _fmt_grid_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)
