import asyncio
import copy
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelInfo
from model_pipeline import Config, HP_BOUNDS, HP_GRID, OPTIMIZER_CHOICES, Trainer, is_finite_number, sanitize_for_json, logger
import pareto


def _parse_scalar(raw_value: Any) -> Any:
    """Parse one scalar value from an LLM response."""
    if isinstance(raw_value, dict):
        for key in ("value", "new_value", "new", "set_to"):
            if key in raw_value:
                return _parse_scalar(raw_value[key])
        return raw_value
    if not isinstance(raw_value, str):
        return raw_value

    value = raw_value.strip().strip("`").strip()
    value = re.split(r"\s+#|\s+//|\s+\(", value, maxsplit=1)[0].strip()
    value = value.strip("\"'")
    lower = value.lower()
    if lower in {"true", "yes", "y"}:
        return True
    if lower in {"false", "no", "n"}:
        return False
    if lower in {"none", "null", "n/a", "na"}:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?", value, flags=re.IGNORECASE):
            return float(value)
    except Exception:
        pass
    return value


def _round_metric(value: Any, digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if is_finite_number(value) else None


def _format_history(history: List[Dict], recent_limit: int = 5) -> Dict[str, Any]:
    """Build compact structured memory from prior rounds."""
    rounds: List[Dict[str, Any]] = []
    for entry in history:
        diag = entry.get("diagnosis") or {}
        rounds.append({
            "round":           entry.get("round", "?"),
            "diagnosis":       diag.get("primary_problem", "unknown"),
            "severity":        diag.get("severity", "unknown"),
            "strategy":        entry.get("strategy", "unknown"),
            "confidence":      entry.get("confidence", "unknown"),
            "changes_applied": entry.get("changes_applied", {}),
            "best_val_loss":   _round_metric(entry.get("val_loss")),
            "avg_val_loss":    _round_metric(entry.get("round_avg_val_loss")),
            "best_rmse":       _round_metric(entry.get("round_best_rmse")),
            "avg_rmse":        _round_metric(entry.get("round_avg_rmse")),
            "outcome":         entry.get("outcome", "unknown"),
            "pareto":          entry.get("pareto") or {},
            "pareto_score":    _round_metric((entry.get("pareto") or {}).get("score")),
            "pareto_weights":  (entry.get("pareto") or {}).get("weights") or {},
        })

    # `best_so_far` is the round with the lowest scalarized multi-objective
    # score (accuracy + weighted cost). Fall back to plain val_loss only when
    # no round has a finite Pareto score yet.
    best_so_far = None
    scored_rounds = [r for r in rounds if is_finite_number(r.get("pareto_score"))]
    if scored_rounds:
        best_so_far = min(scored_rounds, key=lambda r: float(r["pareto_score"]))
    else:
        valid_rounds = [r for r in rounds if is_finite_number(r.get("best_val_loss"))]
        if valid_rounds:
            best_so_far = min(valid_rounds, key=lambda r: float(r["best_val_loss"]))

    return {
        "best_so_far": best_so_far,
        "recent_rounds": rounds[-recent_limit:],
        "older_rounds_omitted": max(0, len(rounds) - recent_limit),
    }


def _parse_changes_text(text: Any) -> Dict[str, Any]:
    """Parse a compact changes field such as 'learning_rate=0.0005, dropout=0.35'."""
    if isinstance(text, dict):
        return {k: _parse_scalar(v) for k, v in text.items() if k in ALLOWED_HP_KEYS}
    if text is None:
        return {}
    s = str(text).strip()
    if not s or s.lower() in {"none", "no changes", "no change", "{}", "[]", "null", "n/a", "na"}:
        return {}

    changes: Dict[str, Any] = {}
    hp_names = "|".join(re.escape(k) for k in sorted(ALLOWED_HP_KEYS, key=len, reverse=True))
    for key in ALLOWED_HP_KEYS:
        pattern = (
            rf"\b{re.escape(key)}\b\s*(?:=|:|->|to)\s*"
            rf"(.+?)(?=(?:[,;]\s*|\s+)({hp_names})\b\s*(?:=|:|->|to)|[,;\n]|$)"
        )
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if match:
            changes[key] = _parse_scalar(match.group(1))
    return changes


def _parse_text_proposal(raw: str) -> Dict[str, Any]:
    """Parse the preferred line-oriented LLM response format.

    Expected shape:
        diagnosis: healthy
        severity: low
        changes: learning_rate=0.0005

    The parser is intentionally forgiving about markdown bullets, bold labels,
    and '=' versus ':' separators.
    """
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    fields: Dict[str, Any] = {}
    direct_changes: Dict[str, Any] = {}
    aliases = {
        "diagnosis": "diagnosis",
        "primary_problem": "diagnosis",
        "problem": "diagnosis",
        "severity": "severity",
        "situation": "situation",
        "changes": "changes",
        "proposed_changes": "changes",
        "hyperparameters": "changes",
        "resets_model": "resets_model",
        "reset_model": "resets_model",
        "reset": "resets_model",
        "strategy": "strategy",
        "reasoning": "reasoning",
        "confidence": "confidence",
        "expected_improvement": "expected_improvement",
        # Multi-objective Pareto cost weights.
        "w_lat": "w_lat",
        "w_stab": "w_stab",
        "w_res": "w_res",
    }

    for line in text.splitlines():
        clean = line.strip()
        clean = re.sub(r"^\s*[-*+]\s+", "", clean)
        clean = clean.replace("**", "").strip()
        if not clean:
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\s*(?:=|:)\s*(.+)$", clean)
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        if key in ALLOWED_HP_KEYS:
            direct_changes[key] = _parse_scalar(value)
            continue
        canonical = aliases.get(key)
        if canonical:
            fields[canonical] = value

    raw_diagnosis = str(fields.get("diagnosis", "unknown")).strip().lower()
    # Small LLMs (e.g. qwen2.5-coder:3b) often emit `diagnosis: healthy(low)`,
    # collapsing severity into the diagnosis line. Pull the leading token out
    # and reuse any parenthetical as a severity fallback.
    diag_match = re.match(r"([a-z_][a-z0-9_]*)\s*\(?\s*([a-z]+)?\s*\)?", raw_diagnosis)
    if diag_match:
        diagnosis = diag_match.group(1).replace(" ", "_")
        embedded_severity = (diag_match.group(2) or "").strip()
    else:
        diagnosis = raw_diagnosis.replace(" ", "_")
        embedded_severity = ""
    severity = str(fields.get("severity", "")).strip().lower() or embedded_severity or "unknown"
    changes = _parse_changes_text(fields.get("changes"))
    changes.update(direct_changes)

    if diagnosis not in ALLOWED_DIAGNOSES:
        raise ValueError("No parseable text proposal found in response")

    # Multi-objective Pareto cost weights — collected only when the LLM
    # provides them; None lets the validator fall back to a uniform split.
    weight_fields = {
        wk: _parse_scalar(fields[wk])
        for wk in ("w_lat", "w_stab", "w_res")
        if wk in fields
    }

    return {
        "diagnosis": {
            "primary_problem": diagnosis,
            "severity": severity,
            "situation": str(fields.get("situation", "")).strip(),
        },
        "proposed_changes": changes,
        "resets_model": bool(_parse_scalar(fields.get("resets_model", False))),
        "strategy": str(fields.get("strategy", "exploit")).strip().lower(),
        "reasoning": str(fields.get("reasoning", "")).strip(),
        "confidence": str(fields.get("confidence", "medium")).strip().lower(),
        "expected_improvement": str(fields.get("expected_improvement", "")).strip(),
        "pareto_weights": weight_fields or None,
    }


def _parse_llm_proposal(raw: str) -> Dict[str, Any]:
    """Parse the line-oriented `key: value` response format.

    JSON is intentionally not supported. Small local LLMs (3B-8B) spend too
    much effort on braces, commas, quoting, escaping, and schema details when
    forced into JSON, and that effort is taken away from the actual control
    task. The system prompt asks for plain `key: value` lines; any non-
    conforming response is treated as a parse failure and triggers the
    standard retry / skip logic.
    """
    return _parse_text_proposal(raw)


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

ALLOWED_DIAGNOSES = {"overfitting", "underfitting", "plateau", "healthy", "no_data"}

# The 9 conventional hyperparameters of the protocol search space. Motion
# loss-shaping levers are deferred from the search space for now (see memory:
# professor-protocol), so they are NOT accepted here — a proposal naming one
# is treated as an unknown key.
ALLOWED_HP_KEYS = set(HP_GRID.keys())

# Parameters that must be integers.
INTEGER_HP_KEYS = {"lstm_hidden", "lstm_layers", "batch_size", "window_size", "patience"}

# Integer / categorical params validated by exact grid membership (the
# existing snap branch int-coerces these safely). The float-valued grid
# params (learning_rate, weight_decay, dropout) are validated against
# HP_GRID with on-grid membership in the hard-validation rework (Step 2/3);
# for now they go through the range-clamp path so the samplers, which only
# ever emit on-grid values, work end-to-end.
DISCRETE_HP_VALUES: Dict[str, set] = {
    "lstm_hidden":      set(HP_GRID["lstm_hidden"]),
    "lstm_layers":      set(HP_GRID["lstm_layers"]),
    "batch_size":       set(HP_GRID["batch_size"]),
    "window_size":      set(HP_GRID["window_size"]),
    "patience":         set(HP_GRID["patience"]),
    "optimizer_choice": set(OPTIMIZER_CHOICES),
}

# Architecture-changing parameters (trigger model rebuild).
ARCH_CHANGE_KEYS = {"lstm_hidden", "lstm_layers"}


def _empty_corrections() -> Dict[str, Any]:
    """Structured record of every silent fix the validator applies."""
    return {
        "unknown_keys_stripped":      [],   # list[str]
        "clamped":                    [],   # list[{key, raw, clamped, lo, hi}]
        "discrete_snapped":           [],   # list[{key, raw, snapped, allowed}]
        "diagnosis_auto_corrected":   None, # {from, to, reason} | None
        "resets_model_corrected":     None, # {from, to, reason} | None
        "any_correction":             False,
    }


class SemanticRepairRequired(ValueError):
    """Strict-mode rejection: the LLM's proposal would have been silently
    repaired by the validator. Carries the structured `corrections` dict so
    the caller can attribute the rejection to a specific violation type
    (out-of-range clamp, invalid diagnosis label, unknown HP key, etc.).
    """
    def __init__(self, corrections: Dict[str, Any]):
        parts: List[str] = []
        if corrections["unknown_keys_stripped"]:
            parts.append(f"unknown_keys={corrections['unknown_keys_stripped']}")
        if corrections["clamped"]:
            parts.append(
                "out_of_range="
                + str([(c["key"], c["raw"], (c["lo"], c["hi"])) for c in corrections["clamped"]])
            )
        if corrections["discrete_snapped"]:
            parts.append(
                "invalid_discrete="
                + str([(c["key"], c["raw"]) for c in corrections["discrete_snapped"]])
            )
        if corrections["diagnosis_auto_corrected"]:
            parts.append(f"diagnosis_mismatch={corrections['diagnosis_auto_corrected']}")
        if corrections["resets_model_corrected"]:
            parts.append(f"reset_flag_mismatch={corrections['resets_model_corrected']}")
        super().__init__("strict_mode_repair_required: " + "; ".join(parts))
        self.corrections = corrections


def _validate_proposal(parsed: Dict[str, Any], context: Dict[str, Any],
                       strict: bool = False) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate and sanitise a parsed LLM proposal.

    Two modes:

    * ``strict=False`` (default — "semantic repair on"): soft issues
      (out-of-range numerics, unknown keys, contradictory diagnosis labels,
      wrong reset flag) are auto-corrected with a warning AND recorded in
      the returned ``corrections`` dict. This is the safety-net behaviour the
      experiment ran with originally.

    * ``strict=True`` ("semantic repair off" — for the no-repair ablation):
      the validator still detects and *records* every would-be correction,
      but at the end it raises ``SemanticRepairRequired`` instead of
      accepting the silently-fixed proposal. The exception carries the
      corrections dict so the caller can attribute the failure to a specific
      violation type. Hard structural failures (e.g. invalid diagnosis enum,
      non-numeric value, all-unknown keys) raise ``ValueError`` in both modes.

    Returns a ``(parsed, corrections)`` tuple.
    """
    corrections = _empty_corrections()
    # â”€â”€ 1. Diagnosis enum (optional) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # The simplified system prompt no longer asks for a `diagnosis` field;
    # only validate it when the model actually provides one with a non-empty
    # primary_problem. Missing / "unknown" â†’ skip this check.
    diag = parsed.get("diagnosis", {})
    if isinstance(diag, dict):
        pp = diag.get("primary_problem")
        if pp and pp != "unknown" and pp not in ALLOWED_DIAGNOSES:
            raise ValueError(
                f"Invalid diagnosis '{pp}'. "
                f"Must be one of: {sorted(ALLOWED_DIAGNOSES)}"
            )

    # â”€â”€ 2. Allowed keys in proposed_changes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    changes: Dict[str, Any] = parsed.get("proposed_changes", {})
    unknown_keys = set(changes.keys()) - ALLOWED_HP_KEYS
    if unknown_keys:
        logger.warning("Stripping unknown HP keys from proposal: %s", unknown_keys)
        corrections["unknown_keys_stripped"] = sorted(unknown_keys)
        for k in unknown_keys:
            del changes[k]
        if not changes:
            raise ValueError(
                f"All proposed keys were invalid ({unknown_keys}). "
                f"Allowed keys: {sorted(ALLOWED_HP_KEYS)}"
            )

    # â”€â”€ 3. Value types and ranges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for key in list(changes.keys()):
        val = _parse_scalar(changes[key])
        changes[key] = val

        # 3a. Discrete choices (categorical or fixed-set params)
        if key in DISCRETE_HP_VALUES:
            allowed = DISCRETE_HP_VALUES[key]
            # For string params (optimizer_choice)
            if isinstance(next(iter(allowed)), str):
                if val not in allowed:
                    raise ValueError(
                        f"Invalid value for '{key}': '{val}'. "
                        f"Must be one of {sorted(allowed)}"
                    )
            else:
                # Numeric discrete (lstm_hidden, lstm_layers)
                try:
                    val = int(val)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Non-numeric value for '{key}': '{val}'"
                    )
                if val not in allowed:
                    # Snap to closest allowed value
                    closest = min(allowed, key=lambda x: abs(x - val))
                    logger.warning(
                        "Clamping %s=%d to nearest allowed value %d (allowed: %s)",
                        key, val, closest, sorted(allowed),
                    )
                    corrections["discrete_snapped"].append({
                        "key":     key,
                        "raw":     val,
                        "snapped": closest,
                        "allowed": sorted(allowed),
                    })
                    val = closest
                changes[key] = val
            continue

        # 3b. Integer coercion for non-discrete integer params
        if key in INTEGER_HP_KEYS:
            try:
                val = int(val)
            except (TypeError, ValueError):
                raise ValueError(f"Non-numeric value for '{key}': '{val}'")
            changes[key] = val

        # 3c. Numeric range clamping
        if key in HP_BOUNDS:
            try:
                val = float(changes[key])
            except (TypeError, ValueError):
                raise ValueError(f"Non-numeric value for '{key}': '{changes[key]}'")
            lo, hi = HP_BOUNDS[key]
            if val < lo or val > hi:
                clamped = max(lo, min(hi, val))
                logger.warning(
                    "Clamping %s=%.6g to [%.6g, %.6g] --> %.6g",
                    key, val, lo, hi, clamped,
                )
                corrections["clamped"].append({
                    "key":     key,
                    "raw":     float(val),
                    "clamped": float(clamped),
                    "lo":      float(lo),
                    "hi":      float(hi),
                })
                val = clamped
            # Preserve int type for integer params after clamping
            changes[key] = int(val) if key in INTEGER_HP_KEYS else val

    parsed["proposed_changes"] = changes

    # â”€â”€ 4. Diagnosis contradicts metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Soft thresholds â€” deliberately more lenient than the system prompt to
    # allow interpretive freedom while catching obviously wrong diagnoses.
    metrics = context.get("metrics", {})
    trends  = context.get("trends", {})
    tools   = context.get("tool_results", {})
    pp = parsed.get("diagnosis", {}).get("primary_problem", "unknown")
    loss_ratio = metrics.get("loss_ratio")
    val_loss   = metrics.get("val_loss")
    epochs_si  = trends.get("epochs_since_improvement")
    total_epochs = context.get("training_progress", {}).get("total_epochs", 0)
    baseline_ref = context.get("baseline_reference", {}) if isinstance(context.get("baseline_reference", {}), dict) else {}
    baseline_summary = baseline_ref.get("training_summary", {}) if isinstance(baseline_ref.get("training_summary", {}), dict) else {}
    current_has_metrics = is_finite_number(val_loss) and int(total_epochs or 0) > 0
    baseline_has_metrics = is_finite_number(baseline_summary.get("final_val_loss"))
    if not current_has_metrics and baseline_has_metrics:
        val_loss = baseline_summary.get("final_val_loss")
        loss_ratio = baseline_summary.get("final_loss_ratio")
        epochs_si = baseline_summary.get("epochs_since_improvement")
    curve = tools.get("training_curve_analysis", {}) if isinstance(tools, dict) else {}
    tool_pattern = curve.get("pattern")
    gap_widening_rate = curve.get("gap_widening_rate")
    val_slope = curve.get("val_slope")
    train_slope = curve.get("train_slope")
    tool_overfit_evidence = (
        tool_pattern == "overfitting_divergence"
        or (
            is_finite_number(gap_widening_rate)
            and float(gap_widening_rate) > 0
            and is_finite_number(val_slope)
            and float(val_slope) >= 0
            and is_finite_number(train_slope)
            and float(train_slope) < 0
        )
    )

    # Soft validation: instead of rejecting the whole proposal when the
    # diagnosis label contradicts the metrics, override the label and keep
    # the model's `proposed_changes`. This prevents tiny models from cascading
    # into "no_data" loops and wasting all rounds.
    def _infer_diagnosis() -> str:
        if not current_has_metrics and not baseline_has_metrics:
            return "no_data"
        if tool_overfit_evidence:
            return "overfitting"
        if is_finite_number(loss_ratio) and float(loss_ratio) >= 1.3:
            return "overfitting"
        if is_finite_number(val_loss) and float(val_loss) >= 0.5:
            return "underfitting"
        if is_finite_number(epochs_si) and int(epochs_si) >= 4:
            return "plateau"
        return "healthy"

    corrected_pp: Optional[str] = None
    correction_reason: Optional[str] = None
    if not current_has_metrics and not baseline_has_metrics and pp != "no_data":
        corrected_pp = "no_data"
        correction_reason = "no_current_or_baseline_metrics"
    elif not current_has_metrics and baseline_has_metrics and pp == "no_data":
        corrected_pp = _infer_diagnosis()
        correction_reason = "baseline_metrics_available_so_not_no_data"
    elif pp == "overfitting" and is_finite_number(loss_ratio) and float(loss_ratio) < 1.3 and not tool_overfit_evidence:
        corrected_pp = _infer_diagnosis()
        correction_reason = f"loss_ratio={float(loss_ratio):.3f}<1.3_no_tool_overfit_evidence"
    elif pp == "underfitting" and is_finite_number(val_loss) and float(val_loss) < 0.5:
        corrected_pp = _infer_diagnosis()
        correction_reason = f"val_loss={float(val_loss):.4f}<0.5_too_low_for_underfitting"
    elif pp == "plateau" and is_finite_number(epochs_si) and int(epochs_si) < 4:
        corrected_pp = _infer_diagnosis()
        correction_reason = f"epochs_since_improvement={int(epochs_si)}<4_too_short_for_plateau"

    if corrected_pp is not None and corrected_pp != pp:
        logger.warning(
            "Auto-correcting diagnosis '%s' --> '%s' (metrics: loss_ratio=%s, val_loss=%s, epochs_si=%s)",
            pp, corrected_pp, loss_ratio, val_loss, epochs_si,
        )
        corrections["diagnosis_auto_corrected"] = {
            "from":   pp,
            "to":     corrected_pp,
            "reason": correction_reason or "unspecified",
        }
        diag = parsed.setdefault("diagnosis", {})
        diag["primary_problem"] = corrected_pp
        pp = corrected_pp

    # Soft check: reasoning should reference at least one motion feature —
    # only when the motion feature is enabled (context carries diagnostics).
    # In --no-motion ablation runs, motion_diagnostics is {} so we skip the
    # warning entirely; the prompt no longer asks the LLM to cite motion.
    if context.get("motion_diagnostics"):
        motion_keywords = (
            "speed", "dwell", "stop", "motion",
        )
        reasoning_lc = str(parsed.get("reasoning", "")).lower()
        if reasoning_lc and not any(kw in reasoning_lc for kw in motion_keywords):
            logger.warning(
                "Proposal reasoning does not cite a motion feature: %r",
                parsed.get("reasoning", ""),
            )

    # â”€â”€ 5. resets_model flag consistency â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    has_arch_change = bool(ARCH_CHANGE_KEYS & set(changes.keys()))
    resets = parsed.get("resets_model", False)

    if has_arch_change and not resets:
        logger.warning(
            "Auto-correcting resets_model=False --> True (proposal contains arch params: %s)",
            ARCH_CHANGE_KEYS & set(changes.keys()),
        )
        corrections["resets_model_corrected"] = {
            "from":   False,
            "to":     True,
            "reason": f"arch_change_present:{sorted(ARCH_CHANGE_KEYS & set(changes.keys()))}",
        }
        parsed["resets_model"] = True
    elif not has_arch_change and resets:
        # window_size is handled separately in the loop; if it's not in changes
        # either, then resets_model=True is wrong.
        if "window_size" not in changes:
            logger.warning(
                "Auto-correcting resets_model=True --> False "
                "(no architecture or window_size change in proposal)"
            )
            corrections["resets_model_corrected"] = {
                "from":   True,
                "to":     False,
                "reason": "no_arch_or_window_size_change",
            }
            parsed["resets_model"] = False

    corrections["any_correction"] = bool(
        corrections["unknown_keys_stripped"]
        or corrections["clamped"]
        or corrections["discrete_snapped"]
        or corrections["diagnosis_auto_corrected"]
        or corrections["resets_model_corrected"]
    )

    # â”€â”€ 6. Multi-objective Pareto cost weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Coerce the LLM's (w_lat, w_stab, w_res) into a clean vector summing to 1.
    # Missing / malformed entries fall back to a uniform 1/3 share. This is a
    # benign normalisation, not a semantic repair, so it never trips strict
    # mode or the corrections counters.
    parsed["pareto_weights"] = pareto.normalise_weights(parsed.get("pareto_weights"))

    # Strict ("no semantic repair") mode: the validator has detected and
    # mutated the proposal to make it legal, but in this ablation arm we
    # want to characterise the *raw* LLM as an optimizer — every silent fix
    # is reported as a constraint violation, the round is rejected, and the
    # retry / skip logic kicks in. The corrections dict travels with the
    # exception so the caller can record what would have been fixed.
    if strict and corrections["any_correction"]:
        raise SemanticRepairRequired(corrections)

    return parsed, corrections


# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------



def _format_payload_as_text(payload: Dict[str, Any]) -> str:
    """Convert the context payload into a flat, labelled text block.

    Many smaller LLMs struggle with deeply nested JSON input.  A
    human-readable text format with clear section headers is easier
    to parse and leads to more accurate responses.
    """
    lines: List[str] = []

    def _v(v, decimals: int = 4) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)

    # â”€â”€ Current Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    m = payload.get("metrics", {})
    lines.append("â”â” CURRENT METRICS â”â”")
    lines.append(f"  val_loss:                   {_v(m.get('val_loss'))}")
    lines.append(f"  train_loss:                 {_v(m.get('train_loss'))}")
    lines.append(f"  val_mae:                    {_v(m.get('val_mae'))}")
    lines.append(f"  loss_ratio:                 {_v(m.get('loss_ratio'))}")
    lines.append(f"  mean_euclidean_distance_m:  {_v(m.get('mean_euclidean_distance_m'))}")
    lines.append("")

    # â”€â”€ Round Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rs = payload.get("round_summary", {})
    lines.append("â”â” ROUND SUMMARY â”â”")
    for k, v in rs.items():
        lines.append(f"  {k}: {_v(v)}")
    lines.append("")

    # â”€â”€ Trends â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    t = payload.get("trends", {})
    lines.append("â”â” TRENDS â”â”")
    for k, v in t.items():
        lines.append(f"  {k}: {_v(v)}")
    lines.append("")

    # â”€â”€ Training Progress â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tp = payload.get("training_progress", {})
    lines.append("â”â” TRAINING PROGRESS â”â”")
    for k, v in tp.items():
        lines.append(f"  {k}: {_v(v)}")
    lines.append("")

    # â”€â”€ Current Hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    hp = payload.get("current_hyperparameters", {})
    lines.append("â”â” CURRENT HYPERPARAMETERS â”â”")
    for k, v in hp.items():
        lines.append(f"  {k}: {_v(v)}")
    lines.append("")

    baseline = payload.get("baseline_reference", {}) or {}
    if baseline:
        lines.append("â”â” COMPLETED BASELINE REFERENCE â”â”")
        desc = baseline.get("description")
        if desc:
            lines.append(f"  description: {desc}")
        b_hp = baseline.get("hyperparameters", {}) or {}
        if b_hp:
            lines.append("  [baseline_hyperparameters]")
            for k, v in b_hp.items():
                lines.append(f"    {k}: {_v(v)}")
        summary = baseline.get("training_summary", {}) or {}
        if summary:
            lines.append("  [training_summary]")
            for k, v in summary.items():
                lines.append(f"    {k}: {_v(v)}")
        test_metrics = baseline.get("test_metrics", {}) or {}
        if test_metrics:
            lines.append("  [test_metrics]")
            for k, v in test_metrics.items():
                lines.append(f"    {k}: {_v(v)}")
        lines.append("")

    # â”€â”€ Motion Diagnostics (per-bin error breakdown) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mp = payload.get("motion_profile", {}) or {}
    if mp:
        lines.append("== MOTION PROFILE (dataset trajectory speed + dwell) ==")
        lines.append(f"  speed_mean_mps:     {_v(mp.get('speed_mean_mps'))}")
        lines.append(f"  speed_std_mps:      {_v(mp.get('speed_std_mps'))}")
        lines.append(f"  speed_median_mps:   {_v(mp.get('speed_median_mps'))}")
        lines.append(f"  speed_iqr_mps:      {_v(mp.get('speed_iqr_mps'))}")
        lines.append(f"  speed_p95_mps:      {_v(mp.get('speed_p95_mps'))}")
        lines.append(f"  speed_min_mps:      {_v(mp.get('speed_min_mps'))}")
        lines.append(f"  speed_max_mps:      {_v(mp.get('speed_max_mps'))}")
        dwell = mp.get("dwell", {}) or {}
        if dwell:
            lines.append(f"  dwell.stop_share:       {_v(dwell.get('stop_share'))}")
            lines.append(f"  dwell.n_episodes:       {_v(dwell.get('n_dwell_episodes'))}")
            lines.append(f"  dwell.stop_go_trans:    {_v(dwell.get('n_stop_go_transitions'))}")
            lines.append(f"  dwell.dwell_s_mean:     {_v(dwell.get('dwell_s_mean'))}")
            lines.append(f"  dwell.dwell_s_p95:      {_v(dwell.get('dwell_s_p95'))}")
            lines.append(f"  dwell.episodes_per_min: {_v(dwell.get('episodes_per_min'))}")
        lines.append("")

    md = payload.get("motion_diagnostics", {}) or {}
    if md:
        lines.append("â”â” MOTION DIAGNOSTICS (val set) â”â”")
        lines.append(f"  overall_rmse:        {_v(md.get('overall_rmse'))}")
        lines.append(f"  overall_mean_euclid: {_v(md.get('overall_mean_euclid'))}")
        lines.append("")

    # â”€â”€ Tool Results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tr = payload.get("tool_results", {})
    if tr:
        lines.append("â”â” DIAGNOSTIC TOOLS â”â”")
        for tool_name, result in tr.items():
            lines.append(f"  [{tool_name}]")
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                parts = [f"{ik}={_v(iv)}" for ik, iv in item.items()
                                         if not isinstance(iv, dict)]
                                lines.append(f"    {k}[{i}]: {', '.join(parts)}")
                            else:
                                lines.append(f"    {k}[{i}]: {_v(item)}")
                    elif isinstance(v, dict):
                        for sk, sv in v.items():
                            lines.append(f"    {k}.{sk}: {_v(sv)}")
                    else:
                        lines.append(f"    {k}: {_v(v)}")
            lines.append("")

    # Compact structured memory from previous LLM decisions.
    memory = payload.get("optimization_history", {})
    lines.append("OPTIMIZATION MEMORY")

    def _history_line(entry: Dict[str, Any]) -> str:
        r = entry.get("round", "?")
        changes = entry.get("changes_applied", {})
        change_str = (
            ", ".join(f"{ck}={_v(cv)}" for ck, cv in changes.items())
            if changes else "none"
        )
        pareto = entry.get("pareto") or {}
        cost_str = ""
        if pareto:
            cost_str = (
                f" latency_ms={_v(pareto.get('latency_ms'))}"
                f" stability_std_m={_v(pareto.get('stability_std_m'))}"
                f" params={_v(pareto.get('params_trainable'))}"
                f" pareto_score={_v(entry.get('pareto_score'))}"
            )
        weights = entry.get("pareto_weights") or {}
        w_str = ""
        if weights:
            w_str = (
                f" weights=[w_lat={_v(weights.get('w_lat'))},"
                f"w_stab={_v(weights.get('w_stab'))},"
                f"w_res={_v(weights.get('w_res'))}]"
            )
        return (
            f"Round {r}: diagnosis={entry.get('diagnosis', '?')}({entry.get('severity', '?')}) "
            f"strategy={entry.get('strategy', '?')} confidence={entry.get('confidence', '?')} "
            f"best_val={_v(entry.get('best_val_loss'))} avg_val={_v(entry.get('avg_val_loss'))} "
            f"best_rmse={_v(entry.get('best_rmse'))} avg_rmse={_v(entry.get('avg_rmse'))}"
            f"{cost_str}{w_str}"
            f" outcome={entry.get('outcome', '?')} changes=[{change_str}]"
        )

    if isinstance(memory, dict) and (memory.get("best_so_far") or memory.get("recent_rounds")):
        best = memory.get("best_so_far")
        if best:
            lines.append("  [best_so_far]")
            lines.append(f"  {_history_line(best)}")
        recent = memory.get("recent_rounds") or []
        if recent:
            omitted = int(memory.get("older_rounds_omitted") or 0)
            lines.append(f"  [recent_rounds; older_omitted={omitted}]")
            for entry in recent:
                lines.append(f"  {_history_line(entry)}")
    else:
        lines.append("  (no prior rounds)")
    lines.append("")

    lines.append("Respond ONLY in the compact line format from the system prompt. One field per line. No JSON, no markdown, no prose.")
    return "\n".join(lines)


# ===========================================================================
# PROTOCOL PATH (Step 6) — qualitative history-context + prompt
# ===========================================================================
# The from-scratch protocol replaces the warm-loop "current training curve"
# context with a history of TRIED SETTINGS and their averaged, QUALITATIVE
# scores. The LLM (and the rule-based controller) propose a small `changes:`
# delta relative to the best-so-far setting (the ANCHOR). Pareto and motion
# levers are deferred; the only objective is mean validation RMSE.

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


# ── Qualitative label helpers (numbers -> words for the LLM prompt) ─────────
# Raw numbers stay in the logs/report; the prompt sees only these labels.

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


def _format_search_space_text(allow_arch_changes: bool = True) -> str:
    order = [
        "learning_rate", "weight_decay", "dropout", "batch_size",
        "lstm_hidden", "lstm_layers", "window_size", "optimizer_choice", "patience",
    ]
    lines = []
    for k in order:
        if not allow_arch_changes and k in ("lstm_hidden", "lstm_layers"):
            continue
        lines.append(f"  - {k}: " + ", ".join(_fmt_grid_val(v) for v in HP_GRID[k]))
    return "\n".join(lines)


def protocol_system_prompt(allow_arch_changes: bool = True) -> str:
    """The from-scratch protocol prompt: tune conventional HPs from the
    qualitative history of tried settings, proposing a small delta vs the
    best-so-far anchor. No motion levers, no Pareto cost weights."""
    arch_note = (
        "" if allow_arch_changes
        else "\n(NOTE: lstm_hidden and lstm_layers are FIXED this run — do not propose them.)"
    )
    return f"""You are tuning an LSTM model for indoor localization of one person. The model receives windows of sensor features and predicts position. Your goal is to reduce validation error while avoiding unstable behavior. Do not repeat a setting that has already been tried.

You may change ONLY these hyperparameters, and ONLY to one of the listed allowed values:
{_format_search_space_text(allow_arch_changes)}{arch_note}

Each setting is trained 3 times (3 fixed seeds) and scored by its MEAN validation RMSE - lower is better. You are shown the best settings so far, the most recent attempts, every setting already tried, and observed patterns, all as qualitative summaries.

Propose a SMALL change relative to the best setting so far (shown as ANCHOR). Respond using EXACTLY these lines, one field per line, no prose, no markdown:

diagnosis: <healthy | possible overfitting tendency | possible underfitting tendency | plateau | unstable | inconclusive>
strategy: <one short phrase, e.g. increase regularization>
changes: <param=value, param=value>
reason: <one sentence>
confidence: <low | medium | high>
"""


def _parse_protocol_proposal(raw: str) -> Dict[str, Any]:
    """Parse the protocol output (diagnosis / strategy / changes / reason /
    confidence). Reuses `_parse_changes_text` for the delta (which already
    filters to allowed grid keys). Raises ValueError if no usable changes line
    and no recognisable diagnosis are present."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    fields: Dict[str, str] = {}
    direct_changes: Dict[str, Any] = {}
    for line in text.splitlines():
        clean = re.sub(r"^\s*[-*+]\s+", "", line.strip()).replace("**", "").strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\s*[:=]\s*(.+)$", clean)
        if not m:
            continue
        key = m.group(1).strip().lower().replace(" ", "_")
        val = m.group(2).strip()
        if key in ALLOWED_HP_KEYS:
            direct_changes[key] = _parse_scalar(val)
        elif key in ("diagnosis", "strategy", "changes", "reason", "reasoning", "confidence"):
            fields[key] = val

    changes = _parse_changes_text(fields.get("changes"))
    changes.update(direct_changes)
    return {
        "diagnosis": _normalize_diagnosis(fields.get("diagnosis")),
        "strategy":  (fields.get("strategy") or "exploit").strip().lower(),
        "proposed_changes": changes,
        "reason":    (fields.get("reason") or fields.get("reasoning") or "").strip(),
        "confidence": (fields.get("confidence") or "medium").strip().lower(),
    }


def _setting_line(setting: Dict[str, Any], allow_arch_changes: bool = True) -> str:
    order = [
        "learning_rate", "weight_decay", "dropout", "batch_size",
        "lstm_hidden", "lstm_layers", "window_size", "optimizer_choice", "patience",
    ]
    parts = []
    for k in order:
        if k in setting:
            parts.append(f"{k}={_fmt_grid_val(setting[k])}")
    return ", ".join(parts)


def format_protocol_payload(
    history: List[Dict[str, Any]],
    anchor_setting: Dict[str, Any],
    max_epochs: int,
    allow_arch_changes: bool = True,
) -> str:
    """Render the qualitative history context the protocol prompt consumes.

    `history` is the list of attempt records produced by the proposer driver
    (Step 6b): each has setting, score (mean val RMSE), val_rmse_std,
    mean_best_epoch, mean_val_loss, mean_train_val_gap, changes_from_anchor,
    diagnosis, output_status. All raw numbers are converted to qualitative
    labels here; the exact values stay in the logs.
    """
    all_scores = [h.get("score") for h in history]
    lines: List[str] = []

    # ── ANCHOR (best-so-far) ────────────────────────────────────────────────
    lines.append("== ANCHOR (best setting so far - propose changes relative to this) ==")
    lines.append("  " + (_setting_line(anchor_setting, allow_arch_changes) or "(fixed-reference defaults)"))
    lines.append("")

    def _level_str(h: Dict[str, Any]) -> str:
        lvl = _qual_level_10(h.get("score"), all_scores)
        return f"{lvl}/10" if lvl is not None else "unknown"

    def _qual_block(h: Dict[str, Any]) -> List[str]:
        q   = _qual_quality(h.get("score"), all_scores)
        var = _qual_variation(h.get("score"), h.get("val_rmse_std"))
        gap = _qual_gap(h.get("mean_val_loss"), h.get("mean_train_val_gap"))
        tim = _qual_epoch_timing(h.get("mean_best_epoch"), max_epochs)
        beh = _behavior_label(var, gap, q)
        return [
            f"    validation quality: {q} (level {_level_str(h)}, 1=best 10=worst)",
            f"    reliability across 3 trainings: {var}",
            f"    train/validation gap: {gap}",
            f"    best epoch timing: {tim}",
            f"    behavior label: {beh.replace('_', ' ')}",
        ]

    # ── Best settings so far (top 5, RANKED best → worst) ─────────────────────
    ranked = sorted(
        (h for h in history if is_finite_number(h.get("score"))),
        key=lambda h: float(h["score"]),
    )[:5]
    lines.append("== BEST SETTINGS SO FAR (ranked #1 = best; level 1/10 = best) ==")
    if ranked:
        for rank, h in enumerate(ranked, start=1):
            lines.append(f"  #{rank} [attempt {h.get('attempt')}] (level {_level_str(h)}) "
                         f"{_setting_line(h.get('setting', {}), allow_arch_changes)}")
            lines.extend(_qual_block(h))
    else:
        lines.append("  (none yet)")
    lines.append("")

    # ── Last 5 attempts, in CHRONOLOGICAL order so the LLM can read the trend ──
    # (oldest → newest). Each line carries the 1-10 quality level and an explicit
    # direction vs the immediately preceding attempt, so the search trajectory is
    # legible without raw numbers.
    lines.append("== LAST ATTEMPTS (oldest → newest; watch the level trend, 1=best 10=worst) ==")
    recent = history[-5:]
    if recent:
        prev_score = None
        for h in recent:
            chg = h.get("changes_from_anchor") or {}
            chg_str = ", ".join(f"{k}={_fmt_grid_val(v)}" for k, v in chg.items()) or "none"
            q = _qual_quality(h.get("score"), all_scores)
            trend = ""
            if is_finite_number(h.get("score")) and is_finite_number(prev_score):
                trend = ("  ↓ improved" if h["score"] < prev_score * 0.98
                         else "  ↑ worsened" if h["score"] > prev_score * 1.02
                         else "  → about the same")
            beh = _behavior_label(_qual_variation(h.get('score'), h.get('val_rmse_std')),
                                  _qual_gap(h.get('mean_val_loss'), h.get('mean_train_val_gap')), q)
            lines.append(f"  [attempt {h.get('attempt')}] level {_level_str(h)} ({q}){trend} | changed: {chg_str}")
            lines.append(f"    reliability: {_qual_variation(h.get('score'), h.get('val_rmse_std'))}"
                         f"  gap: {_qual_gap(h.get('mean_val_loss'), h.get('mean_train_val_gap'))}"
                         f"  best epoch: {_qual_epoch_timing(h.get('mean_best_epoch'), max_epochs)}")
            lines.append(f"    behavior label: {beh.replace('_', ' ')}")
            lines.append(f"    output: {h.get('output_status', 'clean')}")
            if is_finite_number(h.get("score")):
                prev_score = h["score"]
    else:
        lines.append("  (no attempts yet)")
    lines.append("")

    # ── All settings already tried (avoid repeats) ───────────────────────────
    lines.append("== ALREADY TRIED (do NOT repeat) ==")
    if history:
        for h in history:
            lines.append(f"  #{h.get('attempt')}: {_setting_line(h.get('setting', {}), allow_arch_changes)}")
    else:
        lines.append("  (none)")
    lines.append("")

    # ── Observed patterns (auto-mined per-value quality) ─────────────────────
    lines.append("== OBSERVED PATTERNS ==")
    patterns = _mine_patterns(history, all_scores, allow_arch_changes)
    if patterns:
        lines.extend(f"  - {p}" for p in patterns)
    else:
        lines.append("  (not enough data yet)")
    lines.append("")

    lines.append("Respond ONLY in the protocol line format. One field per line. No prose, no markdown.")
    return "\n".join(lines)


def _mine_patterns(history: List[Dict[str, Any]], all_scores: List[float],
                   allow_arch_changes: bool = True) -> List[str]:
    """Cheap per-HP-value pattern miner: flag values that tended to give poor /
    unstable behavior, so the LLM avoids them without reading raw numbers."""
    if len(history) < 4:
        return []
    keys = ["learning_rate", "dropout", "weight_decay", "window_size", "optimizer_choice", "batch_size"]
    if allow_arch_changes:
        keys += ["lstm_hidden", "lstm_layers"]
    out: List[str] = []
    for k in keys:
        buckets: Dict[Any, List[str]] = {}
        for h in history:
            v = (h.get("setting") or {}).get(k)
            if v is None:
                continue
            q = _qual_quality(h.get("score"), all_scores)
            var = _qual_variation(h.get("score"), h.get("val_rmse_std"))
            label = "unstable" if var == "high" else q
            buckets.setdefault(_fmt_grid_val(v), []).append(label)
        for val, labels in buckets.items():
            if len(labels) < 2:
                continue
            if all(l in ("poor", "unstable") for l in labels):
                bad = "unstable" if any(l == "unstable" for l in labels) else "poor"
                out.append(f"{k}={val} tended to be {bad} ({len(labels)} attempts)")
            elif all(l in ("best", "good") for l in labels):
                out.append(f"{k}={val} tended to be strong ({len(labels)} attempts)")
    return out[:8]


def _value_in_grid(key: str, val: Any) -> bool:
    """Hard membership test against HP_GRID (the deferred Step-1 check, landed
    here for the protocol path). Categorical exact; numeric within tolerance."""
    grid = HP_GRID.get(key)
    if grid is None:
        return False
    if key == "optimizer_choice":
        return val in grid
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return False
    return any(abs(fv - float(g)) <= 1e-9 + 1e-6 * abs(float(g)) for g in grid)


def _snap_to_grid(key: str, val: Any) -> Any:
    """Return the grid value the (possibly typed) val corresponds to, so the
    stored setting uses canonical grid values (e.g. int batch_size)."""
    grid = HP_GRID.get(key, [])
    if key == "optimizer_choice":
        return val
    fv = float(val)
    return min(grid, key=lambda g: abs(float(g) - fv))


def validate_protocol_changes(
    changes: Dict[str, Any],
    anchor: Dict[str, Any],
    allow_arch_changes: bool,
    is_tried,
) -> tuple:
    """Hard-validate a proposed delta against the grid and the already-tried set.

    Returns (resolved_setting, ok, reason). `reason` is "" on success, else a
    short machine tag the retry feedback is built from. No silent repair: an
    out-of-grid value, an unknown key, an arch change while frozen, or a repeat
    all fail (the protocol's hard-validation main run).
    """
    if not isinstance(changes, dict):
        return None, False, "no_changes_parsed"
    for k, v in changes.items():
        if k not in ALLOWED_HP_KEYS:
            return None, False, f"unknown_param:{k}"
        if not allow_arch_changes and k in ARCH_CHANGE_KEYS:
            return None, False, f"arch_frozen:{k}"
        if not _value_in_grid(k, v):
            return None, False, f"value_not_in_grid:{k}={v}"
    resolved = {**anchor, **{k: _snap_to_grid(k, v) for k, v in changes.items()}}
    if is_tried(resolved):
        return None, False, "already_tried"
    return resolved, True, ""


def _human_reason(reason: str) -> str:
    """Turn a validate_protocol_changes tag into one line of LLM feedback."""
    if reason.startswith("unknown_param:"):
        return f"'{reason.split(':',1)[1]}' is not a tunable parameter."
    if reason.startswith("arch_frozen:"):
        return f"{reason.split(':',1)[1]} is fixed this run; do not change it."
    if reason.startswith("value_not_in_grid:"):
        return f"{reason.split(':',1)[1]} is not an allowed value; pick one from the list."
    if reason == "already_tried":
        return "that setting was already tried; propose a different change."
    if reason == "no_changes_parsed":
        return "no valid 'changes:' line was found."
    return reason


def _grid_neighbor(key: str, current: Any, direction: int) -> Any:
    """Step one position up (+1) or down (-1) the sorted grid for `key`.
    Returns the current value (no-op) if already at the end."""
    grid = HP_GRID.get(key, [])
    if not grid or key == "optimizer_choice":
        return current
    ordered = sorted(grid, key=float)
    # Find the closest index to current.
    try:
        cf = float(current)
        idx = min(range(len(ordered)), key=lambda i: abs(float(ordered[i]) - cf))
    except (TypeError, ValueError):
        idx = 0
    new_idx = max(0, min(len(ordered) - 1, idx + direction))
    return ordered[new_idx]


# ---------------------------------------------------------------------------
# OptimizerTools
# ---------------------------------------------------------------------------

class OptimizerTools:
    """Lightweight diagnostics used by the optimization loop."""

    def __init__(self, trainer: Trainer, config: Config):
        self.trainer = trainer
        self.config  = config
        self.optimization_history: List[Dict[str, Any]] = []

    def analyze_training_curve(self, window_size: int = 20) -> Dict[str, Any]:
        val_losses   = self.trainer.history.get("val_loss", [])
        train_losses = self.trainer.history.get("train_loss", [])
        if len(val_losses) < 5 or len(train_losses) < 5:
            return {"status": "insufficient_data", "pattern": "unknown"}

        rv = val_losses[-window_size:]   if len(val_losses)   >= window_size else val_losses
        rt = train_losses[-window_size:] if len(train_losses) >= window_size else train_losses
        val_slope   = float(np.polyfit(range(len(rv)), rv, 1)[0])
        train_slope = float(np.polyfit(range(len(rt)), rt, 1)[0])
        n    = min(len(rv), len(rt))
        gaps = [v - t for v, t in zip(rv[-n:], rt[-n:])]
        gap_slope = float(np.polyfit(range(len(gaps)), gaps, 1)[0]) if len(gaps) > 1 else 0.0
        ratio = rv[-1] / rt[-1] if rt[-1] > 0 else 999.0

        # Scale-invariant thresholds: normalise slopes and gap_slope by the
        # current loss level so a model at loss=0.01 and one at loss=1.0 are
        # judged by the same *fractional* change standard, not absolute numbers.
        cur_val   = rv[-1] if rv[-1] > 0 else 1.0
        cur_train = rt[-1] if rt[-1] > 0 else 1.0
        rel_val_slope   = val_slope   / cur_val
        rel_train_slope = train_slope / cur_train
        rel_gap_slope   = gap_slope   / cur_val

        if abs(rel_val_slope) < 1e-3 and abs(rel_train_slope) < 1e-3:
            pattern = "plateau"
        elif val_slope > 0 and train_slope < 0 and rel_gap_slope > 0.01:
            pattern = "overfitting_divergence"
        elif rel_val_slope < -1e-3 and rel_train_slope < -1e-3:
            pattern = "healthy_convergence"
        else:
            pattern = "unknown"

        return {
            "status": "success",
            "pattern": pattern,
            "val_slope": val_slope,
            "train_slope": train_slope,
            "gap_widening_rate": gap_slope,
            "current_loss_ratio": float(ratio),
        }



    def estimate_gradient_norm(self) -> Dict[str, Any]:
        norms: List[float] = []
        for p in self.trainer.model.parameters():
            if p.grad is not None:
                n = p.grad.data.norm(2).item()
                if np.isfinite(n):
                    norms.append(float(n))
        if not norms:
            return {"status": "no_gradients", "avg_gradient_norm": 0.0}
        avg = float(np.mean(norms))
        if avg < 1e-6:
            status = "vanishing"
        elif avg > 100:
            status = "exploding"
        else:
            status = "healthy"
        return {"status": status, "avg_gradient_norm": avg}



    def get_learning_rate_health(self) -> Dict[str, Any]:
        tl = self.trainer.history.get("train_loss", [])
        if len(tl) < 10:
            return {"status": "insufficient_data", "assessment": "unknown"}
        changes    = np.diff(tl[-10:])
        avg_change = float(np.mean(changes))
        volatility = float(np.std(changes))

        # Scale-invariant thresholds: express avg_change and volatility as a
        # fraction of the current training loss so the same relative thresholds
        # apply whether the loss is 0.01 or 1.0.
        current_loss   = float(tl[-1]) if tl[-1] > 0 else 1.0
        rel_avg_change = avg_change / current_loss
        rel_volatility = volatility / current_loss

        if rel_avg_change > 0.01:                              # loss rising > 1% of current per epoch
            assessment = "too_high"
        elif abs(rel_avg_change) < 1e-4 and rel_volatility < 1e-3:  # barely moving
            assessment = "too_low"
        elif rel_volatility > 0.1:                             # swinging > 10% of current per epoch
            assessment = "unstable"
        else:
            assessment = "healthy"
        return {
            "status":                "success",
            "assessment":            assessment,
            "current_learning_rate": float(self.trainer.hyperparams.get("learning_rate", 0.001)),
            "avg_loss_change":       avg_change,
            "loss_volatility":       volatility,
            "rel_avg_loss_change":   rel_avg_change,
            "rel_loss_volatility":   rel_volatility,
        }

    def run_all_tools(self) -> Dict[str, Any]:
        return {
            "training_curve_analysis": self.analyze_training_curve(),
            "gradient_health":         self.estimate_gradient_norm(),
            "learning_rate_health":    self.get_learning_rate_health(),
        }

    def update_history(self, round_data: Dict[str, Any]) -> None:
        self.optimization_history.append(round_data)


# ---------------------------------------------------------------------------
# SingleAgentOptimizer
# ---------------------------------------------------------------------------

class SingleAgentOptimizer:


    @staticmethod
    def _build_system_prompt(model_arch: str = "LSTM", allow_arch_changes: bool = True,
                             enable_motion: bool = True) -> str:
        if enable_motion:
            return f"""You are a motion-aware hyperparameter optimization expert for a small {model_arch} indoor localization model.
Each round, use the training metrics, optimization history, and motion diagnostics
to propose hyperparameter changes that improve validation performance.

Available parameters:
  - learning_rate    : 1e-5 to 1e-2
  - weight_decay     : 0.0 to 1e-3
  - dropout          : 0.0 to 0.5
  - batch_size       : 16 to 256
  - optimizer_choice : "adam" | "adamw" | "sgd"
  - lstm_hidden      : 64 | 128 | 256  (RESETS model)
  - lstm_layers      : 1 | 2 | 3       (RESETS model)
  - window_size      : 10 to 50        (RESETS model; use sparingly)

Motion-aware loss-shaping levers (these reshape the TRAINING OBJECTIVE itself,
not the optimiser — this is where your knowledge of how humans move matters):
  - v_max            : 0.5 to 5.0   plausible top human speed in m/s; predicted speeds above it are penalised
  - lambda_vel       : 0.0 to 1.0   strength of that speed penalty (0 = off)
  - lambda_smooth    : 0.0 to 1.0   strength of a penalty on physically implausible acceleration / jerk (0 = off)
  - bin_weight_slow / bin_weight_medium / bin_weight_fast : 0.5 to 5.0 position-error weight per speed regime (1.0 = neutral)

Motion descriptors:
  The MOTION PROFILE section gives the dataset's real speed distribution
  (mean, std, median, IQR, p95, min/max in m/s) and dwell/stop-go behaviour.
  Use those concrete numbers to set the loss-shaping levers:
   - Set v_max just above speed_p95_mps (a small margin, about 1.1x):
     predicted speeds beyond the realistic top speed in this data are then
     penalised. Do not leave v_max at a generic default if the profile
     shows a different speed range.
   - Raise lambda_vel when predictions look noisy or speed_std_mps is low
     (smooth motion); raise lambda_smooth when the trajectory is smooth and
     dwell episodes are frequent; keep both low when motion is genuinely fast.
   - The three speed bins (slow / medium / fast) are terciles of the data's
     speed. Raise the weight of whichever regime you judge hardest to fit
     (e.g. upweight the fast regime when speed_std_mps is high).
   - Leave a lever neutral (lambda = 0, weight = 1.0) only when the motion
     evidence genuinely does not support changing it.

Multi-objective cost trade-off:
  Beyond validation accuracy, three deployment costs are tracked each round:
    - latency_ms       : forward-pass inference time per window
                         (lower = faster on-device inference)
    - stability_std_m  : std of per-sample position error in metres
                         (lower = more consistent predictions)
    - params_trainable : trainable parameter count (lower = smaller model)
  You must also output a weight for each cost, expressing how much it matters
  for THIS dataset's deployment context. Weights are >= 0 and should sum to 1:
    - w_lat  : weight on latency
    - w_stab : weight on prediction stability
    - w_res  : weight on resource cost (model size)
  Each round is scored as
    val_loss x (1 + w_lat*latency + w_stab*stability + w_res*params),
  with each cost min-max-normalised across rounds. Raise a weight when that
  cost matters more (e.g. w_res high for a memory-constrained sensor, w_lat
  high when fast response is critical); lower it when it barely matters.
  Justify the balance from the dataset's motion profile and deployment needs.

Optimisation rules:
  1. Never propose the current value of a parameter (that is a no-op).
  2. Respect all ranges strictly.

Diagnosis rules:
        plateau      : epochs_since_improvement > 8  AND  loss_ratio <= 1.2
        underfitting : val_loss > 2.0                AND  loss_ratio < 1.2
        overfitting  : loss_ratio > 1.5
        no_data      : metrics missing or invalid
        healthy      : otherwise

Respond using this compact line format ONLY:

diagnosis: <overfitting|underfitting|plateau|healthy|no_data>
severity: <low|medium|high>
situation: <one sentence>
changes: <param=value, param=value>
resets_model: <false|true>
strategy: <exploit|explore|regularise|stabilise>
confidence: <low|medium|high>
reasoning: <one sentence that must reference a motion feature>
expected_improvement: <one sentence>
w_lat: <0.0 to 1.0>
w_stab: <0.0 to 1.0>
w_res: <0.0 to 1.0>
"""
        # Motion feature DISABLED (ablation): identical prompt with all
        # motion-aware sections removed, so the LLM has no awareness of
        # motion regimes when proposing hyperparameter changes.
        return f"""You are a hyperparameter optimization expert for a small {model_arch} indoor localization model.
Each round, use the training metrics and optimization history to propose
hyperparameter changes that improve validation performance.

Available parameters:
  - learning_rate    : 1e-5 to 1e-2
  - weight_decay     : 0.0 to 1e-3
  - dropout          : 0.0 to 0.5
  - batch_size       : 16 to 256
  - optimizer_choice : "adam" | "adamw" | "sgd"
  - lstm_hidden      : 64 | 128 | 256  (RESETS model)
  - lstm_layers      : 1 | 2 | 3       (RESETS model)
  - window_size      : 10 to 50        (RESETS model; use sparingly)

Multi-objective cost trade-off:
  Beyond validation accuracy, three deployment costs are tracked each round:
    - latency_ms       : forward-pass inference time per window
                         (lower = faster on-device inference)
    - stability_std_m  : std of per-sample position error in metres
                         (lower = more consistent predictions)
    - params_trainable : trainable parameter count (lower = smaller model)
  You must also output a weight for each cost, expressing how much it matters
  for THIS dataset's deployment context. Weights are >= 0 and should sum to 1:
    - w_lat  : weight on latency
    - w_stab : weight on prediction stability
    - w_res  : weight on resource cost (model size)
  Each round is scored as
    val_loss x (1 + w_lat*latency + w_stab*stability + w_res*params),
  with each cost min-max-normalised across rounds. Raise a weight when that
  cost matters more (e.g. w_res high for a memory-constrained sensor, w_lat
  high when fast response is critical); lower it when it barely matters.

Optimisation rules:
  1. Never propose the current value of a parameter (that is a no-op).
  2. Respect all ranges strictly.

Diagnosis rules:
        plateau      : epochs_since_improvement > 8  AND  loss_ratio <= 1.2
        underfitting : val_loss > 2.0                AND  loss_ratio < 1.2
        overfitting  : loss_ratio > 1.5
        no_data      : metrics missing or invalid
        healthy      : otherwise

Respond using this compact line format ONLY:

diagnosis: <overfitting|underfitting|plateau|healthy|no_data>
severity: <low|medium|high>
situation: <one sentence>
changes: <param=value, param=value>
resets_model: <false|true>
strategy: <exploit|explore|regularise|stabilise>
confidence: <low|medium|high>
reasoning: <one sentence>
expected_improvement: <one sentence>
w_lat: <0.0 to 1.0>
w_stab: <0.0 to 1.0>
w_res: <0.0 to 1.0>
"""

    def __init__(self, model_name: str = "minimax-m2.5:cloud", max_retries: int = 1,
                 model_arch: str = "LSTM", allow_arch_changes: bool = True,
                 enable_motion: bool = True, semantic_repair: bool = False,
                 llm_timeout_s: float = 300.0):

        self.client = OllamaChatCompletionClient(
            model=model_name,
            temperature=0.2,   # protocol
            model_info=ModelInfo(
                vision=False,
                function_calling=False,
                json_output=False,
                family="unknown",
                structured_output=False,
            ),
        )
        self.max_retries       = max_retries
        self.allow_arch_changes = allow_arch_changes
        self.enable_motion     = enable_motion
        # When False, every validator correction (clamp, snap, diagnosis fix,
        # unknown-key strip, reset-flag flip) becomes a hard rejection — used
        # for the "raw LLM as optimizer" ablation arm.
        self.semantic_repair   = semantic_repair
        # Hard ceiling (seconds) on a single LLM generation. Reasoning models
        # (e.g. qwen3) can emit very long <think> traces or stall outright;
        # without a bound one hung generation freezes the whole experiment.
        # On timeout the attempt fails -> retry, then the round still trains
        # with current HPs (no proposal applied).
        self.llm_timeout_s     = float(llm_timeout_s)
        self._system_prompt    = self._build_system_prompt(
            model_arch, allow_arch_changes, enable_motion,
        )
        self.conversation_log: List[Dict[str, Any]] = []
        
        self.retry_stats: Dict[str, int] = {
            # Per-attempt counters
            "total_attempts":               0,
            "retries":                      0,
            "fallbacks":                    0,
            "parse_failures":               0,
            "validation_failures":          0,
            "empty_changes_rejections":     0,
            # Per-round outcome counters (one increment per round)
            "rounds_clean":                 0,   # accepted on first attempt with no validator corrections
            "rounds_corrected":             0,   # accepted (any attempt) with at least one validator correction
            "rounds_retry_succeeded":       0,   # accepted, but only after one or more retries
            "rounds_skipped":               0,   # exhausted retries -> no proposal applied
            # Per-correction-type counters (sum across all attempts)
            "clamps_count":                 0,
            "discrete_snaps_count":         0,
            "unknown_keys_count":           0,
            "diagnosis_corrections_count":  0,
            "resets_model_corrections_count": 0,
        }
       
        self.token_stats: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "calls_with_usage": 0,
        }
        # Per-round counter: strict-mode rejections (validator detected a
        # would-be repair and bounced the proposal back to the retry loop).
        self.retry_stats["strict_rejections"] = 0
        self.retry_stats["llm_timeouts"] = 0
        logger.info(
            "SingleAgentOptimizer initialised  model=%s  arch=%s  allow_arch_changes=%s  semantic_repair=%s",
            model_name, model_arch, allow_arch_changes, semantic_repair,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def suggest_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """DEPRECATED (warm-loop path). Superseded by `propose_setting`, which
        the protocol runner (`run_proposer_search`) uses. This method and its
        helpers (`_call_with_retry`, `_attempt`, `_log_round`,
        `_build_system_prompt`, and the module-level `_format_payload_as_text`
        / `_format_history` / `_validate_proposal` / `OptimizerTools`) are no
        longer called; kept temporarily for reference and removed in a separate
        pass after a full protocol run is confirmed green.
        """
        history             = context.get("optimization_history", [])
        current_hyperparams = context.get("current_hyperparameters", {})

        proposal = await self._call_with_retry(context, history, current_hyperparams)
        self._log_round(context, proposal)

        return proposal

    async def close(self):
        await self.client.close()

    def save_conversation_log(self, path: str = "conversation_log.json"):
        with open(path, "w") as f:
            json.dump(self.conversation_log, f, indent=2, default=str)
        logger.info("Conversation log saved -> %s  (%d rounds)", path, len(self.conversation_log))

    # ------------------------------------------------------------------
    # Protocol path (Step 6): propose a small delta vs the best-so-far anchor
    # ------------------------------------------------------------------

    async def _protocol_raw_call(self, system_prompt: str, user_text: str):
        """One LLM generation with timeout + token accounting. Returns
        (raw_text, error). error is None on success."""
        msgs = [SystemMessage(content=system_prompt),
                UserMessage(content=user_text, source="user")]
        try:
            response = await asyncio.wait_for(self.client.create(msgs), timeout=self.llm_timeout_s)
        except asyncio.TimeoutError:
            self.retry_stats["llm_timeouts"] += 1
            return None, f"llm_call_timeout after {self.llm_timeout_s:.0f}s"
        except Exception as exc:  # noqa: BLE001 - network/client error
            return None, f"client_error:{exc}"
        try:
            self._record_token_usage(response)
        except Exception:
            pass
        return response.content.strip(), None

    async def propose_setting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Protocol entry point: read the qualitative history, propose a small
        change relative to the ANCHOR (best-so-far), hard-validate it against
        the grid + already-tried set, retry ONCE on failure, else mark the
        attempt rejected (no silent repair). Returns a proposal dict the
        proposer driver consumes."""
        for k in ("valid_first_try", "valid_after_retry", "rejected",
                  "repeats_proposed", "invalid_values"):
            self.retry_stats.setdefault(k, 0)
        if not hasattr(self, "_protocol_prompt"):
            self._protocol_prompt = protocol_system_prompt(self.allow_arch_changes)

        anchor      = context.get("anchor_setting") or {}
        is_tried    = context.get("is_tried") or (lambda s: False)
        allow_arch  = context.get("allow_arch_changes", self.allow_arch_changes)
        max_epochs  = int(context.get("max_epochs", 100))
        history     = context.get("history", [])

        base_user = format_protocol_payload(history, anchor, max_epochs, allow_arch)
        feedback = ""
        last_reason = "unknown"
        last_parsed: Dict[str, Any] = {}
        raw_last = ""

        for attempt in range(self.max_retries + 1):
            user_text = base_user if not feedback else (
                base_user + f"\n\nYOUR PREVIOUS REPLY WAS REJECTED: {feedback}\n"
                "Reply again, in the exact format, with a different valid change."
            )
            raw, err = await self._protocol_raw_call(self._protocol_prompt, user_text)
            if err:
                last_reason = err
                feedback = ("the previous call did not return a usable reply "
                            f"({err}); reply concisely in the exact format.")
                continue
            raw_last = raw
            try:
                parsed = _parse_protocol_proposal(raw)
            except Exception as exc:  # parse failure
                self.retry_stats["parse_failures"] += 1
                last_reason = f"parse_error:{exc}"
                feedback = "your reply could not be parsed; output exactly the 5 fields, one per line."
                continue
            last_parsed = parsed
            resolved, ok, reason = validate_protocol_changes(
                parsed.get("proposed_changes", {}), anchor, allow_arch, is_tried,
            )
            if ok:
                status = "clean" if attempt == 0 else "accepted_after_retry"
                if attempt == 0:
                    self.retry_stats["valid_first_try"] += 1
                else:
                    self.retry_stats["valid_after_retry"] += 1
                return {
                    "valid":            True,
                    "resolved_setting": resolved,
                    "proposed_changes": parsed.get("proposed_changes", {}),
                    "diagnosis":        parsed.get("diagnosis", "inconclusive"),
                    "strategy":         parsed.get("strategy", ""),
                    "reason":           parsed.get("reason", ""),
                    "confidence":       parsed.get("confidence", "medium"),
                    "output_status":    status,
                    "raw":              raw,
                }
            # invalid -> record reason, set feedback, retry
            if reason == "already_tried":
                self.retry_stats["repeats_proposed"] += 1
            elif reason.startswith("value_not_in_grid") or reason.startswith("unknown_param"):
                self.retry_stats["invalid_values"] += 1
            last_reason = reason
            feedback = _human_reason(reason)

        self.retry_stats["rejected"] += 1
        logger.warning("LLM proposal rejected after %d attempt(s): %s",
                       self.max_retries + 1, last_reason)
        return {
            "valid":          False,
            "output_status":  "rejected",
            "proposed_changes": last_parsed.get("proposed_changes", {}),
            "diagnosis":      last_parsed.get("diagnosis", "inconclusive"),
            "failure_reason": last_reason,
            "raw":            raw_last,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self,
        context:             Dict[str, Any],
        history:             List[Dict],
        current_hyperparams: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attempt LLM call; retry on parse failure; mark round as failed if all retries exhausted."""
        failure_reasons: List[str] = []
        attempts_log: List[Dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            self.retry_stats["total_attempts"] += 1
            repair_mode = attempt > 0
            result, err, attempt_record = await self._attempt(
                context, history, current_hyperparams, repair_mode, attempt_number=attempt,
            )
            attempts_log.append(attempt_record)
            if result is not None:
                if attempt > 0:
                    logger.info("SingleAgent retry %d succeeded", attempt)
                    self.retry_stats["rounds_retry_succeeded"] += 1
                # Tag final source so analysis can separate clean LLM output
                # from validator-repaired output without re-deriving it.
                had_corrections = bool(
                    (result.get("_corrections") or {}).get("any_correction")
                )
                if attempt == 0 and not had_corrections:
                    final_source = "llm_clean"
                    self.retry_stats["rounds_clean"] += 1
                elif attempt == 0 and had_corrections:
                    final_source = "llm_corrected"
                    self.retry_stats["rounds_corrected"] += 1
                elif attempt > 0 and not had_corrections:
                    final_source = "retry_clean"
                elif attempt > 0 and had_corrections:
                    final_source = "retry_corrected"
                    self.retry_stats["rounds_corrected"] += 1
                else:
                    final_source = "unknown"
                result["_final_source"] = final_source
                result["_attempts"] = attempts_log
                result["_attempt_count"] = attempt + 1
                return result
            if err:
                failure_reasons.append(err)
            if attempt < self.max_retries:
                self.retry_stats["retries"] += 1
                logger.warning("SingleAgent retry %d/%d...", attempt + 1, self.max_retries)

        self.retry_stats["fallbacks"] += 1
        self.retry_stats["rounds_skipped"] += 1
        reason = " | ".join(failure_reasons) if failure_reasons else "unknown_parse_or_validation_error"
        logger.error("SingleAgent LLM failed after %d retries - skipping round. Reason: %s", self.max_retries, reason)
        return {
            "failed":         True,
            "failure_reason": reason,
            "reasoning":      "",
            "_attempts":      attempts_log,
            "_attempt_count": len(attempts_log),
            "_final_source":  "skipped",
        }

    async def _attempt(
        self,
        context:             Dict[str, Any],
        history:             List[Dict],
        current_hyperparams: Dict[str, Any],
        repair_mode:         bool = False,
        attempt_number:      int  = 0,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        """Single LLM call. Returns (result, error, attempt_record).

        `attempt_record` is always populated and captures the raw LLM output,
        the parsed-but-not-yet-validated proposal, the validator's corrections,
        and the outcome of this attempt. The retry loop accumulates these into
        the final proposal so downstream analysis can distinguish clean LLM
        output from validator-repaired output and count failure modes.
        """
        attempt_record: Dict[str, Any] = {
            "attempt_number": attempt_number,
            "repair_mode":    repair_mode,
            "raw_output":     None,
            "raw_parsed":     None,
            "corrections":    None,
            "error":          None,
            "outcome":        "unknown",
        }
        payload = {
            "metrics":                context.get("metrics", {}),
            "round_summary":          context.get("round_summary", {}),
            "trends":                 context.get("trends", {}),
            "training_progress":      context.get("training_progress", {}),
            "tool_results":           context.get("tool_results", {}),
            "motion_diagnostics":     context.get("motion_diagnostics", {}),
            "motion_profile":         context.get("motion_profile", {}),
            "baseline_reference":     context.get("baseline_reference", {}),

            "current_hyperparameters": current_hyperparams,
            "optimization_history":   _format_history(history),
        }

        prompt = self._system_prompt
        if repair_mode:
            reasoning_hint = (
                "reasoning: <one sentence citing a motion feature>\n"
                if self.enable_motion
                else "reasoning: <one sentence>\n"
            )
            prompt += (
                "\n\n----------------------------------------\n"
                "FORMAT REPAIR: your previous reply could not be parsed.\n"
                "Reply with EXACTLY these 12 lines, one field per line, no prose,\n"
                "no markdown, no parentheses inside the diagnosis value:\n\n"
                "diagnosis: healthy\n"
                "severity: low\n"
                "situation: <one sentence>\n"
                "changes: learning_rate=0.0008, dropout=0.25\n"
                "resets_model: false\n"
                "strategy: exploit\n"
                "confidence: medium\n"
                + reasoning_hint +
                "expected_improvement: <one sentence>\n"
                "w_lat: 0.33\n"
                "w_stab: 0.34\n"
                "w_res: 0.33\n"
                "----------------------------------------"
            )

        # Build the new user turn for this round.
        # Use a flat text format instead of raw JSON â€” easier for all LLMs to parse.
        user_msg = UserMessage(
            content=_format_payload_as_text(sanitize_for_json(payload)),
            source="user",
        )
        # Full message list: system + compact structured memory + current round.
        msgs = [SystemMessage(content=prompt), user_msg]

        response = None
        t0 = time.time()
        raw: Optional[str] = None
        try:
            try:
                response = await asyncio.wait_for(
                    self.client.create(msgs), timeout=self.llm_timeout_s,
                )
            except asyncio.TimeoutError:
                self.retry_stats["llm_timeouts"] += 1
                err = f"llm_call_timeout after {self.llm_timeout_s:.0f}s"
                logger.error("SingleAgent attempt failed: %s", err)
                attempt_record["outcome"] = "llm_timeout"
                attempt_record["error"]   = err
                return None, err, attempt_record
            elapsed  = time.time() - t0
            raw      = response.content.strip()
            attempt_record["raw_output"] = raw
            try:
                parsed = _parse_llm_proposal(raw)
            except Exception as parse_exc:
                self.retry_stats["parse_failures"] += 1
                attempt_record["outcome"] = "parse_error"
                attempt_record["error"]   = str(parse_exc)
                raise
            # Token cost - best effort: autogen returns a RequestUsage on the
            # response when the backend reports it. Ollama populates this for
            # most models but not all, so missing usage is non-fatal.
            call_tokens = self._record_token_usage(response)
            parsed["_token_usage"] = call_tokens

            # Validate required top-level keys.
            # Coerce null / [] / missing -> {} so a well-intentioned "no changes"
            # answer is not punished with a retry.
            pc = parsed.get("proposed_changes", {})
            if pc is None or (isinstance(pc, list) and len(pc) == 0):
                pc = {}
            if not isinstance(pc, dict):
                raise ValueError(
                    f"Invalid type for 'proposed_changes': {type(pc).__name__} (must be object)"
                )
            parsed["proposed_changes"] = pc

            parsed.setdefault("resets_model", False)
            parsed["agent_time_s"]   = elapsed
            parsed["_input_payload"] = payload          # full JSON sent to the LLM
            parsed["_raw_llm_output"] = raw             # exact text the LLM returned

            # Extract/normalise embedded diagnosis (gracefully)
            diag = parsed.get("diagnosis", {})
            if not isinstance(diag, dict):
                diag = {}
            parsed["diagnosis"] = {
                "primary_problem": diag.get("primary_problem", "unknown"),
                "severity":        diag.get("severity",        "unknown"),
                "situation":       diag.get("situation",       ""),
            }

            # Snapshot the parsed-but-not-yet-validated proposal so we can
            # later diff it against the post-validation version and attribute
            # the final accepted action to either the LLM or the validator.
            raw_parsed_snapshot = copy.deepcopy({
                "diagnosis":            parsed.get("diagnosis"),
                "proposed_changes":     parsed.get("proposed_changes"),
                "resets_model":         parsed.get("resets_model"),
                "strategy":             parsed.get("strategy"),
                "reasoning":            parsed.get("reasoning"),
                "confidence":           parsed.get("confidence"),
                "expected_improvement": parsed.get("expected_improvement"),
            })
            attempt_record["raw_parsed"] = raw_parsed_snapshot

            # Structural + semantic validation. In repair mode (default) the
            # validator silently fixes legalisable problems and returns the
            # corrections record. In strict mode it raises
            # SemanticRepairRequired on the same conditions so the round
            # retries / skips — that is the "raw LLM as optimizer" arm.
            strict_mode = not self.semantic_repair
            try:
                parsed, corrections = _validate_proposal(parsed, context, strict=strict_mode)
            except SemanticRepairRequired as repair_exc:
                self.retry_stats["validation_failures"] += 1
                self.retry_stats["strict_rejections"]   += 1
                # Even though we reject, count the per-type constraint
                # violations so the failure-taxonomy table is populated
                # identically in both modes.
                corr = repair_exc.corrections
                self.retry_stats["clamps_count"]         += len(corr["clamped"])
                self.retry_stats["discrete_snaps_count"] += len(corr["discrete_snapped"])
                self.retry_stats["unknown_keys_count"]   += len(corr["unknown_keys_stripped"])
                if corr["diagnosis_auto_corrected"]:
                    self.retry_stats["diagnosis_corrections_count"] += 1
                if corr["resets_model_corrected"]:
                    self.retry_stats["resets_model_corrections_count"] += 1
                attempt_record["outcome"]     = "strict_repair_rejected"
                attempt_record["error"]       = str(repair_exc)
                attempt_record["corrections"] = corr
                raise
            except Exception as val_exc:
                self.retry_stats["validation_failures"] += 1
                attempt_record["outcome"] = "validation_error"
                attempt_record["error"]   = str(val_exc)
                raise

            attempt_record["corrections"] = corrections
            self.retry_stats["clamps_count"]         += len(corrections["clamped"])
            self.retry_stats["discrete_snaps_count"] += len(corrections["discrete_snapped"])
            self.retry_stats["unknown_keys_count"]   += len(corrections["unknown_keys_stripped"])
            if corrections["diagnosis_auto_corrected"]:
                self.retry_stats["diagnosis_corrections_count"] += 1
            if corrections["resets_model_corrected"]:
                self.retry_stats["resets_model_corrections_count"] += 1
            parsed["_corrections"]  = corrections
            parsed["_attempt_number"] = attempt_number

            if not parsed.get("proposed_changes") and parsed["diagnosis"]["primary_problem"] not in ("no_data", "healthy"):
                self.retry_stats["empty_changes_rejections"] += 1
                attempt_record["outcome"] = "empty_changes_rejection"
                attempt_record["error"]   = "empty_proposed_changes"
                return None, "empty_proposed_changes", attempt_record

            attempt_record["outcome"] = "success"
            logger.info(
                "SingleAgent: diagnosis=%s(%s) strategy=%s changes=%s (%.2fs)",
                parsed["diagnosis"]["primary_problem"],
                parsed["diagnosis"]["severity"],
                parsed.get("strategy"),
                parsed.get("proposed_changes"),
                elapsed,
            )
            return parsed, None, attempt_record

        except Exception as exc:
            err = str(exc)
            logger.error("SingleAgent attempt failed: %s", err)
            if response is not None:
                try:
                    logger.error("Raw (first 500): %s", response.content[:500])
                except Exception:
                    pass
            if attempt_record["outcome"] == "unknown":
                # Network / client error before parse or after validation succeeded.
                attempt_record["outcome"] = "client_error"
            attempt_record["error"] = err
            if attempt_record["raw_output"] is None and raw is not None:
                attempt_record["raw_output"] = raw
            return None, err, attempt_record

    def _record_token_usage(self, response: Any) -> Dict[str, int]:
        """Extract prompt/completion token counts from the LLM response.

        Returns the per-call counts (zeroed when the backend didn't report
        usage), and updates the cumulative `self.token_stats`. Designed to
        survive any of the shapes autogen / Ollama may produce: a `usage`
        attribute (RequestUsage), a dict `usage`, or nothing at all.
        """
        self.token_stats["calls"] += 1
        prompt_t = completion_t = 0
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is not None:
            if hasattr(usage, "prompt_tokens"):
                prompt_t     = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_t = int(getattr(usage, "completion_tokens", 0) or 0)
            elif isinstance(usage, dict):
                prompt_t     = int(usage.get("prompt_tokens", 0) or 0)
                completion_t = int(usage.get("completion_tokens", 0) or 0)
            if prompt_t or completion_t:
                self.token_stats["calls_with_usage"] += 1
        total_t = prompt_t + completion_t
        self.token_stats["prompt_tokens"]     += prompt_t
        self.token_stats["completion_tokens"] += completion_t
        self.token_stats["total_tokens"]      += total_t
        return {
            "prompt_tokens":     prompt_t,
            "completion_tokens": completion_t,
            "total_tokens":      total_t,
        }

    def _log_round(self, context: Dict[str, Any], proposal: Dict[str, Any]):
        diag = proposal.get("diagnosis", {})
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round":     context.get("training_progress", {}).get("current_round"),
            # â”€â”€ What the LLM received â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "llm_input": {
                "system_prompt":          self._system_prompt,
                "user_payload":           proposal.get("_input_payload", {}),
            },
            # What the LLM returned (raw + parsed-before-validation + final)
            "llm_raw_output": proposal.get("_raw_llm_output", ""),
            "raw_parsed_proposal": (
                proposal.get("_attempts", [{}])[-1].get("raw_parsed")
                if proposal.get("_attempts") else None
            ),
            "diagnosis": {
                "primary_problem": diag.get("primary_problem", "unknown"),
                "severity":        diag.get("severity",        "unknown"),
                "situation":       diag.get("situation",       ""),
                "time_s":          proposal.get("agent_time_s", 0.0),
            },
            "proposal": {
                "proposed_changes":     proposal.get("proposed_changes"),
                "resets_model":         proposal.get("resets_model"),
                "strategy":             proposal.get("strategy"),
                "reasoning":            proposal.get("reasoning"),
                "confidence":           proposal.get("confidence"),
                "expected_improvement": proposal.get("expected_improvement"),
                "time_s":               proposal.get("agent_time_s", 0.0),
            },
            # Validator audit trail: what got silently fixed on the winning
            # attempt, how many attempts it took, and where the accepted
            # action came from (LLM clean / LLM with corrections / retry / skipped).
            "corrections":    proposal.get("_corrections"),
            "attempts":       proposal.get("_attempts", []),
            "attempt_count":  proposal.get("_attempt_count", 1),
            "final_source":   proposal.get("_final_source", "unknown"),
            "total_time_s": proposal.get("agent_time_s", 0.0),
            "token_usage":  proposal.get("_token_usage",
                                         {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "mode":         "single_agent",
        }
        self.conversation_log.append(entry)
        logger.info(
            "Round %s | SingleAgent: %s(%s) -> %s | %.2fs | tokens p=%d c=%d",
            entry["round"],
            diag.get("primary_problem", "?"),
            diag.get("severity", "?"),
            proposal.get("proposed_changes"),
            entry["total_time_s"],
            entry["token_usage"].get("prompt_tokens", 0),
            entry["token_usage"].get("completion_tokens", 0),
        )


# ---------------------------------------------------------------------------
# RuleBasedOptimizer
# ---------------------------------------------------------------------------

class RuleBasedOptimizer:
    """Deterministic, non-LLM hyperparameter controller.

    Uses the **exact same diagnosis thresholds** as the LLM system prompt so
    that both controllers are operating under identical rules â€” the only
    difference is *how* those rules are acted upon (deterministic table look-up
    vs natural-language reasoning).

    Diagnosis rules (mirrored from the system prompt):
        plateau      : epochs_since_improvement > 8  AND  loss_ratio <= 1.2
        underfitting : val_loss > 2.0                AND  loss_ratio < 1.2
        overfitting  : loss_ratio > 1.5
        no_data      : metrics missing or invalid
        healthy      : otherwise

    Action rules (at most 2 HPs changed per round, no model resets):
        plateau      -> halve learning_rate (floor 1e-5);
                        if already at floor, nudge dropout +0.05 (max 0.5)
        underfitting -> double learning_rate (ceil 1e-2);
                        if dropout > 0.1, reduce it by -0.05
        overfitting  -> increase dropout +0.1 (max 0.5);
                        if learning_rate > 5e-4, reduce it Ã—0.7
        healthy      -> no changes
        no_data      -> no changes

    The class exposes the same async interface as SingleAgentOptimizer so
    ``llm_optimization_loop`` can use it unchanged as a drop-in replacement.
    """

    # â”€â”€ Thresholds (identical to system prompt) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _PLATEAU_PATIENCE   = 8      # epochs_since_improvement threshold
    _PLATEAU_RATIO_MAX  = 1.2    # loss_ratio upper bound for plateau
    _UNDERFIT_LOSS_MIN  = 2.0    # val_loss floor for underfitting
    _UNDERFIT_RATIO_MAX = 1.2    # loss_ratio upper bound for underfitting
    _OVERFIT_RATIO_MIN  = 1.5    # loss_ratio lower bound for overfitting

    # â”€â”€ HP bounds (same as HP_BOUNDS in model_pipeline.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _LR_MIN  = 1e-5
    _LR_MAX  = 1e-2
    _DO_MIN  = 0.0
    _DO_MAX  = 0.5
    _LR_OVERFIT_THRESHOLD = 5e-4   # only reduce LR when overfitting if above this

    # Log-mode tag written into each conversation_log entry. Subclasses
    # (e.g. MotionAwareRuleBasedOptimizer) override this.
    _MODE = "rule_based"

    def __init__(self, allow_arch_changes: bool = False):
        self.allow_arch_changes = allow_arch_changes
        self.conversation_log: List[Dict[str, Any]] = []
        logger.info("RuleBasedOptimizer initialised  allow_arch_changes=%s", allow_arch_changes)

    # ------------------------------------------------------------------
    # Public interface (mirrors SingleAgentOptimizer)
    # ------------------------------------------------------------------

    async def suggest_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """DEPRECATED (warm-loop path). Superseded by `propose_setting`. Kept
        for reference; not called by the protocol runner."""
        t0 = time.time()

        metrics  = context.get("metrics", {})
        trends   = context.get("trends", {})
        hp       = context.get("current_hyperparameters", {})

        diagnosis, severity, situation = self._diagnose(metrics, trends)
        proposed_changes, strategy     = self._act(diagnosis, hp)

        elapsed  = time.time() - t0
        proposal = {
            "proposed_changes": proposed_changes,
            "resets_model":     False,      # rule-based never rebuilds the model
            "diagnosis": {
                "primary_problem": diagnosis,
                "severity":        severity,
                "situation":       situation,
            },
            "strategy":             strategy,
            "reasoning":            self._reasoning(diagnosis, metrics, hp, proposed_changes),
            "confidence":           "high",   # deterministic â†’ always confident
            "expected_improvement": self._expected(diagnosis),
            "agent_time_s":         elapsed,
        }

        self._log_round(context, proposal)
        logger.info(
            "RuleBased: diagnosis=%s(%s) strategy=%s changes=%s (%.4fs)",
            diagnosis, severity, strategy, proposed_changes, elapsed,
        )
        return proposal

    async def close(self):
        """No-op â€” no network connection to close."""
        pass

    def save_conversation_log(self, path: str = "rule_based_log.json"):
        with open(path, "w") as f:
            json.dump(self.conversation_log, f, indent=2, default=str)
        logger.info("Rule-based log saved -> %s  (%d rounds)", path, len(self.conversation_log))

    # ------------------------------------------------------------------
    # Protocol path (Step 6): deterministic delta vs the best-so-far anchor
    # ------------------------------------------------------------------

    def _diagnose_protocol(self, history: List[Dict[str, Any]]) -> str:
        """Soft behavior label for the MOST RECENT attempt, from its averaged
        qualitative signals. Mirrors the labels the LLM prompt uses, so the
        only difference between the arms is how the label is acted on.
        Frozen rules — never tuned after seeing results."""
        if not history:
            return "inconclusive"
        last = history[-1]
        if not last.get("trained", True):
            return "inconclusive"
        all_scores = [h.get("score") for h in history]
        var = _qual_variation(last.get("score"), last.get("val_rmse_std"))
        gap = _qual_gap(last.get("mean_val_loss"), last.get("mean_train_val_gap"))
        q   = _qual_quality(last.get("score"), all_scores)
        return _behavior_label(var, gap, q)

    def _act_protocol(self, diagnosis: str, anchor: Dict[str, Any],
                      allow_arch_changes: bool) -> tuple:
        """Deterministic grid-stepped change vs the anchor (frozen rules):
          unstable / plateau            -> learning_rate one step DOWN
          possible overfitting tendency -> dropout one step UP (else weight_decay UP)
          possible underfitting tendency-> lstm_hidden one step UP (arch on) else lr UP
          healthy                       -> dropout one step UP (small local explore)
          inconclusive                  -> no change (train the anchor / defaults)
        Returns (changes, strategy)."""
        changes: Dict[str, Any] = {}
        lr = anchor.get("learning_rate")
        do = anchor.get("dropout")
        wd = anchor.get("weight_decay")

        if diagnosis in ("unstable", "plateau"):
            nv = _grid_neighbor("learning_rate", lr, -1)
            if nv != lr:
                changes["learning_rate"] = nv
            return changes, "stabilise"
        if diagnosis == "possible_overfitting_tendency":
            nv = _grid_neighbor("dropout", do, +1)
            if nv != do:
                changes["dropout"] = nv
            else:
                wv = _grid_neighbor("weight_decay", wd, +1)
                if wv != wd:
                    changes["weight_decay"] = wv
            return changes, "regularise"
        if diagnosis == "possible_underfitting_tendency":
            if allow_arch_changes:
                hv = _grid_neighbor("lstm_hidden", anchor.get("lstm_hidden"), +1)
                if hv != anchor.get("lstm_hidden"):
                    changes["lstm_hidden"] = hv
                    return changes, "explore"
            nv = _grid_neighbor("learning_rate", lr, +1)
            if nv != lr:
                changes["learning_rate"] = nv
            return changes, "explore"
        if diagnosis == "healthy":
            nv = _grid_neighbor("dropout", do, +1)
            if nv != do:
                changes["dropout"] = nv
            return changes, "exploit"
        return {}, "exploit"   # inconclusive

    async def propose_setting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic protocol proposer: diagnose the last attempt, step one
        grid value vs the anchor. Always returns a valid (in-grid) proposal."""
        anchor     = context.get("anchor_setting") or {}
        history    = context.get("history", [])
        allow_arch = context.get("allow_arch_changes", self.allow_arch_changes)

        diagnosis = self._diagnose_protocol(history)
        changes, strategy = self._act_protocol(diagnosis, anchor, allow_arch)
        resolved = {**anchor, **changes}
        return {
            "valid":            True,
            "resolved_setting": resolved,
            "proposed_changes": changes,
            "diagnosis":        diagnosis,
            "strategy":         strategy,
            "reason":           self._expected(diagnosis),
            "confidence":       "high",
            "output_status":    "clean",
        }

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def _diagnose(
        self,
        metrics: Dict[str, Any],
        trends:  Dict[str, Any],
    ) -> tuple[str, str, str]:
        """Return (label, severity, one-line situation string)."""
        val_loss   = metrics.get("val_loss")
        loss_ratio = metrics.get("loss_ratio")
        epochs_si  = trends.get("epochs_since_improvement")

        # Guard: missing or non-finite metrics
        if not is_finite_number(val_loss) or not is_finite_number(loss_ratio):
            return "no_data", "unknown", "Metrics are missing or invalid â€” no change this round."

        vl = float(val_loss)
        lr = float(loss_ratio)
        es = int(epochs_si) if is_finite_number(epochs_si) else 0

        # Exact thresholds from system prompt
        if es > self._PLATEAU_PATIENCE and lr <= self._PLATEAU_RATIO_MAX:
            severity = "high" if es > self._PLATEAU_PATIENCE * 2 else "medium"
            return (
                "plateau",
                severity,
                f"No improvement for {es} epochs with loss_ratio={lr:.3f}.",
            )

        if vl > self._UNDERFIT_LOSS_MIN and lr < self._UNDERFIT_RATIO_MAX:
            return (
                "underfitting",
                "high" if vl > self._UNDERFIT_LOSS_MIN * 2 else "medium",
                f"val_loss={vl:.4f} > {self._UNDERFIT_LOSS_MIN} with loss_ratio={lr:.3f}.",
            )

        if lr > self._OVERFIT_RATIO_MIN:
            severity = "high" if lr > self._OVERFIT_RATIO_MIN * 1.5 else "medium"
            return (
                "overfitting",
                severity,
                f"loss_ratio={lr:.3f} > {self._OVERFIT_RATIO_MIN} â€” val/train gap widening.",
            )

        return (
            "healthy",
            "low",
            f"val_loss={vl:.4f}  loss_ratio={lr:.3f} â€” training is on track.",
        )

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _act(
        self,
        diagnosis: str,
        hp:        Dict[str, Any],
    ) -> tuple[Dict[str, Any], str]:
        """Return (proposed_changes, strategy)."""
        lr = float(hp.get("learning_rate", 1e-3))
        do = float(hp.get("dropout",       0.1))
        changes: Dict[str, Any] = {}

        if diagnosis == "plateau":
            new_lr = max(lr * 0.5, self._LR_MIN)
            if lr > self._LR_MIN:
                changes["learning_rate"] = round(new_lr, 8)
            else:
                # LR already at floor â€” nudge dropout instead
                new_do = min(do + 0.05, self._DO_MAX)
                if new_do != do:
                    changes["dropout"] = round(new_do, 4)
            return changes, "exploit"

        if diagnosis == "underfitting":
            new_lr = min(lr * 2.0, self._LR_MAX)
            if new_lr != lr:
                changes["learning_rate"] = round(new_lr, 8)
            if do > 0.1:
                new_do = max(do - 0.05, self._DO_MIN)
                changes["dropout"] = round(new_do, 4)
            return changes, "explore"

        if diagnosis == "overfitting":
            new_do = min(do + 0.1, self._DO_MAX)
            if new_do != do:
                changes["dropout"] = round(new_do, 4)
            if lr > self._LR_OVERFIT_THRESHOLD:
                new_lr = round(lr * 0.7, 8)
                changes["learning_rate"] = new_lr
            return changes, "exploit"

        # healthy / no_data â†’ no changes
        return {}, "exploit"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reasoning(
        self,
        diagnosis:       str,
        metrics:         Dict[str, Any],
        hp:              Dict[str, Any],
        proposed_changes: Dict[str, Any],
    ) -> str:
        if not proposed_changes:
            return f"Diagnosis={diagnosis}. No changes warranted this round."
        parts = [f"Diagnosis={diagnosis}."]
        for k, v in proposed_changes.items():
            old = hp.get(k, "?")
            parts.append(f"{k}: {old} â†’ {v}")
        return "  ".join(parts)

    def _expected(self, diagnosis: str) -> str:
        return {
            "plateau":      "Reduced LR should help escape plateau.",
            "underfitting": "Higher LR should accelerate convergence.",
            "overfitting":  "Increased dropout should reduce generalisation gap.",
            "healthy":      "No change needed.",
            "no_data":      "Waiting for training data.",
        }.get(diagnosis, "Unknown.")

    def _log_round(self, context: Dict[str, Any], proposal: Dict[str, Any]):
        diag = proposal.get("diagnosis", {})
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round":     context.get("training_progress", {}).get("current_round"),
            "diagnosis": {
                "primary_problem": diag.get("primary_problem", "unknown"),
                "severity":        diag.get("severity",        "unknown"),
                "situation":       diag.get("situation",       ""),
                "time_s":          proposal.get("agent_time_s", 0.0),
            },
            "proposal": {
                "proposed_changes":     proposal.get("proposed_changes"),
                "resets_model":         proposal.get("resets_model"),
                "strategy":             proposal.get("strategy"),
                "reasoning":            proposal.get("reasoning"),
                "confidence":           proposal.get("confidence"),
                "expected_improvement": proposal.get("expected_improvement"),
                "time_s":               proposal.get("agent_time_s", 0.0),
            },
            "total_time_s": proposal.get("agent_time_s", 0.0),
            "mode":         self._MODE,
        }
        self.conversation_log.append(entry)


# ---------------------------------------------------------------------------
# MotionAwareRuleBasedOptimizer  (controller arm "C2")
# ---------------------------------------------------------------------------

class MotionAwareRuleBasedOptimizer(RuleBasedOptimizer):
    """Deterministic controller WITH motion-aware loss shaping (arm "C2").

    Identical to ``RuleBasedOptimizer`` for the conventional hyperparameters
    — same diagnosis thresholds, same metric-driven HP rules — but it ALSO
    sets the motion-aware loss-shaping levers (``v_max``, ``lambda_vel``,
    ``lambda_smooth``, the speed-bin weights) from FIXED heuristics on the
    dataset motion profile.

    This is the fair comparator for the LLM arm. It has the exact same levers
    and follows the same recipe the LLM prompt describes (``v_max`` = p95
    speed x 1.1), but applies it by deterministic formula rather than by
    reasoning. The thesis question "does the LLM's *interpretation* of motion
    beat a fixed motion heuristic" is answered by C3 (LLM) vs C2 (this class);
    "does motion loss-shaping help at all" by C2 vs C1 (plain RuleBased).
    """

    _MODE = "motion_rule"

    # Fixed loss-shaping heuristic constants.
    _V_MAX_MARGIN    = 1.1   # v_max = p95 speed * this margin (same as the LLM recipe)
    _LAMBDA_VEL      = 0.1   # gentle fixed velocity-plausibility weight
    _LAMBDA_SMOOTH   = 0.1   # gentle fixed smoothness weight
    _BIN_WEIGHT_FAST = 1.5   # up-weight the rarer / harder fast speed regime

    def __init__(self, allow_arch_changes: bool = False):
        super().__init__(allow_arch_changes=allow_arch_changes)
        logger.info(
            "MotionAwareRuleBasedOptimizer initialised (C2 - motion loss-shaping ON)"
        )

    def _motion_loss_shaping(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic loss-shaping levers from the dataset motion profile.

        Mirrors the recipe the LLM prompt gives (``v_max`` = p95 x 1.1) and
        adds gentle fixed velocity / smoothness weights plus a fast-regime
        up-weight. Only keys that differ from the current active value are
        returned, so this is a no-op after the first round (the motion
        profile is fixed across rounds).
        """
        profile = context.get("motion_profile", {}) or {}
        hp      = context.get("current_hyperparameters", {})

        targets: Dict[str, Any] = {
            "lambda_vel":      self._LAMBDA_VEL,
            "lambda_smooth":   self._LAMBDA_SMOOTH,
            "bin_weight_fast": self._BIN_WEIGHT_FAST,
        }
        p95 = profile.get("speed_p95_mps")
        if is_finite_number(p95):
            lo, hi = HP_BOUNDS["v_max"]
            targets["v_max"] = round(min(hi, max(lo, float(p95) * self._V_MAX_MARGIN)), 4)

        # Only propose what differs from the current active value (mirrors the
        # "never propose a no-op" rule used for the metric-driven HP changes).
        changes: Dict[str, Any] = {}
        for k, v in targets.items():
            cur = hp.get(k)
            if not is_finite_number(cur) or abs(float(cur) - float(v)) > 1e-9:
                changes[k] = v
        return changes

    async def suggest_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Metric-driven HP rules (inherited from C1) plus motion loss shaping."""
        t0 = time.time()

        metrics = context.get("metrics", {})
        trends  = context.get("trends", {})
        hp      = context.get("current_hyperparameters", {})

        diagnosis, severity, situation = self._diagnose(metrics, trends)
        hp_changes, strategy           = self._act(diagnosis, hp)

        # The C2 difference vs C1: deterministic motion-aware loss-shaping.
        loss_shaping_changes = self._motion_loss_shaping(context)
        proposed_changes = {**hp_changes, **loss_shaping_changes}

        elapsed   = time.time() - t0
        reasoning = self._reasoning(diagnosis, metrics, hp, hp_changes)
        if loss_shaping_changes:
            reasoning += "  Motion loss-shaping (fixed heuristic): " + ", ".join(
                f"{k}={v}" for k, v in loss_shaping_changes.items()
            )

        proposal = {
            "proposed_changes": proposed_changes,
            "resets_model":     False,
            "diagnosis": {
                "primary_problem": diagnosis,
                "severity":        severity,
                "situation":       situation,
            },
            "strategy":             strategy,
            "reasoning":            reasoning,
            "confidence":           "high",
            "expected_improvement": self._expected(diagnosis),
            "agent_time_s":         elapsed,
        }

        self._log_round(context, proposal)
        logger.info(
            "MotionRule: diagnosis=%s(%s) strategy=%s changes=%s (%.4fs)",
            diagnosis, severity, strategy, proposed_changes, elapsed,
        )
        return proposal
