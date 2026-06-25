"""DEPRECATED warm-loop machinery (pre-protocol path).

Everything in this module belongs to the old warm-loop optimisation path
(`SingleAgentOptimizer.suggest_hyperparameters` and friends), which was
superseded by the from-scratch protocol path (`propose_setting` +
`run_proposer_search`). It is kept temporarily for reference and is slated
for removal after a confirmed full protocol run. Nothing in here is called
by the protocol runner.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import pareto
from pipeline import Config, HP_BOUNDS, Trainer, is_finite_number, logger

from .parsing import _parse_scalar, _round_metric
from .validation import (
    ALLOWED_DIAGNOSES, ALLOWED_HP_KEYS, ARCH_CHANGE_KEYS,
    DISCRETE_HP_VALUES, INTEGER_HP_KEYS,
)


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

    # ── Current Metrics ─────────────────────────────────────────────
    m = payload.get("metrics", {})
    lines.append("━━ CURRENT METRICS ━━")
    lines.append(f"  val_loss:                   {_v(m.get('val_loss'))}")
    lines.append(f"  train_loss:                 {_v(m.get('train_loss'))}")
    lines.append(f"  val_mae:                    {_v(m.get('val_mae'))}")
    lines.append(f"  loss_ratio:                 {_v(m.get('loss_ratio'))}")
    lines.append(f"  mean_euclidean_distance_m:  {_v(m.get('mean_euclidean_distance_m'))}")
    lines.append("")

    # ── Round Summary ───────────────────────────────────────────────
    rs = payload.get("round_summary", {})
    lines.append("━━ ROUND SUMMARY ━━")
    for k, v in rs.items():
        lines.append(f"  {k}: {_v(v)}")
    lines.append("")

    # ── Trends ──────────────────────────────────────────────────────
    t = payload.get("trends", {})
    lines.append("━━ TRENDS ━━")
    for k, v in t.items():
        lines.append(f"  {k}: {_v(v)}")
    lines.append("")

    # ── Training Progress ───────────────────────────────────────────
    tp = payload.get("training_progress", {})
    lines.append("━━ TRAINING PROGRESS ━━")
    for k, v in tp.items():
        lines.append(f"  {k}: {_v(v)}")
    lines.append("")

    # ── Current Hyperparameters ─────────────────────────────────────
    hp = payload.get("current_hyperparameters", {})
    lines.append("━━ CURRENT HYPERPARAMETERS ━━")
    for k, v in hp.items():
        lines.append(f"  {k}: {_v(v)}")
    lines.append("")

    baseline = payload.get("baseline_reference", {}) or {}
    if baseline:
        lines.append("━━ COMPLETED BASELINE REFERENCE ━━")
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

    # ── Motion Profile (dataset trajectory speed + dwell) ───────────
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
        lines.append("━━ MOTION DIAGNOSTICS (val set) ━━")
        lines.append(f"  overall_rmse:        {_v(md.get('overall_rmse'))}")
        lines.append(f"  overall_mean_euclid: {_v(md.get('overall_mean_euclid'))}")
        lines.append("")

    # ── Tool Results ────────────────────────────────────────────────
    tr = payload.get("tool_results", {})
    if tr:
        lines.append("━━ DIAGNOSTIC TOOLS ━━")
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
        pareto_metrics = entry.get("pareto") or {}
        cost_str = ""
        if pareto_metrics:
            cost_str = (
                f" latency_ms={_v(pareto_metrics.get('latency_ms'))}"
                f" stability_std_m={_v(pareto_metrics.get('stability_std_m'))}"
                f" params={_v(pareto_metrics.get('params_trainable'))}"
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
    """Validate and sanitise a parsed LLM proposal (warm-loop path).

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
    # ── 1. Diagnosis enum (optional) ────────────────────────────────────────
    # The simplified system prompt no longer asks for a `diagnosis` field;
    # only validate it when the model actually provides one with a non-empty
    # primary_problem. Missing / "unknown" → skip this check.
    diag = parsed.get("diagnosis", {})
    if isinstance(diag, dict):
        pp = diag.get("primary_problem")
        if pp and pp != "unknown" and pp not in ALLOWED_DIAGNOSES:
            raise ValueError(
                f"Invalid diagnosis '{pp}'. "
                f"Must be one of: {sorted(ALLOWED_DIAGNOSES)}"
            )

    # ── 2. Allowed keys in proposed_changes ─────────────────────────────────
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

    # ── 3. Value types and ranges ───────────────────────────────────────────
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

    # ── 4. Diagnosis contradicts metrics ─────────────────────────────────────
    # Soft thresholds — deliberately more lenient than the system prompt to
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

    # ── 5. resets_model flag consistency ─────────────────────────────────────
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

    # ── 6. Multi-objective Pareto cost weights ────────────────────────────────
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
# OptimizerTools
# ---------------------------------------------------------------------------

class OptimizerTools:
    """Lightweight diagnostics used by the (deprecated) warm optimization loop."""

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
