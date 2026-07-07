"""Protocol prompt + qualitative history payload (Step 6).

The from-scratch protocol replaces the warm-loop "current training curve"
context with a history of TRIED SETTINGS and their averaged, QUALITATIVE
scores. The LLM (and the rule-based controller) propose a small `changes:`
delta relative to the best-so-far setting (the ANCHOR). Pareto and motion
levers are deferred; the only objective is mean validation RMSE.
"""

from typing import Any, Dict, List

from pipeline import HP_GRID, LOSS_SHAPING_GRID, is_finite_number

from .labels import (
    _behavior_label, _fmt_grid_val, _qual_curve_shape, _qual_epoch_timing,
    _qual_gap, _qual_level_10, _qual_motion_error, _qual_motion_profile,
    _qual_quality, _qual_variation,
)

_HP_ORDER = [
    "learning_rate", "weight_decay", "dropout", "batch_size",
    "lstm_hidden", "lstm_layers", "window_size", "optimizer_choice", "patience",
]


def _format_search_space_text(allow_arch_changes: bool = True) -> str:
    lines = []
    for k in _HP_ORDER:
        if not allow_arch_changes and k in ("lstm_hidden", "lstm_layers"):
            continue
        lines.append(f"  - {k}: " + ", ".join(_fmt_grid_val(v) for v in HP_GRID[k]))
    return "\n".join(lines)


def protocol_system_prompt(allow_arch_changes: bool = True, show_curves: bool = False,
                           explore: bool = False, show_motion: bool = False) -> str:
    """The from-scratch protocol prompt: tune conventional HPs from the
    qualitative history of tried settings, proposing a small delta vs the
    best-so-far anchor. No motion levers, no Pareto cost weights.

    `explore=True` (Q4 prompt-variant) swaps the "propose a SMALL change vs the
    ANCHOR" instruction for an exploration-oriented one (change several HPs, make
    large moves into untried regions). Used to test whether the anchoring
    instruction is what caps the LLM's search-grid coverage."""
    arch_note = (
        "" if allow_arch_changes
        else "\n(NOTE: lstm_hidden and lstm_layers are FIXED this run — do not propose them.)"
    )
    curve_note = (
        "\n\nEach setting also reports a TRAINING CURVE shape describing how its "
        "validation error moved over epochs:\n"
        "  - still improving: error was still dropping when training stopped "
        "(more capacity / a higher learning rate may help)\n"
        "  - overfitting upturn: error rose again after reaching its best "
        "(add regularization)\n"
        "  - unstable/noisy: error oscillated epoch-to-epoch (lower the learning "
        "rate or raise batch size)\n"
        "  - clean plateau: error converged and flattened\n"
        "Use the curve shape, not only the final score, to choose the change."
        if show_curves else ""
    )
    motion_note = (
        "\n\nYou are also shown a MOTION PROFILE of how the tracked person moves "
        "(overall pace, fast bursts, stop-go), and for each setting where its "
        "position error concentrates across motion regimes (slow / medium / fast). "
        "Use your knowledge of how people move: if error concentrates in the fast "
        "regime the model may underfit quick movements (more capacity / longer "
        "window), and a smoother / slower regime tolerates more regularization."
        if show_motion else ""
    )
    return f"""You are tuning an LSTM model for indoor localization of one person. The model receives windows of sensor features and predicts position. Your goal is to reduce validation error while avoiding unstable behavior. Do not repeat a setting that has already been tried.

You may change ONLY these hyperparameters, and ONLY to one of the listed allowed values:
{_format_search_space_text(allow_arch_changes)}{arch_note}

Each setting is trained 3 times (3 fixed seeds) and scored by its MEAN validation RMSE - lower is better. You are shown the best settings so far, the most recent attempts, every setting already tried, and observed patterns, all as qualitative summaries.{curve_note}{motion_note}

{_proposal_instruction(explore)}

diagnosis: <healthy | possible overfitting tendency | possible underfitting tendency | plateau | unstable | inconclusive>
strategy: <one short phrase, e.g. increase regularization>
changes: <param=value, param=value>
reason: <one sentence>
confidence: <low | medium | high>
"""


def _proposal_instruction(explore: bool) -> str:
    """The strategic instruction line(s) before the output format. The Q4
    prompt-variant (`explore=True`) replaces the small-delta-vs-anchor framing
    with an exploration-oriented one."""
    if explore:
        return (
            "EXPLORE the search space: propose a setting in a region you have NOT tried yet. "
            "You may change SEVERAL hyperparameters at once and make LARGE moves away from the best "
            "setting so far (shown as ANCHOR) — do not just tweak one value. Prefer allowed values and "
            "combinations that are not in the already-tried list, so each attempt covers new ground. "
            "Respond using EXACTLY these lines, one field per line, no prose, no markdown:"
        )
    return (
        "Propose a SMALL change relative to the best setting so far (shown as ANCHOR). "
        "Respond using EXACTLY these lines, one field per line, no prose, no markdown:"
    )


_LEVER_ORDER = [
    "v_max", "lambda_vel", "lambda_smooth",
    "bin_weight_slow", "bin_weight_medium", "bin_weight_fast",
]


def _format_lever_space_text() -> str:
    return "\n".join(
        f"  - {k}: " + ", ".join(_fmt_grid_val(v) for v in LOSS_SHAPING_GRID[k])
        for k in _LEVER_ORDER
    )


def motion_loss_system_prompt() -> str:
    """Motion-thesis system prompt: the LLM reshapes the TRAINING OBJECTIVE of a
    fixed LSTM (all 9 conventional HPs frozen at baseline) using its knowledge of
    how people move, from interpretable motion summaries of the tracked person.

    It sets six loss-shaping levers (allowed values only) to reduce validation
    error. Raw motion numbers are shown on purpose — the LLM is expected to
    reason quantitatively (e.g. v_max just above the p95 speed)."""
    return f"""You are improving the TRAINING OBJECTIVE of a fixed LSTM that localizes one walking person indoors. The network architecture and all conventional hyperparameters are FROZEN — you change ONLY the loss function, using your knowledge of how humans move.

You are given interpretable MOTION SUMMARIES of the tracked person (walking speed, acceleration, turning, stop-go/dwell, trajectory roughness) and, once settings have been tried, where the model's position error concentrates across motion regimes (slow / medium / fast).

You set six loss-shaping levers, ONLY to one of the listed allowed values:
{_format_lever_space_text()}

What the levers do (this is where human-motion knowledge matters):
  - v_max — plausible top human walking speed in m/s. Predicted steps faster than this are penalised. Set it just above the observed p95 speed (about 1.1x); do not leave a generic default if the data's speed range differs.
  - lambda_vel — strength of that speed-plausibility penalty (0 = off). Raise it when predictions look noisy or the motion is smooth/slow.
  - lambda_smooth — penalty on physically implausible acceleration/jerk (0 = off). Raise it when the trajectory is smooth and dwell episodes are frequent; keep low when motion is genuinely jerky/fast.
  - bin_weight_slow / medium / fast — per-speed-regime position-error weights (1.0 = neutral). Up-weight whichever regime the model fits worst (use the per-regime error), e.g. the fast regime when quick movements are hardest.

Neutral levers (lambda_vel=0, lambda_smooth=0, all bin weights 1.0) reduce to plain MSE. Propose a COMPLETE lever vector (all six) that you expect to lower validation error. Do not repeat a lever vector already tried.

Respond using EXACTLY these lines, one field per line, no prose, no markdown:

diagnosis: <healthy | possible overfitting tendency | possible underfitting tendency | plateau | unstable | inconclusive>
strategy: <one short phrase, e.g. penalise implausible speed, upweight fast regime>
changes: v_max=<val>, lambda_vel=<val>, lambda_smooth=<val>, bin_weight_slow=<val>, bin_weight_medium=<val>, bin_weight_fast=<val>
reason: <one sentence grounded in the motion summaries>
confidence: <low | medium | high>
"""


def _fmt_num(v: Any, n: int = 3) -> str:
    try:
        f = float(v)
        return f"{f:.{n}g}" if is_finite_number(f) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def format_motion_loss_payload(
    history: List[Dict[str, Any]],
    anchor_levers: Dict[str, Any],
    motion_profile: Dict[str, Any],
    anchor_regime_error: Any = None,
) -> str:
    """Render the motion context for the loss-shaping LLM: raw motion summaries,
    the current levers, where error concentrates by regime, and the tried lever
    vectors with their scores (worst→best), so the LLM can reason quantitatively
    and avoid repeats."""
    mp = motion_profile or {}
    lines: List[str] = []
    lines.append("== MOTION SUMMARY of the tracked person (raw numbers) ==")
    lines.append(
        f"  speed m/s: mean {_fmt_num(mp.get('speed_mean_mps'))}, median {_fmt_num(mp.get('speed_median_mps'))}, "
        f"p95 {_fmt_num(mp.get('speed_p95_mps'))}, max {_fmt_num(mp.get('speed_max_mps'))}, std {_fmt_num(mp.get('speed_std_mps'))}")
    lines.append(
        f"  acceleration m/s^2: mean {_fmt_num(mp.get('accel_abs_mean_mps2'))}, p95 {_fmt_num(mp.get('accel_abs_p95_mps2'))}")
    lines.append(
        f"  turning: mean {_fmt_num(mp.get('turn_abs_mean_deg'))} deg/step, p95 {_fmt_num(mp.get('turn_abs_p95_deg'))} deg, "
        f"sharp-turn share {_fmt_num(mp.get('sharp_turn_share'))}")
    lines.append(
        f"  roughness: path sinuosity {_fmt_num(mp.get('path_sinuosity'))} (1=straight), mean jerk {_fmt_num(mp.get('jerk_abs_mean_mps3'))} m/s^3")
    dwell = mp.get("dwell", {}) or {}
    lines.append(
        f"  stop-go: stop share {_fmt_num(dwell.get('stop_share'))}, {_fmt_num(dwell.get('episodes_per_min'))} dwell episodes/min")
    lines.append("")

    lines.append("== CURRENT LOSS-SHAPING LEVERS (propose changes relative to this) ==")
    lines.append("  " + ", ".join(f"{k}={_fmt_grid_val(anchor_levers.get(k))}" for k in _LEVER_ORDER))
    reg = anchor_regime_error or {}
    if reg and reg.get("worst_regime"):
        lines.append(
            f"  best-so-far error by regime (metres): slow {_fmt_num(reg.get('slow'),4)}, "
            f"medium {_fmt_num(reg.get('medium'),4)}, fast {_fmt_num(reg.get('fast'),4)} "
            f"-> worst = {reg.get('worst_regime')} (spread x{_fmt_num(reg.get('spread_ratio'))})")
    lines.append("")

    scored = [h for h in history if is_finite_number(h.get("score"))]
    ranked = sorted(scored, key=lambda h: float(h["score"]), reverse=True)
    lines.append("== TRIED LEVER VECTORS (score = mean validation RMSE in m, lower better; worst → best) ==")
    if ranked:
        for h in ranked:
            lv = h.get("setting", {}) or {}
            lines.append(f"  score={_fmt_num(h['score'],4)}  " +
                         ", ".join(f"{k}={_fmt_grid_val(lv.get(k))}" for k in _LEVER_ORDER))
    else:
        lines.append("  (none yet — the anchor above is the neutral plain-MSE loss)")
    lines.append("")
    lines.append("Propose ONE complete lever vector (all six), different from every vector above, that lowers validation error.")
    lines.append("Respond ONLY in the protocol line format. One field per line. No prose, no markdown.")
    return "\n".join(lines)


def opro_system_prompt(allow_arch_changes: bool = True) -> str:
    """OPRO-style optimizer meta-prompt (Yang et al., 2024, "Large Language
    Models as Optimizers"), added for the Email-5 diagnostic.

    Unlike `protocol_system_prompt` (qualitative labels + small delta vs the
    ANCHOR), this asks the LLM to read a list of past (setting, NUMERIC score)
    pairs sorted worst→best and emit a COMPLETE new setting expected to score
    lower than all of them. The reply still uses the protocol's 5-line format so
    the existing parser/validator/logging apply unchanged — but `changes:` must
    carry ALL hyperparameters (a full candidate), not a one-value tweak.
    """
    arch_note = (
        "" if allow_arch_changes
        else "\n(NOTE: lstm_hidden and lstm_layers are FIXED this run — reuse the listed value, do not vary them.)"
    )
    n_hp = len(_HP_ORDER) - (0 if allow_arch_changes else 2)
    return f"""You are optimizing the hyperparameters of an LSTM model for indoor localization of one person. The model receives windows of sensor features and predicts position. Each setting is trained 3 times (3 fixed seeds) and scored by its MEAN validation RMSE — lower is better.

You may use ONLY these hyperparameters, and ONLY the listed allowed values:
{_format_search_space_text(allow_arch_changes)}{arch_note}

Below (in the message) are hyperparameter settings that have already been evaluated, each with its measured score, sorted from HIGHEST (worst) to LOWEST (best) score. Study how the values relate to the scores, then propose ONE NEW complete setting that is DIFFERENT from every setting listed and that you expect to have a score LOWER than all of them.

Give a COMPLETE setting: specify all {n_hp} hyperparameters in the `changes:` line (every allowed value must come from the lists above). Do not repeat any setting already listed.

Respond using EXACTLY these lines, one field per line, no prose, no markdown:

diagnosis: <healthy | possible overfitting tendency | possible underfitting tendency | plateau | unstable | inconclusive>
strategy: <one short phrase, e.g. lower learning rate and add regularization>
changes: <param=value, param=value, ... for ALL hyperparameters>
reason: <one sentence>
confidence: <low | medium | high>
"""


def format_opro_payload(
    history: List[Dict[str, Any]],
    anchor_setting: Dict[str, Any],
    allow_arch_changes: bool = True,
) -> str:
    """Render the OPRO trajectory: past (setting, NUMERIC score) pairs sorted
    worst→best (highest RMSE first), so the best solution sits closest to the
    instruction — the ordering the OPRO meta-prompt relies on. Raw numeric
    scores are shown on purpose (the point of the OPRO variant vs the
    qualitative-label protocol payload)."""
    scored = [h for h in history if is_finite_number(h.get("score"))]
    # OPRO convention: ascending by quality so the best (lowest RMSE) is last.
    ranked = sorted(scored, key=lambda h: float(h["score"]), reverse=True)

    lines: List[str] = []
    lines.append("== EVALUATED SETTINGS (score = mean validation RMSE, lower is better; sorted worst → best) ==")
    if ranked:
        for h in ranked:
            lines.append(
                f"  score={float(h['score']):.4f}  "
                f"{_setting_line(h.get('setting', {}), allow_arch_changes)}"
            )
    else:
        lines.append("  (none yet — propose a sensible starting setting)")
    lines.append("")
    lines.append("Propose ONE NEW complete setting, different from all above, with a lower score.")
    lines.append("Respond ONLY in the protocol line format. One field per line. No prose, no markdown.")
    return "\n".join(lines)


def _setting_line(setting: Dict[str, Any], allow_arch_changes: bool = True) -> str:
    parts = []
    for k in _HP_ORDER:
        if k in setting:
            parts.append(f"{k}={_fmt_grid_val(setting[k])}")
    return ", ".join(parts)


def format_protocol_payload(
    history: List[Dict[str, Any]],
    anchor_setting: Dict[str, Any],
    max_epochs: int,
    allow_arch_changes: bool = True,
    show_curves: bool = False,
    motion_profile: Any = None,
    show_motion: bool = False,
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

    # ── MOTION PROFILE (Q5) — how the tracked person moves ───────────────────
    if show_motion:
        prof = _qual_motion_profile(motion_profile)
        if prof:
            lines.append("== MOTION PROFILE (how the tracked person moves) ==")
            lines.append("  " + prof)
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
        lines = [
            f"    validation quality: {q} (level {_level_str(h)}, 1=best 10=worst)",
            f"    reliability across 3 trainings: {var}",
            f"    train/validation gap: {gap}",
            f"    best epoch timing: {tim}",
            f"    behavior label: {beh.replace('_', ' ')}",
        ]
        if show_curves:
            shape = _qual_curve_shape(h.get("curve_summary"))
            if shape != "unknown":
                lines.append(f"    training curve: {shape.replace('_', ' ')}")
        if show_motion:
            merr = _qual_motion_error(h.get("motion_error_summary"))
            if merr:
                lines.append(f"    motion error: {merr}")
        return lines

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
            if show_curves:
                shape = _qual_curve_shape(h.get("curve_summary"))
                if shape != "unknown":
                    lines.append(f"    training curve: {shape.replace('_', ' ')}")
            if show_motion:
                merr = _qual_motion_error(h.get("motion_error_summary"))
                if merr:
                    lines.append(f"    motion error: {merr}")
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
