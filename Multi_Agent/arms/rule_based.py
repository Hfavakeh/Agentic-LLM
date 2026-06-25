"""Deterministic (non-LLM) controllers.

`RuleBasedOptimizer` (C1) acts on the SAME soft diagnosis labels the LLM
prompt uses, stepping one grid value vs the anchor by walking a frozen
per-diagnosis priority list. `MotionAwareRuleBasedOptimizer` (C2) adds fixed
motion-heuristic loss shaping on the deprecated warm-loop path.
"""

import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline import HP_BOUNDS, is_finite_number, logger

from .labels import _behavior_label, _qual_gap, _qual_quality, _qual_variation
from .validation import _grid_neighbor

# ---------------------------------------------------------------------------
# Protocol candidate-move tables
# ---------------------------------------------------------------------------
#
# Ordered list of single-HP (param, direction, strategy) moves per diagnosis.
# `_act_protocol` walks the list and returns the first move whose value
# actually differs from the anchor (not at a grid boundary) AND whose
# resolved setting is not already-tried. This replaces the older
# single-move-per-diagnosis rule that would deadlock once its preferred move
# had been tried and not improved — without this priority list the
# controller emits the same proposal every attempt forever whenever the
# anchor doesn't change. Architectural moves are appended only when
# allow_arch_changes is True. Frozen rules — never tuned after seeing
# results, so the only difference vs the LLM remains *how* the diagnosis
# label is acted on.

_PROTOCOL_MOVES_BASE: Dict[str, List[Tuple[str, int, str]]] = {
    "unstable": [
        ("learning_rate", -1, "stabilise"),
        ("batch_size",    -1, "stabilise"),
        ("weight_decay",  +1, "stabilise"),
        ("dropout",       +1, "stabilise"),
        ("patience",      +1, "stabilise"),
        ("learning_rate", +1, "stabilise"),
    ],
    "plateau": [
        ("learning_rate", -1, "escape_plateau"),
        ("learning_rate", +1, "escape_plateau"),
        ("dropout",       -1, "escape_plateau"),
        ("weight_decay",  -1, "escape_plateau"),
        ("batch_size",    -1, "escape_plateau"),
        ("patience",      +1, "escape_plateau"),
    ],
    "possible_overfitting_tendency": [
        ("dropout",       +1, "regularise"),
        ("weight_decay",  +1, "regularise"),
        ("batch_size",    +1, "regularise"),
        ("learning_rate", -1, "regularise"),
    ],
    "possible_underfitting_tendency": [
        ("learning_rate", +1, "explore"),
        ("dropout",       -1, "explore"),
        ("weight_decay",  -1, "explore"),
        ("window_size",   +1, "explore"),
        ("batch_size",    -1, "explore"),
    ],
    "healthy": [
        ("dropout",       +1, "exploit"),
        ("dropout",       -1, "exploit"),
        ("weight_decay",  +1, "exploit"),
        ("weight_decay",  -1, "exploit"),
        ("learning_rate", -1, "exploit"),
        ("learning_rate", +1, "exploit"),
        ("batch_size",    +1, "exploit"),
        ("batch_size",    -1, "exploit"),
        ("patience",      +1, "exploit"),
        ("window_size",   +1, "explore"),
        ("window_size",   -1, "explore"),
    ],
}

# Arch-only moves, prepended for underfitting (capacity-first) and appended
# for healthy (small-local-explore-first). Skipped when allow_arch_changes
# is False.
_PROTOCOL_ARCH_MOVES: Dict[str, List[Tuple[str, int, str]]] = {
    "possible_underfitting_tendency": [
        ("lstm_hidden", +1, "explore"),
        ("lstm_layers", +1, "explore"),
    ],
    "healthy": [
        ("lstm_hidden", +1, "explore"),
        ("lstm_hidden", -1, "explore"),
        ("lstm_layers", +1, "explore"),
    ],
}


class RuleBasedOptimizer:
    """Deterministic, non-LLM hyperparameter controller.

    Uses the **exact same diagnosis thresholds** as the LLM system prompt so
    that both controllers are operating under identical rules — the only
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
                        if learning_rate > 5e-4, reduce it x0.7
        healthy      -> no changes
        no_data      -> no changes

    The class exposes the same async interface as SingleAgentOptimizer so
    the optimization loop can use it unchanged as a drop-in replacement.
    """

    # ── Thresholds (identical to system prompt) ──────────────────────────────
    _PLATEAU_PATIENCE   = 8      # epochs_since_improvement threshold
    _PLATEAU_RATIO_MAX  = 1.2    # loss_ratio upper bound for plateau
    _UNDERFIT_LOSS_MIN  = 2.0    # val_loss floor for underfitting
    _UNDERFIT_RATIO_MAX = 1.2    # loss_ratio upper bound for underfitting
    _OVERFIT_RATIO_MIN  = 1.5    # loss_ratio lower bound for overfitting

    # ── HP bounds (same as HP_BOUNDS in pipeline.search_space) ───────────────
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
            "confidence":           "high",   # deterministic -> always confident
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
        """No-op — no network connection to close."""
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

    def _act_protocol(
        self,
        diagnosis: str,
        anchor: Dict[str, Any],
        allow_arch_changes: bool,
        is_tried: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Pick a single-HP delta vs the anchor by walking the diagnosis's
        priority list (`_PROTOCOL_MOVES_BASE` + optional `_PROTOCOL_ARCH_MOVES`),
        skipping moves at a grid boundary (no actual change) and moves whose
        resolved setting is already tried. Returns the first hit as
        `(changes, strategy)`; returns `(None, "exhausted")` when every
        candidate is a no-op or already-tried so the engine can mark the
        attempt rejected instead of training a duplicate. On `"inconclusive"`
        with an unseen anchor, returns `({}, "exploit")` so attempt 1 trains
        the anchor (defaults) to establish a baseline data point — but if the
        anchor is itself already-tried (e.g. attempt N after rejects), falls
        through to the healthy moves so the budget isn't wasted re-evaluating.
        Frozen rules — never tuned after seeing results.
        """
        if diagnosis == "inconclusive" and (is_tried is None or not is_tried(anchor)):
            return {}, "exploit"

        base  = _PROTOCOL_MOVES_BASE.get(diagnosis) or _PROTOCOL_MOVES_BASE["healthy"]
        archs = _PROTOCOL_ARCH_MOVES.get(diagnosis, []) if allow_arch_changes else []
        # Capacity-first for underfitting; small-local-explore-first elsewhere.
        moves = (archs + list(base)) if diagnosis == "possible_underfitting_tendency" \
                else (list(base) + archs)

        for param, direction, strategy in moves:
            cur = anchor.get(param)
            nv  = _grid_neighbor(param, cur, direction)
            if nv == cur:                 # grid boundary — move doesn't change anything
                continue
            candidate = {**anchor, param: nv}
            if is_tried is not None and is_tried(candidate):
                continue
            return {param: nv}, strategy
        return None, "exhausted"

    async def propose_setting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic protocol proposer: diagnose the last attempt, then
        walk the diagnosis's move-priority list to pick the first single-HP
        delta that actually changes the anchor and isn't already-tried
        (consuming `context["is_tried"]`, exposed by `run_proposer_search`).
        Returns `valid=False, failure_reason="exhausted"` only when every
        priority move would be a no-op or duplicate — in which case the
        attempt is rejected by the engine like any other invalid proposal.
        """
        anchor     = context.get("anchor_setting") or {}
        history    = context.get("history", [])
        allow_arch = context.get("allow_arch_changes", self.allow_arch_changes)
        is_tried   = context.get("is_tried")

        diagnosis = self._diagnose_protocol(history)
        changes, strategy = self._act_protocol(diagnosis, anchor, allow_arch, is_tried)

        if changes is None:
            return {
                "valid":            False,
                "proposed_changes": {},
                "diagnosis":        diagnosis,
                "strategy":         strategy,
                "failure_reason":   "exhausted",
                "output_status":    "rejected",
                "confidence":       "high",
            }

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
    # Diagnosis (warm-loop path — DEPRECATED)
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
            return "no_data", "unknown", "Metrics are missing or invalid — no change this round."

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
                f"loss_ratio={lr:.3f} > {self._OVERFIT_RATIO_MIN} — val/train gap widening.",
            )

        return (
            "healthy",
            "low",
            f"val_loss={vl:.4f}  loss_ratio={lr:.3f} — training is on track.",
        )

    # ------------------------------------------------------------------
    # Action (warm-loop path — DEPRECATED)
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
                # LR already at floor — nudge dropout instead
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

        # healthy / no_data -> no changes
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
            parts.append(f"{k}: {old} -> {v}")
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
        """Metric-driven HP rules (inherited from C1) plus motion loss shaping.
        DEPRECATED warm-loop path — the protocol runner uses the inherited
        `propose_setting` (C2 differs from C1 only on this legacy path)."""
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
