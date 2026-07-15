"""Ollama-backed LLM proposer (SingleAgentOptimizer).

Protocol path: `propose_setting` renders the qualitative history context,
sends the protocol system prompt, parses the 5-line reply, hard-validates
the proposed delta, and retries ONCE on failure (no silent repair).

The warm-loop methods (`suggest_hyperparameters`, `_call_with_retry`,
`_attempt`, `_build_system_prompt`, `_log_round`) are DEPRECATED — superseded
by the protocol path and kept only until a confirmed full run; their module-
level helpers live in `agents.legacy`.
"""

import asyncio
import copy
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from autogen_core.models import ModelInfo, SystemMessage, UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

from pipeline import logger, sanitize_for_json

from .legacy import SemanticRepairRequired, _format_history, _format_payload_as_text, _validate_proposal
from .parsing import (
    _parse_llm_proposal, _parse_loss_shaping_proposal, _parse_protocol_proposal,
)
from .prompts import (
    format_motion_loss_payload, format_opro_payload, format_protocol_payload,
    motion_loss_system_prompt, opro_system_prompt, protocol_system_prompt,
)
from .validation import (
    _human_reason, validate_loss_shaping_changes, validate_protocol_changes,
)


# Outcome fields that, together, define a setting's rendered quality. The Q3
# placebo permutes these as a bundle across the history records (keeping each
# record's `setting` in place) so the BEST-SETTINGS ranking, LAST-ATTEMPTS
# trend and OBSERVED-PATTERNS signal are scrambled while the prompt format,
# the already-tried list, and the true ANCHOR stay intact.
_OUTCOME_KEYS = (
    "score", "val_rmse", "val_loss", "val_rmse_std", "mean_best_epoch",
    "mean_val_loss", "mean_train_val_gap",
)


def _apply_history_ablation(history: List[Dict[str, Any]], mode: str,
                            rng: random.Random) -> List[Dict[str, Any]]:
    """Q3 history-use placebo: return a perturbed COPY of the rendered history.

    Only the context shown to the LLM is altered — the engine's training and
    the driver's real best-so-far anchor are untouched. ``mode``:
      - "none"     → history unchanged (normal run)
      - "empty"    → no prior attempts shown
      - "shuffled" → each record keeps its setting but is given another
                     record's outcome bundle, so the quality signal is
                     misleading. A derangement is attempted (no record keeps
                     its own outcome) when more than one scorable record exists.
    """
    if mode == "none" or not history:
        return history
    if mode == "empty":
        return []
    if mode != "shuffled":
        logger.warning("Unknown history_ablation=%r; treating as 'none'", mode)
        return history

    out = copy.deepcopy(history)
    # Permute outcomes only among records that actually have a finite score;
    # rejected/untrained rows carry no outcome to scramble.
    idxs = [i for i, h in enumerate(out)
            if isinstance(h.get("score"), (int, float))
            and h.get("score") not in (float("inf"),)]
    if len(idxs) < 2:
        return out
    perm = idxs[:]
    for _ in range(8):  # a few tries to get a derangement; fall back if not
        rng.shuffle(perm)
        if all(p != i for p, i in zip(perm, idxs)):
            break
    bundles = [{k: out[i].get(k) for k in _OUTCOME_KEYS} for i in idxs]
    for dst, src_pos in zip(idxs, perm):
        src_bundle = bundles[idxs.index(src_pos)]
        for k in _OUTCOME_KEYS:
            out[dst][k] = src_bundle[k]
    return out


class SingleAgentOptimizer:

    @staticmethod
    def _build_system_prompt(model_arch: str = "LSTM", allow_arch_changes: bool = True,
                             enable_motion: bool = True) -> str:
        """DEPRECATED warm-loop system prompt (kept for reference only)."""
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
                 llm_timeout_s: float = 300.0, history_ablation: str = "none",
                 payload_curves: bool = False, explore_prompt: bool = False,
                 payload_motion: bool = False, opro_prompt: bool = False,
                 motion_show_profile: bool = True):

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

        # Q3 history-use placebo + protocol-path transcript (Step 0
        # instrumentation). `protocol_log` captures, per proposer call, the
        # EXACT rendered payload, every raw reply, the parsed proposal, the
        # anchor, and a history snapshot — the data needed to test whether the
        # LLM uses history. `_ablation_rng` is fixed-seeded so the "shuffled"
        # placebo is reproducible within a run.
        self.history_ablation  = history_ablation
        self._ablation_rng     = random.Random(12345)
        self.protocol_log: List[Dict[str, Any]] = []

        # Q2: when True, the rendered history adds a per-epoch TRAINING CURVE
        # shape label per setting and the system prompt explains how to use it.
        self.payload_curves    = payload_curves
        # Q4 prompt-variant: when True, the system prompt instructs broad
        # exploration of untried regions instead of a small change vs the anchor.
        self.explore_prompt    = explore_prompt
        # Q5: when True, the payload gains a MOTION PROFILE block and a per-setting
        # per-regime (slow/med/fast) error label, and the prompt explains them.
        self.payload_motion    = payload_motion
        self.motion_show_profile = motion_show_profile
        # OPRO variant (Email-5): when True the LLM arm uses the OPRO meta-prompt
        # + raw (setting, score) trajectory payload instead of the qualitative
        # protocol prompt. Takes precedence over explore_prompt/payload_curves/
        # payload_motion (those shape the qualitative payload, unused by OPRO).
        self.opro_prompt       = opro_prompt
        if opro_prompt and (explore_prompt or payload_curves or payload_motion):
            logger.warning(
                "opro_prompt=True overrides explore_prompt/payload_curves/"
                "payload_motion for the LLM arm (OPRO uses its own prompt+payload)."
            )

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
            "SingleAgentOptimizer initialised  model=%s  arch=%s  allow_arch_changes=%s  semantic_repair=%s  history_ablation=%s",
            model_name, model_arch, allow_arch_changes, semantic_repair, history_ablation,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def suggest_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """DEPRECATED (warm-loop path). Superseded by `propose_setting`, which
        the protocol runner (`run_proposer_search`) uses. This method and its
        helpers (`_call_with_retry`, `_attempt`, `_log_round`,
        `_build_system_prompt`, and `agents.legacy`'s `_format_payload_as_text`
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

    def save_protocol_log(self, path: str = "protocol_log.json"):
        """Persist the protocol-path transcript (Step 0): per proposer call, the
        exact rendered payload, raw replies, parsed proposal, anchor, history
        snapshot, and the active history_ablation mode. This is the evidence
        base for the Q3 history-use analysis."""
        with open(path, "w") as f:
            json.dump(self.protocol_log, f, indent=2, default=str)
        logger.info("Protocol log saved -> %s  (%d calls)", path, len(self.protocol_log))

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
            self._protocol_prompt = (
                opro_system_prompt(self.allow_arch_changes)
                if self.opro_prompt else
                protocol_system_prompt(
                    self.allow_arch_changes, self.payload_curves,
                    self.explore_prompt, self.payload_motion)
            )

        anchor      = context.get("anchor_setting") or {}
        is_tried    = context.get("is_tried") or (lambda s: False)
        allow_arch  = context.get("allow_arch_changes", self.allow_arch_changes)
        max_epochs  = int(context.get("max_epochs", 100))
        history     = context.get("history", [])
        motion_profile = context.get("motion_profile", {})

        # Q3 placebo: perturb the RENDERED history (not the true anchor / engine
        # training) so we can test whether the LLM's proposals actually depend
        # on the history it is shown.
        rendered_history = _apply_history_ablation(
            history, self.history_ablation, self._ablation_rng,
        )
        if self.opro_prompt:
            base_user = format_opro_payload(rendered_history, anchor, allow_arch)
        else:
            base_user = format_protocol_payload(rendered_history, anchor, max_epochs, allow_arch,
                                                show_curves=self.payload_curves,
                                                motion_profile=motion_profile,
                                                show_motion=self.payload_motion)
        feedback = ""
        last_reason = "unknown"
        last_parsed: Dict[str, Any] = {}
        raw_last = ""

        # Step 0 transcript: capture exactly what the LLM saw and replied so the
        # history-use analysis has a ground truth to work from.
        log_entry: Dict[str, Any] = {
            "timestamp":        datetime.now().isoformat(),
            "attempt_index":    len(self.protocol_log) + 1,
            "history_ablation": self.history_ablation,
            "opro_prompt":      self.opro_prompt,
            "anchor":           dict(anchor),
            "n_history":        len(history),
            "history_snapshot": [
                {"attempt": h.get("attempt"), "setting": h.get("setting"),
                 "score": h.get("score"), "output_status": h.get("output_status")}
                for h in history
            ],
            "rendered_payload": base_user,
            "sub_attempts":     [],
            "outcome":          "pending",
        }

        def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
            log_entry["outcome"] = "accepted" if result.get("valid") else "rejected"
            log_entry["final_reason"] = result.get("failure_reason", "")
            log_entry["resolved_setting"] = result.get("resolved_setting")
            self.protocol_log.append(log_entry)
            return result

        for attempt in range(self.max_retries + 1):
            user_text = base_user if not feedback else (
                base_user + f"\n\nYOUR PREVIOUS REPLY WAS REJECTED: {feedback}\n"
                "Reply again, in the exact format, with a different valid change."
            )
            sub: Dict[str, Any] = {"attempt": attempt, "raw": None,
                                   "parsed_changes": None, "status": None}
            raw, err = await self._protocol_raw_call(self._protocol_prompt, user_text)
            if err:
                last_reason = err
                feedback = ("the previous call did not return a usable reply "
                            f"({err}); reply concisely in the exact format.")
                sub["status"] = err
                log_entry["sub_attempts"].append(sub)
                continue
            raw_last = raw
            sub["raw"] = raw
            try:
                parsed = _parse_protocol_proposal(raw)
            except Exception as exc:  # parse failure
                self.retry_stats["parse_failures"] += 1
                last_reason = f"parse_error:{exc}"
                feedback = "your reply could not be parsed; output exactly the 5 fields, one per line."
                sub["status"] = last_reason
                log_entry["sub_attempts"].append(sub)
                continue
            last_parsed = parsed
            sub["parsed_changes"] = parsed.get("proposed_changes", {})
            sub["diagnosis"] = parsed.get("diagnosis")
            resolved, ok, reason = validate_protocol_changes(
                parsed.get("proposed_changes", {}), anchor, allow_arch, is_tried,
            )
            sub["status"] = "ok" if ok else reason
            log_entry["sub_attempts"].append(sub)
            if ok:
                status = "clean" if attempt == 0 else "accepted_after_retry"
                if attempt == 0:
                    self.retry_stats["valid_first_try"] += 1
                else:
                    self.retry_stats["valid_after_retry"] += 1
                return _finish({
                    "valid":            True,
                    "resolved_setting": resolved,
                    "proposed_changes": parsed.get("proposed_changes", {}),
                    "diagnosis":        parsed.get("diagnosis", "inconclusive"),
                    "strategy":         parsed.get("strategy", ""),
                    "reason":           parsed.get("reason", ""),
                    "confidence":       parsed.get("confidence", "medium"),
                    "output_status":    status,
                    "raw":              raw,
                })
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
        return _finish({
            "valid":          False,
            "output_status":  "rejected",
            "proposed_changes": last_parsed.get("proposed_changes", {}),
            "diagnosis":      last_parsed.get("diagnosis", "inconclusive"),
            "failure_reason": last_reason,
            "raw":            raw_last,
        })

    # ------------------------------------------------------------------
    # Motion experiment path: propose a loss-shaping lever vector
    # ------------------------------------------------------------------

    async def propose_loss_shaping(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Motion-thesis entry point (C3): read the motion summaries + per-regime
        error + tried lever vectors, propose a COMPLETE loss-shaping lever vector,
        hard-validate it against LOSS_SHAPING_GRID and the already-tried set,
        retry ONCE, else reject. Mirrors `propose_setting` but over the six
        loss-shaping levers (the 9 HPs are frozen at baseline)."""
        for k in ("valid_first_try", "valid_after_retry", "rejected",
                  "repeats_proposed", "invalid_values"):
            self.retry_stats.setdefault(k, 0)
        if not hasattr(self, "_motion_prompt"):
            self._motion_prompt = motion_loss_system_prompt()

        anchor    = context.get("anchor_levers") or {}
        is_tried  = context.get("is_tried") or (lambda s: False)
        history   = context.get("history", [])
        motion_profile = context.get("motion_profile", {})
        regime_err = context.get("anchor_regime_error")

        base_user = format_motion_loss_payload(history, anchor, motion_profile, regime_err,
                                               show_profile=self.motion_show_profile)
        feedback = ""
        last_reason = "unknown"
        last_parsed: Dict[str, Any] = {}
        raw_last = ""

        log_entry: Dict[str, Any] = {
            "timestamp":        datetime.now().isoformat(),
            "attempt_index":    len(self.protocol_log) + 1,
            "mode":             "motion_loss_shaping",
            "anchor_levers":    dict(anchor),
            "n_history":        len(history),
            "rendered_payload": base_user,
            "sub_attempts":     [],
            "outcome":          "pending",
        }

        def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
            log_entry["outcome"] = "accepted" if result.get("valid") else "rejected"
            log_entry["final_reason"] = result.get("failure_reason", "")
            log_entry["resolved_setting"] = result.get("resolved_setting")
            self.protocol_log.append(log_entry)
            return result

        for attempt in range(self.max_retries + 1):
            user_text = base_user if not feedback else (
                base_user + f"\n\nYOUR PREVIOUS REPLY WAS REJECTED: {feedback}\n"
                "Reply again, in the exact format, with a different valid lever vector.")
            sub: Dict[str, Any] = {"attempt": attempt, "raw": None, "parsed_changes": None, "status": None}
            raw, err = await self._protocol_raw_call(self._motion_prompt, user_text)
            if err:
                last_reason = err
                feedback = (f"the previous call did not return a usable reply ({err}); "
                            "reply concisely in the exact format.")
                sub["status"] = err
                log_entry["sub_attempts"].append(sub)
                continue
            raw_last = raw
            sub["raw"] = raw
            try:
                parsed = _parse_loss_shaping_proposal(raw)
            except Exception as exc:
                self.retry_stats["parse_failures"] += 1
                last_reason = f"parse_error:{exc}"
                feedback = "your reply could not be parsed; output exactly the 5 fields, one per line."
                sub["status"] = last_reason
                log_entry["sub_attempts"].append(sub)
                continue
            last_parsed = parsed
            sub["parsed_changes"] = parsed.get("proposed_changes", {})
            resolved, ok, reason = validate_loss_shaping_changes(
                parsed.get("proposed_changes", {}), anchor, is_tried,
            )
            sub["status"] = "ok" if ok else reason
            log_entry["sub_attempts"].append(sub)
            if ok:
                status = "clean" if attempt == 0 else "accepted_after_retry"
                if attempt == 0:
                    self.retry_stats["valid_first_try"] += 1
                else:
                    self.retry_stats["valid_after_retry"] += 1
                return _finish({
                    "valid":            True,
                    "resolved_setting": resolved,
                    "proposed_changes": parsed.get("proposed_changes", {}),
                    "diagnosis":        parsed.get("diagnosis", "inconclusive"),
                    "strategy":         parsed.get("strategy", ""),
                    "reason":           parsed.get("reason", ""),
                    "confidence":       parsed.get("confidence", "medium"),
                    "output_status":    status,
                    "raw":              raw,
                })
            if reason == "already_tried":
                self.retry_stats["repeats_proposed"] += 1
            elif reason.startswith("value_not_in_grid") or reason.startswith("unknown_lever"):
                self.retry_stats["invalid_values"] += 1
            last_reason = reason
            feedback = _human_reason(reason)

        self.retry_stats["rejected"] += 1
        logger.warning("LLM loss-shaping proposal rejected after %d attempt(s): %s",
                       self.max_retries + 1, last_reason)
        return _finish({
            "valid":          False,
            "output_status":  "rejected",
            "proposed_changes": last_parsed.get("proposed_changes", {}),
            "diagnosis":      last_parsed.get("diagnosis", "inconclusive"),
            "failure_reason": last_reason,
            "raw":            raw_last,
        })

    # ------------------------------------------------------------------
    # Internal helpers (warm-loop path — DEPRECATED)
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
        # Use a flat text format instead of raw JSON — easier for all LLMs to parse.
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
            # ── What the LLM received ───────────────────────────────────────
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
