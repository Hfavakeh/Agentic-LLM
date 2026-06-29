"""Proposer-driven search (LLM + rule-based) on the from-scratch engine."""

from typing import Any, Dict, List, Optional

from pipeline import Config, is_finite_number, logger

from .engine import _coerce_setting, _default_setting, evaluate_setting, setting_signature


async def run_proposer_search(
    base_cfg: Config,
    dataset: Dict[str, Any],
    agent,                       # SingleAgentOptimizer or RuleBasedOptimizer
    n_attempts: int,
    arm_name: str = "proposer",
) -> Dict[str, Any]:
    """Generic 25-attempt driver for the proposer arms (LLM, rule-based).

    Each attempt: build the qualitative history context (anchored on the best
    setting so far), ask the agent for a small change, and — if the agent
    returns a valid in-grid, not-already-tried setting — score it with the
    from-scratch engine (`evaluate_setting`, 3 trainings, mean val RMSE). A
    rejected proposal (LLM hard-validation failure / repeat after one retry)
    still CONSUMES the attempt with no training, per the protocol.
    """
    allow_arch = bool(getattr(base_cfg, "allow_arch_changes", True))
    max_epochs = base_cfg.epochs_per_round
    defaults = _coerce_setting(_default_setting(base_cfg, allow_arch), allow_arch)

    history: List[Dict[str, Any]] = []
    tried: set = set()
    best_score = float("inf")
    best_rec: Optional[Dict[str, Any]] = None
    best_setting: Optional[Dict[str, Any]] = None
    best_curve: List[float] = []

    def is_tried(s: Dict[str, Any]) -> bool:
        return setting_signature(s, allow_arch) in tried

    for a in range(1, n_attempts + 1):
        anchor = best_setting if best_setting is not None else defaults
        context = {
            "history":            history,
            "anchor_setting":     anchor,
            "max_epochs":         max_epochs,
            "allow_arch_changes": allow_arch,
            "is_tried":           is_tried,
        }
        proposal = await agent.propose_setting(context)

        # Defense-in-depth: even after the proposer's own check, ensure we
        # never re-train an already-tried setting. Rule-based now consults
        # `is_tried` inside `_act_protocol`; LLM proposals go through
        # `validate_protocol_changes` which also rejects `already_tried`. This
        # guard catches anything that slips past — without it a stale
        # controller could burn the whole 25-attempt budget re-evaluating one
        # duplicate setting (the bug seen in pre-fix rule-based runs).
        if proposal.get("valid"):
            try:
                candidate = _coerce_setting(proposal["resolved_setting"], allow_arch)
                if is_tried(candidate):
                    proposal = {
                        **proposal,
                        "valid": False,
                        "failure_reason": "already_tried",
                        "output_status": "rejected",
                    }
            except Exception:
                pass  # malformed proposal falls through to the invalid branch below

        if not proposal.get("valid"):
            history.append({
                "attempt": a, "round": a, "setting": dict(anchor),
                "score": float("inf"), "val_loss": float("nan"), "val_rmse": float("nan"),
                "val_rmse_std": float("nan"), "mean_best_epoch": float("nan"),
                "mean_val_loss": float("nan"), "mean_train_val_gap": float("nan"),
                "changes_from_anchor": proposal.get("proposed_changes", {}) or {},
                "diagnosis": proposal.get("diagnosis", "inconclusive"),
                "strategy": "", "reason": proposal.get("failure_reason", ""),
                "confidence": "", "output_status": "rejected",
                "trained": False, "per_seed": [],
            })
            best_curve.append(best_score if is_finite_number(best_score) else float("nan"))
            logger.info("%s attempt %d/%d | REJECTED (%s) | best=%.4f",
                        arm_name, a, n_attempts, proposal.get("failure_reason", "?"),
                        best_score if is_finite_number(best_score) else float("nan"))
            continue

        setting = _coerce_setting(proposal["resolved_setting"], allow_arch)
        result = evaluate_setting(base_cfg, dataset, setting)
        tried.add(setting_signature(setting, allow_arch))
        rec = {
            "attempt": a, "round": a, "setting": setting,
            "score": result["score"], "val_rmse": result["score"],
            "val_loss": result["mean_val_loss"],          # legacy plots/threshold
            "val_rmse_std": result["val_rmse_std"],
            "mean_best_epoch": result["mean_best_epoch"],
            "mean_train_loss": result["mean_train_loss"],
            "mean_val_loss": result["mean_val_loss"],
            "mean_train_val_gap": result["mean_train_val_gap"],
            "curve_summary": result.get("curve_summary", {}),
            "runtime_s": result["runtime_s"],
            "changes_from_anchor": proposal.get("proposed_changes", {}) or {},
            "diagnosis": proposal.get("diagnosis"), "strategy": proposal.get("strategy"),
            "reason": proposal.get("reason"), "confidence": proposal.get("confidence"),
            "output_status": proposal.get("output_status", "clean"),
            "trained": True, "per_seed": result["per_seed"],
        }
        history.append(rec)
        if is_finite_number(result["score"]) and result["score"] < best_score:
            best_score = result["score"]
            best_rec = rec
            best_setting = setting
        best_curve.append(best_score if is_finite_number(best_score) else float("nan"))
        logger.info("%s attempt %d/%d | val_rmse=%.4f ± %.4f | best=%.4f | %s",
                    arm_name, a, n_attempts, result["score"], result["val_rmse_std"],
                    best_score, proposal.get("output_status", "clean"))

    return {"attempts": history, "best": best_rec, "best_curve": best_curve,
            "best_setting": best_setting or {}}


def _proposer_opt_log(search: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a proposer search result to the legacy optimization_log shape that
    the plots and cross-run metrics consume."""
    rounds = []
    for h in search["attempts"]:
        rounds.append({
            "round": h["attempt"],
            "val_loss": h.get("val_loss"),
            "val_rmse": h.get("val_rmse"),
            "changes_applied": h.get("changes_from_anchor", {}),
            "diagnosis": h.get("diagnosis"),
            "output_status": h.get("output_status"),
            "trained": h.get("trained", True),
        })
    return {
        "initial_hyperparameters": defaults,
        "rounds": rounds,
        "best_setting": search.get("best_setting", {}),
        "final_summary": {
            "rounds_completed": len(rounds),
            "best_val_rmse": (search["best"] or {}).get("val_rmse"),
        },
    }
