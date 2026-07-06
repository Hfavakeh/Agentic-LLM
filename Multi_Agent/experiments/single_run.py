"""One experiment run: baseline → LLM → random → rule-based → Optuna →
protocol reports → final evaluation on fresh seeds."""

import copy
import json
from typing import Any, Dict, List

from arms import (
    MotionAwareRuleBasedOptimizer, RuleBasedOptimizer, SingleAgentOptimizer,
    TRAIN_SEEDS, build_dataset_and_loaders, run_baseline, run_optuna_search,
    run_proposer_search, run_random_search, train_and_test_setting,
)
from arms.driver import _proposer_opt_log
from arms.engine import _coerce_setting, _default_setting
from pipeline import Config, logger
from reporting import (
    plot_action_impact_scatter, plot_hyperparameter_trajectory,
    plot_optimization_trajectories, write_budget_table,
    write_final_eval_report, write_per_setting_report,
)

from .final_eval import FINAL_EVAL_SEEDS_POOL, run_final_evaluation


async def run_experiment(
    config: Config,
    run_id: int,
    llm_model: str,
) -> Dict[str, Any]:
    results = {}

    # ── 1. Baseline (fixed reference) ─────────────────────────────────────────
    results["baseline"] = run_baseline(config, run_id)
    b_history = results["baseline"]["history"]

    # ── 2. LLM controller (from-scratch engine + qualitative history prompt) ──
    l_cfg = copy.deepcopy(config)
    l_dataset, _, _, _ = build_dataset_and_loaders(l_cfg)
    model_arch = getattr(config, "model_arch", "LSTM")
    agent = SingleAgentOptimizer(
        model_name=llm_model,
        model_arch=model_arch,
        allow_arch_changes=config.allow_arch_changes,
        enable_motion=getattr(config, "enable_motion", True),
        semantic_repair=getattr(config, "semantic_repair", False),
        history_ablation=getattr(config, "history_ablation", "none"),
        payload_curves=getattr(config, "payload_curves", False),
        explore_prompt=getattr(config, "explore_prompt", False),
        payload_motion=getattr(config, "payload_motion", False),
        opro_prompt=getattr(config, "opro_prompt", False),
    )
    logger.info("--- Run %d | LLM CONTROLLER (model=%s, attempts=%d × %d trainings) ---",
                run_id, llm_model, l_cfg.optimization_rounds, len(TRAIN_SEEDS))

    l_search = await run_proposer_search(
        l_cfg, l_dataset, agent, n_attempts=l_cfg.optimization_rounds, arm_name="LLM",
    )
    l_best_setting = (l_search["best"] or {}).get("setting", {})
    l_defaults = _coerce_setting(_default_setting(l_cfg, l_cfg.allow_arch_changes), l_cfg.allow_arch_changes)
    opt_log = _proposer_opt_log(l_search, l_defaults)

    # Hard-validation accounting (protocol report) + token cost.
    rs = getattr(agent, "retry_stats", {})
    logger.info(
        "LLM validity | first-try=%d  after-retry=%d  rejected=%d  repeats=%d  invalid-values=%d  timeouts=%d",
        rs.get("valid_first_try", 0), rs.get("valid_after_retry", 0), rs.get("rejected", 0),
        rs.get("repeats_proposed", 0), rs.get("invalid_values", 0), rs.get("llm_timeouts", 0),
    )
    ts = getattr(agent, "token_stats", {})
    logger.info("LLM token cost | total=%d (prompt=%d completion=%d) over %d call(s)",
                ts.get("total_tokens", 0), ts.get("prompt_tokens", 0),
                ts.get("completion_tokens", 0), ts.get("calls", 0))
    await agent.close()

    # Save the per-attempt transcript (proposals, diagnoses, validity outcomes).
    with open(config.output_dir / f"conversation_log_run{run_id}.json", "w") as f:
        json.dump(l_search["attempts"], f, indent=2, default=str)

    # Step 0 instrumentation: the protocol-path transcript (exact rendered
    # payload + raw replies per call) — the evidence base for the Q3
    # history-use analysis and the history_ablation placebo.
    agent.save_protocol_log(str(config.output_dir / f"protocol_log_run{run_id}.json"))

    # Test only AFTER selection (interim single-seed; final eval → 30 seeds).
    l_te = train_and_test_setting(l_cfg, l_dataset, l_best_setting, seed=TRAIN_SEEDS[0])
    l_metrics, l_eval = l_te["metrics"], l_te["evaluator"]
    l_preds, l_targets = l_te["preds"], l_te["targets"]
    l_eval.plot_training_history(l_te["history"], filename=f"llm_history_run{run_id}.png")
    l_eval.plot_predictions(l_preds, l_targets,        filename=f"llm_predictions_run{run_id}.png")
    l_eval.plot_error_distribution(l_preds, l_targets, filename=f"llm_errors_run{run_id}.png")

    results["llm"] = {
        "metrics":          l_metrics,
        "history":          l_te["history"],
        "optimization_log": opt_log,
        "conversation_log": l_search["attempts"],
        "best_setting":     l_best_setting,
        "search":           l_search,
        "validity_stats":   dict(rs),
    }
    logger.info("LLM RMSE=%.4f  R²=%.4f", l_metrics["rmse"], l_metrics["r2"])

    # ── 3. Random search (from-scratch engine: 25 settings × 3 trainings) ─────
    r_cfg = copy.deepcopy(config)
    r_dataset, _, _, _ = build_dataset_and_loaders(r_cfg)
    logger.info("--- Run %d | RANDOM SEARCH (attempts=%d × %d trainings) ---",
                run_id, r_cfg.optimization_rounds, len(TRAIN_SEEDS))
    r_search = run_random_search(
        r_cfg, r_dataset, n_attempts=r_cfg.optimization_rounds, opt_seed=r_cfg.seed,
    )
    r_best_setting = (r_search["best"] or {}).get("setting", {})

    # Test only AFTER selection (protocol). Interim single-seed eval to populate
    # the legacy metrics/plot slots; the final evaluation replaces this with the
    # 30-seed result + paired differences.
    r_te = train_and_test_setting(r_cfg, r_dataset, r_best_setting, seed=TRAIN_SEEDS[0])
    r_metrics, r_eval = r_te["metrics"], r_te["evaluator"]
    r_preds, r_targets = r_te["preds"], r_te["targets"]
    r_eval.plot_training_history(r_te["history"], filename=f"random_history_run{run_id}.png")
    r_eval.plot_predictions(r_preds, r_targets,        filename=f"random_predictions_run{run_id}.png")
    r_eval.plot_error_distribution(r_preds, r_targets, filename=f"random_errors_run{run_id}.png")

    results["random"] = {
        "metrics":      r_metrics,
        "history":      r_te["history"],
        "random_log":   r_search["attempts"],
        "best_curve":   r_search["best_curve"],
        "best_setting": r_best_setting,
        "search":       r_search,
    }
    logger.info("Random RMSE=%.4f  R²=%.4f", r_metrics["rmse"], r_metrics["r2"])

    # ── 4. Rule-based controller (from-scratch engine, same history context) ──
    rb_cfg = copy.deepcopy(config)
    rb_dataset, _, _, _ = build_dataset_and_loaders(rb_cfg)
    # C1 (metric-only) by default; C2 (motion-aware) when --motion-rule is set.
    # Both write to the `rule_based` slot so C1 vs C2 compare across two runs.
    if getattr(config, "motion_rule", False):
        rb_agent = MotionAwareRuleBasedOptimizer(allow_arch_changes=config.allow_arch_changes)
        logger.info("--- Run %d | RULE-BASED = C2 (motion-aware) (attempts=%d × %d) ---",
                    run_id, rb_cfg.optimization_rounds, len(TRAIN_SEEDS))
    else:
        rb_agent = RuleBasedOptimizer(
            allow_arch_changes=config.allow_arch_changes,
            payload_curves=getattr(config, "payload_curves", False),
        )
        logger.info("--- Run %d | RULE-BASED (attempts=%d × %d trainings) ---",
                    run_id, rb_cfg.optimization_rounds, len(TRAIN_SEEDS))

    rb_search = await run_proposer_search(
        rb_cfg, rb_dataset, rb_agent, n_attempts=rb_cfg.optimization_rounds, arm_name="Rule-Based",
    )
    rb_best_setting = (rb_search["best"] or {}).get("setting", {})
    rb_defaults = _coerce_setting(_default_setting(rb_cfg, rb_cfg.allow_arch_changes), rb_cfg.allow_arch_changes)
    rb_opt_log = _proposer_opt_log(rb_search, rb_defaults)
    await rb_agent.close()

    rb_te = train_and_test_setting(rb_cfg, rb_dataset, rb_best_setting, seed=TRAIN_SEEDS[0])
    rb_metrics, rb_eval = rb_te["metrics"], rb_te["evaluator"]
    rb_preds, rb_targets = rb_te["preds"], rb_te["targets"]
    rb_eval.plot_training_history(rb_te["history"], filename=f"rule_based_history_run{run_id}.png")
    rb_eval.plot_predictions(rb_preds, rb_targets,    filename=f"rule_based_predictions_run{run_id}.png")
    rb_eval.plot_error_distribution(rb_preds, rb_targets, filename=f"rule_based_errors_run{run_id}.png")

    with open(config.output_dir / f"optimization_log_rule_based_run{run_id}.json", "w") as f:
        json.dump(rb_opt_log, f, indent=2, default=str)

    results["rule_based"] = {
        "metrics":          rb_metrics,
        "history":          rb_te["history"],
        "optimization_log": rb_opt_log,
        "best_setting":     rb_best_setting,
        "search":           rb_search,
    }
    logger.info("Rule-Based RMSE=%.4f  R²=%.4f", rb_metrics["rmse"], rb_metrics["r2"])

    # ── 5. Optuna (TPE / Bayesian) search ──────────────────────────────────────
    # "Competent conventional optimizer" arm requested by the professor on
    # 2026-05-20. Skipped with a warning if optuna is not installed so
    # legacy environments keep running. Trial budget matches the other arms
    # (optimization_rounds x epochs_per_round).
    o_cfg = copy.deepcopy(config)
    o_dataset, _, _, _ = build_dataset_and_loaders(o_cfg)
    o_te = None
    logger.info("--- Run %d | OPTUNA (TPE) (attempts=%d × %d trainings) ---",
                run_id, o_cfg.optimization_rounds, len(TRAIN_SEEDS))
    try:
        o_search = run_optuna_search(
            o_cfg, o_dataset, n_attempts=o_cfg.optimization_rounds,
        )
        o_best_setting = (o_search["best"] or {}).get("setting", {})
        # Test only AFTER selection (interim single-seed; final eval → 30 seeds).
        o_te = train_and_test_setting(o_cfg, o_dataset, o_best_setting, seed=TRAIN_SEEDS[0])
        o_metrics, o_eval = o_te["metrics"], o_te["evaluator"]
        o_preds, o_targets = o_te["preds"], o_te["targets"]
        o_eval.plot_training_history(o_te["history"], filename=f"optuna_history_run{run_id}.png")
        o_eval.plot_predictions(o_preds, o_targets,        filename=f"optuna_predictions_run{run_id}.png")
        o_eval.plot_error_distribution(o_preds, o_targets, filename=f"optuna_errors_run{run_id}.png")

        results["optuna"] = {
            "metrics":       o_metrics,
            "history":       o_te["history"],
            "optuna_log":    o_search["attempts"],
            "best_curve":    o_search["best_curve"],
            "best_setting":  o_best_setting,
            "study_summary": o_search["study_summary"],
            "search":        o_search,
        }
        logger.info("Optuna RMSE=%.4f  R²=%.4f", o_metrics["rmse"], o_metrics["r2"])
    except ImportError:
        logger.warning(
            "Optuna arm skipped - `optuna` is not installed. "
            "Run `pip install optuna` to enable the Bayesian-optimization baseline."
        )
        results["optuna"] = None
    except Exception as exc:
        logger.exception("Optuna arm failed (%s); continuing without it.", exc)
        results["optuna"] = None

    # ── Per-run trajectory plot ───────────────────────────────────────────────
    trajectory_histories: Dict[str, Dict[str, List[float]]] = {
        "Baseline":      b_history,
        "LLM":           l_te["history"],
        "Random Search": r_te["history"],
        "Rule-Based":    rb_te["history"],
    }
    if results.get("optuna") is not None and o_te is not None:
        trajectory_histories["Optuna (TPE)"] = o_te["history"]
    plot_optimization_trajectories(
        config=config,
        histories=trajectory_histories,
        filename=f"trajectories_run{run_id}.png",
        title=f"Run {run_id} — Validation Loss Trajectories",
    )

    plot_hyperparameter_trajectory(
        config=config,
        opt_log=opt_log,
        filename=f"hyperparameter_trajectory_run{run_id}.png",
        title=f"Run {run_id} - Hyperparameter Trajectory",
    )
    plot_action_impact_scatter(
        config=config,
        opt_log=opt_log,
        filename=f"action_impact_scatter_run{run_id}.png",
        title=f"Run {run_id} - Action Magnitude vs Delta Best Val Loss",
    )

    log_path = config.output_dir / f"optimization_log_run{run_id}.json"
    with open(log_path, "w") as f:
        json.dump(opt_log, f, indent=2, default=str)

    # Protocol reports: per-setting metrics table + budget table.
    write_per_setting_report(config, results, run_id)
    write_budget_table(config, results, run_id)

    # ── Final evaluation (protocol point 8) ───────────────────────────────────
    # Take each method's single best setting and evaluate on a larger set of
    # FRESH, never-used seeds. This is the headline result; it replaces the
    # interim single-seed test numbers above.
    if getattr(config, "enable_final_eval", True):
        n_seeds = int(getattr(config, "n_final_eval_seeds", 30))
        fe_seeds = FINAL_EVAL_SEEDS_POOL[:n_seeds]
        allow_arch = bool(getattr(config, "allow_arch_changes", True))
        best_settings: Dict[str, Dict[str, Any]] = {
            "fixed_reference": _coerce_setting(_default_setting(config, allow_arch), allow_arch),
            "llm":        (results.get("llm") or {}).get("best_setting", {}),
            "random":     (results.get("random") or {}).get("best_setting", {}),
            "rule_based": (results.get("rule_based") or {}).get("best_setting", {}),
        }
        if isinstance(results.get("optuna"), dict):
            best_settings["optuna"] = results["optuna"].get("best_setting", {})

        logger.info("--- Run %d | FINAL EVALUATION (%d fresh seeds) ---", run_id, len(fe_seeds))
        final_eval = run_final_evaluation(config, best_settings, fe_seeds)
        write_final_eval_report(config, final_eval, run_id)
        results["final_eval"] = final_eval

        # Promote the 30-seed means into each arm's headline metrics so the
        # cross-run summary reflects the final result, keeping the interim
        # single-seed numbers under `interim_metrics`.
        arm_map = {"fixed_reference": "baseline", "llm": "llm", "random": "random",
                   "rule_based": "rule_based", "optuna": "optuna"}
        for fe_arm, agg in final_eval["per_arm"].items():
            slot_name = arm_map.get(fe_arm)
            slot = results.get(slot_name)
            if isinstance(slot, dict):
                slot["interim_metrics"] = slot.get("metrics", {})
                slot["metrics"] = {
                    **slot.get("metrics", {}),
                    "rmse": agg["mean_rmse"],
                    "rmse_std": agg["std_rmse"],
                    "r2":   agg["mean_r2"],
                }
                slot["final_eval"] = agg

    return results
