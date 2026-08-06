"""
main.py
=======
Thin CLI entry point for the multi-agent experiment.

The actual logic lives in the packages:
  pipeline/     — config, data, model, trainer, evaluator (training stack)
  agents/       — LLM + rule-based proposers, parsing, validation, prompts
  search/       — from-scratch engine + random / Optuna / proposer search arms
  reporting/    — plots, protocol reports, cross-run metrics
  experiments/  — per-run orchestration, final evaluation, multi-seed runner

Flow
----
For each run:
  1. Baseline       : train with fixed defaults, evaluate on test set.
  2. LLM controller : 25 attempts of proposer-driven search (3 trainings each).
  3. Random search  : same budget, uniform draws from the shared grid.
  4. Rule-based     : same budget, deterministic controller.
  5. Optuna (TPE)   : same budget, Bayesian sampler.
  6. Reports + final evaluation of each arm's best setting on fresh seeds.

Usage
-----
# Run the full experiment
python main.py

# Quick smoke test: 1 run, 2 rounds, 5 epochs
python main.py --smoke-test

# Override key settings without editing Config
python main.py --rounds 10 --runs 3 --epochs 50 --model qwen3:8b
"""

import argparse
import asyncio
from pathlib import Path
from typing import List

from experiments import EXPERIMENT_SEEDS, run_full_experiment, run_multi_seed_experiment
from experiments.final_eval import FINAL_EVAL_SEEDS_POOL
from pipeline import Config, logger


def parse_args():
    p = argparse.ArgumentParser(
        description="LSTM indoor localisation single-agent optimisation experiment"
    )

    p.add_argument("--model",  default="minimax-m2.5:cloud",
                   help="Ollama model tag for the LLM agent (default: minimax-m2.5:cloud)")

    p.add_argument("--rounds",  type=int, default=None,
                   help="Optimisation rounds per run (overrides Config)")
    p.add_argument("--runs",    type=int, default=None,
                   help="Number of experiment runs (overrides Config)")
    p.add_argument("--epochs",  type=int, default=None,
                   help="Training epochs per round (overrides Config)")
    p.add_argument("--data",    default=None,
                   help="Path to CSV dataset (overrides Config)")
    p.add_argument("--output",  default=None,
                   help="Output directory (overrides Config)")
    p.add_argument("--final-eval-seeds", type=int, default=None, dest="final_eval_seeds",
                   help="Number of fresh seeds for the final evaluation (overrides Config, default 30)")
    p.add_argument("--no-final-eval", action="store_true", dest="no_final_eval",
                   help="Skip the final-evaluation stage (search + reports only)")

    p.add_argument("--smoke-test", action="store_true",
                   help="Quick run: 1 experiment, 2 rounds, 5 epochs each")

    # ── Multi-seed control ────────────────────────────────────────────────
    p.add_argument("--seeds", type=int, nargs="+", default=None, metavar="SEED",
                   help=(
                       "List of random seeds for multi-seed replication. "
                       f"Defaults to EXPERIMENT_SEEDS={EXPERIMENT_SEEDS}. "
                       "Pass a single seed to run just that seed without the "
                       "multi-seed runner (e.g. --seeds 42). "
                       "Example: --seeds 42 123 456"
                   ))
    p.add_argument("--single-seed", action="store_true",
                   help=(
                       "Disable the multi-seed runner and run only the first seed "
                       "(or the value of --seeds if a single seed is given). "
                       "Equivalent to the pre-multi-seed behaviour."
                   ))

    p.add_argument(
        "--no-arch-changes", action="store_true", dest="no_arch_changes",
        help=(
            "Arch-free mode: prevent the LLM from proposing lstm_hidden or lstm_layers "
            "changes. Useful to isolate the effect of architectural search. "
            "When set, those parameters are stripped from every proposal and the "
            "system prompt is updated so the LLM knows they are off-limits."
        ),
    )

    p.add_argument(
        "--no-motion", action="store_true", dest="no_motion",
        help=(
            "Disable the interpretable motion feature for an ablation run: "
            "(a) turn off motion-weighted MSE (plain MSE instead), "
            "(b) skip the per-round motion diagnostic computation, "
            "(c) drop the motion-aware sections from the LLM system prompt so "
            "the agent has no awareness of motion regimes. "
            "Pair with the same --seeds / --model as the motion-on run to A/B."
        ),
    )

    p.add_argument(
        "--semantic-repair", action="store_true", dest="semantic_repair",
        help=(
            "Turn the validator's silent auto-corrections back ON for the LLM "
            "arm. The protocol's MAIN run is HARD validation (repair OFF by "
            "default): invalid LLM output is rejected, retried once, then the "
            "round trains with no HP change. This flag enables the SEPARATE "
            "'LLM + semantic repair' arm, which must be reported on its own. "
            "Pair with the same --seeds / --model as the hard run to A/B."
        ),
    )

    p.add_argument(
        "--payload-motion", action="store_true", dest="payload_motion",
        help=(
            "Q5: add a qualitative MOTION PROFILE (pace, fast bursts, stop-go) "
            "and per-setting per-regime (slow/med/fast) validation-error labels "
            "to the LLM payload; the engine computes the per-regime error only "
            "when this is on. Search space unchanged. Tests whether motion-aware "
            "summaries change the LLM's decisions / per-regime errors. A/B "
            "against the no-motion run on the same --model / --seeds."
        ),
    )

    p.add_argument(
        "--explore-prompt", action="store_true", dest="explore_prompt",
        help=(
            "Q4 prompt-variant (LLM arm only): replace the 'propose a SMALL "
            "change vs the ANCHOR' instruction with one that asks the LLM to "
            "explore untried regions (change several HPs, make large moves). "
            "Tests whether the anchoring instruction is what caps the LLM's "
            "search-grid coverage. A/B against the default-prompt run on the "
            "same --model / --seeds."
        ),
    )

    p.add_argument(
        "--motion-experiment", action="store_true", dest="motion_experiment",
        help=(
            "Run the MOTION-THESIS experiment instead of the 9-HP bake-off: the "
            "9 conventional HPs are frozen at baseline and the arms search the "
            "six loss-shaping levers (v_max, lambda_vel/smooth, bin weights) from "
            "motion summaries. Arms: baseline (plain MSE), C2 (motion heuristic), "
            "random, C3 (LLM). Tests whether the LLM's motion interpretation "
            "beats a fixed motion heuristic (C3 vs C2) and whether motion "
            "loss-shaping helps at all (C2 vs baseline)."
        ),
    )

    p.add_argument(
        "--motion-arms", nargs="+", dest="motion_arms",
        choices=["baseline", "motion_rule", "random", "llm"], default=["llm"],
        help=(
            "Which motion-experiment arms to run (default: llm only). The "
            "baseline (plain MSE, HPs frozen) equals the HP bake-off's baseline "
            "so it never needs recomputing; add 'motion_rule' for the C2 "
            "heuristic comparator (cheap, one deterministic vector) when you need "
            "the C3-vs-C2 comparison. Only used with --motion-experiment."
        ),
    )

    p.add_argument(
        "--motion-no-profile", action="store_true", dest="motion_no_profile",
        help=(
            "Motion control ablation (Email-7 point 5): the LLM loss-shaping arm "
            "sees the per-regime error feedback but NOT the motion-summary block "
            "(speed / turning / stop-go interpretation). Isolates whether the "
            "motion interpretation adds anything over the raw error signal. Use "
            "with --motion-experiment --motion-arms llm."
        ),
    )

    p.add_argument(
        "--opro-prompt", action="store_true", dest="opro_prompt",
        help=(
            "OPRO prompt-variant (LLM arm only; Email-5 diagnostic): replace the "
            "qualitative-label payload + small-delta-vs-anchor framing with the "
            "OPRO recipe (Yang et al., 2024) — show past (setting, NUMERIC score) "
            "pairs sorted worst→best and ask the LLM for a COMPLETE new setting "
            "expected to score lower than all of them. Tests whether SoTA "
            "optimizer-prompting (raw candidates + scores) beats the protocol's "
            "label prompt. Overrides --explore-prompt/--payload-curves/"
            "--payload-motion for the LLM arm. A/B against the default-prompt run "
            "on the same --model / --seeds."
        ),
    )

    p.add_argument(
        "--payload-curves", action="store_true", dest="payload_curves",
        help=(
            "Q2: add a per-epoch TRAINING CURVE shape label (still improving / "
            "overfitting upturn / unstable-noisy / clean plateau) to each tried "
            "setting's qualitative summary, and let the rule-based controller "
            "diagnose from that shape. Tests whether per-epoch curve information "
            "(vs best-epoch aggregates only) improves the LLM or rule-based arm. "
            "Run with and without this flag, same --model / --seeds, to A/B."
        ),
    )

    p.add_argument(
        "--llm-timeout", type=float, default=None, dest="llm_timeout",
        metavar="SECONDS",
        help=(
            "Seconds to wait for one LLM call before abandoning it (default "
            "300). A timed-out call is retried once and the attempt is then "
            "rejected without training, so a ceiling below the server's real "
            "latency starves the LLM arm while every other arm keeps its full "
            "25 attempts — the equal-budget comparison silently stops holding. "
            "Raise it for large models or a CPU-only server: gemma4:12b needed "
            "~170-300 s per call on the 2026-07-26 run and lost 57% of its "
            "attempts to the 300 s default."
        ),
    )

    p.add_argument(
        "--payload-repr", default="labels", dest="payload_repr",
        choices=["labels", "numeric", "numeric_ci", "ranks"],
        help=(
            "Email-8 representation arm (LLM arm only): how each past outcome is "
            "rendered in the history the LLM reads. 'labels' (default) is the "
            "protocol's qualitative summary — unchanged, so earlier runs stay "
            "comparable. 'numeric' shows the measured mean validation RMSE in "
            "metres, best epoch and gap. 'numeric_ci' adds the spread across the "
            "3 trainings and the 3 individual seed scores, so a negligible "
            "difference can be told from a substantial one. 'ranks' shows the "
            "ordering only (#1 of N), no magnitudes. The system prompt is kept "
            "in lock-step with whichever is chosen. A/B on the same --model / "
            "--seeds to test whether the label rendering, not the LLM, is what "
            "loses the parameter-outcome relationship."
        ),
    )

    p.add_argument(
        "--no-anchor", action="store_true", dest="no_anchor",
        help=(
            "Email-8 anchor ablation (LLM arm only): remove the best-so-far "
            "ANCHOR block from the payload, testing whether it encourages "
            "imitation rather than inference. With no anchor there is nothing to "
            "take a delta against, so proposals must name EVERY hyperparameter; "
            "incomplete ones are rejected rather than silently completed from the "
            "hidden anchor (which would leave the arm anchored after all). The "
            "engine's true best-so-far is unaffected."
        ),
    )

    p.add_argument(
        "--history-window", type=int, default=None, dest="history_window",
        metavar="N",
        help=(
            "Email-8 recent-vs-full history arm (LLM arm only): show OUTCOMES "
            "for only the last N attempts instead of all of them. The "
            "already-tried list stays complete, so this measures whether old "
            "outcomes carry usable knowledge without also imposing a repeat "
            "penalty. Try 5 against the default (all 25)."
        ),
    )

    p.add_argument(
        "--no-auto-conclusions", action="store_true", dest="no_auto_conclusions",
        help=(
            "Email-8 conclusions-vs-observations arm (LLM arm only): drop the "
            "automatically generated interpretations — the mined OBSERVED "
            "PATTERNS block, the per-setting behavior labels, and the "
            "improved/worsened trend arrows — leaving the observations "
            "themselves. Tests whether the pipeline's own conclusions help or "
            "replace the LLM's reasoning."
        ),
    )

    p.add_argument(
        "--history-ablation", default="none", dest="history_ablation",
        choices=["none", "shuffled", "empty"],
        help=(
            "Q3 history-use placebo for the LLM arm. Perturbs the RENDERED "
            "history context the LLM sees (NOT the engine training or the true "
            "anchor): 'shuffled' reassigns each setting's outcome to another "
            "setting's so the best-settings ranking / trend / patterns are "
            "misleading; 'empty' shows no prior attempts. If proposals do not "
            "degrade vs 'none', the LLM is ignoring history. Pair with the same "
            "--seeds / --model as the 'none' run to A/B."
        ),
    )

    p.add_argument(
        "--motion-rule", action="store_true", dest="motion_rule",
        help=(
            "Swap the deterministic rule-based arm from C1 (metric-only) to "
            "C2 (motion-aware): C2 sets the loss-shaping levers (v_max, "
            "lambda_vel/smooth, bin weights) from fixed heuristics on the "
            "dataset motion profile. The arm still writes to the `rule_based` "
            "output slot, so C1 vs C2 compare across two runs on the same "
            "seeds. Use to A/B the LLM's motion reasoning against a fixed "
            "motion heuristic."
        ),
    )

    return p.parse_args()


def build_config(args) -> Config:
    # The dataset must be chosen at CONSTRUCTION, not patched on afterwards.
    # `Config.__post_init__` is what reads DATASET_SPECS and derives `hz` /
    # `window_size` (and injects the baseline window into HP_GRID); assigning
    # `csv_path` to an already-built Config leaves all of that at the radar
    # default, so a `--data` run would silently train IR at hz=4 / window=12
    # instead of its spec's hz=5 / window=5. Passing it here also puts the
    # dataset in place before the --smoke-test early return below.
    cfg = Config(csv_path=args.data) if getattr(args, "data", None) else Config()

    # Apply arch-change toggle early so it is honoured in every code path.
    cfg.allow_arch_changes = not getattr(args, "no_arch_changes", False)

    # Master switch for the interpretable motion feature.
    if getattr(args, "no_motion", False):
        cfg.enable_motion = False
        logger.info("Motion feature DISABLED (ablation run): no motion diagnostics, motion-agnostic prompt.")

    # Hard validation is the default main run (cfg.semantic_repair=False).
    # --semantic-repair opts into the separate repair-on arm.
    if getattr(args, "semantic_repair", False):
        cfg.semantic_repair = True
        logger.info(
            "Semantic repair ENABLED (separate repair-on arm): the validator "
            "silently fixes out-of-range / discrete / diagnosis / reset-flag "
            "issues. Report this arm separately from the hard-validation main run."
        )
    else:
        logger.info(
            "Semantic repair OFF (protocol main run): invalid LLM output is "
            "rejected, retried once, then the round trains with no HP change."
        )

    # Q5: motion-aware summaries (motion profile + per-regime error) in the
    # LLM payload.
    cfg.payload_motion = bool(getattr(args, "payload_motion", False))
    if cfg.payload_motion:
        logger.info(
            "Payload motion ENABLED (Q5): motion profile + per-regime error "
            "labels added to the LLM payload; engine computes per-regime val error."
        )

    # Q4 prompt-variant: exploration-oriented system prompt (LLM arm).
    cfg.explore_prompt = bool(getattr(args, "explore_prompt", False))
    if cfg.explore_prompt:
        logger.info(
            "Explore prompt ENABLED (Q4 variant): the LLM is asked to explore "
            "untried regions instead of a small change vs the anchor."
        )

    # Motion-thesis experiment: freeze HPs, search loss-shaping levers from
    # motion summaries. Implies the motion payload (per-regime error) is on.
    cfg.motion_experiment = bool(getattr(args, "motion_experiment", False))
    if cfg.motion_experiment:
        cfg.payload_motion = True
        logger.info(
            "MOTION EXPERIMENT enabled: 9 HPs frozen at baseline; arms search the "
            "6 loss-shaping levers (baseline / C2 heuristic / random / C3 LLM)."
        )

    # Email-7 point 5 control: LLM sees per-regime error only, no motion summary.
    cfg.motion_no_profile = bool(getattr(args, "motion_no_profile", False))
    if cfg.motion_no_profile:
        logger.info(
            "MOTION NO-PROFILE ablation: the LLM loss-shaping arm sees per-regime "
            "error feedback but NOT the motion-summary block."
        )

    # OPRO prompt-variant (Email-5): raw (setting, score) trajectory + ask for a
    # full new candidate. Overrides explore_prompt for the LLM arm.
    cfg.opro_prompt = bool(getattr(args, "opro_prompt", False))
    if cfg.opro_prompt:
        logger.info(
            "OPRO prompt ENABLED (Email-5 variant): the LLM sees past settings "
            "with their NUMERIC scores (sorted worst→best) and proposes a "
            "complete new setting expected to score lower."
        )
        if cfg.explore_prompt:
            logger.warning("--opro-prompt overrides --explore-prompt for the LLM arm.")

    # Q2: per-epoch training-curve summaries in the payload + curve-aware
    # rule-based diagnosis.
    cfg.payload_curves = bool(getattr(args, "payload_curves", False))
    if cfg.payload_curves:
        logger.info(
            "Payload curves ENABLED (Q2): per-epoch training-curve shape labels "
            "added to the LLM payload and the rule-based diagnosis."
        )

    # Per-call LLM timeout. Kept visible in the header log because a run whose
    # ceiling is too low still "succeeds" — it just quietly trains fewer
    # settings for the LLM arm than for every other arm.
    if getattr(args, "llm_timeout", None) is not None:
        if args.llm_timeout <= 0:
            raise SystemExit("--llm-timeout must be positive")
        cfg.llm_timeout_s = float(args.llm_timeout)
        logger.info(
            "LLM call timeout = %.0fs (default 300s). A timed-out call is "
            "retried once, then the attempt is rejected without training.",
            cfg.llm_timeout_s,
        )

    # Email-8 representation arms (LLM arm). Each changes ONE property of the
    # rendered history; the defaults reproduce the protocol payload exactly.
    cfg.payload_repr = getattr(args, "payload_repr", "labels")
    cfg.payload_anchor = not bool(getattr(args, "no_anchor", False))
    cfg.history_window = getattr(args, "history_window", None)
    cfg.payload_auto_conclusions = not bool(getattr(args, "no_auto_conclusions", False))
    if cfg.payload_repr != "labels":
        logger.info(
            "Payload representation = %s (Email-8): past outcomes are rendered as "
            "%s instead of qualitative labels.", cfg.payload_repr,
            {"numeric": "measured numbers",
             "numeric_ci": "measured numbers with the 3-training spread",
             "ranks": "rank order only"}[cfg.payload_repr],
        )
    if not cfg.payload_anchor:
        logger.info(
            "ANCHOR block REMOVED from the payload (Email-8): the LLM must "
            "propose COMPLETE settings; incomplete proposals are rejected so the "
            "arm cannot be re-anchored by the validator's merge."
        )
    if cfg.history_window:
        logger.info(
            "History window = %d (Email-8): outcomes shown for the last %d "
            "attempts only; the already-tried list stays complete.",
            cfg.history_window, cfg.history_window,
        )
    if not cfg.payload_auto_conclusions:
        logger.info(
            "Auto-conclusions REMOVED from the payload (Email-8): no mined "
            "patterns, behavior labels or trend arrows — observations only."
        )
    if cfg.opro_prompt and (cfg.payload_repr != "labels" or not cfg.payload_anchor
                            or cfg.history_window or not cfg.payload_auto_conclusions):
        logger.warning(
            "--opro-prompt uses its own payload and OVERRIDES the Email-8 "
            "representation flags for the LLM arm."
        )

    # Q3 history-use placebo (LLM arm). Default "none" = normal run.
    cfg.history_ablation = getattr(args, "history_ablation", "none")
    if cfg.history_ablation != "none":
        logger.info(
            "History ablation = %s (Q3 placebo): the LLM's RENDERED history is "
            "perturbed to test whether it uses history. Engine training and the "
            "true best-so-far anchor are unaffected.", cfg.history_ablation,
        )

    # Swap the rule-based arm to the motion-aware controller (C2).
    if getattr(args, "motion_rule", False):
        cfg.motion_rule = True
        logger.info(
            "Rule-based arm = C2 (motion-aware): loss-shaping levers set from "
            "fixed motion heuristics instead of metric-only HP rules."
        )

    # Resolve and create the output dir up front so every path (default,
    # smoke-test, --single-seed) writes into an existing directory.
    if args.output is not None:
        cfg.output_dir = Path(args.output)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        cfg.num_experiment_runs = 1
        cfg.optimization_rounds = 2
        cfg.epochs_per_round    = 5
        cfg.epochs              = 5
        cfg.n_final_eval_seeds  = 2     # keep the final-eval stage quick in smoke
        logger.info("Smoke-test mode: 1 run × 2 rounds × 5 epochs × 2 final-eval seeds")
        return cfg

    # Use `is not None` so that 0 is a valid explicit override.
    if args.rounds is not None:
        cfg.optimization_rounds = args.rounds
    if args.runs is not None:
        cfg.num_experiment_runs = args.runs
    if args.epochs is not None:
        cfg.epochs_per_round = args.epochs
        cfg.epochs           = args.epochs
    # NOTE: --data is applied at construction (see the top of this function), not
    # here — reassigning csv_path at this point would not re-derive hz/window_size.
    if getattr(args, "final_eval_seeds", None) is not None:
        cfg.n_final_eval_seeds = args.final_eval_seeds
    if getattr(args, "no_final_eval", False):
        cfg.enable_final_eval = False

    return cfg


async def run_motion_over_seeds(config: Config, seeds: List[int], llm_model: str,
                                arms_to_run=("llm",)):
    """Motion-thesis experiment across seeds: per seed, build a fresh dataset +
    LLM agent, run the loss-shaping bake-off, and write per-seed results."""
    import copy as _copy

    from arms.engine import build_dataset_and_loaders
    from arms.llm import SingleAgentOptimizer
    from arms.motion import run_motion_experiment, save_motion_results

    n_final = int(getattr(config, "n_final_eval_seeds", 30))
    fe_seeds = list(FINAL_EVAL_SEEDS_POOL[:n_final])
    for seed in seeds:
        logger.info("##### MOTION experiment | seed %d #####", seed)
        cfg = _copy.deepcopy(config)
        cfg.output_dir = Path(config.output_dir) / f"seed_{seed}"
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        dataset, *_ = build_dataset_and_loaders(cfg)
        agent = SingleAgentOptimizer(
            model_name=llm_model,
            allow_arch_changes=cfg.allow_arch_changes,
            semantic_repair=getattr(cfg, "semantic_repair", False),
            llm_timeout_s=getattr(cfg, "llm_timeout_s", 300.0),
            payload_motion=True,
            motion_show_profile=not getattr(cfg, "motion_no_profile", False),
        )
        try:
            results = await run_motion_experiment(
                cfg, dataset, agent,
                n_attempts=cfg.optimization_rounds,
                final_eval_seeds=fe_seeds,
                run_id=1,
                arms_to_run=tuple(arms_to_run),
                search_seed=seed,
            )
            save_motion_results(results, cfg.output_dir, run_id=1)
            agent.save_protocol_log(str(cfg.output_dir / "motion_protocol_log_run1.json"))
        finally:
            await agent.close()


async def main():
    args   = parse_args()
    config = build_config(args)

    # ── Resolve seed list ─────────────────────────────────────────────────
    seeds: List[int] = args.seeds if args.seeds is not None else EXPERIMENT_SEEDS

    # In smoke-test or single-seed mode, only run the first seed directly
    # without the multi-seed runner overhead.
    use_multi_seed = not args.single_seed and not args.smoke_test and len(seeds) > 1

    logger.info("==" * 30)
    logger.info("LSTM Single-Agent Optimisation Experiment")
    logger.info("  agent model    : %s", args.model)
    logger.info("  runs           : %d", config.num_experiment_runs)
    logger.info("  rounds/run     : %d", config.optimization_rounds)
    logger.info("  epochs/round   : %d", config.epochs_per_round)
    logger.info("  total epochs   : %d (same for all algorithms)",
                config.epochs_per_round * config.optimization_rounds)
    logger.info("  output dir     : %s", config.output_dir)
    logger.info("  allow_arch_changes : %s", config.allow_arch_changes)
    if use_multi_seed:
        logger.info("  multi-seed     : %s (%d seeds)", seeds, len(seeds))
    else:
        logger.info("  seed           : %d (single-seed mode)", seeds[0])
    logger.info("==" * 30)

    if getattr(config, "motion_experiment", False):
        # ── Motion-thesis experiment (loss-shaping levers, HPs frozen) ────
        await run_motion_over_seeds(
            config, seeds, args.model,
            arms_to_run=getattr(args, "motion_arms", ["llm"]),
        )
    elif use_multi_seed:
        # ── Multi-Seed Runner ────────────────────────────────────────────
        await run_multi_seed_experiment(
            base_config=config,
            seeds=seeds,
            llm_model=args.model,
        )
    else:
        # ── Single-seed path ─────────────────────────────────────────────
        await run_full_experiment(
            config=config,
            llm_model=args.model,
            seed=seeds[0],
        )


if __name__ == "__main__":
    asyncio.run(main())
