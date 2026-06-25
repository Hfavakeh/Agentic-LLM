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
    cfg = Config()

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
    if args.data is not None:
        cfg.csv_path = args.data
    if getattr(args, "final_eval_seeds", None) is not None:
        cfg.n_final_eval_seeds = args.final_eval_seeds
    if getattr(args, "no_final_eval", False):
        cfg.enable_final_eval = False

    return cfg


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

    if use_multi_seed:
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
