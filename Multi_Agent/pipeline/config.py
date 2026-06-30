"""Experiment configuration — the single source of truth for all run settings."""

from dataclasses import dataclass
from pathlib import Path

import torch

from .datasets import get_dataset_spec
from .logging_setup import logger
from .search_space import HP_BOUNDS, HP_GRID


@dataclass
class Config:
    """Single source of truth for all experiment settings."""

    # Data
    csv_path: str = "preprocessed-RadarEXP1(in).csv"
    window_size: int = 12
    target_columns: int = 2
    hz: float = 4.0

    # Model (LSTM)
    lstm_hidden: int = 64
    lstm_layers: int = 2
    dropout: float = 0.3

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100              # protocol: max 100 epochs per training
    weight_decay: float = 1e-3

    # Early stopping is ON for all arms (protocol). `early_stopping_patience`
    # is the fixed-reference baseline's patience and the fallback when a
    # setting does not specify one; the searchable `patience` HP (8/12/16)
    # overrides it per setting. Same stop rule for every arm: monitor
    # validation loss, stop after `patience` non-improving epochs, restore the
    # best-epoch weights.
    early_stopping_patience: int = 12
    enable_early_stopping: bool = True
    optimizer_choice: str = "adam"
    enable_warm_start: bool = False

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    strict_prediction_checks: bool = True

    # Output. Default lives under outputs/ so the Docker volume mount
    # (./outputs:/app/outputs) persists results to the host without needing
    # an explicit --output flag.
    output_dir: Path = Path("outputs") / "default-run"

    # Optimisation
    optimization_rounds: int = 25      # protocol: 25 attempts per method
    epochs_per_round: int = 100        # protocol: max 100 epochs per training
    num_experiment_runs: int = 1
    allow_arch_changes: bool = True

    # Final evaluation (protocol point 8): after each method picks its single
    # best setting, evaluate that setting on a larger set of FRESH, never-used
    # seeds and report mean/std/best/worst, paired differences vs the LLM.
    enable_final_eval: bool = True
    n_final_eval_seeds: int = 30

    # `enable_motion` is the master switch for the interpretable motion feature:
    # when False the motion-aware LLM prompt and diagnostic payload are turned
    # off (used for ablation A/B vs the motion-on run).
    enable_motion: bool = True

    # `motion_rule` swaps the deterministic rule-based arm from C1 (metric-only
    # RuleBasedOptimizer) to C2 (MotionAwareRuleBasedOptimizer), which sets the
    # loss-shaping levers from fixed motion heuristics. Used to A/B the LLM's
    # motion reasoning against a fixed motion heuristic.
    motion_rule: bool = False

    # `semantic_repair` is the master switch for the validator's silent
    # auto-corrections (clamping out-of-range HP values, snapping discrete
    # choices, stripping unknown HP keys, flipping the reset flag, relabeling
    # contradictory diagnoses). The protocol's MAIN LLM run is HARD validation,
    # so this defaults to False: every would-be fix is reported as a constraint
    # violation and the round retries / skips, exposing the raw LLM as an
    # optimizer. Set True (via --semantic-repair) only for the SEPARATE
    # repair-on arm, which must be reported on its own.
    semantic_repair: bool = False

    # `payload_curves` is the Q2 lever: when True, every tried setting's
    # qualitative summary gains a per-epoch TRAINING CURVE shape label
    # (still improving / overfitting upturn / unstable-noisy / clean plateau)
    # derived from its validation trajectory, and the curve-aware rule-based
    # controller diagnoses from that shape. Tests whether per-epoch curve
    # information (vs best-epoch aggregates only) improves the LLM or the
    # rule-based arm. Affects only the LLM and rule-based arms.
    payload_curves: bool = False

    # `explore_prompt` is the Q4 prompt-variant lever (LLM arm only): when True,
    # the system prompt drops the "propose a SMALL change vs the ANCHOR"
    # instruction in favour of broad exploration of untried regions. Tests
    # whether the anchoring instruction is what caps the LLM's search-grid
    # coverage (vs random/Optuna ~100%).
    explore_prompt: bool = False

    # `history_ablation` is the Q3 history-use placebo probe. It perturbs the
    # RENDERED history context the LLM proposer sees (NOT the engine's training)
    # to test whether the LLM actually uses history or mostly ignores it:
    #   "none"     — real history (default, normal run)
    #   "shuffled" — each setting's outcome is reassigned to another setting's,
    #                so the BEST-SETTINGS ranking / LAST-ATTEMPTS trend /
    #                OBSERVED-PATTERNS signal is misleading (format unchanged)
    #   "empty"    — no history shown at all (only the ANCHOR)
    # If proposals do not degrade under shuffled/empty vs none, the LLM is
    # ignoring history. Only affects the LLM arm.
    history_ablation: str = "none"

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        spec = get_dataset_spec(self.csv_path)
        if spec is not None:
            self.hz = float(spec["hz"])
            self.window_size = int(round(spec["hz"] * spec["window_seconds"]))

        # ── Fair-comparison fix: baseline window must be reachable by the search ──
        # The fixed-reference baseline trains at the dataset's spec window_size
        # (radar=12, cap=15, IR=5), but the protocol search grid is a fixed list
        # ([10,20,30,40,50]) that contains NONE of those. If the baseline's window
        # is not in the grid, no search arm (random / Optuna / LLM / rule-based)
        # can ever draw it, so none can reproduce — let alone beat — the baseline
        # config. That made every arm lose to the baseline. Inject the baseline
        # window into the shared grid so all arms and the baseline draw from one
        # reachable search space. HP_GRID is the single source of truth read live
        # by every arm/sampler/validator, so updating it once here (at Config
        # construction, before any arm runs) keeps them all aligned.
        if int(self.window_size) not in HP_GRID["window_size"]:
            HP_GRID["window_size"] = sorted(
                set(HP_GRID["window_size"]) | {int(self.window_size)}
            )
            HP_BOUNDS["window_size"] = (
                min(HP_GRID["window_size"]), max(HP_GRID["window_size"]),
            )
            logger.info(
                "Search grid: injected baseline window_size=%d -> window_size grid now %s",
                int(self.window_size), HP_GRID["window_size"],
            )
