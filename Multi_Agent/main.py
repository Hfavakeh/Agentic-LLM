"""
main.py
=======
Single entry point for the multi-agent experiment.

Flow
----
For each run:
  1. Baseline  : train with fixed defaults, evaluate on test set.
  2. LLM/Multi-agent:
       a. Read completed baseline results, propose initial HP changes
       b. Apply changes and train
       c. Apply changes, train again  (repeated for N rounds)
       d. Evaluate on test set.
  3. Random search: same number of rounds, random HP each time.
  4. Compare and log.

Usage
-----
# Run the full multi-agent experiment (Analyst + Strategist)
python main.py

# Quick smoke test: 1 run, 2 rounds, 5 epochs
python main.py --smoke-test

# Override key settings without editing Config
python main.py --rounds 10 --runs 3 --epochs 50 --model qwen3:8b
"""

import argparse
import asyncio
import copy
import csv
import json
import numbers
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model_pipeline import (
    LSTM_Localizer, Config, DataProcessor, HP_BOUNDS, HP_GRID, OPTIMIZER_CHOICES,
    LOSS_SHAPING_KEYS, Evaluator, TimeSeriesDataset, Trainer,
    is_finite_number, logger, safe_trapz, set_seed,
)
from Agent import (
    SingleAgentOptimizer, OptimizerTools, RuleBasedOptimizer,
    MotionAwareRuleBasedOptimizer,
)
import pareto
from motion_descriptors import (
    error_payload as motion_error_payload,
    extract_motion_features, summarize_motion, llm_payload as motion_llm_payload,
)


# ---------------------------------------------------------------------------
# 1. SEED CONTROL
# ---------------------------------------------------------------------------

# 30 fixed, distinct, non-sequential seeds. Hard-coded (not randomly sampled)
# so every run of the experiment is reproducible.
EXPERIMENT_SEEDS: List[int] = [
    17, 42, 73, 128, 256, 314, 451, 512, 666, 777,
   
]

# Protocol seeds. The same three fixed seeds train EVERY setting in EVERY arm,
# so the only thing that differs between attempts/arms is the setting itself.
# The optimizer's own randomness (random-search RNG, Optuna sampler) uses a
# SEPARATE seed and must not be confused with these.
TRAIN_SEEDS: tuple = (101, 102, 103)

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(dataset: Dict, batch_size: int):
    """Build train / val / test DataLoaders from the dataset dict."""
    def _loader(X, y, prev_y, prev_prev_y, shuffle):
        return DataLoader(
            TimeSeriesDataset(X, y, prev_y=prev_y, prev_prev_y=prev_prev_y),
            batch_size=batch_size,
            shuffle=shuffle,
        )
    return (
        _loader(
            dataset["X_train"], dataset["y_train"],
            dataset.get("prev_y_train"),
            dataset.get("prev_prev_y_train"),
            shuffle=True,
        ),
        _loader(
            dataset["X_val"], dataset["y_val"],
            dataset.get("prev_y_val"),
            dataset.get("prev_prev_y_val"),
            shuffle=False,
        ),
        _loader(
            dataset["X_test"], dataset["y_test"],
            dataset.get("prev_y_test"),
            dataset.get("prev_prev_y_test"),
            shuffle=False,
        ),
    )


def build_dataset_and_loaders(config: Config):
    """Create an isolated dataset and loaders bundle for one arm/run."""
    dataset = DataProcessor.build_dataset(config)
    # Motion profile — speed distribution (mean/std/median/IQR/p95/extremes)
    # and dwell/stop-go behaviour of the dataset's trajectory. Computed once
    # (the trajectory is fixed) and surfaced in every round's context so a
    # controller can ground its loss-shaping decisions (v_max, bin weights)
    # in the data's actual motion regime rather than guessing.
    try:
        feats = extract_motion_features(dataset["raw_y_train"], hz=config.hz)
        dataset["motion_profile"] = motion_llm_payload(
            summarize_motion(feats, hz=config.hz)
        )
    except Exception as exc:
        logger.warning("Motion profile computation failed: %s", exc)
        dataset["motion_profile"] = {}
    train_loader, val_loader, test_loader = make_loaders(dataset, config.batch_size)
    return dataset, train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Setting evaluator — the protocol's core unit of work
# ---------------------------------------------------------------------------
#
# One "attempt" in any arm = one HP setting trained `len(TRAIN_SEEDS)` times
# FROM SCRATCH (seeds 101/102/103), with the per-setting score defined as the
# MEAN validation RMSE over those trainings. The three trainings reduce noise
# so the optimizer (random / Optuna / rule-based / LLM) decides on the average,
# not on one lucky/unlucky run. The test set is never touched here.

# Search-space keys an arm may set on a setting. Motion loss-shaping levers are
# deferred from the search space, so they are dropped if a proposal includes
# them.
_SETTING_KEYS = {
    "learning_rate", "weight_decay", "dropout", "batch_size",
    "lstm_hidden", "lstm_layers", "window_size", "optimizer_choice", "patience",
}


def _coerce_setting(setting: Dict[str, Any], allow_arch_changes: bool = True) -> Dict[str, Any]:
    """Keep only known search-space keys with non-None values; drop motion
    levers and (when arch is frozen) the architecture keys."""
    out: Dict[str, Any] = {}
    for k, v in (setting or {}).items():
        if k not in _SETTING_KEYS or v is None:
            continue
        if not allow_arch_changes and k in ("lstm_hidden", "lstm_layers"):
            continue
        out[k] = v
    return out


def setting_signature(setting: Dict[str, Any], allow_arch_changes: bool = True) -> tuple:
    """Hashable identity of a setting, for detecting already-tried repeats."""
    s = _coerce_setting(setting, allow_arch_changes)
    return tuple(sorted((k, round(float(v), 12) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                        for k, v in s.items()))


def evaluate_setting(
    base_cfg: Config,
    dataset: Dict[str, Any],
    setting: Dict[str, Any],
    train_seeds: tuple = TRAIN_SEEDS,
    max_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one HP setting once per seed in *train_seeds*, from scratch, and
    return its averaged validation result.

    score == val_rmse_mean == mean over seeds of the best-epoch validation
    RMSE in METRES (sqrt of the position MSE on inverse-transformed
    preds/targets, evaluated at the early-stopping best epoch whose weights are
    restored and carried to test). Lower is better. This is the single number
    the optimizer ranks settings by; it uses the same RMSE functional and units
    as the headline test RMSE (Evaluator.compute_metrics), so search selection
    and final evaluation agree.

    The dataset's windows are rebuilt in place when the setting's window_size
    differs from the current one (raw splits are cached, so no CSV reload and
    no change to the train/val/test split). Loaders are rebuilt per setting for
    its batch_size. Each seed gets a fresh model + optimizer; early stopping
    (Step 2) caps each training at <= max_epochs and restores the best epoch.
    """
    setting = _coerce_setting(setting, getattr(base_cfg, "allow_arch_changes", True))
    max_epochs = int(max_epochs or base_cfg.epochs_per_round)

    proc = dataset["processor"]
    ws = int(setting.get("window_size", proc.config.window_size))
    if ws != proc.config.window_size:
        DataProcessor.rebuild_with_window_size(dataset, ws)
    bs = int(setting.get("batch_size", base_cfg.batch_size))

    per_seed: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for seed in train_seeds:
        set_seed(int(seed))
        cfg = copy.deepcopy(base_cfg)
        cfg.window_size            = ws
        cfg.batch_size             = bs
        cfg.epochs                 = max_epochs
        cfg.lstm_hidden            = int(setting.get("lstm_hidden", cfg.lstm_hidden))
        cfg.lstm_layers            = int(setting.get("lstm_layers", cfg.lstm_layers))
        cfg.dropout                = float(setting.get("dropout", cfg.dropout))
        cfg.learning_rate          = float(setting.get("learning_rate", cfg.learning_rate))
        cfg.weight_decay           = float(setting.get("weight_decay", cfg.weight_decay))
        cfg.optimizer_choice       = setting.get("optimizer_choice", cfg.optimizer_choice)
        cfg.early_stopping_patience = int(setting.get("patience", cfg.early_stopping_patience))

        train_loader, val_loader, _ = make_loaders(dataset, bs)
        model = LSTM_Localizer(
            input_features=dataset["input_dim"], target_dim=dataset["target_dim"],
            lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers, dropout=cfg.dropout,
        ).to(cfg.device)
        trainer = Trainer(
            model=model, config=cfg, scaler_y=proc.scaler_y,
            train_loader=train_loader, val_loader=val_loader,
            dataset_dict=dataset, checkpoint_prefix="setting",
        )
        _ts = time.perf_counter()
        trainer.train()
        runtime_s = time.perf_counter() - _ts

        # Dropout-off train loss on the restored best-epoch weights. The running
        # train_loss in history is accumulated under model.train() (dropout on)
        # and averaged over the epoch's changing weights, so it is biased high
        # vs the eval-mode val_loss — making train/val gap spuriously negative
        # even with zero overfitting. Re-measuring the train set in eval mode on
        # the same best weights as val makes the gap a true generalisation
        # signal. (Does not count toward runtime_s, measured above.)
        clean_tr_loss, _ = trainer.validate(train_loader)

        h = trainer.history
        vl  = h.get("val_loss", [])
        best_epoch = int(getattr(trainer, "best_epoch_in_call", 0) or (len(vl)))
        idx = max(0, min(best_epoch - 1, len(vl) - 1)) if vl else None
        # Score in METRES (same functional as compute_metrics' test RMSE) at the
        # epoch whose weights are actually restored/carried to test, so the
        # ranking objective matches the headline metric and the model it grades.
        pos_m = h.get("val_position_loss_m", [])
        pos_at = (float(pos_m[idx]) if idx is not None and idx < len(pos_m)
                  and is_finite_number(pos_m[idx]) else float("nan"))
        if is_finite_number(pos_at):
            val_rmse = float(np.sqrt(pos_at))
        else:
            finite_m = [v for v in pos_m if is_finite_number(v)]
            val_rmse = float(np.sqrt(min(finite_m))) if finite_m else float("inf")
        tr_at = float(clean_tr_loss) if is_finite_number(clean_tr_loss) else float("nan")
        vl_at = float(vl[idx]) if idx is not None and is_finite_number(vl[idx]) else float("nan")
        per_seed.append({
            "seed":           int(seed),
            "val_rmse":       val_rmse,
            "best_epoch":     best_epoch,
            "epochs_trained": len(vl),
            "train_loss":     tr_at,
            "val_loss":       vl_at,
            "train_val_gap":  (vl_at - tr_at) if (is_finite_number(vl_at) and is_finite_number(tr_at)) else float("nan"),
            "runtime_s":      runtime_s,
        })

    def _mean(key: str) -> float:
        vals = [p[key] for p in per_seed if is_finite_number(p[key])]
        return float(np.mean(vals)) if vals else float("nan")

    rmses = [p["val_rmse"] for p in per_seed if is_finite_number(p["val_rmse"])]
    score = float(np.mean(rmses)) if rmses else float("inf")
    std   = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0
    return {
        "score":           score,            # = val_rmse_mean (what the optimizer ranks on)
        "val_rmse_mean":   score,
        "val_rmse_std":    std,
        "mean_best_epoch": _mean("best_epoch"),
        "mean_train_loss": _mean("train_loss"),
        "mean_val_loss":   _mean("val_loss"),
        "mean_train_val_gap": _mean("train_val_gap"),
        "runtime_s":       float(time.perf_counter() - t0),
        "n_trainings":     len(per_seed),
        "setting":         setting,
        "per_seed":        per_seed,
    }


# ---------------------------------------------------------------------------
# Random search helper
# ---------------------------------------------------------------------------

def sample_random_hparams(allow_arch_changes: bool = True, rng=None) -> Dict[str, Any]:
    """Sample a random HP configuration from the same search space as the LLM.

    Every hyperparameter range matches the agent system prompt so
    Random Search and LLM explore identical territories.

    Parameters
    ----------
    allow_arch_changes : bool, default True
        When False, ``lstm_hidden`` and ``lstm_layers`` are omitted from the
        sample so the random arm cannot redraw the architecture every round.
        This mirrors the LLM arm's behaviour under ``--no-arch-changes`` (see
        main.py:1040–1050) — under that ablation the LLM's validator strips
        the same two keys from any proposal, so excluding them here keeps the
        two arms' effective search spaces aligned. Other arch-adjacent levers
        (``window_size``, ``batch_size``) are still drawn because the LLM can
        also still change them — ``window_size`` triggers a model rebuild and
        ``batch_size`` only rebuilds the DataLoader.
    """
    # Draw each value uniformly from the discrete grid so random search and the
    # LLM/Optuna arms explore identical territory (motion levers deferred).
    # `rng` is the OPTIMIZER's own RNG (separate from the 101/102/103 training
    # seeds): evaluate_setting calls set_seed() internally, which would reset
    # the global RNG between attempts, so the search must draw from an
    # independent Random instance to stay reproducible.
    r = rng if rng is not None else random
    sample: Dict[str, Any] = {
        "learning_rate":    float(r.choice(HP_GRID["learning_rate"])),
        "weight_decay":     float(r.choice(HP_GRID["weight_decay"])),
        "dropout":          float(r.choice(HP_GRID["dropout"])),
        "window_size":      int(r.choice(HP_GRID["window_size"])),
        "batch_size":       int(r.choice(HP_GRID["batch_size"])),
        "patience":         int(r.choice(HP_GRID["patience"])),
        "optimizer_choice": r.choice(HP_GRID["optimizer_choice"]),
    }
    if allow_arch_changes:
        sample["lstm_hidden"] = int(r.choice(HP_GRID["lstm_hidden"]))
        sample["lstm_layers"] = int(r.choice(HP_GRID["lstm_layers"]))
    return sample


def train_and_test_setting(
    base_cfg: Config,
    dataset: Dict[str, Any],
    setting: Dict[str, Any],
    seed: int,
    max_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one setting from scratch on a single seed and evaluate it on the
    TEST set. Used ONLY after a method has selected its best setting (the
    protocol allows the test set only at that point). Step 5 calls this across
    the 30 fresh final-evaluation seeds; the interim search arms call it once on
    a training seed just to populate the legacy `metrics`/plots slots.
    """
    setting = _coerce_setting(setting, getattr(base_cfg, "allow_arch_changes", True))
    max_epochs = int(max_epochs or base_cfg.epochs_per_round)
    proc = dataset["processor"]
    ws = int(setting.get("window_size", proc.config.window_size))
    if ws != proc.config.window_size:
        DataProcessor.rebuild_with_window_size(dataset, ws)
    bs = int(setting.get("batch_size", base_cfg.batch_size))

    set_seed(int(seed))
    cfg = copy.deepcopy(base_cfg)
    cfg.window_size             = ws
    cfg.batch_size              = bs
    cfg.epochs                  = max_epochs
    cfg.lstm_hidden             = int(setting.get("lstm_hidden", cfg.lstm_hidden))
    cfg.lstm_layers             = int(setting.get("lstm_layers", cfg.lstm_layers))
    cfg.dropout                 = float(setting.get("dropout", cfg.dropout))
    cfg.learning_rate           = float(setting.get("learning_rate", cfg.learning_rate))
    cfg.weight_decay            = float(setting.get("weight_decay", cfg.weight_decay))
    cfg.optimizer_choice        = setting.get("optimizer_choice", cfg.optimizer_choice)
    cfg.early_stopping_patience = int(setting.get("patience", cfg.early_stopping_patience))

    train_loader, val_loader, test_loader = make_loaders(dataset, bs)
    model = LSTM_Localizer(
        input_features=dataset["input_dim"], target_dim=dataset["target_dim"],
        lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers, dropout=cfg.dropout,
    ).to(cfg.device)
    trainer = Trainer(
        model=model, config=cfg, scaler_y=proc.scaler_y,
        train_loader=train_loader, val_loader=val_loader,
        dataset_dict=dataset, checkpoint_prefix="eval",
    )
    trainer.train()   # early stopping restores the best-epoch weights
    evaluator = Evaluator(trainer.model, cfg, proc.scaler_y)
    preds, targets = evaluator.predict(test_loader)
    metrics = evaluator.compute_metrics(preds, targets)
    return {
        "metrics":    metrics,
        "history":    trainer.history,
        "best_epoch": int(getattr(trainer, "best_epoch_in_call", 0) or 0),
        "trainer":    trainer,
        "evaluator":  evaluator,
        "preds":      preds,
        "targets":    targets,
    }


def _attempt_record(attempt: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """Per-attempt search-log row, shared by all engine-driven arms. Keeps the
    legacy `round`/`val_loss` keys so existing plots and cross-run metrics keep
    working, and adds the protocol's `val_rmse` (= the setting's score)."""
    return {
        "round":              attempt,                     # legacy alias
        "attempt":            attempt,
        "proposed_changes":   result["setting"],
        "setting":            result["setting"],
        "val_loss":           result["mean_val_loss"],     # legacy plots/threshold
        "val_rmse":           result["score"],             # protocol score
        "val_rmse_std":       result["val_rmse_std"],
        "mean_best_epoch":    result["mean_best_epoch"],
        "mean_train_loss":    result["mean_train_loss"],
        "mean_val_loss":      result["mean_val_loss"],
        "mean_train_val_gap": result["mean_train_val_gap"],
        "runtime_s":          result["runtime_s"],
        "n_trainings":        result["n_trainings"],
        "per_seed":           result["per_seed"],
        # Random / Optuna always train every attempt (no validation gate).
        "trained":            True,
        "output_status":      "clean",
    }


def run_random_search(
    base_cfg: Config,
    dataset: Dict[str, Any],
    n_attempts: int,
    opt_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Random-search arm on the from-scratch engine: `n_attempts` settings, each
    scored by mean validation RMSE over the 3 training seeds. Picks the setting
    with the best (lowest) mean val RMSE. `opt_seed` seeds ONLY the sampler.
    """
    allow_arch = bool(getattr(base_cfg, "allow_arch_changes", True))
    opt_rng = random.Random(opt_seed if opt_seed is not None else getattr(base_cfg, "seed", 0))
    attempts: List[Dict[str, Any]] = []
    best_curve: List[float] = []
    best_score = float("inf")
    best_rec: Optional[Dict[str, Any]] = None
    for a in range(1, n_attempts + 1):
        setting = sample_random_hparams(allow_arch_changes=allow_arch, rng=opt_rng)
        result = evaluate_setting(base_cfg, dataset, setting)
        rec = _attempt_record(a, result)
        attempts.append(rec)
        if is_finite_number(result["score"]) and result["score"] < best_score:
            best_score = result["score"]
            best_rec = rec
        best_curve.append(best_score if is_finite_number(best_score) else float("nan"))
        logger.info(
            "Random attempt %d/%d | val_rmse=%.4f ± %.4f | best=%.4f",
            a, n_attempts, result["score"], result["val_rmse_std"], best_score,
        )
    return {"attempts": attempts, "best": best_rec, "best_curve": best_curve,
            "opt_seed": (opt_seed if opt_seed is not None else getattr(base_cfg, "seed", 0))}


# ---------------------------------------------------------------------------
# Optuna (TPE / Bayesian) search arm
# ---------------------------------------------------------------------------
#
# The "competent conventional optimizer" arm. It shares the SAME from-scratch
# engine (evaluate_setting: 3 trainings per setting, mean val RMSE score) and
# the SAME discrete HP_GRID as every other arm — the only difference vs random
# is the sampler: Optuna's TPE learns from past trial scores to bias later
# draws toward promising regions of HP space.

def _suggest_from_optuna_trial(trial, allow_arch_changes: bool = True) -> Dict[str, Any]:
    """Translate one Optuna trial into a setting dict with the same shape
    ``sample_random_hparams`` produces. Values are drawn from ``HP_GRID`` via
    suggest_categorical so the reachable set is identical across all arms.
    """
    # Optuna draws from the SAME discrete grid as the random and LLM arms.
    # Using suggest_categorical over the exact value lists (rather than
    # continuous suggest_float) keeps the reachable set identical across arms,
    # so the random ↔ Optuna comparison isolates TPE's smart sampling alone.
    s: Dict[str, Any] = {
        "learning_rate":    trial.suggest_categorical("learning_rate", HP_GRID["learning_rate"]),
        "weight_decay":     trial.suggest_categorical("weight_decay", HP_GRID["weight_decay"]),
        "dropout":          trial.suggest_categorical("dropout", HP_GRID["dropout"]),
        "window_size":      int(trial.suggest_categorical("window_size", HP_GRID["window_size"])),
        "batch_size":       int(trial.suggest_categorical("batch_size", HP_GRID["batch_size"])),
        "patience":         int(trial.suggest_categorical("patience", HP_GRID["patience"])),
        "optimizer_choice": trial.suggest_categorical("optimizer_choice", HP_GRID["optimizer_choice"]),
    }
    if allow_arch_changes:
        s["lstm_hidden"] = int(trial.suggest_categorical("lstm_hidden", HP_GRID["lstm_hidden"]))
        s["lstm_layers"] = int(trial.suggest_categorical("lstm_layers", HP_GRID["lstm_layers"]))
    return s


OPTUNA_SAMPLER_SEED = 1000        # protocol: fixed, reported optimizer seed
OPTUNA_N_STARTUP_TRIALS = 5       # protocol: first 5 trials random, rest TPE-guided


def run_optuna_search(
    base_cfg: Config,
    dataset: Dict[str, Any],
    n_attempts: int,
    sampler_seed: int = OPTUNA_SAMPLER_SEED,
) -> Dict[str, Any]:
    """Optuna TPE arm on the from-scratch engine. Each trial proposes a setting,
    which `evaluate_setting` scores by mean validation RMSE over the 3 training
    seeds; that score is `tell`-ed back so TPE biases later trials toward good
    regions. The sampler seed (default 1000) is the OPTIMIZER seed — separate
    from the 101/102/103 training seeds — and is reported in study_summary.

    Raises ImportError when `optuna` is not installed; the caller skips the arm.
    """
    import optuna  # local import so missing dep doesn't break other arms
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    allow_arch = bool(getattr(base_cfg, "allow_arch_changes", True))
    if not allow_arch:
        logger.info(
            "Optuna arm: --no-arch-changes is active — lstm_hidden / lstm_layers "
            "excluded from the TPE search space (matches the random and LLM arms)."
        )

    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed, n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    attempts: List[Dict[str, Any]] = []
    best_curve: List[float] = []
    best_score = float("inf")
    best_rec: Optional[Dict[str, Any]] = None

    for a in range(1, n_attempts + 1):
        trial = study.ask()
        setting = _suggest_from_optuna_trial(trial, allow_arch_changes=allow_arch)
        result = evaluate_setting(base_cfg, dataset, setting)
        # Objective = mean validation RMSE. Non-finite scores are reported as a
        # large finite sentinel so TPE down-ranks that region instead of erroring.
        tell_value = float(result["score"]) if is_finite_number(result["score"]) else 1e9
        study.tell(trial, tell_value)

        rec = _attempt_record(a, result)
        rec["tpe_tell_value"] = tell_value
        attempts.append(rec)
        if is_finite_number(result["score"]) and result["score"] < best_score:
            best_score = result["score"]
            best_rec = rec
        best_curve.append(best_score if is_finite_number(best_score) else float("nan"))
        logger.info(
            "Optuna attempt %d/%d | val_rmse=%.4f ± %.4f | best=%.4f",
            a, n_attempts, result["score"], result["val_rmse_std"], best_score,
        )

    try:
        bt = study.best_trial
        study_summary = {
            "sampler": type(sampler).__name__,
            "sampler_seed": sampler_seed,
            "n_startup_trials": OPTUNA_N_STARTUP_TRIALS,
            "best_value": float(bt.value) if bt.value is not None else None,
            "best_params": dict(bt.params),
            "best_trial_number": int(bt.number),
            "n_trials": int(len(study.trials)),
        }
    except ValueError:
        study_summary = {
            "sampler": type(sampler).__name__,
            "sampler_seed": sampler_seed,
            "n_startup_trials": OPTUNA_N_STARTUP_TRIALS,
            "best_value": None, "best_params": None,
            "best_trial_number": None, "n_trials": int(len(study.trials)),
        }
    return {"attempts": attempts, "best": best_rec, "best_curve": best_curve,
            "study_summary": study_summary}


# ---------------------------------------------------------------------------
# Proposer-driven search (LLM + rule-based) on the from-scratch engine
# ---------------------------------------------------------------------------

def _default_setting(cfg: Config, allow_arch: bool) -> Dict[str, Any]:
    """The fixed-reference configuration, used as the first anchor before any
    setting has been scored."""
    s: Dict[str, Any] = {
        "learning_rate":    cfg.learning_rate,
        "weight_decay":     cfg.weight_decay,
        "dropout":          cfg.dropout,
        "batch_size":       cfg.batch_size,
        "window_size":      cfg.window_size,
        "optimizer_choice": cfg.optimizer_choice,
        "patience":         getattr(cfg, "early_stopping_patience", 12),
    }
    if allow_arch:
        s["lstm_hidden"] = cfg.lstm_hidden
        s["lstm_layers"] = cfg.lstm_layers
    return s


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


# ---------------------------------------------------------------------------
# Per-setting reports + budget table (protocol point 2 & 9)
# ---------------------------------------------------------------------------

_SETTING_COLS = [
    "learning_rate", "weight_decay", "dropout", "batch_size",
    "lstm_hidden", "lstm_layers", "window_size", "optimizer_choice", "patience",
]
_REPORT_COLS = (
    ["arm", "attempt", "output_status", "trained",
     "val_rmse_mean", "val_rmse_std", "mean_best_epoch",
     "mean_train_loss", "mean_val_loss", "train_val_gap", "runtime_s", "diagnosis"]
    + _SETTING_COLS
)


def _attempt_to_row(arm: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one attempt record into a per-setting report row. Reports the
    protocol's columns: mean val RMSE, std, mean best epoch, mean train/val
    loss, train/val gap, runtime."""
    setting = rec.get("setting", {}) or {}
    row = {
        "arm":             arm,
        "attempt":         rec.get("attempt"),
        "output_status":   rec.get("output_status", ""),
        "trained":         rec.get("trained", True),
        "val_rmse_mean":   rec.get("val_rmse", rec.get("score")),
        "val_rmse_std":    rec.get("val_rmse_std"),
        "mean_best_epoch": rec.get("mean_best_epoch"),
        "mean_train_loss": rec.get("mean_train_loss"),
        "mean_val_loss":   rec.get("mean_val_loss"),
        "train_val_gap":   rec.get("mean_train_val_gap"),
        "runtime_s":       rec.get("runtime_s"),
        "diagnosis":       rec.get("diagnosis", ""),
    }
    for k in _SETTING_COLS:
        row[k] = setting.get(k)
    return row


def write_per_setting_report(config: Config, results: Dict[str, Any], run_id: int) -> None:
    """Write one CSV row per attempted setting, per arm, with the protocol's
    per-setting metrics. Raw numbers live here (and in the JSON logs); the LLM
    prompt only ever saw the qualitative labels."""
    rows: List[Dict[str, Any]] = []

    # Fixed-reference baseline: one row (final-eval-only; single training).
    b = results.get("baseline") or {}
    b_pos = [v for v in (b.get("history", {}) or {}).get("val_position_loss", []) if is_finite_number(v)]
    b_setting = _default_setting(config, getattr(config, "allow_arch_changes", True))
    base_row = {c: None for c in _REPORT_COLS}
    base_row.update({
        "arm": "fixed_reference", "attempt": 1, "output_status": "final_eval_only",
        "trained": True,
        "val_rmse_mean": float(np.sqrt(min(b_pos))) if b_pos else None,
    })
    for k in _SETTING_COLS:
        base_row[k] = b_setting.get(k)
    rows.append(base_row)

    for arm in ("llm", "random", "rule_based", "optuna"):
        slot = results.get(arm)
        if not isinstance(slot, dict):
            continue
        attempts = (slot.get("search") or {}).get("attempts", [])
        for rec in attempts:
            rows.append(_attempt_to_row(arm, rec))

    path = config.output_dir / f"per_setting_report_run{run_id}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_REPORT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _REPORT_COLS})
    logger.info("Per-setting report -> %s  (%d rows)", path, len(rows))


def write_budget_table(config: Config, results: Dict[str, Any], run_id: int) -> None:
    """Write the protocol budget table (point 9) + the LLM validity accounting."""
    n = config.optimization_rounds
    e = config.epochs_per_round
    t = len(TRAIN_SEEDS)

    def _trained(arm: str) -> int:
        slot = results.get(arm) or {}
        return sum(1 for a in (slot.get("search") or {}).get("attempts", []) if a.get("trained"))

    lines = [
        f"# Budget table (run {run_id})",
        "",
        "| Method | Attempts | Trainings per valid setting | Epochs/train (max) | Val during search | Test during search | Settings actually trained |",
        "|---|---|---|---|---|---|---|",
        f"| Fixed reference | 1 | final-eval only | {e} | no | no | 1 |",
        f"| Random search | {n} | {t} | {e} | yes | no | {_trained('random')} |",
        f"| Optuna (TPE) | {n} | {t} | {e} | yes | no | {_trained('optuna')} |",
        f"| Rule-based | {n} | {t} | {e} | yes | no | {_trained('rule_based')} |",
        f"| LLM controller | {n} | {t} per valid setting | {e} | yes | no | {_trained('llm')} |",
        "",
    ]

    vs = (results.get("llm") or {}).get("validity_stats", {}) or {}
    if vs:
        lines += [
            "## LLM controller validity",
            "",
            f"- valid on first try: {vs.get('valid_first_try', 0)}",
            f"- valid after retry: {vs.get('valid_after_retry', 0)}",
            f"- rejected (invalid after retry): {vs.get('rejected', 0)}",
            f"- repeated settings proposed: {vs.get('repeats_proposed', 0)}",
            f"- invalid parameter values proposed: {vs.get('invalid_values', 0)}",
            f"- LLM timeouts: {vs.get('llm_timeouts', 0)}",
            f"- parse failures: {vs.get('parse_failures', 0)}",
            f"- actual valid settings trained: {_trained('llm')}",
            "",
        ]

    path = config.output_dir / f"budget_table_run{run_id}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Budget table -> %s", path)
    for ln in lines:
        if ln.startswith("|") or ln.startswith("- ") or ln.startswith("#"):
            logger.info("  %s", ln)


# ---------------------------------------------------------------------------
# Final evaluation on fresh seeds (protocol point 8)
# ---------------------------------------------------------------------------

# Fresh, never-used seeds for the final evaluation. Disjoint from the 3 training
# seeds (101/102/103) and the Optuna sampler seed (1000), so the final result is
# NOT computed on any seed used during optimization.
FINAL_EVAL_SEEDS_POOL: List[int] = list(range(201, 281))


def run_final_evaluation(
    base_cfg: Config,
    best_settings: Dict[str, Dict[str, Any]],
    seeds: List[int],
) -> Dict[str, Any]:
    """Evaluate each method's single best setting on `seeds` fresh seeds.

    For each (arm, seed): train the setting from scratch and evaluate on TEST.
    Report per arm: mean/std test RMSE, best & worst seed, mean best epoch,
    mean runtime, trainable #params. Also report paired differences vs the LLM
    (same seeds, so the difference is per-seed paired)."""
    dataset, _, _, _ = build_dataset_and_loaders(base_cfg)
    per_arm: Dict[str, Any] = {}
    per_seed_rmse: Dict[str, Dict[int, float]] = {}

    for arm, setting in best_settings.items():
        if not setting:
            logger.info("Final eval: arm '%s' has no best setting — skipped.", arm)
            continue
        rmses: List[float] = []
        r2s: List[float] = []
        best_epochs: List[float] = []
        runtimes: List[float] = []
        n_params: Optional[int] = None
        seed_rows: List[Dict[str, Any]] = []
        per_seed_rmse[arm] = {}
        logger.info("Final eval | %s | best setting=%s", arm, setting)
        for seed in seeds:
            t0 = time.perf_counter()
            te = train_and_test_setting(base_cfg, dataset, setting, seed=int(seed))
            rt = time.perf_counter() - t0
            m = te["metrics"]
            if n_params is None:
                try:
                    n_params = int(te["trainer"].model.count_parameters()[1])
                except Exception:
                    n_params = None
            rmses.append(float(m["rmse"]))
            r2s.append(float(m["r2"]))
            best_epochs.append(float(te["best_epoch"]))
            runtimes.append(rt)
            per_seed_rmse[arm][int(seed)] = float(m["rmse"])
            seed_rows.append({"seed": int(seed), "rmse": float(m["rmse"]),
                              "r2": float(m["r2"]), "best_epoch": int(te["best_epoch"])})
        arr = np.array(rmses, dtype=float)
        bi = int(np.argmin(arr)); wi = int(np.argmax(arr))
        per_arm[arm] = {
            "mean_rmse": float(np.mean(arr)),
            "std_rmse":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "best_seed":  {"seed": int(seeds[bi]), "rmse": float(arr[bi])},
            "worst_seed": {"seed": int(seeds[wi]), "rmse": float(arr[wi])},
            "mean_r2":         float(np.mean(r2s)),
            "mean_best_epoch": float(np.mean(best_epochs)),
            "mean_runtime_s":  float(np.mean(runtimes)),
            "n_params":        n_params,
            "n_seeds":         len(seeds),
            "setting":         setting,
            "per_seed":        seed_rows,
        }
        logger.info("Final eval | %s | mean test RMSE=%.4f ± %.4f over %d seeds (params=%s)",
                    arm, per_arm[arm]["mean_rmse"], per_arm[arm]["std_rmse"], len(seeds), n_params)

    # Paired differences vs the LLM (same seeds → per-seed paired).
    paired: Dict[str, Any] = {}
    if "llm" in per_seed_rmse:
        for arm in per_seed_rmse:
            if arm == "llm":
                continue
            diffs = [per_seed_rmse["llm"][s] - per_seed_rmse[arm][s]
                     for s in seeds if s in per_seed_rmse["llm"] and s in per_seed_rmse[arm]]
            if diffs:
                d = np.array(diffs, dtype=float)
                paired[f"llm_minus_{arm}"] = {
                    "mean": float(np.mean(d)),
                    "std":  float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                    "n":    len(d),
                    "llm_better_count": int(np.sum(d < 0)),  # negative = LLM lower RMSE
                }
    return {"per_arm": per_arm, "paired_vs_llm": paired, "seeds": list(seeds)}


def write_final_eval_report(config: Config, final_eval: Dict[str, Any], run_id: int) -> None:
    """Write the final-evaluation table + paired differences (markdown + JSON)."""
    per_arm = final_eval.get("per_arm", {})
    paired  = final_eval.get("paired_vs_llm", {})
    order = ["fixed_reference", "llm", "random", "rule_based", "optuna"]
    label = {"fixed_reference": "Fixed reference", "llm": "LLM controller",
             "random": "Random search", "rule_based": "Rule-based", "optuna": "Optuna (TPE)"}

    lines = [
        f"# Final evaluation (run {run_id}) — best setting per method on {len(final_eval.get('seeds', []))} fresh seeds",
        "",
        "| Method | mean test RMSE | std | best seed | worst seed | mean best epoch | mean runtime (s) | #params |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in order:
        a = per_arm.get(arm)
        if not a:
            continue
        lines.append(
            f"| {label[arm]} | {a['mean_rmse']:.4f} | {a['std_rmse']:.4f} | "
            f"{a['best_seed']['rmse']:.4f}@{a['best_seed']['seed']} | "
            f"{a['worst_seed']['rmse']:.4f}@{a['worst_seed']['seed']} | "
            f"{a['mean_best_epoch']:.1f} | {a['mean_runtime_s']:.1f} | {a['n_params']} |"
        )
    lines.append("")
    if paired:
        lines += ["## Paired differences (LLM − method, per seed; negative = LLM better)", ""]
        for key, p in paired.items():
            arm = key.replace("llm_minus_", "")
            lines.append(f"- LLM − {label.get(arm, arm)}: {p['mean']:+.4f} ± {p['std']:.4f} "
                         f"(LLM better on {p['llm_better_count']}/{p['n']} seeds)")
        lines.append("")

    md_path = config.output_dir / f"final_evaluation_run{run_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    json_path = config.output_dir / f"final_evaluation_run{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(final_eval, f, indent=2, default=str)
    logger.info("Final-eval report -> %s", md_path)
    for ln in lines:
        if ln.startswith("|") or ln.startswith("- ") or ln.startswith("#"):
            logger.info("  %s", ln)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_optimization_trajectories(
    config: Config,
    histories: Dict[str, Dict[str, List[float]]],
    filename: str,
    title: str = "Optimization Trajectories",
) -> None:
    """
    Plot val-loss trajectories for any number of named histories.

    histories: {"Baseline": history_dict, "LLM": history_dict, ...}
    """
    # Cycle styles by index so the plot never silently drops arms: zipping
    # against a fixed 4-element list truncated to 4 lines, dropping the
    # last-inserted history (Optuna).
    styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, h) in enumerate(histories.items()):
        vl = h.get("val_loss", [])
        if vl:
            ax.plot(vl, lw=2, ls=styles[i % len(styles)], label=label)
    ax.set(xlabel="Epoch", ylabel="Validation Loss", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved -> %s", path)


def _build_param_series_from_log(
    opt_log: Dict[str, Any],
    param: str,
    default_value: float,
) -> Dict[str, List[float]]:
    def _as_float(v: Any) -> Optional[float]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if is_finite_number(f) else None

    init = _as_float(opt_log.get("initial_hyperparameters", {}).get(param))
    current = init if init is not None else float(default_value)
    rounds = [0]
    values = [current]

    for rd in sorted(opt_log.get("rounds", []), key=lambda x: x.get("round", 0)):
        r = int(rd.get("round", 0))
        changes = rd.get("changes_applied", {}) or {}
        if param in changes:
            # Rejected / malformed proposals (e.g. an LLM emitting prose where a
            # number belongs) are still logged but were never validly applied,
            # so they must not move the trajectory line.
            parsed = _as_float(changes[param])
            if parsed is not None:
                current = parsed
        rounds.append(r)
        values.append(current)

    return {"rounds": rounds, "values": values}


def plot_hyperparameter_trajectory(
    config: Config,
    opt_log: Dict[str, Any],
    filename: str,
    title: str = "Hyperparameter Trajectory",
) -> None:
    lr_series     = _build_param_series_from_log(opt_log, "learning_rate", 1e-3)
    do_series     = _build_param_series_from_log(opt_log, "dropout",       0.1)
    hidden_series = _build_param_series_from_log(opt_log, "lstm_hidden",   128)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(lr_series["rounds"], lr_series["values"], marker="o", lw=2, color="tab:blue")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(do_series["rounds"], do_series["values"], marker="o", lw=2, color="tab:orange")
    axes[1].set_ylabel("Dropout")
    axes[1].grid(alpha=0.3)

    axes[2].step(hidden_series["rounds"], hidden_series["values"], where="post", lw=2, color="tab:green")
    axes[2].scatter(hidden_series["rounds"], hidden_series["values"], s=28, color="tab:green")
    axes[2].set_ylabel("LSTM Hidden Size")
    axes[2].set_xlabel("Round")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Hyperparameter trajectory plot saved -> %s", path)


def plot_action_impact_scatter(
    config: Config,
    opt_log: Dict[str, Any],
    filename: str,
    title: str = "Agent Action Magnitude vs Delta Best Val Loss",
) -> None:
    xs, ys = [], []
    for rd in opt_log.get("rounds", []):
        mag = rd.get("change_magnitude")
        delta_best = rd.get("delta_best_val_loss")
        if is_finite_number(mag) and is_finite_number(delta_best):
            xs.append(float(mag))
            ys.append(float(delta_best))

    if not xs:
        logger.info("Skipping action-impact scatter plot (no finite points).")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, s=60, alpha=0.85, color="tab:red", edgecolors="black", linewidths=0.5)
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Proposed Change Magnitude (normalized)")
    ax.set_ylabel("Delta Best Validation Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Action-impact scatter plot saved -> %s", path)


def plot_training_dynamics(
    config: Config,
    all_results: List[Dict[str, Any]],
    filename: str = "training_dynamics.png",
) -> None:
    """
    PLOT 1 — Training Dynamics  (epoch-level)
    ==========================================
    Shows how val_loss evolves across *epochs* during the full training run
    of each arm.  This captures the learning curve shape, convergence speed,
    and overfitting behaviour — not which HP config was used.

    X-axis : Epoch
    Y-axis : Validation loss
    One faint line per experiment run; bold line = mean across runs.
    """
    arms   = ["baseline", "llm", "random", "rule_based"]
    labels = {"baseline": "Baseline (Expert)", "llm": "LLM", "random": "Random Search", "rule_based": "Rule-Based"}
    colors = {"baseline": "steelblue", "llm": "darkorange", "random": "seagreen", "rule_based": "mediumpurple"}

    fig, axes = plt.subplots(1, len(arms), figsize=(18, 5), sharey=True)
    for ax, arm in zip(axes, arms):
        run_vls = [r[arm]["history"].get("val_loss", []) for r in all_results]
        max_len = max((len(v) for v in run_vls), default=0)
        if max_len == 0:
            ax.set(title=labels[arm], xlabel="Epoch")
            continue
        padded = [v + [float("nan")] * (max_len - len(v)) for v in run_vls]
        arr    = np.array(padded, dtype=float)
        xs     = np.arange(max_len)

        for row in padded:
            ax.plot(xs, row, lw=1, alpha=0.25, color=colors[arm])

        mean = np.nanmean(arr, axis=0)
        ax.plot(xs, mean, lw=2.5, color=colors[arm], label=f"Mean ({len(all_results)} runs)")
        ax.set(xlabel="Epoch", title=labels[arm])
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Validation Loss")
    fig.suptitle("Training Dynamics — Epoch-Level Validation Loss", fontsize=13)
    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Training dynamics plot saved -> %s", path)


def plot_hp_search_dynamics(
    config: Config,
    all_results: List[Dict[str, Any]],
    filename: str = "hp_search_dynamics.png",
) -> None:
    """
    PLOT 2 — Hyperparameter Search Dynamics  (round-level)
    =======================================================
    Shows the val_loss *achieved by each HP configuration* tried during
    the optimization process.  Each tick on the x-axis is one new HP
    configuration (one optimization round), NOT individual training epochs.

    X-axis : Optimization round (1…N)
    Y-axis : Val_loss at end of that round's training budget
    One faint line per experiment run; bold line = mean across runs.

    Arms
    ----
    llm    — per-round val_loss from opt_log
    random — per-random-trial val_loss from random_log
    expert — flat horizontal reference (single config, no search)
    """
    arms   = ["baseline", "llm", "random", "rule_based"]
    labels = {"baseline": "Expert", "llm": "LLM", "random": "Random Search", "rule_based": "Rule-Based"}
    colors = {"baseline": "steelblue", "llm": "darkorange", "random": "seagreen", "rule_based": "mediumpurple"}

    def _opt_round_vals(result, arm: str) -> List[float]:
        opt_log = result.get(arm, {}).get("optimization_log", {})
        if not opt_log:
            return []
        vals: List[float] = []
        for rd in sorted(opt_log.get("rounds", []), key=lambda x: x.get("round", 0)):
            vl = rd.get("val_loss")
            if vl is not None and is_finite_number(vl):
                vals.append(float(vl))
        return vals

    def _random_round_vals(result) -> List[float]:
        rlog = result.get("random", {}).get("random_log", [])
        return [float(rd["val_loss"]) for rd in rlog
                if rd.get("val_loss") is not None and is_finite_number(rd["val_loss"])]

    def _baseline_round_vals(result, n_rounds: int) -> List[float]:
        """Flat line — expert trains once, no per-round HP changes."""
        bvl = result.get("baseline", {}).get("best_curve", [])
        if bvl:
            return list(bvl)
        hist = result.get("baseline", {}).get("history", {}).get("val_loss", [])
        vl   = hist[-1] if hist else float("nan")
        return [float(vl)] * (n_rounds + 1)

    n_rounds_global = max((len(_opt_round_vals(r, "llm")) for r in all_results), default=1) - 1

    extractors = {
        "baseline":   lambda r: _baseline_round_vals(r, n_rounds_global),
        "llm":        lambda r: _opt_round_vals(r, "llm"),
        "random":     _random_round_vals,
        "rule_based": lambda r: _opt_round_vals(r, "rule_based"),
    }

    fig, axes = plt.subplots(1, len(arms), figsize=(18, 5), sharey=True)
    for ax, arm in zip(axes, arms):
        run_vls = [extractors[arm](r) for r in all_results]
        max_len = max((len(v) for v in run_vls), default=0)
        if max_len == 0:
            ax.set(title=labels[arm], xlabel="Optimization Round")
            continue
        padded = [v + [float("nan")] * (max_len - len(v)) for v in run_vls]
        arr    = np.array(padded, dtype=float)
        xs     = np.arange(max_len)

        for row in padded:
            ax.plot(xs, row, lw=1, alpha=0.25, color=colors[arm])

        mean = np.nanmean(arr, axis=0)
        ax.plot(xs, mean, lw=2.5, color=colors[arm], label=f"Mean ({len(all_results)} runs)")
        ax.set_xticks(xs)
        ax.set(xlabel="Optimization Round", title=labels[arm])
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Validation Loss")
    fig.suptitle("HP Search Dynamics — Val Loss per Optimization Round", fontsize=13)
    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("HP search dynamics plot saved -> %s", path)


# Keep alias so any external code using the old name still works
def plot_cross_run_summary(
    config: Config,
    all_results: List[Dict[str, Any]],
    filename: str = "cross_run_summary.png",
) -> None:
    """Calls both separated plots. Kept for backwards compatibility."""
    plot_training_dynamics(config, all_results,
                           filename=filename.replace(".png", "_training.png"))
    plot_hp_search_dynamics(config, all_results,
                            filename=filename.replace(".png", "_hp_search.png"))


# ---------------------------------------------------------------------------
# Cross-run metrics
# ---------------------------------------------------------------------------

def compute_cross_run_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute compact cross-run quality and efficiency summaries.

    Per-arm statistics reported
    ---------------------------
    rmse : mean ± std across runs
    r2   : mean ± std across runs   ← NEW
    """
    def _arm_metric(run: Dict[str, Any], arm: str, key: str) -> float:
        """Pull a metric from an arm safely. Returns NaN when the arm is
        absent (e.g. Optuna skipped because the package wasn't installed) so
        downstream `np.nanmean` / `_safe_stats` simply ignore it.
        """
        slot = run.get(arm)
        if not isinstance(slot, dict):
            return np.nan
        return slot.get("metrics", {}).get(key, np.nan)

    b_rmses  = [_arm_metric(r, "baseline",   "rmse") for r in all_results]
    l_rmses  = [_arm_metric(r, "llm",        "rmse") for r in all_results]
    r_rmses  = [_arm_metric(r, "random",     "rmse") for r in all_results]
    rb_rmses = [_arm_metric(r, "rule_based", "rmse") for r in all_results]
    o_rmses  = [_arm_metric(r, "optuna",     "rmse") for r in all_results]

    b_r2s  = [_arm_metric(r, "baseline",   "r2") for r in all_results]
    l_r2s  = [_arm_metric(r, "llm",        "r2") for r in all_results]
    r_r2s  = [_arm_metric(r, "random",     "r2") for r in all_results]
    rb_r2s = [_arm_metric(r, "rule_based", "r2") for r in all_results]
    o_r2s  = [_arm_metric(r, "optuna",     "r2") for r in all_results]

    baseline_best_losses = []
    aucs: Dict[str, List[float]] = {
        "baseline": [], "llm": [], "random": [], "rule_based": [], "optuna": [],
    }

    for run in all_results:
        bv  = run["baseline"]["history"].get("val_loss", [])
        lv  = run["llm"]["history"].get("val_loss", [])
        rv  = run["random"]["history"].get("val_loss", [])
        rbv = run["rule_based"]["history"].get("val_loss", [])
        ov  = (run.get("optuna") or {}).get("history", {}).get("val_loss", []) or []
        if bv:
            baseline_best_losses.append(min(bv))
            aucs["baseline"].append(safe_trapz(bv))
        if lv:
            aucs["llm"].append(safe_trapz(lv))
        if rv:
            aucs["random"].append(safe_trapz(rv))
        if rbv:
            aucs["rule_based"].append(safe_trapz(rbv))
        if ov:
            aucs["optuna"].append(safe_trapz(ov))

    threshold = float(np.nanmedian(baseline_best_losses)) if baseline_best_losses else None
    llm_trials, random_trials, rb_trials, optuna_trials = [], [], [], []
    if threshold is not None:
        for run in all_results:
            opt_log = run["llm"]["optimization_log"]
            lv = list(
                float(x.get("val_loss"))
                for x in opt_log.get("rounds", [])
                if is_finite_number(x.get("val_loss"))
            )
            rv = [
                float(x.get("val_loss"))
                for x in run["random"].get("random_log", [])
                if is_finite_number(x.get("val_loss"))
            ]
            rbv = list(
                float(x.get("val_loss"))
                for x in run["rule_based"].get("optimization_log", {}).get("rounds", [])
                if is_finite_number(x.get("val_loss"))
            )
            optuna_slot = run.get("optuna") or {}
            ov = [
                float(x.get("val_loss"))
                for x in optuna_slot.get("optuna_log", [])
                if is_finite_number(x.get("val_loss"))
            ]
            llm_trials.append(
                next((i + 1 for i, v in enumerate(lv) if v is not None and v <= threshold), None)
            )
            random_trials.append(
                next((i + 1 for i, v in enumerate(rv) if v is not None and v <= threshold), None)
            )
            rb_trials.append(
                next((i + 1 for i, v in enumerate(rbv) if v is not None and v <= threshold), None)
            )
            optuna_trials.append(
                next((i + 1 for i, v in enumerate(ov) if v is not None and v <= threshold), None)
            )

    def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
        filtered = [v for v in values if v is not None]
        return float(np.nanmean(filtered)) if filtered else None

    def _safe_stats(vals: List[float]) -> Dict[str, Optional[float]]:
        """mean + std of finite values in *vals*."""
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return {"mean": None, "std": None}
        arr = np.array(finite, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    mean_b  = float(np.nanmean(b_rmses))
    mean_l  = float(np.nanmean(l_rmses))
    mean_r  = float(np.nanmean(r_rmses))
    mean_rb = float(np.nanmean(rb_rmses))
    mean_o  = float(np.nanmean(o_rmses)) if any(np.isfinite(o_rmses)) else float("nan")
    has_optuna = bool(np.any(np.isfinite(o_rmses)))
    return {
        "optimization_quality": {
            # ── RMSE ─────────────────────────────────────────────────────
            "mean_rmse": {
                "baseline":   mean_b,
                "llm":        mean_l,
                "random":     mean_r,
                "rule_based": mean_rb,
                "optuna":     mean_o if has_optuna else None,
            },
            "rmse": {
                "baseline":   _safe_stats(b_rmses),
                "llm":        _safe_stats(l_rmses),
                "random":     _safe_stats(r_rmses),
                "rule_based": _safe_stats(rb_rmses),
                "optuna":     _safe_stats(o_rmses),
            },
            # ── R² ───────────────────────────────────────────────────────
            "mean_r2": {
                "baseline":   float(np.nanmean(b_r2s)),
                "llm":        float(np.nanmean(l_r2s)),
                "random":     float(np.nanmean(r_r2s)),
                "rule_based": float(np.nanmean(rb_r2s)),
                "optuna":     float(np.nanmean(o_r2s)) if has_optuna else None,
            },
            "r2": {
                "baseline":   _safe_stats(b_r2s),
                "llm":        _safe_stats(l_r2s),
                "random":     _safe_stats(r_r2s),
                "rule_based": _safe_stats(rb_r2s),
                "optuna":     _safe_stats(o_r2s),
            },
            # ── Comparative ──────────────────────────────────────────────
            "pct_improvement_over_baseline": (
                (mean_b - mean_l) / mean_b * 100
                if mean_b and np.isfinite(mean_b) else None
            ),
            "pct_improvement_rule_based_over_baseline": (
                (mean_b - mean_rb) / mean_b * 100
                if mean_b and np.isfinite(mean_b) else None
            ),
            "pct_improvement_optuna_over_baseline": (
                (mean_b - mean_o) / mean_b * 100
                if (mean_b and np.isfinite(mean_b) and has_optuna and np.isfinite(mean_o)) else None
            ),
            "win_rate": (
                float(np.nanmean([l < b for l, b in zip(l_rmses, b_rmses)]))
                if l_rmses else None
            ),
            "win_rate_rule_based": (
                float(np.nanmean([rb < b for rb, b in zip(rb_rmses, b_rmses)]))
                if rb_rmses else None
            ),
            "win_rate_optuna": (
                float(np.nanmean([o < b for o, b in zip(o_rmses, b_rmses) if np.isfinite(o)]))
                if has_optuna else None
            ),
        },
        "optimization_efficiency": {
            "trials_to_threshold": {
                "llm":        _safe_mean(llm_trials),
                "random":     _safe_mean(random_trials),
                "rule_based": _safe_mean(rb_trials),
                "optuna":     _safe_mean(optuna_trials) if has_optuna else None,
                "threshold":  threshold,
            },
            "auc_val_loss_curve": {
                "baseline":   _safe_mean(aucs["baseline"]),
                "llm":        _safe_mean(aucs["llm"]),
                "random":     _safe_mean(aucs["random"]),
                "rule_based": _safe_mean(aucs["rule_based"]),
                "optuna":     _safe_mean(aucs["optuna"]) if aucs["optuna"] else None,
            },
        },
    }


# ---------------------------------------------------------------------------
# Single experiment (baseline + LLM + random)
# ---------------------------------------------------------------------------

async def run_experiment(
    config: Config,
    run_id: int,
    llm_model: str,
) -> Dict[str, Any]:
    results = {}

    def _new_model(dataset: Dict, cfg: Config):
        return LSTM_Localizer(
            input_features=dataset["input_dim"],
            target_dim=dataset["target_dim"],
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
        ).to(cfg.device)

    def _new_trainer(model, cfg, dataset, train_loader, val_loader, prefix):
        return Trainer(
            model=model, config=cfg, scaler_y=dataset["processor"].scaler_y,
            train_loader=train_loader, val_loader=val_loader,
            dataset_dict=dataset, checkpoint_prefix=prefix,
        )

    # ── 1. Baseline ──────────────────────────────────────────────────────────
    # Use exactly optimization_rounds * epochs_per_round (no +1 extra budget).
    b_cfg        = copy.deepcopy(config)
    total_epochs = b_cfg.epochs_per_round
    b_dataset, b_train_loader, b_val_loader, b_test_loader = build_dataset_and_loaders(b_cfg)
    logger.info("--- Run %d | BASELINE (total_epochs=%d) ---", run_id, total_epochs)
    b_trainer = _new_trainer(
        _new_model(b_dataset, b_cfg),
        b_cfg,
        b_dataset,
        b_train_loader,
        b_val_loader,
        f"baseline_run{run_id}",
    )
    b_trainer.train_n_epochs(total_epochs)
    b_history = b_trainer.history

    # Evaluate the best model seen during training, not the (possibly
    # overshot) final-epoch model — early stopping is disabled.
    b_trainer.restore_best_checkpoint()
    b_eval = Evaluator(b_trainer.model, b_cfg, b_dataset["processor"].scaler_y)
    b_preds, b_targets = b_eval.predict(b_test_loader)
    b_metrics = b_eval.compute_metrics(b_preds, b_targets)

    b_eval.plot_training_history(b_history, filename=f"baseline_history_run{run_id}.png")
    b_eval.plot_predictions(b_preds, b_targets,    filename=f"baseline_predictions_run{run_id}.png")
    b_eval.plot_error_distribution(b_preds, b_targets, filename=f"baseline_errors_run{run_id}.png")
    b_trainer.save_checkpoint(f"baseline_best_run{run_id}.pt")

    # Baseline best_curve: flat reference at the final best val_loss
    # (single continuous training pass — no per-round adaptation)
    b_best_vl    = b_trainer.best_val_loss
    b_best_curve = [float(b_best_vl)] * max(b_cfg.optimization_rounds, 1)

    results["baseline"] = {"metrics": b_metrics, "history": b_history,
                           "best_curve": b_best_curve}
    logger.info("Baseline RMSE=%.4f  R\u00b2=%.4f", b_metrics["rmse"], b_metrics["r2"])

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

    # Test only AFTER selection (interim single-seed; Step 5 → 30 seeds).
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
    logger.info("LLM RMSE=%.4f  R\u00b2=%.4f", l_metrics["rmse"], l_metrics["r2"])

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
    # the legacy metrics/plot slots; Step 5 replaces this with the 30-seed
    # final evaluation + paired differences.
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
    logger.info("Random RMSE=%.4f  R\u00b2=%.4f", r_metrics["rmse"], r_metrics["r2"])

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
        rb_agent = RuleBasedOptimizer(allow_arch_changes=config.allow_arch_changes)
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
    logger.info("Rule-Based RMSE=%.4f  R\u00b2=%.4f", rb_metrics["rmse"], rb_metrics["r2"])

    # \u2500\u2500 5. Optuna (TPE / Bayesian) search \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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
        # Test only AFTER selection (interim single-seed; Step 5 → 30 seeds).
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
        logger.info("Optuna RMSE=%.4f  R\u00b2=%.4f", o_metrics["rmse"], o_metrics["r2"])
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


# ---------------------------------------------------------------------------
# Full experiment (N runs → aggregate → report)
# ---------------------------------------------------------------------------

async def run_full_experiment(
    config: Config,
    llm_model: str,
    *,
    seed: Optional[int] = None,
):
    """Run the full experiment (all runs) for a single seed.

    If *seed* is provided it overrides config.seed so that each seed
    replication is truly independent.
    """
    if seed is not None:
        config.seed = seed
    set_seed(config.seed)

    all_results = []

    for run_id in range(1, config.num_experiment_runs + 1):
        set_seed(config.seed + run_id)
        logger.info(
            "\n%s\nRUN %d / %d\n%s",
            "=" * 60, run_id, config.num_experiment_runs, "=" * 60,
        )
        result = await run_experiment(
            config=config,
            run_id=run_id,
            llm_model=llm_model,
        )
        all_results.append(result)

    cross = compute_cross_run_metrics(all_results)
    with open(config.output_dir / "cross_run_metrics.json", "w") as f:
        json.dump(cross, f, indent=2, default=str)

    # Cross-run summary: all runs overlaid, one panel per optimiser.
    plot_cross_run_summary(config, all_results, filename="cross_run_summary.png")

    q = cross["optimization_quality"]
    e = cross["optimization_efficiency"]

    logger.info('\\n%s\\nEXPERIMENT COMPLETE\\n%s', '=' * 60, '=' * 60)
    logger.info('Results saved to: %s', config.output_dir)
    logger.info(
        'RMSE | LLM %.4f (+-%.4f)  Baseline %.4f (+-%.4f)  Random %.4f (+-%.4f)  Rule-Based %.4f (+-%.4f)',
        q['rmse']['llm']['mean']        or 0, q['rmse']['llm']['std']        or 0,
        q['rmse']['baseline']['mean']   or 0, q['rmse']['baseline']['std']   or 0,
        q['rmse']['random']['mean']     or 0, q['rmse']['random']['std']     or 0,
        q['rmse']['rule_based']['mean'] or 0, q['rmse']['rule_based']['std'] or 0,
    )
    logger.info(
        'R2   | LLM %.4f (+-%.4f)  Baseline %.4f (+-%.4f)  Random %.4f (+-%.4f)  Rule-Based %.4f (+-%.4f)',
        q['r2']['llm']['mean']        or 0, q['r2']['llm']['std']        or 0,
        q['r2']['baseline']['mean']   or 0, q['r2']['baseline']['std']   or 0,
        q['r2']['random']['mean']     or 0, q['r2']['random']['std']     or 0,
        q['r2']['rule_based']['mean'] or 0, q['r2']['rule_based']['std'] or 0,
    )
    logger.info('LLM improvement over baseline:        %.1f%%', q['pct_improvement_over_baseline'] or 0)
    logger.info('Rule-Based improvement over baseline: %.1f%%', q['pct_improvement_rule_based_over_baseline'] or 0)
    if q.get('pct_improvement_optuna_over_baseline') is not None:
        logger.info('Optuna improvement over baseline:     %.1f%%', q['pct_improvement_optuna_over_baseline'])
    logger.info('LLM win rate across runs:        %.0f%%', (q['win_rate'] or 0) * 100)
    logger.info('Rule-Based win rate across runs: %.0f%%', (q['win_rate_rule_based'] or 0) * 100)
    if q.get('win_rate_optuna') is not None:
        logger.info('Optuna win rate across runs:     %.0f%%', q['win_rate_optuna'] * 100)
    logger.info(
        'Avg trials to threshold | LLM %.1f  Random %.1f  Rule-Based %.1f  Optuna %s',
        e['trials_to_threshold']['llm']        or 0,
        e['trials_to_threshold']['random']     or 0,
        e['trials_to_threshold']['rule_based'] or 0,
        (f"{e['trials_to_threshold']['optuna']:.1f}"
         if e['trials_to_threshold'].get('optuna') is not None else "n/a"),
    )


# ---------------------------------------------------------------------------
# 2. WRAP YOUR OPTIMIZATION  (per-seed isolation)
# ---------------------------------------------------------------------------

async def run_one_seed(
    base_config: Config,
    seed: int,
    llm_model: str,
) -> Dict[str, Any]:
    """
    Run the complete experiment pipeline for a single *seed*.

    Each seed gets its own sub-directory (``<output_dir>/seed_<seed>``)
    so that plots, checkpoints and logs from different seeds never
    collide, making every replication fully isolated and reproducible.

    Returns a dict that mirrors the structure of a single-seed
    ``all_results`` list, tagged with the seed value.
    """

    cfg = copy.deepcopy(base_config)
    cfg.output_dir = base_config.output_dir / f"seed_{seed}"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n%s\nSEED %d  →  %s\n%s", "#" * 60, seed, cfg.output_dir, "#" * 60)

    await run_full_experiment(
        config=cfg,
        llm_model=llm_model,
        seed=seed,
    )

    # Collect the final cross-run metrics produced by run_full_experiment.
    metrics_path = cfg.output_dir / "cross_run_metrics.json"
    cross_metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as fh:
            cross_metrics = json.load(fh)

    return {"seed": seed, "output_dir": str(cfg.output_dir), "cross_run_metrics": cross_metrics}


# ---------------------------------------------------------------------------
# 3. MULTI-SEED RUNNER
# ---------------------------------------------------------------------------

async def run_multi_seed_experiment(
    base_config: Config,
    seeds: List[int],
    llm_model: str,
) -> None:
    """
    Run the full experiment pipeline for every seed in *seeds* and
    aggregate the results into a single ``multi_seed_summary.json``
    written to ``base_config.output_dir``.

    Aggregated statistics include mean ± std across seeds for:
    - RMSE (baseline, LLM, random)
    - LLM % improvement over baseline
    - LLM win rate
    - Trials-to-threshold (LLM, random)
    """
    logger.info("\n%s\nMULTI-SEED EXPERIMENT\nSeeds: %s\n%s",
                "=" * 60, seeds, "=" * 60)

    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        result = await run_one_seed(
            base_config=base_config,
            seed=seed,
            llm_model=llm_model,
        )
        seed_results.append(result)

    # ── Aggregate cross-seed metrics ──────────────────────────────────────
    def _collect(key_path: List[str]) -> List[Optional[float]]:
        """Extract a nested value from each seed's cross_run_metrics."""
        vals = []
        for r in seed_results:
            obj = r.get("cross_run_metrics", {})
            for k in key_path:
                if isinstance(obj, dict):
                    obj = obj.get(k)
                else:
                    obj = None
                    break
            vals.append(float(obj) if obj is not None and np.isfinite(float(obj)) else None)
        return vals

    def _stats(vals: List[Optional[float]]) -> Dict[str, Optional[float]]:
        finite = [v for v in vals if v is not None]
        if not finite:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        arr = np.array(finite)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "n":    len(arr),
        }

    summary: Dict[str, Any] = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "per_seed": seed_results,
        "aggregated": {
            "rmse": {
                "baseline":   _stats(_collect(["optimization_quality", "mean_rmse", "baseline"])),
                "llm":        _stats(_collect(["optimization_quality", "mean_rmse", "llm"])),
                "random":     _stats(_collect(["optimization_quality", "mean_rmse", "random"])),
                "rule_based": _stats(_collect(["optimization_quality", "mean_rmse", "rule_based"])),
                "optuna":     _stats(_collect(["optimization_quality", "mean_rmse", "optuna"])),
            },
            "r2": {
                "baseline":   _stats(_collect(["optimization_quality", "mean_r2", "baseline"])),
                "llm":        _stats(_collect(["optimization_quality", "mean_r2", "llm"])),
                "random":     _stats(_collect(["optimization_quality", "mean_r2", "random"])),
                "rule_based": _stats(_collect(["optimization_quality", "mean_r2", "rule_based"])),
                "optuna":     _stats(_collect(["optimization_quality", "mean_r2", "optuna"])),
            },
            "pct_improvement_over_baseline": _stats(
                _collect(["optimization_quality", "pct_improvement_over_baseline"])
            ),
            "pct_improvement_rule_based_over_baseline": _stats(
                _collect(["optimization_quality", "pct_improvement_rule_based_over_baseline"])
            ),
            "pct_improvement_optuna_over_baseline": _stats(
                _collect(["optimization_quality", "pct_improvement_optuna_over_baseline"])
            ),
            "win_rate":            _stats(_collect(["optimization_quality", "win_rate"])),
            "win_rate_rule_based": _stats(_collect(["optimization_quality", "win_rate_rule_based"])),
            "win_rate_optuna":     _stats(_collect(["optimization_quality", "win_rate_optuna"])),
            "trials_to_threshold": {
                "llm":        _stats(_collect(["optimization_efficiency", "trials_to_threshold", "llm"])),
                "random":     _stats(_collect(["optimization_efficiency", "trials_to_threshold", "random"])),
                "rule_based": _stats(_collect(["optimization_efficiency", "trials_to_threshold", "rule_based"])),
                "optuna":     _stats(_collect(["optimization_efficiency", "trials_to_threshold", "optuna"])),
            },
            "auc_val_loss_curve": {
                "baseline":   _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "baseline"])),
                "llm":        _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "llm"])),
                "random":     _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "random"])),
                "rule_based": _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "rule_based"])),
                "optuna":     _stats(_collect(["optimization_efficiency", "auc_val_loss_curve", "optuna"])),
            },
        },
    }

    out_path = base_config.output_dir / "multi_seed_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Multi-seed summary saved → %s", out_path)

    # ── Console report ────────────────────────────────────────────────────
    agg = summary["aggregated"]
    logger.info("\n%s\nMULTI-SEED SUMMARY  (%d seeds)\n%s", "=" * 60, len(seeds), "=" * 60)
    for arm in ["baseline", "llm", "random", "rule_based"]:
        rs = agg["rmse"][arm]
        r2s = agg["r2"][arm]
        logger.info(
            "RMSE %-8s | mean=%.4f  std=%.4f  [%.4f, %.4f]",
            arm, rs["mean"] or 0, rs["std"] or 0, rs["min"] or 0, rs["max"] or 0,
        )
        logger.info(
            "R²   %-8s | mean=%.4f  std=%.4f  [%.4f, %.4f]",
            arm, r2s["mean"] or 0, r2s["std"] or 0, r2s["min"] or 0, r2s["max"] or 0,
        )
    imp = agg["pct_improvement_over_baseline"]
    logger.info(
        "LLM improvement over baseline | mean=%.1f%%  std=%.1f%%",
        imp["mean"] or 0, imp["std"] or 0,
    )
    wr = agg["win_rate"]
    logger.info(
        "LLM win rate across seeds     | mean=%.0f%%  std=%.0f%%",
        (wr["mean"] or 0) * 100, (wr["std"] or 0) * 100,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        logger.info("Smoke-test mode: 1 run \u00d7 2 rounds \u00d7 5 epochs \u00d7 2 final-eval seeds")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
