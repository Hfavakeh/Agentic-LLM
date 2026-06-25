"""Optuna (TPE / Bayesian) search arm.

The "competent conventional optimizer" arm. It shares the SAME from-scratch
engine (evaluate_setting: 3 trainings per setting, mean val RMSE score) and
the SAME discrete HP_GRID as every other arm — the only difference vs random
is the sampler: Optuna's TPE learns from past trial scores to bias later
draws toward promising regions of HP space.
"""

from typing import Any, Dict, List, Optional

from pipeline import Config, HP_GRID, is_finite_number, logger

from .engine import _attempt_record, evaluate_setting

OPTUNA_SAMPLER_SEED = 1000        # protocol: fixed, reported optimizer seed
OPTUNA_N_STARTUP_TRIALS = 5       # protocol: first 5 trials random, rest TPE-guided


def _suggest_from_optuna_trial(trial, allow_arch_changes: bool = True) -> Dict[str, Any]:
    """Translate one Optuna trial into a setting dict with the same shape
    ``sample_random_hparams`` produces. Values are drawn from ``HP_GRID`` via
    suggest_categorical so the reachable set is identical across all arms.
    """
    # Optuna draws from the SAME discrete grid as the random and LLM arms.
    # Using suggest_categorical over the exact value lists (rather than
    # continuous suggest_float) keeps the reachable set identical across arms,
    # so the random <-> Optuna comparison isolates TPE's smart sampling alone.
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
