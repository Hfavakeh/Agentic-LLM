"""Random-search arm: uniform draws from the shared discrete HP grid."""

import random
from typing import Any, Dict, List, Optional

from pipeline import Config, HP_GRID, is_finite_number, logger

from .engine import _attempt_record, evaluate_setting


def sample_random_hparams(allow_arch_changes: bool = True, rng=None) -> Dict[str, Any]:
    """Sample a random HP configuration from the same search space as the LLM.

    Every hyperparameter range matches the agent system prompt so
    Random Search and LLM explore identical territories.

    Parameters
    ----------
    allow_arch_changes : bool, default True
        When False, ``lstm_hidden`` and ``lstm_layers`` are omitted from the
        sample so the random arm cannot redraw the architecture every round.
        This mirrors the LLM arm's behaviour under ``--no-arch-changes`` —
        under that ablation the LLM's validator strips the same two keys from
        any proposal, so excluding them here keeps the two arms' effective
        search spaces aligned. Other arch-adjacent levers (``window_size``,
        ``batch_size``) are still drawn because the LLM can also still change
        them — ``window_size`` triggers a model rebuild and ``batch_size``
        only rebuilds the DataLoader.
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
