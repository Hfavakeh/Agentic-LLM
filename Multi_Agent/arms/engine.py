"""The from-scratch evaluation engine — the protocol's core unit of work.

One "attempt" in any arm = one HP setting trained ``len(TRAIN_SEEDS)`` times
FROM SCRATCH (seeds 101/102/103), with the per-setting score defined as the
MEAN validation RMSE over those trainings. The three trainings reduce noise
so the optimizer (random / Optuna / rule-based / LLM) decides on the average,
not on one lucky/unlucky run. The test set is never touched here —
``train_and_test_setting`` is called only AFTER a method selects its best.
"""

import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader

from motion_descriptors import (
    extract_motion_features, llm_payload as motion_llm_payload, summarize_motion,
)
from pipeline import (
    Config, DataProcessor, Evaluator, LSTM_Localizer, TimeSeriesDataset, Trainer,
    is_finite_number, logger, set_seed,
)

# Protocol seeds. The same three fixed seeds train EVERY setting in EVERY arm,
# so the only thing that differs between attempts/arms is the setting itself.
# The optimizer's own randomness (random-search RNG, Optuna sampler) uses a
# SEPARATE seed and must not be confused with these.
TRAIN_SEEDS: tuple = (101, 102, 103)

# Search-space keys an arm may set on a setting. Motion loss-shaping levers are
# deferred from the search space, so they are dropped if a proposal includes
# them.
_SETTING_KEYS = {
    "learning_rate", "weight_decay", "dropout", "batch_size",
    "lstm_hidden", "lstm_layers", "window_size", "optimizer_choice", "patience",
}


# ---------------------------------------------------------------------------
# Setting identity / coercion
# ---------------------------------------------------------------------------

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
# Core evaluation
# ---------------------------------------------------------------------------

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
        # ── Per-epoch curve SHAPE features (Q2) ───────────────────────────────
        # Computed from the metres validation curve (the early-stopping monitor,
        # same metric as the score). These capture what the best-epoch scalars
        # cannot: whether the curve was still descending at the budget limit,
        # whether validation rose after its minimum (overfitting onset over
        # epochs), and how noisy the trajectory was. Converted to a qualitative
        # label only in the prompt; raw numbers stay here / in the logs.
        curve = _curve_features([v for v in pos_m if is_finite_number(v)],
                                epochs_trained=len(vl), max_epochs=max_epochs)
        per_seed.append({
            "seed":           int(seed),
            "val_rmse":       val_rmse,
            "best_epoch":     best_epoch,
            "epochs_trained": len(vl),
            "train_loss":     tr_at,
            "val_loss":       vl_at,
            "train_val_gap":  (vl_at - tr_at) if (is_finite_number(vl_at) and is_finite_number(tr_at)) else float("nan"),
            "runtime_s":      runtime_s,
            "curve":          curve,
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
        "curve_summary":   _aggregate_curves([p.get("curve") for p in per_seed]),
        "runtime_s":       float(time.perf_counter() - t0),
        "n_trainings":     len(per_seed),
        "setting":         setting,
        "per_seed":        per_seed,
    }


def _curve_features(vc: List[float], epochs_trained: int, max_epochs: int) -> Optional[Dict[str, float]]:
    """Per-epoch shape features of one seed's metres validation curve.

    `vc` is the finite val-position-RMSE-in-metres trajectory over epochs.
    Returns None when the curve is too short to read a shape from.
      upturn : (val_last - val_best)/val_best  — overfitting rise after the minimum
      osc    : mean |Δ| between consecutive epochs / mean level (post-warmup) — noise
      still  : 1.0 if the best epoch is at the very tail (still descending), else 0.0
      hit_max: 1.0 if training reached the epoch budget (early stopping never fired)
      best_frac: best epoch position as a fraction of epochs trained
    """
    n = len(vc)
    if n < 4:
        return None
    bidx  = int(np.argmin(vc))
    vbest = float(vc[bidx])
    vlast = float(vc[-1])
    upturn = (vlast - vbest) / vbest if vbest > 0 else 0.0
    tail = vc[2:] if n > 4 else vc                       # drop the noisy warmup epochs
    diffs = np.abs(np.diff(tail))
    level = float(np.mean(tail)) if len(tail) else 0.0
    osc = float(np.mean(diffs) / level) if level > 0 and len(diffs) else 0.0
    return {
        "upturn":    float(upturn),
        "osc":       osc,
        "still":     1.0 if bidx >= n - 2 else 0.0,
        "hit_max":   1.0 if epochs_trained >= max_epochs else 0.0,
        "best_frac": (bidx + 1) / n,
    }


def _aggregate_curves(curves: List[Optional[Dict[str, float]]]) -> Dict[str, float]:
    """Average the per-seed curve features into the run-level summary the
    qualitative `_qual_curve_shape` label reads."""
    valid = [c for c in curves if isinstance(c, dict)]
    if not valid:
        return {}
    def _m(k: str) -> float:
        vals = [c[k] for c in valid if is_finite_number(c.get(k))]
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "mean_upturn":          _m("upturn"),
        "mean_oscillation":     _m("osc"),
        "frac_still_improving": _m("still"),
        "frac_hit_max":         _m("hit_max"),
        "mean_best_frac":       _m("best_frac"),
        "n_curves":             len(valid),
    }


def train_and_test_setting(
    base_cfg: Config,
    dataset: Dict[str, Any],
    setting: Dict[str, Any],
    seed: int,
    max_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one setting from scratch on a single seed and evaluate it on the
    TEST set. Used ONLY after a method has selected its best setting (the
    protocol allows the test set only at that point). The final evaluation
    calls this across the 30 fresh seeds; the interim search arms call it once
    on a training seed just to populate the legacy `metrics`/plots slots.
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
