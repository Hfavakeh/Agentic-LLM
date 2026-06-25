"""Fixed-reference baseline arm.

Not a search method — it trains the `Config` defaults once (no per-round
adaptation) and evaluates on the test set. Extracted into its own module so
every arm in the bake-off (baseline, LLM, random, rule-based, Optuna) lives
side by side in `arms/`.
"""

import copy
from typing import Any, Dict

from pipeline import Config, Evaluator, LSTM_Localizer, Trainer, logger

from .engine import build_dataset_and_loaders


def run_baseline(config: Config, run_id: int) -> Dict[str, Any]:
    """Train the fixed-reference defaults once and evaluate on test.

    Uses exactly `optimization_rounds * epochs_per_round` epochs (no +1 extra
    budget). Evaluates the best model seen during training (restored from the
    checkpoint), not the possibly-overshot final-epoch model. Returns the
    `results["baseline"]` slot: metrics, full history, and a flat best-curve
    reference (a single continuous training pass, so no per-round curve).
    """
    b_cfg = copy.deepcopy(config)
    total_epochs = b_cfg.epochs_per_round
    b_dataset, b_train_loader, b_val_loader, b_test_loader = build_dataset_and_loaders(b_cfg)
    logger.info("--- Run %d | BASELINE (total_epochs=%d) ---", run_id, total_epochs)

    model = LSTM_Localizer(
        input_features=b_dataset["input_dim"],
        target_dim=b_dataset["target_dim"],
        lstm_hidden=b_cfg.lstm_hidden,
        lstm_layers=b_cfg.lstm_layers,
        dropout=b_cfg.dropout,
    ).to(b_cfg.device)
    trainer = Trainer(
        model=model, config=b_cfg, scaler_y=b_dataset["processor"].scaler_y,
        train_loader=b_train_loader, val_loader=b_val_loader,
        dataset_dict=b_dataset, checkpoint_prefix=f"baseline_run{run_id}",
    )
    trainer.train_n_epochs(total_epochs)
    history = trainer.history

    # Evaluate the best model seen during training, not the (possibly
    # overshot) final-epoch model — early stopping is disabled here.
    trainer.restore_best_checkpoint()
    evaluator = Evaluator(trainer.model, b_cfg, b_dataset["processor"].scaler_y)
    preds, targets = evaluator.predict(b_test_loader)
    metrics = evaluator.compute_metrics(preds, targets)

    evaluator.plot_training_history(history, filename=f"baseline_history_run{run_id}.png")
    evaluator.plot_predictions(preds, targets, filename=f"baseline_predictions_run{run_id}.png")
    evaluator.plot_error_distribution(preds, targets, filename=f"baseline_errors_run{run_id}.png")
    trainer.save_checkpoint(f"baseline_best_run{run_id}.pt")

    # Baseline best_curve: flat reference at the final best val_loss
    # (single continuous training pass — no per-round adaptation).
    best_curve = [float(trainer.best_val_loss)] * max(b_cfg.optimization_rounds, 1)

    logger.info("Baseline RMSE=%.4f  R²=%.4f", metrics["rmse"], metrics["r2"])
    return {"metrics": metrics, "history": history, "best_curve": best_curve}
