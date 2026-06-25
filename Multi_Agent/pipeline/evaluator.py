"""Test-set prediction, metric computation, and per-run diagnostic plots."""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import Config
from .logging_setup import logger


class Evaluator:
    """Predict, compute metrics, and produce plots."""

    def __init__(self, model: nn.Module, config: Config, scaler_y: StandardScaler):
        self.model    = model
        self.config   = config
        self.scaler_y = scaler_y

    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                X_batch = batch["X"].to(self.config.device)
                y_batch = batch["y"]
                preds.append(self.model(X_batch).cpu().numpy())
                targets.append(y_batch.numpy())
        preds   = np.vstack(preds)
        targets = np.vstack(targets)
        nan_mask = np.isnan(preds)
        if nan_mask.any():
            nan_count = int(nan_mask.sum())
            sample_idx = np.argwhere(nan_mask)[:5].tolist()
            if self.config.strict_prediction_checks:
                raise ValueError(
                    f"NaN detected in predictions ({nan_count} values). "
                    f"Sample indices: {sample_idx}"
                )
            logger.warning(
                "NaN in predictions (%d values). strict_prediction_checks=False, replacing with 0. "
                "Sample indices: %s",
                nan_count, sample_idx
            )
            preds = np.nan_to_num(preds, nan=0.0)
        return self.scaler_y.inverse_transform(preds), self.scaler_y.inverse_transform(targets)

    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
        r2 = float(r2_score(targets, predictions, multioutput="uniform_average"))
        distances = np.sqrt(np.sum((predictions - targets) ** 2, axis=1))
        return {
            "rmse": rmse, "r2": r2,
            "mean_distance":   float(np.mean(distances)),
            "median_distance": float(np.median(distances)),
            "max_distance":    float(np.max(distances)),
            "std_distance":    float(np.std(distances)),
        }

    # -- Plots --

    def plot_training_history(self, history: Dict, filename: str = "training_history.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax, key, title in [
            (ax1, "loss", "Training Loss (MSE)"),
            (ax2, "mae",  "Training MAE"),
        ]:
            ax.plot(history[f"train_{key}"], label="Train")
            ax.plot(history[f"val_{key}"],   label="Validation")
            ax.set(xlabel="Epoch", title=title)
            ax.legend()
            ax.grid(alpha=0.3)
        self._save_fig(fig, filename)

    def plot_error_distribution(self, preds: np.ndarray, targets: np.ndarray,
                                filename: str = "error_distribution.png"):
        if np.isnan(preds).any() or np.isnan(targets).any():
            logger.error("Skipping error distribution plot — data contains NaN.")
            return
        errors = np.linalg.norm(preds - targets, axis=1)
        if not np.isfinite(errors).all():
            logger.error("Skipping error distribution plot — non-finite errors.")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(errors, bins=50, alpha=0.7)
        ax1.set(title="Euclidean Error Distribution", xlabel="Error", ylabel="Frequency")
        ax1.grid(alpha=0.3)
        sorted_err = np.sort(errors)
        ax2.plot(sorted_err, np.arange(1, len(sorted_err) + 1) / len(sorted_err))
        ax2.axhline(0.95, linestyle="--")
        ax2.set(title="Cumulative Error Distribution", xlabel="Error", ylabel="CDF")
        ax2.grid(alpha=0.3)
        self._save_fig(fig, filename)

    def plot_predictions(self, preds: np.ndarray, targets: np.ndarray,
                         filename: str = "predictions_vs_true.png"):
        if np.isnan(preds).any() or np.isnan(targets).any():
            logger.error("Skipping predictions plot — data contains NaN.")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for ax, dim, label in [(ax1, 0, "X"), (ax2, 1, "Y")]:
            lo, hi = targets[:, dim].min(), targets[:, dim].max()
            ax.scatter(targets[:, dim], preds[:, dim], alpha=0.6)
            ax.plot([lo, hi], [lo, hi], "r--")
            ax.set(xlabel=f"True {label}", ylabel=f"Pred {label}",
                   title=f"{label}: Prediction vs Truth")
            ax.grid(alpha=0.3)
        self._save_fig(fig, filename)

    def _save_fig(self, fig, filename: str):
        path = self.config.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Plot saved -> %s", path)
