"""Training loop: early stopping, hyperparameter hot-swap, checkpointing,
and the (currently deferred) motion-aware loss shaping."""

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import Config
from .logging_setup import logger
from .model import LSTM_Localizer

# ---------------------------------------------------------------------------
# Motion-aware loss shaping (Phase-3 scientific-core feature)
# ---------------------------------------------------------------------------

# Default loss-shaping config. With these values `_compute_total_loss`
# reduces *exactly* to plain MSE, so motion shaping is fully opt-in and
# baseline behaviour is unchanged until a controller proposes non-default
# values. A controller (LLM or motion-aware rule-based) overrides these to
# steer the training objective:
#   v_max         — physical human speed ceiling in m/s (velocity prior).
#   lambda_vel    — weight of the velocity-plausibility penalty (0 = off).
#   lambda_smooth — weight of the acceleration/smoothness penalty (0 = off).
#   bin_weights   — per-speed-bin MSE weights [slow, medium, fast]
#                   (None or all-equal = uniform = plain MSE).
#   bin_edges     — speed thresholds (m/s) separating the bins; len == len(bin_weights) - 1.
DEFAULT_LOSS_SHAPING: Dict[str, Any] = {
    "v_max":         2.0,
    "lambda_vel":    0.0,
    "lambda_smooth": 0.0,
    "bin_weights":   None,
    "bin_edges":     None,
}


class Trainer:
    """Training, validation, hyperparameter updates, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        scaler_y: StandardScaler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset_dict: Dict[str, Any],
        checkpoint_prefix: str = "trainer",
    ):
        self.model             = model
        self.config            = config
        self.scaler_y          = scaler_y
        self.train_loader      = train_loader
        self.val_loader        = val_loader
        self.dataset_dict      = dataset_dict
        self.checkpoint_prefix = checkpoint_prefix

        self.optimizer = self._create_optimizer(
            model.parameters(), config.learning_rate, config.weight_decay,
            getattr(config, 'optimizer_choice', 'adam'),
        )
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [],
            "val_distance_m": [],
            # Plain position MSE per epoch — diverges from `val_loss` once
            # motion-aware loss shaping is active; used for true RMSE.
            "val_position_loss": [],
            # Same plain position MSE but in ORIGINAL units (metres), i.e.
            # computed on inverse-transformed preds/targets. This is the
            # selection objective so search ranking matches the meters-space
            # test RMSE the headline reports (compute_metrics).
            "val_position_loss_m": [],
        }
        self.hyperparams = {
            "learning_rate":    config.learning_rate,
            "dropout":          config.dropout,
            "weight_decay":     config.weight_decay,
            "batch_size":       config.batch_size,
            "lstm_hidden":      model.lstm.hidden_size,
            "lstm_layers":      model.lstm.num_layers,
            "optimizer_choice": getattr(config, 'optimizer_choice', 'adam'),
            # Searchable early-stopping patience (8/12/16). Drives the stop
            # rule in train(); a setting that proposes `patience` overrides the
            # config default without touching the model or optimiser.
            "patience":         int(getattr(config, 'early_stopping_patience', 12)),
        }
        self.best_val_loss: float              = float("inf")
        self.round_best_val_loss: float        = float("inf")
        self.best_metric: float                = float("inf")
        self.round_best_metric: float          = float("inf")
        self.patience_counter: int             = 0
        self.best_checkpoint_path: Optional[Path] = None
        self._cached_val_distance_m: Optional[float] = None   # updated each validate() call
        self._cached_val_distance_std_m: Optional[float] = None  # std of per-sample Euclidean error (m)
        self._cached_val_position_loss: Optional[float] = None  # plain MSE (scaled space), updated each validate()
        self._cached_val_position_loss_m: Optional[float] = None  # plain MSE in metres, updated each validate()
        # Motion-aware loss shaping. Starts at defaults (= plain MSE); a
        # controller overrides entries via apply_loss_shaping_update().
        # `bin_edges` are fixed speed thresholds from the dataset (terciles of
        # training-set speed) — not controller-tunable.
        self.loss_shaping: Dict[str, Any] = dict(DEFAULT_LOSS_SHAPING)
        self.loss_shaping["bin_edges"] = dataset_dict.get("bin_edges")

    # ------------------------------------------------------------------
    # Convenience wrappers (used by the optimisation loops)
    # ------------------------------------------------------------------

    def update_hyperparameters(self, proposed: Dict[str, Any]):
        changes = proposed.get("proposed_changes", proposed)
        resets  = proposed.get("resets_model", False)
        return self.apply_hyperparameter_update(changes, resets_model=resets)

    def apply_loss_shaping_update(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply motion-aware loss-shaping changes to ``self.loss_shaping``.

        Accepts the flat scalar keys a controller proposes — ``v_max``,
        ``lambda_vel``, ``lambda_smooth`` and the three ``bin_weight_*`` keys —
        and folds the bin weights into the ``bin_weights`` list that
        ``_compute_total_loss`` consumes. ``bin_edges`` is dataset-derived and
        never touched here. Returns the subset of keys actually applied (for
        per-round logging). Values are assumed already range-validated by the
        controller's validator.
        """
        applied: Dict[str, Any] = {}
        bin_slot = {"bin_weight_slow": 0, "bin_weight_medium": 1, "bin_weight_fast": 2}
        for key, val in (changes or {}).items():
            if key in ("v_max", "lambda_vel", "lambda_smooth"):
                self.loss_shaping[key] = float(val)
                applied[key] = float(val)
            elif key in bin_slot:
                weights = self.loss_shaping.get("bin_weights")
                weights = [1.0, 1.0, 1.0] if not weights else list(weights)
                weights[bin_slot[key]] = float(val)
                self.loss_shaping["bin_weights"] = weights
                applied[key] = float(val)
        if applied:
            logger.info(
                "Loss-shaping update: %s | active: v_max=%.3g lambda_vel=%.3g "
                "lambda_smooth=%.3g bin_weights=%s",
                applied,
                self.loss_shaping.get("v_max"),
                self.loss_shaping.get("lambda_vel"),
                self.loss_shaping.get("lambda_smooth"),
                self.loss_shaping.get("bin_weights"),
            )
        return applied

    def train_n_epochs(self, n: int):
        """Train for n epochs, restoring config.epochs even if an exception occurs."""
        original = self.config.epochs
        try:
            self.config.epochs = n
            self.train(self.train_loader, self.val_loader)
        finally:
            self.config.epochs = original

    def pareto_metrics(self) -> Dict[str, float]:
        """Per-round cost metrics for the multi-objective Pareto scalarizer.

        Three axes (consumed by `pareto.py`):
          * latency_ms       — mean forward-pass inference time per window,
                               in ms (deployment compute cost).
          * stability_std_m  — std of per-sample Euclidean error in metres
                               (prediction consistency across the trajectory).
          * params_trainable — trainable parameter count (resource proxy).

        Call once per round, after training, so the measurement reflects the
        round's final model/architecture.
        """
        _, params_trainable = self.model.count_parameters()

        # Forward-pass latency: mean wall-time of one forward pass over the
        # val set, expressed per input window (pure inference compute cost).
        self.model.eval()
        n_windows = 0
        t_forward = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                X = batch["X"].to(self.config.device)
                t0 = time.perf_counter()
                self.model(X)
                t_forward += time.perf_counter() - t0
                n_windows += X.size(0)
        forward_ms = (t_forward / n_windows * 1000.0) if n_windows else float("nan")

        stability = self._cached_val_distance_std_m
        return {
            "latency_ms":       float(forward_ms),
            "stability_std_m":  float(stability) if stability is not None else float("nan"),
            "params_trainable": int(params_trainable),
        }

    def sync_data_dependencies(
        self,
        *,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        scaler_y: Optional[StandardScaler] = None,
    ) -> None:
        """Refresh loaders/scalers after dataset rebuilds such as window-size changes."""
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
        if scaler_y is not None:
            self.scaler_y = scaler_y
        # A window-size rebuild re-computes speeds, hence the speed-bin edges.
        # Refresh them from the (in-place updated) dataset dict so the
        # motion-weighted loss term keeps using correct thresholds.
        if self.dataset_dict.get("bin_edges") is not None:
            self.loss_shaping["bin_edges"] = self.dataset_dict["bin_edges"]

    # ------------------------------------------------------------------
    # Core training / validation
    # ------------------------------------------------------------------

    def _compute_total_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        prev_y: Optional[torch.Tensor] = None,
        prev_prev_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Position loss with optional motion-aware shaping.

        With the default loss-shaping config (`lambda_vel=0`,
        `lambda_smooth=0`, `bin_weights=None`) this reduces *exactly* to
        plain MSE, so motion shaping is fully opt-in and baseline behaviour
        is unchanged until a controller sets non-default values.

        Shaping terms (each opt-in, configured via `self.loss_shaping`):
          * speed-bin-weighted MSE — per-sample MSE reweighted by the true
            motion regime (slow / medium / fast) that the sample belongs to.
          * velocity-plausibility prior — penalises predicted speeds above a
            physical human ceiling `v_max` (m/s).
          * smoothness prior — penalises implausible acceleration (jerk),
            computed from the three consecutive points prev_prev -> prev -> pred.

        Velocity and acceleration are evaluated in real-world metres by
        un-scaling displacements with `scaler_y.scale_` (the StandardScaler
        mean cancels for differences), so `v_max` is a genuine physical
        quantity a controller can reason about.

        Returns (total_loss, position_mse): `total_loss` is the shaped
        objective used for back-prop; `position_mse` is the plain unweighted
        MSE, recorded separately so that reported RMSE stays a true RMSE
        rather than sqrt(composite shaped loss).
        """
        ls = self.loss_shaping
        per_sample_mse = ((preds - targets) ** 2).mean(dim=1)            # (B,)
        position_mse = per_sample_mse.mean()                            # plain MSE

        def _scale_y() -> torch.Tensor:
            return torch.as_tensor(
                self.scaler_y.scale_, dtype=preds.dtype, device=preds.device,
            )

        # ── Base position loss (optionally speed-bin weighted) ──
        bin_weights = ls.get("bin_weights")
        bin_edges   = ls.get("bin_edges")
        if (bin_weights and bin_edges and prev_y is not None
                and len(set(bin_weights)) > 1):
            true_speed = (((targets - prev_y) * _scale_y()).norm(dim=1)
                          * float(self.config.hz))                       # (B,) m/s
            edges = torch.as_tensor(bin_edges, dtype=preds.dtype, device=preds.device)
            bin_idx = torch.bucketize(true_speed, edges).clamp(max=len(bin_weights) - 1)
            w = torch.as_tensor(bin_weights, dtype=preds.dtype, device=preds.device)
            sample_w = w[bin_idx]
            base_loss = (per_sample_mse * sample_w).sum() / sample_w.sum().clamp_min(1e-12)
        else:
            base_loss = position_mse

        total = base_loss

        # ── Velocity-plausibility prior ──
        lam_vel = float(ls.get("lambda_vel", 0.0) or 0.0)
        if lam_vel > 0.0 and prev_y is not None:
            pred_speed = (((preds - prev_y) * _scale_y()).norm(dim=1)
                          * float(self.config.hz))                       # (B,) m/s
            v_max = float(ls.get("v_max", 2.0))
            total = total + lam_vel * torch.relu(pred_speed - v_max).pow(2).mean()

        # ── Smoothness / acceleration prior ──
        lam_sm = float(ls.get("lambda_smooth", 0.0) or 0.0)
        if lam_sm > 0.0 and prev_y is not None and prev_prev_y is not None:
            accel = ((preds - prev_y) - (prev_y - prev_prev_y)) * _scale_y()
            total = total + lam_sm * accel.pow(2).sum(dim=1).mean()

        return (total, position_mse)

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = total_mae = total_samples = 0

        for batch in loader:
            X_batch = batch["X"].to(self.config.device)
            y_batch = batch["y"].to(self.config.device)
            prev_y = (batch["prev_y"].to(self.config.device)
                      if "prev_y" in batch else None)
            prev_prev_y = (batch["prev_prev_y"].to(self.config.device)
                           if "prev_prev_y" in batch else None)
            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            (loss, _pos_mse) = self._compute_total_loss(preds, y_batch, prev_y, prev_prev_y)
            if not torch.isfinite(loss):
                logger.error("Non-finite loss — aborting epoch.")
                return float("nan"), float("nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            n = y_batch.size(0)
            total_loss += loss.item() * n
            total_mae  += torch.mean(torch.abs(preds - y_batch)).item() * n
            total_samples += n

        return total_loss / total_samples, total_mae / total_samples

    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = total_mae = total_pos = total_samples = 0
        preds_s: List[np.ndarray] = []
        targets_s: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                X_batch = batch["X"].to(self.config.device)
                y_batch = batch["y"].to(self.config.device)
                prev_y = (batch["prev_y"].to(self.config.device)
                          if "prev_y" in batch else None)
                prev_prev_y = (batch["prev_prev_y"].to(self.config.device)
                               if "prev_prev_y" in batch else None)
                preds = self.model(X_batch)
                (loss, pos_mse) = self._compute_total_loss(preds, y_batch, prev_y, prev_prev_y)
                if not torch.isfinite(loss):
                    logger.error("Non-finite validation loss.")
                    self._cached_val_distance_m = None
                    self._cached_val_distance_std_m = None
                    self._cached_val_position_loss = None
                    self._cached_val_position_loss_m = None
                    return float("nan"), float("nan")
                n = y_batch.size(0)
                total_loss += loss.item() * n
                total_pos  += pos_mse.item() * n
                total_mae  += torch.mean(torch.abs(preds - y_batch)).item() * n
                total_samples += n
                preds_s.append(preds.cpu().numpy())
                targets_s.append(y_batch.cpu().numpy())

        # Cache Euclidean distance so _compute_val_distance_m does not need a
        # second validation forward pass.
        try:
            p_inv = self.scaler_y.inverse_transform(np.vstack(preds_s))
            t_inv = self.scaler_y.inverse_transform(np.vstack(targets_s))
            per_sample_dist = np.sqrt(np.sum((p_inv - t_inv) ** 2, axis=1))
            self._cached_val_distance_m     = float(np.mean(per_sample_dist))
            self._cached_val_distance_std_m = float(np.std(per_sample_dist))
            # Metres-space position MSE, same functional as compute_metrics'
            # RMSE (mean of squared error over all elements). sqrt of this is
            # the selection objective, so search ranks on the same quantity the
            # headline test RMSE uses — just on the validation set.
            self._cached_val_position_loss_m = float(np.mean((p_inv - t_inv) ** 2))
        except Exception:
            self._cached_val_distance_m     = None
            self._cached_val_distance_std_m = None
            self._cached_val_position_loss_m = None

        # Plain position MSE (prior-free) — kept separate from the shaped
        # `val_loss` so reported RMSE = sqrt(position MSE) stays a true RMSE.
        self._cached_val_position_loss = total_pos / total_samples

        return total_loss / total_samples, total_mae / total_samples

    def train(self, train_loader=None, val_loader=None) -> Dict:
        """Full training loop with early stopping.

        Early-stopping rule (identical for every arm): monitor the validation
        position MSE in METRES (`val_position_loss_m` — the same metric the
        search score and the headline test RMSE use); if it does not improve for
        `patience` consecutive epochs, stop and restore the best-epoch weights.
        Falls back to the scaled `val_loss` only on epochs where the metres
        metric is unavailable. `patience` is the per-setting HP (8/12/16),
        falling back to the config default. State is local to this call, so each
        from-scratch training is judged on its own best epoch.
        """
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader

        es_enabled  = bool(getattr(self.config, "enable_early_stopping", False))
        es_patience = int(self.hyperparams.get("patience")
                          or getattr(self.config, "early_stopping_patience", 12))
        es_best     = float("inf")
        es_best_state = None
        es_counter  = 0
        self.best_epoch_in_call = 0   # 1-based best epoch of THIS training

        for epoch in range(self.config.epochs):
            tr_loss, tr_mae = self.train_epoch(self.train_loader)
            vl_loss, vl_mae = self.validate(self.val_loader)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_mae"].append(tr_mae)
            self.history["val_mae"].append(vl_mae)
            self.history["val_distance_m"].append(
                float(self._cached_val_distance_m) if self._cached_val_distance_m is not None else float("nan")
            )
            self.history["val_position_loss"].append(
                float(self._cached_val_position_loss)
                if self._cached_val_position_loss is not None else float("nan")
            )
            self.history["val_position_loss_m"].append(
                float(self._cached_val_position_loss_m)
                if self._cached_val_position_loss_m is not None else float("nan")
            )

            logger.info(
                "Epoch %d/%d | loss %.4f/%.4f | mae %.4f/%.4f",
                epoch + 1, self.config.epochs,
                tr_loss, vl_loss,
                tr_mae, vl_mae,
            )

            if not (np.isfinite(tr_loss) and np.isfinite(vl_loss)):
                logger.error("Stopping — non-finite epoch metrics.")
                break

            primary_metric = (
                float(self._cached_val_distance_m)
                if self._cached_val_distance_m is not None and np.isfinite(self._cached_val_distance_m)
                else float(vl_loss)
            )

            if vl_loss < self.best_val_loss - 1e-6:
                self.best_val_loss    = vl_loss

            if primary_metric < self.best_metric - 1e-9:
                self.best_metric = primary_metric
                # Stable per-prefix name (no timestamp): each new Trainer
                # overwrites its own prefix file instead of leaving a fresh
                # snapshot behind every training, which otherwise accumulates
                # thousands of orphaned .pt files (~26 GB/run). The only reader,
                # restore_best_checkpoint(), runs on the same trainer right
                # after training, so overwriting across trainers is safe.
                name = f"{self.checkpoint_prefix}_best.pt"
                if self.best_checkpoint_path and self.best_checkpoint_path.exists():
                    try:
                        self.best_checkpoint_path.unlink()
                    except PermissionError:
                        logger.warning(
                            "Could not delete old checkpoint (file locked by another process): %s",
                            self.best_checkpoint_path,
                        )
                self.save_checkpoint(name)
                self.best_checkpoint_path = self.config.output_dir / name

            if primary_metric < self.round_best_metric - 1e-9:
                self.round_best_metric = primary_metric
                self.round_best_val_loss = vl_loss
                self.patience_counter    = 0
            else:
                # `patience_counter` is also exposed as the LLM's
                # `epochs_since_improvement` plateau signal.
                self.patience_counter += 1

            # ── Early stopping (metres position-MSE based, same rule for all
            # arms). Monitoring the same metric the search ranks on and the
            # headline reports means the kept epoch, the score, and the test
            # metric all agree. Fall back to scaled val_loss only if the metres
            # metric is unavailable this epoch. ──
            es_metric = (
                float(self._cached_val_position_loss_m)
                if (self._cached_val_position_loss_m is not None
                    and np.isfinite(self._cached_val_position_loss_m))
                else float(vl_loss)
            )
            if es_enabled and np.isfinite(es_metric):
                if es_metric < es_best - 1e-9:
                    es_best       = float(es_metric)
                    es_best_state = copy.deepcopy(self.model.state_dict())
                    es_counter    = 0
                    self.best_epoch_in_call = epoch + 1
                else:
                    es_counter += 1
                    if es_counter >= es_patience:
                        logger.info(
                            "Early stopping at epoch %d/%d (no val position-MSE "
                            "(m) improvement for %d epochs; best epoch %d, "
                            "val_pos_mse_m %.5f, RMSE_m %.5f).",
                            epoch + 1, self.config.epochs, es_patience,
                            self.best_epoch_in_call, es_best, float(np.sqrt(es_best)),
                        )
                        break

        # Restore the best-epoch weights so the model this call leaves behind is
        # the one used for scoring / checkpointing — true to the protocol's
        # "mean best epoch" reporting and to early stopping's restore semantics.
        if es_enabled and es_best_state is not None:
            self.model.load_state_dict(es_best_state)

        return self.history

    # ------------------------------------------------------------------
    # Optimizer factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_optimizer(params, lr: float, weight_decay: float,
                          optimizer_choice: str = "adam") -> torch.optim.Optimizer:
        """Create an optimizer from a string choice name."""
        if optimizer_choice == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if optimizer_choice == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Hyperparameter updates
    # ------------------------------------------------------------------

    def apply_hyperparameter_update(self, updates: Dict[str, Any], resets_model: bool = False):
        """Apply LLM-proposed changes with bounds-checking and optional warm-start.

        dropout is NOT an architectural parameter — changing it does not alter
        weight tensor shapes, so it must not trigger a full model rebuild.
        """
        updates = copy.deepcopy(updates or {})

        # Only shape-altering parameters require a rebuild.
        arch_params = {"lstm_hidden", "lstm_layers"}
        is_arch = resets_model or any(k in arch_params for k in updates)

        if is_arch:
            self._rebuild_model(updates)
        else:
            self._update_optimizer_only(updates)

        logger.info("Applied: %s | Current: %s", updates, self.hyperparams)

    def _rebuild_model(self, updates: Dict):
        logger.info("Architectural change — rebuilding model.")
        new_p = {**copy.deepcopy(self.hyperparams), **updates}

        # Save weights for warm-start before rebuilding.
        saved = self.model.state_dict() if self.config.enable_warm_start else None

        self.model = LSTM_Localizer(
            input_features=self.dataset_dict["input_dim"],
            target_dim=self.dataset_dict["target_dim"],
            lstm_hidden=new_p.get("lstm_hidden", self.config.lstm_hidden),
            lstm_layers=new_p.get("lstm_layers", self.config.lstm_layers),
            dropout=new_p.get("dropout", self.config.dropout),
        ).to(self.config.device)

        self.optimizer = self._create_optimizer(
            self.model.parameters(),
            new_p.get("learning_rate", self.config.learning_rate),
            new_p.get("weight_decay", self.config.weight_decay),
            new_p.get("optimizer_choice", self.config.optimizer_choice),
        )

        # NOTE: `best_metric` is deliberately NOT reset here. It is the
        # *global* best across the whole run and gates which checkpoint is
        # kept as `best_checkpoint_path` (the others are unlinked). Resetting
        # it on every arch change made the next post-rebuild epoch always
        # "improve" vs infinity — overwriting and deleting the true global
        # best checkpoint, so `restore_best_checkpoint()` could only ever
        # recover the best model *since the last arch change*. The per-round
        # signals (`round_best_*`) are still reset so the LLM's
        # `epochs_since_improvement` / round reporting restart cleanly.
        self.best_val_loss        = float("inf")
        self.round_best_val_loss  = float("inf")
        self.round_best_metric    = float("inf")
        self.patience_counter     = 0

        if saved is not None:
            try:
                self.model.load_state_dict(saved, strict=False)
                logger.info("Warm-start successful.")
            except Exception as e:
                logger.warning("Warm-start failed: %s. Starting fresh.", e)

        self.hyperparams = new_p

    def _update_optimizer_only(self, updates: Dict):
        """Apply non-architectural updates (lr, weight_decay, dropout) in-place.

        Resets `patience_counter` so the LLM-visible
        `epochs_since_improvement` signal restarts each round, even though
        early stopping itself is disabled.
        """
        logger.info("Non-architectural update — preserving weights.")
        self.patience_counter    = 0
        self.round_best_val_loss = float("inf")
        self.round_best_metric   = float("inf")
        self.hyperparams.update(updates)

        # Rebuild optimizer entirely when the optimizer type changes.
        if "optimizer_choice" in updates:
            self.optimizer = self._create_optimizer(
                self.model.parameters(),
                self.hyperparams["learning_rate"],
                self.hyperparams["weight_decay"],
                updates["optimizer_choice"],
            )
        else:
            for pg in self.optimizer.param_groups:
                if "learning_rate" in updates:
                    pg["lr"] = updates["learning_rate"]
                if "weight_decay" in updates:
                    pg["weight_decay"] = updates["weight_decay"]

        # Update dropout probability in every Dropout layer without touching weights.
        if "dropout" in updates:
            new_p = float(updates["dropout"])
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = new_p
            # LSTM inter-layer dropout is only active when num_layers > 1.
            if self.model.lstm.num_layers > 1:
                self.model.lstm.dropout = new_p

    def _compute_val_distance_m(self) -> Optional[float]:
        """Return mean Euclidean distance (metres) on the validation set.

        Value is cached by validate() so no extra forward pass is needed.
        Returns None if the cache is empty (e.g. before first validation).
        """
        return self._cached_val_distance_m

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str):
        path = self.config.output_dir / filename
        # Serialise config as plain strings so torch.load(weights_only=True)
        # succeeds on PyTorch ≥ 2.6 without needing WindowsPath/PosixPath globals.
        safe_config = {
            k: str(v) if hasattr(v, "__fspath__") else v
            for k, v in asdict(self.config).items()
        }
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "hyperparams":     self.hyperparams,
            "best_val_loss":   self.best_val_loss,
            "best_metric":     self.best_metric,
            "history":         self.history,
            "config":          safe_config,
        }, path)
        logger.info("Checkpoint saved -> %s", path)

    def restore_best_checkpoint(self) -> bool:
        """Reload the best-checkpoint weights into the live model before eval.

        Early stopping is disabled, so over a multi-round run the model drifts
        past its optimum — `self.model` at the end of training is NOT the best
        model seen. The trainer checkpoints the best model (lowest validation
        Euclidean distance) to `best_checkpoint_path`; this loads it back so
        evaluation reflects each arm's peak rather than its overshot final
        state. The best checkpoint may carry a different architecture than the
        final model (an arch change after the best round), so the model is
        rebuilt from the checkpoint's stored hyperparameters before loading.

        Returns True on success; False (leaving the final model in place) when
        no checkpoint was recorded or it could not be loaded.
        """
        path = self.best_checkpoint_path
        if path is None or not path.exists():
            logger.warning("No best checkpoint recorded — evaluating final model.")
            return False
        ckpt = None
        last_exc: Optional[Exception] = None
        for _weights_only in (True, False):
            try:
                ckpt = torch.load(
                    path, map_location=self.config.device,
                    weights_only=_weights_only,
                )
                break
            except Exception as exc:  # noqa: BLE001 - fall through to next mode
                last_exc = exc
        if ckpt is None:
            logger.warning(
                "Could not load best checkpoint (%s) — evaluating final model.",
                last_exc,
            )
            return False
        hp = ckpt.get("hyperparams", {}) or {}
        self.model = LSTM_Localizer(
            input_features=self.dataset_dict["input_dim"],
            target_dim=self.dataset_dict["target_dim"],
            lstm_hidden=int(hp.get("lstm_hidden", self.config.lstm_hidden)),
            lstm_layers=int(hp.get("lstm_layers", self.config.lstm_layers)),
            dropout=float(hp.get("dropout", self.config.dropout)),
        ).to(self.config.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        logger.info("Restored best checkpoint for evaluation -> %s", path)
        return True
