"""Data loading, splitting, windowing, and scaling."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .config import Config
from .datasets import get_dataset_spec
from .logging_setup import logger


class TimeSeriesDataset(Dataset):
    """Thin Dataset wrapper for numpy arrays."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prev_pos: Optional[np.ndarray] = None,
        prev_prev_pos: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.prev_pos = (
            torch.tensor(prev_pos, dtype=torch.float32) if prev_pos is not None else None
        )
        self.prev_prev_pos = (
            torch.tensor(prev_prev_pos, dtype=torch.float32)
            if prev_prev_pos is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"X": self.X[idx], "y": self.y[idx]}
        if self.prev_pos is not None:
            item["prev_pos"] = self.prev_pos[idx]
        if self.prev_prev_pos is not None:
            item["prev_prev_pos"] = self.prev_prev_pos[idx]
        return item


def compute_speed_bin_edges(
    y_scaled: np.ndarray,
    prev_pos_scaled: np.ndarray,
    scaler_y: StandardScaler,
    hz: float,
) -> List[float]:
    """Tercile speed thresholds (m/s) from a scaled training split.

    The two returned thresholds split per-sample true speed into slow /
    medium / fast regimes. Computed once at dataset build time; a controller
    proposes the per-bin *weights*, never these edges. Displacements are
    un-scaled to real metres (the StandardScaler mean cancels for a
    difference) so the thresholds are genuine m/s.
    """
    disp  = (np.asarray(y_scaled) - np.asarray(prev_pos_scaled)) * scaler_y.scale_
    speed = np.linalg.norm(disp, axis=1) * float(hz)
    return [
        float(np.quantile(speed, 1.0 / 3.0)),
        float(np.quantile(speed, 2.0 / 3.0)),
    ]


def compute_motion_reference(
    y_scaled: np.ndarray,
    prev_pos_scaled: np.ndarray,
    prev_prev_pos_scaled: np.ndarray,
    scaler_y: StandardScaler,
    hz: float,
) -> Dict[str, float]:
    """Reference kinematics of the TRUE trajectory, from a scaled train split.

    Returns the RMS step speed (m/s) and RMS acceleration (m/s^2) of the
    ground-truth positions. `Trainer._compute_total_loss` divides the velocity
    and smoothness penalties by these, which is what makes `lambda_vel` and
    `lambda_smooth` dimensionless and comparable across the 3 / 4 / 5 Hz
    datasets — see that docstring for the full unit derivation.

    RMS (not mean) because both penalties are squared quantities: the
    smoothness penalty is then ~1.0 when the prediction merely reproduces the
    walker's own acceleration. Computed on train only, like `bin_edges`.
    """
    sigma = scaler_y.scale_
    step  = (np.asarray(y_scaled) - np.asarray(prev_pos_scaled)) * sigma      # metres
    accel = ((np.asarray(y_scaled) - 2.0 * np.asarray(prev_pos_scaled)
              + np.asarray(prev_prev_pos_scaled)) * sigma) * float(hz) ** 2   # m/s^2
    speed = np.linalg.norm(step, axis=1) * float(hz)                          # m/s
    return {
        "speed_ref_mps":  float(np.sqrt(np.mean(speed ** 2))),
        "accel_ref_mps2": float(np.sqrt(np.mean(np.sum(accel ** 2, axis=1)))),
    }


class DataProcessor:
    """Load, split, window, and scale radar sensor data."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.config.csv_path, header=None)
        data = df.values.astype(np.float32)
        n_features = data.shape[1] - self.config.target_columns
        X, y = data[:, :n_features], data[:, n_features:]
        logger.info(
            "Loaded %d samples | %d raw features | %d targets",
            data.shape[0], n_features, self.config.target_columns,
        )
        return X, y

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sliding-window sequences from a single contiguous split.

        Returns (seq_X, seq_y, seq_prev_pos, seq_prev_prev_pos). `seq_prev_pos` and
        `seq_prev_prev_pos` are the two positions immediately preceding each
        target — three consecutive points (prev_prev -> prev -> target) are
        what the LLM-proposed smoothness/acceleration prior needs.
        """
        n = len(X) - self.config.window_size
        seq_X = [X[i : i + self.config.window_size] for i in range(n)]
        seq_y = [y[i + self.config.window_size]     for i in range(n)]
        seq_prev_pos      = [y[i + self.config.window_size - 1] for i in range(n)]
        seq_prev_prev_pos = [y[i + self.config.window_size - 2] for i in range(n)]
        seq_X = np.asarray(seq_X, dtype=np.float32)
        seq_y = np.asarray(seq_y, dtype=np.float32)
        seq_prev_pos      = np.asarray(seq_prev_pos, dtype=np.float32)
        seq_prev_prev_pos = np.asarray(seq_prev_prev_pos, dtype=np.float32)
        logger.info(
            "Sequences: %d | Input: %s | Target: %s | Prev: %s | Prev2: %s",
            len(seq_X), seq_X.shape, seq_y.shape, seq_prev_pos.shape, seq_prev_prev_pos.shape,
        )
        return seq_X, seq_y, seq_prev_pos, seq_prev_prev_pos

    def temporal_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        spec = get_dataset_spec(self.config.csv_path)
        if spec is None:
            n = len(X)
            t1, t2 = int(0.80 * n), int(0.90 * n)
            logger.warning(
                "No DATASET_SPEC for %s — falling back to contiguous 80/10/10 split.",
                self.config.csv_path,
            )
            logger.info("Split | Train: %d | Val: %d | Test: %d", t1, t2 - t1, n - t2)
            return X[:t1], X[t1:t2], X[t2:], y[:t1], y[t1:t2], y[t2:]

        s = spec["splits"]
        n = len(X)
        for name, (a, b) in s.items():
            if a < 0 or b > n or a >= b:
                raise ValueError(
                    f"Spec split '{name}' = [{a}:{b}] out of range for dataset of length {n}"
                )
        tr, va, te = s["train"], s["val"], s["test"]
        logger.info(
            "Split per spec | Train: [%d:%d] (%d) | Val: [%d:%d] (%d) | Test: [%d:%d] (%d)",
            tr[0], tr[1], tr[1] - tr[0],
            va[0], va[1], va[1] - va[0],
            te[0], te[1], te[1] - te[0],
        )
        return (
            X[tr[0]:tr[1]], X[va[0]:va[1]], X[te[0]:te[1]],
            y[tr[0]:tr[1]], y[va[0]:va[1]], y[te[0]:te[1]],
        )

    def scale_splits(self, X_tr, X_val, X_te, y_tr, y_val, y_te) -> Tuple[np.ndarray, ...]:
        """Fit scalers on train only; transform all splits."""
        n_feat = X_tr.shape[-1]
        self.scaler_X.fit(X_tr.reshape(-1, n_feat))
        self.scaler_y.fit(y_tr)
        X_tr_scaled  = self.scaler_X.transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
        X_te_scaled  = self.scaler_X.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape)
        y_tr_scaled  = self.scaler_y.transform(y_tr)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_te_scaled  = self.scaler_y.transform(y_te)
        return X_tr_scaled, X_val_scaled, X_te_scaled, y_tr_scaled, y_val_scaled, y_te_scaled

    @staticmethod
    def build_dataset(config: Config) -> Dict[str, Any]:
        """Full pipeline: load → split → sequence → scale → return dict."""
        proc = DataProcessor(config)
        X, y = proc.load_and_preprocess()
        # Split raw data BEFORE windowing to prevent leakage across splits
        rX_tr, rX_val, rX_te, ry_tr, ry_val, ry_te = proc.temporal_split(X, y)
        # Window each split independently
        X_tr, y_tr, prev_pos_tr, prev2_pos_tr    = proc.create_sequences(rX_tr, ry_tr)
        X_val, y_val, prev_pos_val, prev2_pos_val = proc.create_sequences(rX_val, ry_val)
        X_te, y_te, prev_pos_te, prev2_pos_te    = proc.create_sequences(rX_te, ry_te)
        X_tr, X_val, X_te, y_tr, y_val, y_te = proc.scale_splits(
            X_tr, X_val, X_te, y_tr, y_val, y_te
        )
        prev_pos_tr = proc.scaler_y.transform(prev_pos_tr)
        prev_pos_val = proc.scaler_y.transform(prev_pos_val)
        prev_pos_te = proc.scaler_y.transform(prev_pos_te)
        prev2_pos_tr = proc.scaler_y.transform(prev2_pos_tr)
        prev2_pos_val = proc.scaler_y.transform(prev2_pos_val)
        prev2_pos_te = proc.scaler_y.transform(prev2_pos_te)
        bin_edges = compute_speed_bin_edges(y_tr, prev_pos_tr, proc.scaler_y, config.hz)
        motion_ref = compute_motion_reference(
            y_tr, prev_pos_tr, prev2_pos_tr, proc.scaler_y, config.hz
        )
        return {
            "X_train": X_tr, "y_train": y_tr,
            "X_val":   X_val, "y_val":  y_val,
            "X_test":  X_te,  "y_test": y_te,
            "prev_pos_train": prev_pos_tr,
            "prev_pos_val": prev_pos_val,
            "prev_pos_test": prev_pos_te,
            "prev_prev_pos_train": prev2_pos_tr,
            "prev_prev_pos_val":   prev2_pos_val,
            "prev_prev_pos_test":  prev2_pos_te,
            "bin_edges": bin_edges,
            "speed_ref_mps":  motion_ref["speed_ref_mps"],
            "accel_ref_mps2": motion_ref["accel_ref_mps2"],
            "input_dim":  X_tr.shape[-1],
            "target_dim": y_tr.shape[-1],
            "processor":  proc,
            # Raw (pre-windowed) splits — kept for re-windowing when window_size changes
            "raw_X_train": rX_tr, "raw_y_train": ry_tr,
            "raw_X_val":   rX_val, "raw_y_val":  ry_val,
            "raw_X_test":  rX_te,  "raw_y_test": ry_te,
        }

    @staticmethod
    def rebuild_with_window_size(dataset: Dict[str, Any], new_window_size: int) -> Dict[str, Any]:
        """Re-window and re-scale from stored raw splits.

        Called when the agent proposes a new window_size.  The raw
        (pre-windowed) splits are already stored in *dataset* so we
        never need to reload the CSV or re-split.
        """
        proc: DataProcessor = dataset["processor"]
        proc.config.window_size = new_window_size
        # Re-create fresh scalers to avoid contamination from the old window size.
        proc.scaler_X = StandardScaler()
        proc.scaler_y = StandardScaler()
        X_tr, y_tr, prev_pos_tr, prev2_pos_tr    = proc.create_sequences(dataset["raw_X_train"], dataset["raw_y_train"])
        X_val, y_val, prev_pos_val, prev2_pos_val = proc.create_sequences(dataset["raw_X_val"],   dataset["raw_y_val"])
        X_te, y_te, prev_pos_te, prev2_pos_te    = proc.create_sequences(dataset["raw_X_test"],  dataset["raw_y_test"])
        X_tr, X_val, X_te, y_tr, y_val, y_te = proc.scale_splits(
            X_tr, X_val, X_te, y_tr, y_val, y_te
        )
        prev_pos_tr = proc.scaler_y.transform(prev_pos_tr)
        prev_pos_val = proc.scaler_y.transform(prev_pos_val)
        prev_pos_te = proc.scaler_y.transform(prev_pos_te)
        prev2_pos_tr = proc.scaler_y.transform(prev2_pos_tr)
        prev2_pos_val = proc.scaler_y.transform(prev2_pos_val)
        prev2_pos_te = proc.scaler_y.transform(prev2_pos_te)
        bin_edges = compute_speed_bin_edges(y_tr, prev_pos_tr, proc.scaler_y, proc.config.hz)
        motion_ref = compute_motion_reference(
            y_tr, prev_pos_tr, prev2_pos_tr, proc.scaler_y, proc.config.hz
        )
        dataset.update({
            "X_train": X_tr, "y_train": y_tr,
            "X_val":   X_val, "y_val":  y_val,
            "X_test":  X_te,  "y_test": y_te,
            "prev_pos_train": prev_pos_tr,
            "prev_pos_val": prev_pos_val,
            "prev_pos_test": prev_pos_te,
            "prev_prev_pos_train": prev2_pos_tr,
            "prev_prev_pos_val":   prev2_pos_val,
            "prev_prev_pos_test":  prev2_pos_te,
            "bin_edges": bin_edges,
            "speed_ref_mps":  motion_ref["speed_ref_mps"],
            "accel_ref_mps2": motion_ref["accel_ref_mps2"],
            "input_dim":  X_tr.shape[-1],
            "target_dim": y_tr.shape[-1],
        })
        logger.info("Dataset re-windowed with window_size=%d  (train=%d, val=%d, test=%d)",
                    new_window_size, len(X_tr), len(X_val), len(X_te))
        return dataset
