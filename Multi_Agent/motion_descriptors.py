"""Interpretable motion descriptors for the radar localisation dataset.

The CSV stores raw sensor features followed by 2 target columns (x, y).
The targets form a 2D trajectory of an indoor walker, which we use to derive
human-interpretable motion descriptors:

  * `speed`         — Euclidean step length per sample, ‖p_t − p_{t-1}‖,
                      converted to m/s using the dataset's sampling rate.
                      Reported as raw numerical values so the LLM can reason
                      about the *type of speed* (slow vs. fast walking) on its
                      own, instead of being constrained to fixed categories.
  * `dwell_state`   — {stop, go} based on a near-zero speed threshold; used
                      only to characterise hesitation/pause behaviour, not for
                      categorising movement intensity.

These descriptors feed:
  1. Offline dataset characterisation (this script's CLI).
  2. The LLM payload — speed quantiles + dwell stats so the optimiser can
     reason about the speed distribution it is actually fitting.
  3. Per-quantile error breakdowns (data-driven low / mid / high speed
     buckets, with the numerical speed range labelling each bucket).

Run as a CLI:
    python motion_descriptors.py [--csv preprocessed-RadarEXP1(in).csv]
                                 [--out analysis/motion]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Speed below this (m/s) counts as "stopped" for the dwell statistic.
# Kept small so it captures genuine pauses without classifying slow walking
# as a stop.
STATIONARY_THRESHOLD_MPS = 0.05

DEFAULT_HZ = 4.0  # radar default; pass --hz / hz= for cap (3) or ir (5)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_motion_features(targets: np.ndarray, hz: float = DEFAULT_HZ) -> pd.DataFrame:
    """Compute per-sample motion descriptors from a 2D trajectory.

    Args:
        targets: array of shape (N, 2), columns = (x, y) in metres.
        hz:      sampling rate of the trajectory in Hz. Required to convert
                 per-sample deltas into m/s so that the descriptors are
                 dataset-agnostic.

    Returns:
        DataFrame with one row per sample. `speed` is in m/s. The first row
        has NaN for velocity-derived columns.
    """
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(f"targets must have shape (N, 2); got {targets.shape}")

    x, y = targets[:, 0], targets[:, 1]

    dx = np.concatenate([[np.nan], np.diff(x)])
    dy = np.concatenate([[np.nan], np.diff(y)])

    speed_mps = np.hypot(dx, dy) * hz

    # Acceleration: rate of change of speed (m/s^2). First two samples NaN
    # (needs two consecutive valid speeds).
    accel_mps2 = np.concatenate([[np.nan], np.diff(speed_mps)]) * hz

    # Jerk: rate of change of acceleration (m/s^3) — a per-sample smoothness /
    # roughness signal (large |jerk| = jagged motion).
    jerk_mps3 = np.concatenate([[np.nan], np.diff(accel_mps2)]) * hz

    # Turning: change in heading direction between consecutive steps, in
    # degrees. Heading is atan2(dy, dx); the per-step turn is the wrapped
    # difference of consecutive headings. Only meaningful while actually
    # moving, so turns at stationary samples (speed below threshold, where the
    # heading is just jitter) are masked to NaN.
    heading = np.arctan2(dy, dx)                       # radians, NaN at sample 0
    dheading = np.concatenate([[np.nan], np.diff(heading)])
    dheading = (dheading + np.pi) % (2.0 * np.pi) - np.pi   # wrap to (-pi, pi]
    turn_deg = np.degrees(dheading)
    moving = speed_mps >= STATIONARY_THRESHOLD_MPS
    turn_deg = np.where(moving & np.roll(moving, 1), turn_deg, np.nan)

    # Dwell / stop-go: a sample is "stop" when speed is below the stationary
    # threshold (NaN treated as stop, since the first sample has no defined
    # velocity). `dwell_run_len_samples` is the length of the current
    # contiguous run of identical states up to and including this sample.
    state = np.where(
        np.isnan(speed_mps) | (speed_mps < STATIONARY_THRESHOLD_MPS),
        "stop", "go",
    )
    run_len = np.empty(len(state), dtype=np.int64)
    if len(state):
        run_len[0] = 1
        for i in range(1, len(state)):
            run_len[i] = run_len[i - 1] + 1 if state[i] == state[i - 1] else 1

    return pd.DataFrame({
        "x":                     x,
        "y":                     y,
        "dx":                    dx,
        "dy":                    dy,
        "speed":                 speed_mps,  # m/s
        "accel":                 accel_mps2, # m/s^2
        "jerk":                  jerk_mps3,  # m/s^3
        "heading_rad":           heading,
        "turn_deg":              turn_deg,   # per-step heading change (deg)
        "dwell_state":           state,
        "dwell_run_len_samples": run_len,
    })


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_dwell(df: pd.DataFrame, hz: float = DEFAULT_HZ) -> Dict[str, object]:
    """Stop-go and dwell statistics over a motion-features DataFrame.

    A *dwell* is a contiguous run of "stop" samples (speed below the
    stationary threshold). Returns:
      - stop_share / go_share          : fraction of samples in each state
      - n_dwell_episodes               : count of dwell runs (stop episodes)
      - n_stop_go_transitions          : stop→go boundaries (== dwell episodes
                                         that ended; useful as a "starts" count)
      - dwell_duration_s_{mean,p50,p95,max} : per-episode dwell length in sec
      - mean_episodes_per_min          : dwell episodes per minute of trajectory
    """
    if "dwell_state" not in df.columns or len(df) == 0:
        return {
            "stop_share": float("nan"), "go_share": float("nan"),
            "n_dwell_episodes": 0, "n_stop_go_transitions": 0,
            "dwell_duration_s_mean": float("nan"),
            "dwell_duration_s_p50":  float("nan"),
            "dwell_duration_s_p95":  float("nan"),
            "dwell_duration_s_max":  float("nan"),
            "mean_episodes_per_min": float("nan"),
        }

    state = df["dwell_state"].to_numpy()
    n_total = len(state)

    change_idx = np.where(state[1:] != state[:-1])[0] + 1
    boundaries = np.concatenate([[0], change_idx, [n_total]])

    dwell_lengths = []
    stop_to_go = 0
    for i in range(len(boundaries) - 1):
        seg_state = state[boundaries[i]]
        seg_len   = boundaries[i + 1] - boundaries[i]
        if seg_state == "stop":
            dwell_lengths.append(seg_len)
            if i + 1 < len(boundaries) - 1:
                stop_to_go += 1

    dwell_arr = np.asarray(dwell_lengths, dtype=float) / float(hz)  # seconds
    stop_share = float(np.mean(state == "stop"))
    duration_s = n_total / float(hz)
    eps_per_min = (len(dwell_arr) / duration_s * 60.0) if duration_s > 0 else float("nan")

    def _safe(arr, fn):
        return float(fn(arr)) if len(arr) else float("nan")

    return {
        "stop_share":              stop_share,
        "go_share":                1.0 - stop_share,
        "n_dwell_episodes":        int(len(dwell_arr)),
        "n_stop_go_transitions":   int(stop_to_go),
        "dwell_duration_s_mean":   _safe(dwell_arr, np.mean),
        "dwell_duration_s_p50":    _safe(dwell_arr, lambda a: np.quantile(a, 0.5)),
        "dwell_duration_s_p95":    _safe(dwell_arr, lambda a: np.quantile(a, 0.95)),
        "dwell_duration_s_max":    _safe(dwell_arr, np.max),
        "mean_episodes_per_min":   float(eps_per_min),
    }


def summarize_dynamics(df: pd.DataFrame) -> Dict[str, object]:
    """Acceleration, turning and trajectory-roughness statistics (the motion
    features the professor named beyond speed / stop-go).

    - acceleration: magnitude of speed change (m/s^2) — mean, std, p95 of |accel|.
    - turning: per-step heading change (deg, while moving) — mean, p95 of |turn|
      and the share of sharp turns (>45 deg).
    - roughness: `path_sinuosity` = total path length / net start-end
      displacement (1.0 = straight line; larger = more winding), plus mean
      |jerk| (m/s^3), a per-sample jaggedness signal.
    """
    accel = df["accel"].dropna().abs() if "accel" in df.columns else pd.Series(dtype=float)
    jerk  = df["jerk"].dropna().abs()  if "jerk"  in df.columns else pd.Series(dtype=float)
    turn  = df["turn_deg"].dropna().abs() if "turn_deg" in df.columns else pd.Series(dtype=float)

    # Path sinuosity from the per-step displacements.
    sinuosity = float("nan")
    if {"dx", "dy", "x", "y"}.issubset(df.columns) and len(df) > 1:
        step = np.hypot(df["dx"].to_numpy(), df["dy"].to_numpy())
        path_len = float(np.nansum(step))
        net = float(np.hypot(df["x"].iloc[-1] - df["x"].iloc[0],
                             df["y"].iloc[-1] - df["y"].iloc[0]))
        sinuosity = path_len / net if net > 1e-9 else float("nan")

    def _f(v):
        return float(v) if np.isfinite(v) else float("nan")

    return {
        "accel_abs_mean_mps2": _f(accel.mean()) if len(accel) else float("nan"),
        "accel_std_mps2":      _f(df["accel"].dropna().std()) if "accel" in df.columns and df["accel"].notna().any() else float("nan"),
        "accel_abs_p95_mps2":  _f(accel.quantile(0.95)) if len(accel) else float("nan"),
        "turn_abs_mean_deg":   _f(turn.mean()) if len(turn) else float("nan"),
        "turn_abs_p95_deg":    _f(turn.quantile(0.95)) if len(turn) else float("nan"),
        "sharp_turn_share":    _f((turn > 45.0).mean()) if len(turn) else float("nan"),
        "path_sinuosity":      sinuosity,
        "jerk_abs_mean_mps3":  _f(jerk.mean()) if len(jerk) else float("nan"),
    }


def summarize_motion(df: pd.DataFrame, hz: float = DEFAULT_HZ) -> Dict[str, object]:
    """Return interpretable summary stats over a motion-features DataFrame.

    Assumes `df['speed']` is in m/s, as produced by `extract_motion_features`.
    """
    speed = df["speed"].dropna()

    def _q(arr: pd.Series, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[str, float]:
        if len(arr) == 0:
            return {f"p{int(q*100)}": float("nan") for q in qs}
        return {f"p{int(q*100)}": float(arr.quantile(q)) for q in qs}

    return {
        "n_samples":             int(len(df)),
        "n_speed_valid":         int(len(speed)),
        "speed_mean_mps":        float(speed.mean()) if len(speed) else float("nan"),
        "speed_std_mps":         float(speed.std())  if len(speed) else float("nan"),
        "speed_min_mps":         float(speed.min())  if len(speed) else float("nan"),
        "speed_max_mps":         float(speed.max())  if len(speed) else float("nan"),
        "speed_quantiles_mps":   _q(speed),
        "dynamics":              summarize_dynamics(df),
        "dwell":                 summarize_dwell(df, hz=hz),
    }


# ---------------------------------------------------------------------------
# Overall error payload — kept intentionally minimal. No speed-bucket split:
# the LLM gets only aggregate validation error here, and reasons about speed
# regimes from the dataset-level descriptors in `llm_payload`.
# ---------------------------------------------------------------------------

def error_payload(
    preds: np.ndarray,
    targets: np.ndarray,
    hz: float = DEFAULT_HZ,
    **_ignored,
) -> Dict[str, object]:
    """Compact, LLM-ready summary of overall model error.

    Returns:
        {
          "overall_rmse":        float,
          "overall_mean_euclid": float,
        }
    """
    sample_indices = np.arange(len(targets) - len(preds), len(targets))
    err = preds - targets[sample_indices]
    sq  = (err ** 2).sum(axis=1)
    euc = np.sqrt(sq)
    return {
        "overall_rmse":        round(float(np.sqrt(sq.mean())), 4) if len(sq) else None,
        "overall_mean_euclid": round(float(euc.mean()),         4) if len(euc) else None,
    }


def per_regime_error(
    preds: np.ndarray,
    targets: np.ndarray,
    hz: float = DEFAULT_HZ,
    **_ignored,
) -> Dict[str, object]:
    """Mean position error (metres) split by MOTION REGIME (Q5).

    Speed at each predicted sample is derived from the target trajectory
    (``||Δposition|| * hz``, matching ``extract_motion_features``) and bucketed
    into slow / medium / fast by the terciles of that speed. The mean Euclidean
    error is reported per regime, plus which regime is worst and how large the
    spread is (worst/best ratio). This is the "errors in different motion
    regimes" signal the professor asked about; it is rendered as a qualitative
    label in the prompt, never as raw numbers.

    Returns {"slow","medium","fast": float|None, "worst_regime": str|None,
    "spread_ratio": float|None, "n": int}.
    """
    idx = np.arange(len(targets) - len(preds), len(targets))
    tgt = np.asarray(targets)[idx]
    euc = np.sqrt(((np.asarray(preds) - tgt) ** 2).sum(axis=1))
    # per-sample speed from consecutive targets (first sample -> 0)
    d = np.diff(tgt, axis=0)
    speed = np.concatenate([[0.0], np.hypot(d[:, 0], d[:, 1]) * hz]) if len(tgt) > 1 else np.zeros(len(tgt))
    out: Dict[str, object] = {"slow": None, "medium": None, "fast": None,
                              "worst_regime": None, "spread_ratio": None, "n": int(len(euc))}
    if len(euc) < 6:
        return out
    lo, hi = np.quantile(speed, [1 / 3, 2 / 3])
    masks = {"slow": speed <= lo, "medium": (speed > lo) & (speed <= hi), "fast": speed > hi}
    means = {}
    for name, m in masks.items():
        if m.any():
            means[name] = float(euc[m].mean())
            out[name] = round(means[name], 4)
    if means:
        worst = max(means, key=means.get)
        best_v = min(means.values())
        out["worst_regime"] = worst
        out["spread_ratio"] = round(max(means.values()) / best_v, 3) if best_v > 0 else None
    return out


# ---------------------------------------------------------------------------
# Compact LLM-payload form
# ---------------------------------------------------------------------------

def llm_payload(summary: Dict[str, object]) -> Dict[str, object]:
    """Distil `summarize_motion(...)` into a compact dict for the LLM prompt.

    Keeps raw speed statistics (mean / std / quantile spread / extremes) plus
    the dwell block, so the LLM can reason about the speed distribution and
    hesitation behaviour without us pre-categorising motion regimes.
    """
    sq = summary["speed_quantiles_mps"]

    def _r(v, n=4):
        try:
            return round(float(v), n) if np.isfinite(float(v)) else None
        except (TypeError, ValueError):
            return None

    dwell = summary.get("dwell", {}) or {}
    dyn = summary.get("dynamics", {}) or {}
    return {
        "speed_mean_mps":   _r(summary.get("speed_mean_mps")),
        "speed_std_mps":    _r(summary.get("speed_std_mps")),
        "speed_min_mps":    _r(summary.get("speed_min_mps")),
        "speed_max_mps":    _r(summary.get("speed_max_mps")),
        "speed_median_mps": _r(sq["p50"]),
        "speed_iqr_mps":    _r(sq["p75"] - sq["p25"]),
        "speed_p95_mps":    _r(sq["p95"]),
        # Acceleration / turning / roughness — the motion features the professor
        # named beyond speed and stop-go.
        "accel_abs_mean_mps2": _r(dyn.get("accel_abs_mean_mps2"), 3),
        "accel_abs_p95_mps2":  _r(dyn.get("accel_abs_p95_mps2"), 3),
        "turn_abs_mean_deg":   _r(dyn.get("turn_abs_mean_deg"), 2),
        "turn_abs_p95_deg":    _r(dyn.get("turn_abs_p95_deg"), 2),
        "sharp_turn_share":    _r(dyn.get("sharp_turn_share"), 3),
        "path_sinuosity":      _r(dyn.get("path_sinuosity"), 3),
        "jerk_abs_mean_mps3":  _r(dyn.get("jerk_abs_mean_mps3"), 3),
        # Dwell / stop-go — describes hesitation/pause behaviour.
        "dwell": {
            "stop_share":            _r(dwell.get("stop_share"), 3),
            "n_dwell_episodes":      int(dwell.get("n_dwell_episodes", 0) or 0),
            "n_stop_go_transitions": int(dwell.get("n_stop_go_transitions", 0) or 0),
            "dwell_s_mean":          _r(dwell.get("dwell_duration_s_mean"), 2),
            "dwell_s_p95":           _r(dwell.get("dwell_duration_s_p95"), 2),
            "episodes_per_min":      _r(dwell.get("mean_episodes_per_min"), 2),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_summary_md(summary: Dict[str, object], title: str = "") -> str:
    out = []
    if title:
        out.append(f"# Motion descriptors — {title}\n")
    out.append(f"_{summary['n_samples']} samples — "
               f"{summary['n_speed_valid']} valid speed._\n")

    out.append("## Speed distribution\n")
    sq = summary["speed_quantiles_mps"]
    out.append(f"- mean = {summary['speed_mean_mps']:.4f} m/s, "
               f"std = {summary['speed_std_mps']:.4f} m/s")
    out.append(f"- min  = {summary['speed_min_mps']:.4f} m/s, "
               f"max = {summary['speed_max_mps']:.4f} m/s")
    out.append(f"- quantiles (m/s): p5={sq['p5']:.4f}, p25={sq['p25']:.4f}, "
               f"p50={sq['p50']:.4f}, p75={sq['p75']:.4f}, p95={sq['p95']:.4f}\n")

    dwell = summary.get("dwell", {}) or {}
    if dwell:
        out.append("## Dwell / stop-go\n")
        out.append(f"- stop share = {dwell.get('stop_share', float('nan'))*100:.1f}%, "
                   f"go share = {dwell.get('go_share', float('nan'))*100:.1f}%")
        out.append(f"- {dwell.get('n_dwell_episodes', 0)} dwell episodes, "
                   f"{dwell.get('n_stop_go_transitions', 0)} stop→go transitions, "
                   f"{dwell.get('mean_episodes_per_min', float('nan')):.2f} episodes/min")
        out.append(f"- dwell duration (s): mean={dwell.get('dwell_duration_s_mean', float('nan')):.2f}, "
                   f"p50={dwell.get('dwell_duration_s_p50', float('nan')):.2f}, "
                   f"p95={dwell.get('dwell_duration_s_p95', float('nan')):.2f}, "
                   f"max={dwell.get('dwell_duration_s_max', float('nan')):.2f}\n")

    return "\n".join(out)


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Save a speed histogram as a PNG (matplotlib only — optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    speed_mps = df["speed"].dropna().values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(speed_mps, bins=60, color="#3b82f6", edgecolor="white")
    ax.set_title("Speed")
    ax.set_xlabel("m/s")
    ax.set_ylabel("count")

    fig.tight_layout()
    fig.savefig(out_dir / "speed_histogram.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="preprocessed-RadarEXP1(in).csv",
                    help="Path to the dataset CSV (no header). Last 2 columns = (x, y).")
    ap.add_argument("--out", default="analysis/motion",
                    help="Output directory.")
    ap.add_argument("--hz", type=float, default=None,
                    help="Sampling rate in Hz. If omitted, auto-resolved from "
                         "DATASET_SPECS by csv basename, falling back to 4 Hz (radar).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out)

    if args.hz is not None:
        hz = float(args.hz)
    else:
        try:
            from pipeline.datasets import get_dataset_spec  # lazy: keeps this script torch-free
            spec = get_dataset_spec(str(csv_path))
        except Exception:
            spec = None
        hz = float(spec["hz"]) if spec else DEFAULT_HZ
    print(f"[info] using hz = {hz}")

    df_raw = pd.read_csv(csv_path, header=None)
    targets = df_raw.iloc[:, -2:].values.astype(float)
    print(f"[info] loaded {len(targets)} samples from {csv_path}")

    motion_df = extract_motion_features(targets, hz=hz)
    summary = summarize_motion(motion_df, hz=hz)

    out_dir.mkdir(parents=True, exist_ok=True)
    motion_df.to_csv(out_dir / "per_sample.csv", index=False)
    (out_dir / "summary.md").write_text(
        render_summary_md(summary, title=csv_path.name), encoding="utf-8"
    )
    import json
    (out_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items()
                    if not isinstance(v, dict) or k in ("speed_quantiles_mps", "dwell")},
                   indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "llm_payload.json").write_text(
        json.dumps(llm_payload(summary), indent=2),
        encoding="utf-8",
    )
    plot_distributions(motion_df, out_dir)

    print(f"[done] wrote per_sample.csv summary.md summary.json llm_payload.json -> {out_dir}")
    print("\n--- LLM payload preview ---")
    print(json.dumps(llm_payload(summary), indent=2))


if __name__ == "__main__":
    main()
