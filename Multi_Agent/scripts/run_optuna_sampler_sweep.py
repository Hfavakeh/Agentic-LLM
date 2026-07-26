"""Optuna sampler-seed sweep — run on smaug to verify Optuna's headline isn't
just an artefact of seed=1000.

For each sampler_seed in SAMPLER_SEEDS, this runs the EXACT same Optuna arm the
bake-off uses (25 attempts × 3 trainings via evaluate_setting on the metres
objective), then retrains the best setting on the 30 fresh evaluation seeds
(FINAL_EVAL_SEEDS_POOL[:30]) — i.e. the protocol's headline path. The only
thing varying across runs is the TPESampler seed.

Output: outputs-optuna-sweep/optuna_sampler_sweep.json with per-sampler-seed
records (best setting, search val_rmse, 30-seed test mean/std/min/max), plus an
aggregated mean ± std (min – max) over the K sampler seeds.

Usage on smaug (inside the project's Docker / venv):
    python run_optuna_sampler_sweep.py
or with overrides:
    python run_optuna_sampler_sweep.py --sampler-seeds 1000 2000 3000 4000 5000 \\
                                       --output outputs-optuna-sweep
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_REPO_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
import argparse, copy, json, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from arms import build_dataset_and_loaders, run_optuna_search, train_and_test_setting
from experiments.final_eval import FINAL_EVAL_SEEDS_POOL
from pipeline import Config, logger


def _strip_for_json(setting: Dict[str, Any]) -> Dict[str, Any]:
    """Numpy / torch types -> json-safe primitives."""
    out: Dict[str, Any] = {}
    for k, v in setting.items():
        if isinstance(v, (np.integer,)): out[k] = int(v)
        elif isinstance(v, (np.floating,)): out[k] = float(v)
        else: out[k] = v
    return out


def run_one_sweep(base_cfg: Config, dataset, sampler_seed: int,
                  n_attempts: int, fe_seeds: List[int]) -> Dict[str, Any]:
    logger.info("=" * 72)
    logger.info("Optuna sweep | sampler_seed=%d", sampler_seed)
    logger.info("=" * 72)
    t0 = time.perf_counter()
    search = run_optuna_search(
        base_cfg, dataset, n_attempts=n_attempts, sampler_seed=sampler_seed,
    )
    search_time = time.perf_counter() - t0
    best = search["best"] or {}
    best_setting = best.get("setting") or {}
    search_val_rmse = best.get("val_rmse")
    if not best_setting:
        logger.error("sampler_seed=%d produced no valid best setting", sampler_seed)
        return {"sampler_seed": sampler_seed, "error": "no_best_setting"}

    logger.info(
        "sampler_seed=%d | best_search_val_rmse=%.4f | running %d-fresh-seed final eval ...",
        sampler_seed, float(search_val_rmse), len(fe_seeds),
    )
    rmses, r2s, best_epochs, per_seed_rows = [], [], [], []
    t1 = time.perf_counter()
    for s in fe_seeds:
        te = train_and_test_setting(base_cfg, dataset, best_setting, seed=int(s))
        m = te["metrics"]
        rmses.append(float(m["rmse"]))
        r2s.append(float(m["r2"]))
        best_epochs.append(int(te["best_epoch"]))
        per_seed_rows.append({
            "seed": int(s), "rmse": float(m["rmse"]), "r2": float(m["r2"]),
            "best_epoch": int(te["best_epoch"]),
        })
    fe_time = time.perf_counter() - t1
    arr = np.array(rmses, dtype=float)

    rec = {
        "sampler_seed": sampler_seed,
        "search": {
            "n_attempts": n_attempts,
            "best_setting": _strip_for_json(best_setting),
            "best_search_val_rmse_m": float(search_val_rmse),
            "search_time_s": float(search_time),
        },
        "final_eval_30_seeds": {
            "n": len(fe_seeds),
            "mean_rmse_m": float(arr.mean()),
            "std_rmse_m":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min_rmse_m":  float(arr.min()),
            "max_rmse_m":  float(arr.max()),
            "mean_r2":     float(np.mean(r2s)),
            "mean_best_epoch": float(np.mean(best_epochs)),
            "final_eval_time_s": float(fe_time),
            "per_seed": per_seed_rows,
        },
    }
    logger.info(
        "sampler_seed=%d DONE | search_val_rmse=%.4f | test mean=%.4f std=%.4f "
        "min=%.4f max=%.4f | time search=%.0fs eval=%.0fs",
        sampler_seed, float(search_val_rmse),
        rec["final_eval_30_seeds"]["mean_rmse_m"],
        rec["final_eval_30_seeds"]["std_rmse_m"],
        rec["final_eval_30_seeds"]["min_rmse_m"],
        rec["final_eval_30_seeds"]["max_rmse_m"],
        search_time, fe_time,
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler-seeds", type=int, nargs="+",
                    default=[17, 42, 73, 128, 256, 314, 451, 512, 666, 777],
                    help="Optuna TPESampler seeds to sweep.")
    ap.add_argument("--attempts", type=int, default=25,
                    help="Optuna trial budget per sampler seed (protocol default 25).")
    ap.add_argument("--final-eval-seeds", type=int, default=30,
                    help="Number of fresh seeds for the headline final eval (protocol default 30).")
    ap.add_argument("--output", type=str, default="outputs-optuna-sweep",
                    help="Output directory for the sweep JSON.")
    ap.add_argument("--data", type=str, default=None,
                    help="Optional CSV path override (Config.csv_path default if omitted).")
    args = ap.parse_args()

    # Build the same dataset the bake-off uses.
    cfg = Config()
    if args.data:
        cfg.csv_path = args.data
        # Re-trigger DATASET_SPECS hz/window override
        cfg.__post_init__() if hasattr(cfg, "__post_init__") else None
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = out_dir

    dataset, _, _, _ = build_dataset_and_loaders(cfg)
    fe_seeds = list(FINAL_EVAL_SEEDS_POOL[:args.final_eval_seeds])

    records: List[Dict[str, Any]] = []
    for ss in args.sampler_seeds:
        rec = run_one_sweep(cfg, dataset, sampler_seed=int(ss),
                            n_attempts=int(args.attempts), fe_seeds=fe_seeds)
        records.append(rec)
        # Save incrementally so a partial run is still useful.
        with open(out_dir / "optuna_sampler_sweep.json", "w") as f:
            json.dump({"records": records}, f, indent=2, default=str)

    # Aggregate.
    means = [r["final_eval_30_seeds"]["mean_rmse_m"] for r in records
             if "final_eval_30_seeds" in r]
    if means:
        arr = np.array(means)
        agg = {
            "n_sampler_seeds": len(means),
            "mean_of_means":   float(arr.mean()),
            "std_of_means":    float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min_of_means":    float(arr.min()),
            "max_of_means":    float(arr.max()),
        }
        out = {"records": records, "aggregated": agg}
        with open(out_dir / "optuna_sampler_sweep.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info("=" * 72)
        logger.info("AGGREGATE across %d sampler seeds:", len(means))
        logger.info("  mean ± std (min - max) = %.4f ± %.4f (%.4f - %.4f) m",
                    agg["mean_of_means"], agg["std_of_means"],
                    agg["min_of_means"], agg["max_of_means"])
        logger.info("=" * 72)
    else:
        logger.error("No successful sampler-seed runs to aggregate.")


if __name__ == "__main__":
    main()
