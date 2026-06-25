"""Final evaluation on fresh seeds (protocol point 8)."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from arms.engine import build_dataset_and_loaders, train_and_test_setting
from pipeline import Config, logger

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
