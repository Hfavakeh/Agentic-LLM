"""Automated log processing for the LSTM-LLM-optimizer experiments.

Walks every `outputs-<DATE>-<MODEL>/seed_<N>/` directory, parses the per-run
JSONs (`optimization_log_run*.json`, `optimization_log_rule_based_run*.json`,
`cross_run_metrics.json`) into long-form pandas DataFrames, and emits:

  1. `analysis/all_rounds.csv`           — every round of every agent of every seed
  2. `analysis/all_runs.csv`             — one row per (experiment, seed, agent)
  3. `analysis/experiments.csv`          — one row per experiment (LLM model)
  4. `analysis/<experiment>/report.md`   — per-experiment narrative report
  5. `analysis/summary.md`               — cross-experiment LLM-as-optimizer analysis

Discovery is recursive: any directory at or below `--root` that contains a
`seed_<N>/` subdir with an optimisation log counts as an experiment, whatever
it is named. So `--root results` sweeps every group folder (motion/,
q3-history/, archive/model-sweep/, ...) in one pass, and the descriptively
named run dirs are included alongside the dated `outputs-<date>-<model>/` ones.

Run:
    python scripts/analyze_logs.py --root results --out analysis
    python scripts/analyze_logs.py --root results/motion --out analysis/motion
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric definition (single source of truth for the report headers)
# ---------------------------------------------------------------------------
#
# The reported quality metric is computed once per (seed, agent) at the end of
# every run, on the held-out test set, and is the same for every arm.
#
#   rmse = sqrt(mean((preds - targets) ** 2))     # all output dims combined
#
# Predictions and targets are inverse-transformed by `scaler_y` before the
# metric is computed (see model_pipeline.py::Trainer.compute_metrics), so the
# value is in the *original target units* — i.e. metres of position error for
# the indoor-localisation task. Lower is better.
#
# The summary tables below report:
#   - mean across N seeds   (one run per seed; the per-seed file's own std is 0
#                            by construction and is ignored here)
#   - std  across N seeds   (sample SD, ddof=1)
#   - paired differences vs baseline / rule-based / other arms, with 95% CIs
#     from BOTH a paired t-distribution and a 10 000-iter percentile bootstrap,
#     plus a Wilcoxon signed-rank p-value as a distribution-free companion.
#
# Pairing unit: seed. Two experiments can be paired by matching seed values.
METRIC_NAME = "Test-set RMSE"
METRIC_UNITS = "metres (original target units, post inverse-transform)"
METRIC_STAGE = "test"
METRIC_DIRECTION = "lower is better"
METRIC_AGGREGATION = "± is sample standard deviation across N seeds (ddof=1)"
BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20250520


EXPERIMENT_DIR_RE = re.compile(r"^outputs-(?P<date>\d{4})-(?P<model>.+)$")
SEED_DIR_RE = re.compile(r"^seed_(?P<seed>\d+)$")
RUN_FILE_RE = re.compile(r"^optimization_log(?:_(?P<agent>rule_based))?_run(?P<run>\d+)\.json$")

# Two diagnosis vocabularies appear across the run history:
#   - the original HARD set, written as a {"primary_problem": ..., "severity": ...} dict
#   - the protocol's SOFT set (arms/labels.py::PROTOCOL_DIAGNOSES), written as a bare string
# Both are accepted so old and new runs can be analysed side by side.
HARD_DIAGNOSES = {"overfitting", "underfitting", "plateau", "healthy", "no_data"}
SOFT_DIAGNOSES = {
    "healthy", "possible_overfitting_tendency", "possible_underfitting_tendency",
    "plateau", "unstable", "inconclusive",
}
ALLOWED_DIAGNOSES = HARD_DIAGNOSES | SOFT_DIAGNOSES


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Experiment:
    name: str            # e.g. "0503-llama3-8b"
    date: str            # "0503"
    model: str           # "llama3-8b"
    path: Path           # absolute experiment dir


# Directories that never contain experiment output — skipped while walking.
_SKIP_DIR_NAMES = {
    "logs", "reports", "scripts", "analysis", "_attic", "docs",
    "pipeline", "arms", "experiments", "reporting",
    "__pycache__", ".git", ".venv", "venv", ".idea", ".vscode",
}


def _has_seed_logs(d: Path) -> bool:
    """True if `d` directly contains a seed_<N>/ dir holding an optimisation log."""
    try:
        subdirs = [s for s in d.iterdir() if s.is_dir()]
    except OSError:
        return False
    for s in subdirs:
        if not SEED_DIR_RE.match(s.name):
            continue
        try:
            if any(RUN_FILE_RE.match(f.name) for f in s.iterdir()):
                return True
        except OSError:
            continue
    return False


def discover_experiments(root: Path, max_depth: int = 3) -> List[Experiment]:
    """Find every experiment directory at or below `root`.

    An experiment is any directory containing at least one `seed_<N>/` subdir
    with an optimisation log — this is what identifies it, not its name, so
    both the dated `outputs-<date>-<model>/` sweeps and the descriptively
    named runs (`motion-qwen3/`, `q3-empty-llama/`, `prompt-phi/`) are picked
    up. Recursion lets `--root results` sweep every group folder at once.
    """
    found: List[Experiment] = []

    def walk(d: Path, depth: int) -> None:
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for child in children:
            if child.name in _SKIP_DIR_NAMES or SEED_DIR_RE.match(child.name):
                continue
            if _has_seed_logs(child):
                m = EXPERIMENT_DIR_RE.match(child.name)
                if m:
                    date, model = m.group("date"), m.group("model")
                    name = f"{date}-{model}"
                else:
                    # Non-dated run dir: keep the folder name as the label.
                    date, model = "", child.name
                    name = child.name
                found.append(Experiment(name=name, date=date, model=model, path=child))
                continue  # an experiment is a leaf — don't descend into it
            if depth < max_depth:
                walk(child, depth + 1)

    walk(root, 0)

    # Two groups can hold same-named dirs; prefix those with their group path
    # so experiment names stay unique (they key the report subdirs and --pair).
    counts = Counter(e.name for e in found)
    out: List[Experiment] = []
    for e in found:
        if counts[e.name] > 1:
            try:
                group = e.path.relative_to(root).parent.as_posix().replace("/", "-")
            except ValueError:
                group = e.path.parent.name
            if group and group != ".":
                e = Experiment(name=f"{group}-{e.name}", date=e.date,
                               model=e.model, path=e.path)
        out.append(e)
    return sorted(out, key=lambda x: x.name)


def discover_runs(exp: Experiment) -> List[Tuple[int, int, str, Path]]:
    """Yield (seed, run, agent, path) for every optimisation log under this experiment."""
    out = []
    for seed_dir in sorted(exp.path.iterdir()):
        if not seed_dir.is_dir():
            continue
        m = SEED_DIR_RE.match(seed_dir.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        for f in sorted(seed_dir.iterdir()):
            mr = RUN_FILE_RE.match(f.name)
            if not mr:
                continue
            agent = mr.group("agent") or "llm"
            run = int(mr.group("run"))
            out.append((seed, run, agent, f))
    return out


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _safe_load(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] cannot read {p}: {e}")
        return None


def _is_finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _normalize_diagnosis(diag: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (primary_problem, severity) from either diagnosis schema.

    Older runs store a dict — {"primary_problem": ..., "severity": ...}.
    Protocol runs store a bare soft label string, e.g. "inconclusive", and
    carry no severity. Anything else yields (None, None).
    """
    if isinstance(diag, dict):
        return diag.get("primary_problem"), diag.get("severity")
    if isinstance(diag, str):
        d = diag.strip()
        return (d or None), None
    return None, None


def extract_rounds(
    exp: Experiment,
    seed: int,
    run: int,
    agent: str,
    payload: dict,
) -> List[Dict[str, Any]]:
    """Flatten the per-round records into a list of long-form dicts."""
    rounds = payload.get("rounds", []) or []
    out: List[Dict[str, Any]] = []
    for r in rounds:
        primary, severity = _normalize_diagnosis(r.get("diagnosis"))
        pareto = r.get("pareto") or {}
        weights = pareto.get("weights") or {}
        out.append({
            "experiment":   exp.name,
            "model":        exp.model,
            "date":         exp.date,
            "seed":         seed,
            "run":          run,
            "agent":        agent,
            "round":        r.get("round"),
            "skipped":      bool(r.get("skipped", False)),
            "failure_reason": r.get("failure_reason"),
            "primary_problem": primary,
            "severity":     severity,
            "diagnosis_valid": (primary in ALLOWED_DIAGNOSES) if primary is not None else None,
            "n_changes":    len(r.get("changes_applied") or {}),
            "changes":      r.get("changes_applied") or {},
            "resets_model": r.get("resets_model"),
            "val_loss":     r.get("val_loss"),
            "primary_metric": r.get("primary_metric"),
            "val_mae":      r.get("val_mae"),
            "outcome":      r.get("outcome"),
            "improvement":  r.get("improvement"),
            "delta_best_val_loss": r.get("delta_best_val_loss"),
            "change_magnitude":    r.get("change_magnitude"),
            "confidence":   r.get("llm_confidence"),
            "reasoning":    r.get("llm_reasoning") or r.get("reasoning"),
            # Pareto axes + LLM weights (Block 2/3 outputs).
            "latency_ms":       pareto.get("latency_ms"),
            "stability_std_m":  pareto.get("stability_std_m"),
            "params_trainable": pareto.get("params_trainable"),
            "train_epoch_s":    pareto.get("train_epoch_s"),
            "pareto_score":     pareto.get("score"),
            "w_lat":  weights.get("w_lat"),
            "w_stab": weights.get("w_stab"),
            "w_res":  weights.get("w_res"),
        })
    return out


def extract_run_summary(
    exp: Experiment,
    seed: int,
    run: int,
    agent: str,
    payload: dict,
) -> Dict[str, Any]:
    fs = payload.get("final_summary") or {}
    rs = fs.get("retry_stats") or {}
    ts = fs.get("token_stats") or {}
    rounds_done = fs.get("rounds_completed") or 0
    total_tokens = ts.get("total_tokens") or 0
    return {
        "experiment":        exp.name,
        "model":             exp.model,
        "seed":              seed,
        "run":               run,
        "agent":              agent,
        "best_val_loss":     fs.get("best_val_loss"),
        "best_round":        fs.get("best_round"),
        "total_rounds":      fs.get("total_rounds"),
        "rounds_completed":  rounds_done,
        "rounds_skipped":    fs.get("rounds_skipped"),
        "retry_attempts":    rs.get("total_attempts"),
        "retries":           rs.get("retries"),
        "fallbacks":         rs.get("fallbacks"),
        "prompt_tokens":     ts.get("prompt_tokens"),
        "completion_tokens": ts.get("completion_tokens"),
        "total_tokens":      total_tokens,
        "llm_calls":         ts.get("calls"),
        "tokens_per_round":  (total_tokens / rounds_done) if rounds_done else None,
    }


def extract_cross_run(exp: Experiment, seed: int, payload: dict) -> Dict[str, Any]:
    """Pull the head-to-head rmse/r2/efficiency from cross_run_metrics.json."""
    oq = payload.get("optimization_quality") or {}
    eff = payload.get("optimization_efficiency") or {}
    rmse = oq.get("mean_rmse") or {}
    r2   = oq.get("mean_r2")   or {}
    auc  = eff.get("auc_val_loss_curve") or {}
    ttt  = eff.get("trials_to_threshold") or {}
    return {
        "experiment": exp.name,
        "model":      exp.model,
        "seed":       seed,
        "rmse_baseline":   rmse.get("baseline"),
        "rmse_llm":        rmse.get("llm"),
        "rmse_random":     rmse.get("random"),
        "rmse_rule_based": rmse.get("rule_based"),
        "rmse_optuna":     rmse.get("optuna"),
        "r2_baseline":     r2.get("baseline"),
        "r2_llm":          r2.get("llm"),
        "r2_random":       r2.get("random"),
        "r2_rule_based":   r2.get("rule_based"),
        "r2_optuna":       r2.get("optuna"),
        "auc_baseline":    auc.get("baseline"),
        "auc_llm":         auc.get("llm"),
        "auc_random":      auc.get("random"),
        "auc_rule_based":  auc.get("rule_based"),
        "auc_optuna":      auc.get("optuna"),
        "trials_to_threshold_llm":        ttt.get("llm"),
        "trials_to_threshold_random":     ttt.get("random"),
        "trials_to_threshold_rule_based": ttt.get("rule_based"),
        "trials_to_threshold_optuna":     ttt.get("optuna"),
        "pct_improvement_llm_vs_baseline":        oq.get("pct_improvement_over_baseline"),
        "pct_improvement_rule_based_vs_baseline": oq.get("pct_improvement_rule_based_over_baseline"),
        "pct_improvement_optuna_vs_baseline":     oq.get("pct_improvement_optuna_over_baseline"),
        "win_rate_llm":        oq.get("win_rate"),
        "win_rate_rule_based": oq.get("win_rate_rule_based"),
        "win_rate_optuna":     oq.get("win_rate_optuna"),
    }


# ---------------------------------------------------------------------------
# Aggregation / analytics
# ---------------------------------------------------------------------------

def per_experiment_stats(
    rounds_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    exp_name: str,
) -> Dict[str, Any]:
    """Compute LLM-as-optimizer summary statistics for a single experiment."""
    er  = rounds_df[rounds_df["experiment"] == exp_name]
    eru = runs_df[runs_df["experiment"] == exp_name]
    ec  = cross_df[cross_df["experiment"] == exp_name]

    llm_rounds  = er[er["agent"] == "llm"]
    llm_active  = llm_rounds[~llm_rounds["skipped"].fillna(False)]
    rule_rounds = er[er["agent"] == "rule_based"]

    diag_counts = Counter(llm_active["primary_problem"].dropna().tolist())
    invalid_labels = [d for d in llm_active["primary_problem"].dropna().tolist() if d not in ALLOWED_DIAGNOSES]

    llm_runs  = eru[eru["agent"] == "llm"]

    n_skipped = int(llm_rounds["skipped"].fillna(False).sum())
    total_attempts = int(llm_runs["retry_attempts"].fillna(0).sum())
    total_retries  = int(llm_runs["retries"].fillna(0).sum())
    total_fallbacks = int(llm_runs["fallbacks"].fillna(0).sum())

    # Action-impact: mean delta_best_val_loss conditional on n_changes > 0 vs == 0
    impact_changed = llm_active[llm_active["n_changes"] > 0]["delta_best_val_loss"].dropna()
    impact_nochg   = llm_active[llm_active["n_changes"] == 0]["delta_best_val_loss"].dropna()

    outcome_counts = Counter(llm_active["outcome"].dropna().tolist())

    # Diagnosis -> outcome breakdown
    diag_outcome = (
        llm_active.dropna(subset=["primary_problem", "outcome"])
        .groupby(["primary_problem", "outcome"])
        .size()
        .unstack(fill_value=0)
    )

    # Validator interventions: rejected proposals (skipped=True with a reason).
    # For each rejected (seed, round), look for a non-skipped sibling row to
    # determine whether the round was retried successfully or fell through
    # to a no-op fallback. n_changes==0 on the surviving row ⇒ fallback.
    rejected = llm_rounds[llm_rounds["skipped"].fillna(False)].copy()
    interventions: List[Dict[str, Any]] = []
    if not rejected.empty:
        non_skipped = llm_rounds[~llm_rounds["skipped"].fillna(False)]
        for _, row in rejected.iterrows():
            sibling = non_skipped[
                (non_skipped["seed"] == row["seed"]) &
                (non_skipped["round"] == row["round"])
            ]
            if sibling.empty:
                result = "fallback (no row)"
            else:
                n_changes = int(sibling.iloc[0].get("n_changes") or 0)
                result = "retried (HP change applied)" if n_changes > 0 else "fallback (no HP change)"
            interventions.append({
                "seed":           row["seed"],
                "round":          row["round"],
                "failure_reason": row.get("failure_reason") or "—",
                "result":         result,
            })
        interventions.sort(key=lambda x: (x["seed"], x["round"]))

    return {
        "experiment":          exp_name,
        "n_seeds":             int(eru["seed"].nunique()),
        "n_llm_rounds_total":  int(len(llm_rounds)),
        "n_llm_rounds_active": int(len(llm_active)),
        "n_llm_skipped":       n_skipped,
        "retry_attempts":      total_attempts,
        "retries":             total_retries,
        "fallbacks":           total_fallbacks,
        "retry_rate":          (total_retries / total_attempts) if total_attempts else None,
        "fallback_rate":       (total_fallbacks / total_attempts) if total_attempts else None,
        "diagnosis_counts":    dict(diag_counts),
        "invalid_diagnosis_labels": list(set(invalid_labels)),
        "outcome_counts":      dict(outcome_counts),
        "mean_delta_when_changed":  float(impact_changed.mean()) if len(impact_changed) else None,
        "mean_delta_when_no_change": float(impact_nochg.mean()) if len(impact_nochg) else None,
        # Cross-agent head-to-head means across seeds:
        "rmse_baseline_mean":   float(ec["rmse_baseline"].mean())   if len(ec) else None,
        "rmse_llm_mean":        float(ec["rmse_llm"].mean())        if len(ec) else None,
        "rmse_random_mean":     float(ec["rmse_random"].mean())     if len(ec) else None,
        "rmse_rule_based_mean": float(ec["rmse_rule_based"].mean()) if len(ec) else None,
        "win_rate_llm_mean":        float(ec["win_rate_llm"].mean())        if len(ec) else None,
        "win_rate_rule_based_mean": float(ec["win_rate_rule_based"].mean()) if len(ec) else None,
        "pct_improvement_llm_vs_baseline_mean":
            float(ec["pct_improvement_llm_vs_baseline"].mean()) if len(ec) else None,
        "pct_improvement_rule_based_vs_baseline_mean":
            float(ec["pct_improvement_rule_based_vs_baseline"].mean()) if len(ec) else None,
        "_diag_outcome_table": diag_outcome,
        "validator_interventions": interventions,
    }


# ---------------------------------------------------------------------------
# Paired-difference statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedResult:
    n: int                       # number of paired seeds used
    mean_diff: float             # mean of (a - b)
    sd_diff: float               # sample SD of (a - b), ddof=1
    t_ci_low: float              # 95% CI low — paired t
    t_ci_high: float             # 95% CI high — paired t
    t_pvalue: float              # two-sided paired t p-value
    boot_ci_low: float           # 95% CI low — percentile bootstrap
    boot_ci_high: float          # 95% CI high — percentile bootstrap
    wilcoxon_pvalue: Optional[float]   # two-sided Wilcoxon signed-rank p-value


def _paired_diffs(a: List[float], b: List[float]) -> Optional[PairedResult]:
    """Compute paired-difference statistics for two seed-aligned lists.

    `a` and `b` must be already aligned by seed (i.e. a[i] and b[i] come from
    the same seed). Pairs containing NaN/None are dropped. Returns None if
    fewer than 2 valid pairs survive.
    """
    arr_a = np.asarray([float("nan") if v is None else float(v) for v in a], dtype=float)
    arr_b = np.asarray([float("nan") if v is None else float(v) for v in b], dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask.sum() < 2:
        return None
    diff = arr_a[mask] - arr_b[mask]
    n = int(diff.size)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / math.sqrt(n) if n > 1 else float("nan")

    # Paired t CI + two-sided p-value. Use scipy.stats if available, else
    # fall back to a normal approximation (good enough for n>=10).
    try:
        from scipy import stats as _stats  # type: ignore
        tcrit = float(_stats.t.ppf(0.975, df=n - 1))
        t_stat = mean / se if se > 0 else 0.0
        t_p = float(2.0 * (1.0 - _stats.t.cdf(abs(t_stat), df=n - 1))) if se > 0 else 1.0
        # Wilcoxon needs at least one non-zero difference; emit None otherwise
        # (printed as "—" in the table) instead of letting nan leak through.
        if not np.any(diff != 0):
            w_p = None
        else:
            try:
                wp = float(_stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
                w_p = None if not math.isfinite(wp) else wp
            except Exception:
                w_p = None
    except Exception:
        # Normal approximation fallback — flags a missing scipy install but
        # keeps the table usable.
        tcrit = 1.96
        t_stat = mean / se if se > 0 else 0.0
        # crude two-sided z-test p-value
        t_p = float(math.erfc(abs(t_stat) / math.sqrt(2.0))) if se > 0 else 1.0
        w_p = None

    t_low = mean - tcrit * se
    t_high = mean + tcrit * se

    # Percentile bootstrap (seeded for reproducibility)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_means = rng.choice(diff, size=(BOOTSTRAP_ITERS, n), replace=True).mean(axis=1)
    boot_low = float(np.percentile(boot_means, 2.5))
    boot_high = float(np.percentile(boot_means, 97.5))

    return PairedResult(
        n=n,
        mean_diff=mean,
        sd_diff=sd,
        t_ci_low=float(t_low),
        t_ci_high=float(t_high),
        t_pvalue=float(t_p),
        boot_ci_low=boot_low,
        boot_ci_high=boot_high,
        wilcoxon_pvalue=w_p,
    )


def _arm_series_by_seed(cross_df: pd.DataFrame, experiment: str, arm_col: str) -> Dict[int, float]:
    """Return {seed: rmse} for one experiment+arm. NaN values are skipped."""
    sub = cross_df[cross_df["experiment"] == experiment]
    out: Dict[int, float] = {}
    for _, row in sub.iterrows():
        v = row.get(arm_col)
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(vf):
            out[int(row["seed"])] = vf
    return out


def _align_pairs(a_map: Dict[int, float], b_map: Dict[int, float]) -> Tuple[List[float], List[float], List[int]]:
    """Align two {seed: value} maps. Returns (a_values, b_values, seeds_used)."""
    common = sorted(set(a_map) & set(b_map))
    return [a_map[s] for s in common], [b_map[s] for s in common], common


def within_experiment_pairs(cross_df: pd.DataFrame, experiment: str) -> List[Tuple[str, PairedResult]]:
    """Build the headline paired contrasts within one experiment.

    Always reports (when data is present):
        LLM        − Baseline
        Rule-Based − Baseline
        LLM        − Rule-Based
        LLM        − Random

    When the Optuna arm has seeds in this experiment, additionally:
        Optuna     − Baseline       (vs the strong conventional optimizer)
        LLM        − Optuna         (does the LLM beat Bayesian HPO?)
        Rule-Based − Optuna
        Optuna     − Random         (sanity-check that TPE beats random)
    """
    base = _arm_series_by_seed(cross_df, experiment, "rmse_baseline")
    llm = _arm_series_by_seed(cross_df, experiment, "rmse_llm")
    rul = _arm_series_by_seed(cross_df, experiment, "rmse_rule_based")
    rnd = _arm_series_by_seed(cross_df, experiment, "rmse_random")
    opt = _arm_series_by_seed(cross_df, experiment, "rmse_optuna") \
        if "rmse_optuna" in cross_df.columns else {}

    contrasts: List[Tuple[str, Dict[int, float], Dict[int, float]]] = [
        ("LLM − Baseline",        llm, base),
        ("Rule-Based − Baseline", rul, base),
        ("LLM − Rule-Based",      llm, rul),
        ("LLM − Random",          llm, rnd),
    ]
    if opt:
        contrasts.extend([
            ("Optuna − Baseline",     opt, base),
            ("LLM − Optuna",          llm, opt),
            ("Rule-Based − Optuna",   rul, opt),
            ("Optuna − Random",       opt, rnd),
        ])

    out: List[Tuple[str, PairedResult]] = []
    for label, a_map, b_map in contrasts:
        a_vals, b_vals, _ = _align_pairs(a_map, b_map)
        res = _paired_diffs(a_vals, b_vals)
        if res is not None:
            out.append((label, res))
    return out


def cross_experiment_pair(
    cross_df: pd.DataFrame,
    experiment_a: str,
    experiment_b: str,
    arm: str = "rmse_llm",
) -> Optional[PairedResult]:
    """Paired contrast between the same arm across two experiments.

    Used for ablations like 'LLM repair-on vs LLM repair-off' — pass the two
    experiment names and the arm column (defaults to rmse_llm).
    """
    a_map = _arm_series_by_seed(cross_df, experiment_a, arm)
    b_map = _arm_series_by_seed(cross_df, experiment_b, arm)
    a_vals, b_vals, _ = _align_pairs(a_map, b_map)
    return _paired_diffs(a_vals, b_vals)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame, index: bool = False, floatfmt: str = ".4f") -> str:
    """Markdown table without requiring `tabulate`. Works on a small DataFrame."""
    if df is None or df.empty:
        return "_(empty)_"
    work = df.reset_index() if index else df.copy()

    def _cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:{floatfmt[1:] if floatfmt.startswith(':') else floatfmt}}"
        # Escape pipes so a free-text LLM output containing "|" doesn't
        # break the markdown table layout.
        return str(v).replace("|", "\\|")

    cols = [str(c) for c in work.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(_cell(row[c]) for c in work.columns) + " |")
    return "\n".join(lines)


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _fmt_pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x*100:.1f}%" if abs(x) < 5 else f"{x:.2f}"  # win rates are 0..1, pct_improvement is already %


def _render_metric_definition() -> str:
    return (
        "## Metric definition\n"
        f"- **Metric:** {METRIC_NAME} — `sqrt(mean((preds − targets)²))` over all output dims combined.\n"
        f"- **Stage:** evaluated on the held-out **{METRIC_STAGE}** set (not val).\n"
        f"- **Units:** {METRIC_UNITS}.\n"
        f"- **Direction:** {METRIC_DIRECTION}.\n"
        f"- **Aggregation:** {METRIC_AGGREGATION}; N = number of seeds (one run per seed).\n"
        f"- **CIs:** paired-t and {BOOTSTRAP_ITERS:,}-iter percentile bootstrap (seed {BOOTSTRAP_SEED}); "
        "Wilcoxon signed-rank p-value reported as a non-parametric companion.\n"
    )


def _fmt_signed(x: Optional[float], digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{digits}f}"


def _render_paired_table(rows: List[Tuple[str, PairedResult]]) -> str:
    if not rows:
        return "_(insufficient paired data — need at least 2 seeds with finite values in both arms)_\n"
    out = [
        "| Contrast (A − B) | n | Mean Δ | SD Δ | 95% CI (paired-t) | 95% CI (bootstrap) | paired-t p | Wilcoxon p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, r in rows:
        t_ci = f"[{_fmt_signed(r.t_ci_low)}, {_fmt_signed(r.t_ci_high)}]"
        b_ci = f"[{_fmt_signed(r.boot_ci_low)}, {_fmt_signed(r.boot_ci_high)}]"
        w_p = "—" if r.wilcoxon_pvalue is None else f"{r.wilcoxon_pvalue:.4f}"
        out.append(
            f"| {label} | {r.n} | {_fmt_signed(r.mean_diff)} | {_fmt(r.sd_diff)} | "
            f"{t_ci} | {b_ci} | {r.t_pvalue:.4f} | {w_p} |"
        )
    return "\n".join(out) + "\n"


def render_experiment_report(stats: Dict[str, Any], cross_df: Optional[pd.DataFrame] = None) -> str:
    out = []
    out.append(f"# Experiment report — `{stats['experiment']}`\n")
    out.append(f"_{stats['n_seeds']} seed(s), {stats['n_llm_rounds_total']} LLM rounds total ({stats['n_llm_skipped']} skipped, {stats['n_llm_rounds_active']} active)._\n")

    out.append(_render_metric_definition())

    out.append("## Head-to-head (mean ± SD RMSE across seeds)\n")
    out.append("| Agent | RMSE (mean ± SD) | n | Δ vs baseline |")
    out.append("|---|---|---|---|")
    base = stats.get("rmse_baseline_mean")

    # Per-seed mean+SD across seeds — recompute from cross_df because the
    # per-seed JSONs only carry std=0 (single run per seed). We include
    # `optuna` here too; rows for an arm with no data are skipped below.
    arm_keys = ("baseline", "llm", "random", "rule_based", "optuna")
    sd_map: Dict[str, Optional[float]] = {k: None for k in arm_keys}
    n_map: Dict[str, int] = {k: 0 for k in arm_keys}
    mean_map: Dict[str, Optional[float]] = {k: None for k in arm_keys}
    if cross_df is not None and not cross_df.empty:
        sub = cross_df[cross_df["experiment"] == stats["experiment"]]
        for arm in arm_keys:
            col = f"rmse_{arm}"
            if col in sub.columns:
                vals = sub[col].dropna().astype(float).to_numpy()
                n_map[arm] = int(vals.size)
                if vals.size > 0:
                    mean_map[arm] = float(np.mean(vals))
                if vals.size > 1:
                    sd_map[arm] = float(np.std(vals, ddof=1))
                elif vals.size == 1:
                    sd_map[arm] = 0.0

    for label, arm_key in (
        ("Baseline",   "baseline"),
        ("LLM",        "llm"),
        ("Random",     "random"),
        ("Rule-Based", "rule_based"),
        ("Optuna (TPE)", "optuna"),
    ):
        n = n_map[arm_key]
        if n == 0:
            # Don't print an empty row for an arm that has no data (Optuna
            # is the common case for legacy experiments).
            continue
        v = mean_map[arm_key]
        sd = sd_map[arm_key]
        cell = "—" if v is None else (
            f"{v:.4f} ± {sd:.4f}" if sd is not None else f"{v:.4f}"
        )
        delta = ((v - base) / base * 100.0) if (v is not None and base) else None
        delta_s = f"{delta:+.2f}%" if delta is not None else "—"
        out.append(f"| {label} | {cell} | {n} | {delta_s} |")
    out.append("")

    # Paired-seed comparisons within this experiment.
    if cross_df is not None and not cross_df.empty:
        out.append("## Paired-seed comparisons (within-experiment)\n")
        out.append(
            "_Each row pairs the two arms by seed, then computes the mean of "
            "(A − B). A negative mean Δ means A is **better** (lower RMSE) than "
            "B; a CI that excludes 0 indicates the difference is statistically "
            "distinguishable from zero at the 95% level._\n"
        )
        pairs = within_experiment_pairs(cross_df, stats["experiment"])
        out.append(_render_paired_table(pairs))

    out.append(f"- LLM win rate (vs baseline, across seeds): **{_fmt_pct(stats.get('win_rate_llm_mean'))}**")
    out.append(f"- Rule-Based win rate (vs baseline, across seeds): **{_fmt_pct(stats.get('win_rate_rule_based_mean'))}**\n")

    out.append("## LLM loop reliability\n")
    out.append(f"- Total LLM attempts: **{stats['retry_attempts']}**")
    out.append(f"- Retries: **{stats['retries']}** "
               f"({_fmt_pct(stats.get('retry_rate'))} of attempts)")
    out.append(f"- Fallbacks (round skipped): **{stats['fallbacks']}** "
               f"({_fmt_pct(stats.get('fallback_rate'))} of attempts)\n")

    out.append("## Diagnosis distribution (active rounds only)\n")
    diag = stats.get("diagnosis_counts") or {}
    if diag:
        total = sum(diag.values()) or 1
        out.append("| Diagnosis | Count | Share |")
        out.append("|---|---|---|")
        for k in sorted(diag, key=lambda x: -diag[x]):
            out.append(f"| `{k}` | {diag[k]} | {diag[k]/total*100:.1f}% |")
    else:
        out.append("_No diagnoses recorded._")
    out.append("")
    invalid = stats.get("invalid_diagnosis_labels") or []
    if invalid:
        out.append(f"⚠️ Invalid diagnosis labels emitted (post-validator): `{invalid}`\n")
    else:
        out.append("✅ All emitted diagnoses are within the allowed enum.\n")

    out.append("## Action impact\n")
    out.append(f"- Mean Δ best-val-loss when LLM changed an HP: **{_fmt(stats.get('mean_delta_when_changed'))}**  (negative = improvement)")
    out.append(f"- Mean Δ best-val-loss when LLM proposed no change: **{_fmt(stats.get('mean_delta_when_no_change'))}**\n")

    out.append("## Outcome of LLM rounds\n")
    oc = stats.get("outcome_counts") or {}
    if oc:
        total = sum(oc.values()) or 1
        out.append("| Outcome | Count | Share |")
        out.append("|---|---|---|")
        for k in sorted(oc, key=lambda x: -oc[x]):
            out.append(f"| `{k}` | {oc[k]} | {oc[k]/total*100:.1f}% |")
        out.append("")

    diag_out = stats.get("_diag_outcome_table")
    if isinstance(diag_out, pd.DataFrame) and not diag_out.empty:
        out.append("## Diagnosis × outcome\n")
        out.append(_df_to_markdown(diag_out, index=True))
        out.append("")

    interventions = stats.get("validator_interventions") or []
    out.append("## Validator interventions\n")
    if not interventions:
        out.append("_No proposals were rejected by the validator in this experiment._\n")
    else:
        out.append(f"_{len(interventions)} rejected proposal(s) caught by the validator. "
                   f"Each row is one round where the LLM emitted a logically inconsistent "
                   f"diagnosis or out-of-schema output, which was rejected before being applied._\n")
        out.append("| Seed | Round | Failure reason | Result |")
        out.append("|---|---|---|---|")
        for ev in interventions:
            reason = str(ev["failure_reason"]).replace("|", "\\|")
            out.append(f"| {ev['seed']} | {ev['round']} | {reason} | {ev['result']} |")
        out.append("")

    return "\n".join(out)


def render_summary_report(
    experiments_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> str:
    out: List[str] = []
    out.append("# Cross-experiment LLM-as-optimizer summary\n")
    out.append(f"_{len(experiments_df)} experiments, "
               f"{int(rounds_df[rounds_df['agent']=='llm'].shape[0])} LLM rounds total._\n")

    out.append(_render_metric_definition())

    out.append("## Per-experiment ranking by LLM RMSE vs baseline\n")
    rank_cols = [
        "experiment", "n_seeds",
        "rmse_baseline_mean", "rmse_llm_mean",
        "pct_improvement_llm_vs_baseline_mean",
        "pct_improvement_rule_based_vs_baseline_mean",
        "win_rate_llm_mean",
        "retry_rate", "fallback_rate",
    ]
    avail = [c for c in rank_cols if c in experiments_df.columns]
    sorted_df = experiments_df.sort_values(
        "pct_improvement_llm_vs_baseline_mean", ascending=False, na_position="last"
    )
    out.append(_df_to_markdown(sorted_df[avail], index=False, floatfmt=".4f"))
    out.append("")

    out.append("## Diagnosis distribution across all LLM runs\n")
    llm_active = rounds_df[(rounds_df["agent"]=="llm") & (~rounds_df["skipped"].fillna(False))]
    diag = llm_active["primary_problem"].value_counts()
    if not diag.empty:
        out.append(_df_to_markdown(diag.rename_axis("diagnosis").reset_index(name="count"), index=False))
        invalid = sorted(set(d for d in llm_active["primary_problem"].dropna() if d not in ALLOWED_DIAGNOSES))
        if invalid:
            out.append(f"\n⚠️ Invalid labels seen across all experiments: `{invalid}`\n")
    out.append("")

    out.append("## Reliability — per model (LLM agent)\n")
    out.append("_Only experiments where retry_stats was recorded are shown; "
               "older runs predate that telemetry._\n")
    rel = (
        runs_df[runs_df["agent"] == "llm"]
        .groupby("model")[["retry_attempts", "retries", "fallbacks"]]
        .sum()
    )
    rel = rel[rel["retry_attempts"] > 0].assign(
        retry_rate=lambda d: d["retries"] / d["retry_attempts"],
        fallback_rate=lambda d: d["fallbacks"] / d["retry_attempts"],
    ).reset_index().sort_values("fallback_rate", ascending=False, na_position="last")
    out.append(_df_to_markdown(rel, index=False, floatfmt=".4f"))
    out.append("")

    out.append("## Action impact across all LLM rounds\n")
    impact_changed = llm_active[llm_active["n_changes"] > 0]["delta_best_val_loss"].dropna()
    impact_nochg   = llm_active[llm_active["n_changes"] == 0]["delta_best_val_loss"].dropna()
    out.append(f"- Mean Δ best-val-loss when LLM changed HP(s): **{_fmt(impact_changed.mean() if len(impact_changed) else None)}** (n={len(impact_changed)})")
    out.append(f"- Mean Δ best-val-loss when LLM made no change: **{_fmt(impact_nochg.mean() if len(impact_nochg) else None)}** (n={len(impact_nochg)})")
    out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_dataframes(root: Path, max_depth: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rounds: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    cross: List[Dict[str, Any]] = []
    experiments = discover_experiments(root, max_depth=max_depth)
    for exp in experiments:
        # cross_run_metrics.json (one per seed_dir)
        for seed_dir in sorted(exp.path.iterdir()):
            if not seed_dir.is_dir():
                continue
            sm = SEED_DIR_RE.match(seed_dir.name)
            if not sm:
                continue
            seed = int(sm.group("seed"))
            crm = seed_dir / "cross_run_metrics.json"
            if crm.exists():
                payload = _safe_load(crm)
                if payload is not None:
                    cross.append(extract_cross_run(exp, seed, payload))

        # rounds + per-run summary
        for seed, run, agent, path in discover_runs(exp):
            payload = _safe_load(path)
            if payload is None:
                continue
            rounds.extend(extract_rounds(exp, seed, run, agent, payload))
            runs.append(extract_run_summary(exp, seed, run, agent, payload))

    rounds_df = pd.DataFrame(rounds)
    runs_df   = pd.DataFrame(runs)
    cross_df  = pd.DataFrame(cross)
    return rounds_df, runs_df, cross_df


def build_experiments_df(rounds_df: pd.DataFrame, runs_df: pd.DataFrame, cross_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for exp in sorted(set(runs_df["experiment"].dropna())):
        s = per_experiment_stats(rounds_df, runs_df, cross_df, exp)
        # Strip non-serializable bits before tabulating
        row = {k: v for k, v in s.items() if not k.startswith("_") and not isinstance(v, dict) and not isinstance(v, list)}
        rows.append(row)
    return pd.DataFrame(rows)


def _import_plt():
    """Lazy import — analyze_logs is sometimes run on hosts without a GUI backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401 — caller uses returned module
    return plt


def plot_pareto_front(rounds_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """Latency-vs-val_loss scatter coloured by parameter count (resource axis).

    Each round of every LLM-arm run is one dot. Lower-left corner is the goal.
    Skips silently when none of the Block-2 fields are populated (e.g. legacy
    runs that pre-date Pareto telemetry).
    """
    df = rounds_df[(rounds_df["agent"] == "llm")] if "agent" in rounds_df.columns else rounds_df
    df = df.dropna(subset=["latency_ms", "val_loss", "params_trainable"])
    if df.empty:
        return None

    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["latency_ms"], df["val_loss"],
        c=df["params_trainable"], s=42, cmap="viridis",
        edgecolor="white", linewidths=0.5, alpha=0.85,
    )
    ax.set_xlabel("latency_ms (window + forward pass)")
    ax.set_ylabel("val_loss")
    ax.set_title("Pareto front — latency vs. accuracy (colour = params)")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("params_trainable")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_weights_evolution(rounds_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """Stacked-area plot of the LLM's (w_lat, w_stab, w_res) over rounds.

    Averages weights across seeds and runs at each round index, so a single
    line per axis tells the dominant story. If multiple experiments are mixed
    in `rounds_df`, caller should slice first.
    """
    df = rounds_df[(rounds_df["agent"] == "llm")] if "agent" in rounds_df.columns else rounds_df
    df = df.dropna(subset=["round", "w_lat", "w_stab", "w_res"])
    if df.empty:
        return None

    grouped = (
        df.groupby("round", as_index=False)[["w_lat", "w_stab", "w_res"]]
          .mean()
          .sort_values("round")
    )

    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(
        grouped["round"],
        grouped["w_lat"], grouped["w_stab"], grouped["w_res"],
        labels=["w_lat", "w_stab", "w_res"],
        colors=["#3b82f6", "#f59e0b", "#10b981"],
        alpha=0.85,
    )
    ax.set_xlabel("round")
    ax.set_ylabel("weight share")
    ax.set_ylim(0, 1)
    ax.set_title("LLM Pareto weights over rounds (mean across runs)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render_cross_experiment_pair_report(
    cross_df: pd.DataFrame,
    exp_a: str,
    exp_b: str,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
) -> str:
    """Cross-experiment ablation report (e.g. repair-on vs repair-off).

    For each arm (LLM, baseline, rule-based, random) we pair the same seed
    across the two experiments and compute the paired difference (A − B).
    The LLM row is the headline number for a repair-on/off ablation.
    """
    la = label_a or exp_a
    lb = label_b or exp_b
    out: List[str] = []
    out.append(f"# Paired ablation — `{la}` vs `{lb}`\n")
    out.append(_render_metric_definition())
    out.append(
        "## Cross-experiment paired differences\n"
        f"_Each row pairs the same arm across the two experiments by seed, "
        f"computing `{la} − {lb}`. A negative mean Δ means **`{la}` has lower "
        f"(better) RMSE** than `{lb}` on the same seeds._\n"
    )
    rows: List[Tuple[str, PairedResult]] = []
    for arm_label, arm_col in (
        ("LLM",          "rmse_llm"),
        ("Baseline",     "rmse_baseline"),
        ("Rule-Based",   "rmse_rule_based"),
        ("Random",       "rmse_random"),
        ("Optuna (TPE)", "rmse_optuna"),
    ):
        if arm_col not in cross_df.columns:
            continue
        res = cross_experiment_pair(cross_df, exp_a, exp_b, arm=arm_col)
        if res is not None:
            rows.append((arm_label, res))
    out.append(_render_paired_table(rows))

    # Also dump the raw seed-aligned values for transparency.
    a_seeds = _arm_series_by_seed(cross_df, exp_a, "rmse_llm")
    b_seeds = _arm_series_by_seed(cross_df, exp_b, "rmse_llm")
    common = sorted(set(a_seeds) & set(b_seeds))
    if common:
        out.append("## Per-seed LLM RMSE (raw values used in the LLM-row pairing above)\n")
        out.append(f"| Seed | {la} | {lb} | Δ ({la} − {lb}) |")
        out.append("|---|---|---|---|")
        for s in common:
            d = a_seeds[s] - b_seeds[s]
            out.append(f"| {s} | {a_seeds[s]:.4f} | {b_seeds[s]:.4f} | {_fmt_signed(d)} |")
        out.append("")
    return "\n".join(out)


def _pair_to_csv_rows(label: str, contrast: str, r: PairedResult) -> Dict[str, Any]:
    return {
        "pairing": label,
        "contrast": contrast,
        "n": r.n,
        "mean_diff": r.mean_diff,
        "sd_diff": r.sd_diff,
        "t_ci_low": r.t_ci_low,
        "t_ci_high": r.t_ci_high,
        "t_pvalue": r.t_pvalue,
        "boot_ci_low": r.boot_ci_low,
        "boot_ci_high": r.boot_ci_high,
        "wilcoxon_pvalue": r.wilcoxon_pvalue,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent),
                    help=("Directory to search for experiment run dirs. Searched "
                          "recursively, so `--root results` sweeps every group; "
                          "defaults to the repo root."))
    ap.add_argument("--max-depth", type=int, default=3,
                    help="How many directory levels below --root to search (default 3).")
    ap.add_argument("--out", default="analysis",
                    help="Output directory (relative to --root unless absolute).")
    ap.add_argument(
        "--pair",
        nargs="+",
        action="append",
        metavar="EXPNAME",
        default=None,
        help=(
            "Cross-experiment paired ablation. Pass 2-4 args: EXP_A EXP_B "
            "[LABEL_A LABEL_B]. EXP_A/EXP_B are experiment names (the "
            "'<date>-<model>' part of `outputs-<date>-<model>/`). May be "
            "repeated to emit several ablation reports in one run."
        ),
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] root        = {root}")
    print(f"[info] output dir  = {out_dir}")

    rounds_df, runs_df, cross_df = build_dataframes(root, max_depth=args.max_depth)
    if rounds_df.empty:
        print("[warn] no rounds extracted — nothing to do.")
        print(f"[warn] searched {root} to depth {args.max_depth}; no directory "
              f"there contains a seed_<N>/ subdir with an optimisation log. "
              f"Try a different --root or a larger --max-depth.")
        return

    print(f"[info] discovered  : {rounds_df['experiment'].nunique()} experiments | "
          f"{rounds_df['seed'].nunique()} seeds | {len(rounds_df)} rounds | {len(runs_df)} runs")

    # ---- Long-form CSVs ----
    rounds_df.to_csv(out_dir / "all_rounds.csv", index=False)
    runs_df.to_csv(out_dir / "all_runs.csv", index=False)
    cross_df.to_csv(out_dir / "cross_run.csv", index=False)

    # ---- Per-experiment stats table ----
    experiments_df = build_experiments_df(rounds_df, runs_df, cross_df)
    experiments_df.to_csv(out_dir / "experiments.csv", index=False)

    # ---- Per-experiment Markdown reports + Pareto plots ----
    paired_rows: List[Dict[str, Any]] = []
    for exp_name in sorted(set(runs_df["experiment"].dropna())):
        stats = per_experiment_stats(rounds_df, runs_df, cross_df, exp_name)
        md = render_experiment_report(stats, cross_df=cross_df)
        exp_out = out_dir / exp_name
        exp_out.mkdir(parents=True, exist_ok=True)
        (exp_out / "report.md").write_text(md, encoding="utf-8")

        # Capture within-experiment paired contrasts in a long-form CSV.
        for label, r in within_experiment_pairs(cross_df, exp_name):
            row = _pair_to_csv_rows(exp_name, label, r)
            row["kind"] = "within"
            paired_rows.append(row)

        exp_rounds = rounds_df[rounds_df["experiment"] == exp_name]
        try:
            pf = plot_pareto_front(exp_rounds, exp_out / "pareto_front.png")
            if pf:
                print(f"[plot] {exp_name}: pareto_front.png")
        except Exception as exc:
            print(f"[warn] pareto_front plot failed for {exp_name}: {exc}")
        try:
            we = plot_weights_evolution(exp_rounds, exp_out / "weights_evolution.png")
            if we:
                print(f"[plot] {exp_name}: weights_evolution.png")
        except Exception as exc:
            print(f"[warn] weights_evolution plot failed for {exp_name}: {exc}")

    # ---- Cross-experiment ablation reports (--pair) ----
    known_exps = set(cross_df["experiment"].dropna().unique()) if not cross_df.empty else set()
    for spec in args.pair or []:
        if len(spec) < 2:
            print(f"[warn] --pair needs at least EXP_A EXP_B, got {spec}; skipping.")
            continue
        exp_a, exp_b = spec[0], spec[1]
        label_a = spec[2] if len(spec) >= 3 else None
        label_b = spec[3] if len(spec) >= 4 else None
        missing = [e for e in (exp_a, exp_b) if e not in known_exps]
        if missing:
            print(f"[warn] --pair: experiment(s) not found: {missing}. Known: {sorted(known_exps)}")
            continue
        report = render_cross_experiment_pair_report(cross_df, exp_a, exp_b, label_a, label_b)
        slug = f"paired_{exp_a}_vs_{exp_b}"
        (out_dir / f"{slug}.md").write_text(report, encoding="utf-8")
        # Append to long-form CSV
        for arm_label, arm_col in (
            ("LLM",          "rmse_llm"),
            ("Baseline",     "rmse_baseline"),
            ("Rule-Based",   "rmse_rule_based"),
            ("Random",       "rmse_random"),
            ("Optuna (TPE)", "rmse_optuna"),
        ):
            if arm_col not in cross_df.columns:
                continue
            r = cross_experiment_pair(cross_df, exp_a, exp_b, arm=arm_col)
            if r is None:
                continue
            row = _pair_to_csv_rows(f"{label_a or exp_a} vs {label_b or exp_b}",
                                    f"{arm_label} ({exp_a} − {exp_b})", r)
            row["kind"] = "cross"
            paired_rows.append(row)
        print(f"[pair] wrote {slug}.md")

    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(out_dir / "paired_diffs.csv", index=False)
        print(f"[done] paired_diffs.csv ({len(paired_rows)} rows)")

    # ---- Cross-experiment summary ----
    summary_md = render_summary_report(experiments_df, rounds_df, runs_df)
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"[done] wrote: all_rounds.csv all_runs.csv cross_run.csv experiments.csv summary.md")
    print(f"[done] per-experiment reports under {out_dir}/<experiment>/report.md")


if __name__ == "__main__":
    main()
