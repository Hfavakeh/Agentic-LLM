"""Plotting helpers for per-run and cross-run summaries."""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from pipeline import Config, is_finite_number, logger


def plot_optimization_trajectories(
    config: Config,
    histories: Dict[str, Dict[str, List[float]]],
    filename: str,
    title: str = "Optimization Trajectories",
) -> None:
    """
    Plot val-loss trajectories for any number of named histories.

    histories: {"Baseline": history_dict, "LLM": history_dict, ...}
    """
    # Cycle styles by index so the plot never silently drops arms: zipping
    # against a fixed 4-element list truncated to 4 lines, dropping the
    # last-inserted history (Optuna).
    styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, h) in enumerate(histories.items()):
        vl = h.get("val_loss", [])
        if vl:
            ax.plot(vl, lw=2, ls=styles[i % len(styles)], label=label)
    ax.set(xlabel="Epoch", ylabel="Validation Loss", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved -> %s", path)


def _build_param_series_from_log(
    opt_log: Dict[str, Any],
    param: str,
    default_value: float,
) -> Dict[str, List[float]]:
    def _as_float(v: Any) -> Optional[float]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if is_finite_number(f) else None

    init = _as_float(opt_log.get("initial_hyperparameters", {}).get(param))
    current = init if init is not None else float(default_value)
    rounds = [0]
    values = [current]

    for rd in sorted(opt_log.get("rounds", []), key=lambda x: x.get("round", 0)):
        r = int(rd.get("round", 0))
        changes = rd.get("changes_applied", {}) or {}
        if param in changes:
            # Rejected / malformed proposals (e.g. an LLM emitting prose where a
            # number belongs) are still logged but were never validly applied,
            # so they must not move the trajectory line.
            parsed = _as_float(changes[param])
            if parsed is not None:
                current = parsed
        rounds.append(r)
        values.append(current)

    return {"rounds": rounds, "values": values}


def plot_hyperparameter_trajectory(
    config: Config,
    opt_log: Dict[str, Any],
    filename: str,
    title: str = "Hyperparameter Trajectory",
) -> None:
    lr_series     = _build_param_series_from_log(opt_log, "learning_rate", 1e-3)
    do_series     = _build_param_series_from_log(opt_log, "dropout",       0.1)
    hidden_series = _build_param_series_from_log(opt_log, "lstm_hidden",   128)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(lr_series["rounds"], lr_series["values"], marker="o", lw=2, color="tab:blue")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(do_series["rounds"], do_series["values"], marker="o", lw=2, color="tab:orange")
    axes[1].set_ylabel("Dropout")
    axes[1].grid(alpha=0.3)

    axes[2].step(hidden_series["rounds"], hidden_series["values"], where="post", lw=2, color="tab:green")
    axes[2].scatter(hidden_series["rounds"], hidden_series["values"], s=28, color="tab:green")
    axes[2].set_ylabel("LSTM Hidden Size")
    axes[2].set_xlabel("Round")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Hyperparameter trajectory plot saved -> %s", path)


def plot_action_impact_scatter(
    config: Config,
    opt_log: Dict[str, Any],
    filename: str,
    title: str = "Agent Action Magnitude vs Delta Best Val Loss",
) -> None:
    xs, ys = [], []
    for rd in opt_log.get("rounds", []):
        mag = rd.get("change_magnitude")
        delta_best = rd.get("delta_best_val_loss")
        if is_finite_number(mag) and is_finite_number(delta_best):
            xs.append(float(mag))
            ys.append(float(delta_best))

    if not xs:
        logger.info("Skipping action-impact scatter plot (no finite points).")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, s=60, alpha=0.85, color="tab:red", edgecolors="black", linewidths=0.5)
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Proposed Change Magnitude (normalized)")
    ax.set_ylabel("Delta Best Validation Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Action-impact scatter plot saved -> %s", path)


def plot_training_dynamics(
    config: Config,
    all_results: List[Dict[str, Any]],
    filename: str = "training_dynamics.png",
) -> None:
    """
    PLOT 1 — Training Dynamics  (epoch-level)
    ==========================================
    Shows how val_loss evolves across *epochs* during the full training run
    of each arm.  This captures the learning curve shape, convergence speed,
    and overfitting behaviour — not which HP config was used.

    X-axis : Epoch
    Y-axis : Validation loss
    One faint line per experiment run; bold line = mean across runs.
    """
    arms   = ["baseline", "llm", "random", "rule_based"]
    labels = {"baseline": "Baseline (Expert)", "llm": "LLM", "random": "Random Search", "rule_based": "Rule-Based"}
    colors = {"baseline": "steelblue", "llm": "darkorange", "random": "seagreen", "rule_based": "mediumpurple"}

    fig, axes = plt.subplots(1, len(arms), figsize=(18, 5), sharey=True)
    for ax, arm in zip(axes, arms):
        run_vls = [r[arm]["history"].get("val_loss", []) for r in all_results]
        max_len = max((len(v) for v in run_vls), default=0)
        if max_len == 0:
            ax.set(title=labels[arm], xlabel="Epoch")
            continue
        padded = [v + [float("nan")] * (max_len - len(v)) for v in run_vls]
        arr    = np.array(padded, dtype=float)
        xs     = np.arange(max_len)

        for row in padded:
            ax.plot(xs, row, lw=1, alpha=0.25, color=colors[arm])

        mean = np.nanmean(arr, axis=0)
        ax.plot(xs, mean, lw=2.5, color=colors[arm], label=f"Mean ({len(all_results)} runs)")
        ax.set(xlabel="Epoch", title=labels[arm])
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Validation Loss")
    fig.suptitle("Training Dynamics — Epoch-Level Validation Loss", fontsize=13)
    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Training dynamics plot saved -> %s", path)


def plot_hp_search_dynamics(
    config: Config,
    all_results: List[Dict[str, Any]],
    filename: str = "hp_search_dynamics.png",
) -> None:
    """
    PLOT 2 — Hyperparameter Search Dynamics  (round-level)
    =======================================================
    Shows the val_loss *achieved by each HP configuration* tried during
    the optimization process.  Each tick on the x-axis is one new HP
    configuration (one optimization round), NOT individual training epochs.

    X-axis : Optimization round (1…N)
    Y-axis : Val_loss at end of that round's training budget
    One faint line per experiment run; bold line = mean across runs.

    Arms
    ----
    llm    — per-round val_loss from opt_log
    random — per-random-trial val_loss from random_log
    expert — flat horizontal reference (single config, no search)
    """
    arms   = ["baseline", "llm", "random", "rule_based"]
    labels = {"baseline": "Expert", "llm": "LLM", "random": "Random Search", "rule_based": "Rule-Based"}
    colors = {"baseline": "steelblue", "llm": "darkorange", "random": "seagreen", "rule_based": "mediumpurple"}

    def _opt_round_vals(result, arm: str) -> List[float]:
        opt_log = result.get(arm, {}).get("optimization_log", {})
        if not opt_log:
            return []
        vals: List[float] = []
        for rd in sorted(opt_log.get("rounds", []), key=lambda x: x.get("round", 0)):
            vl = rd.get("val_loss")
            if vl is not None and is_finite_number(vl):
                vals.append(float(vl))
        return vals

    def _random_round_vals(result) -> List[float]:
        rlog = result.get("random", {}).get("random_log", [])
        return [float(rd["val_loss"]) for rd in rlog
                if rd.get("val_loss") is not None and is_finite_number(rd["val_loss"])]

    def _baseline_round_vals(result, n_rounds: int) -> List[float]:
        """Flat line — expert trains once, no per-round HP changes."""
        bvl = result.get("baseline", {}).get("best_curve", [])
        if bvl:
            return list(bvl)
        hist = result.get("baseline", {}).get("history", {}).get("val_loss", [])
        vl   = hist[-1] if hist else float("nan")
        return [float(vl)] * (n_rounds + 1)

    n_rounds_global = max((len(_opt_round_vals(r, "llm")) for r in all_results), default=1) - 1

    extractors = {
        "baseline":   lambda r: _baseline_round_vals(r, n_rounds_global),
        "llm":        lambda r: _opt_round_vals(r, "llm"),
        "random":     _random_round_vals,
        "rule_based": lambda r: _opt_round_vals(r, "rule_based"),
    }

    fig, axes = plt.subplots(1, len(arms), figsize=(18, 5), sharey=True)
    for ax, arm in zip(axes, arms):
        run_vls = [extractors[arm](r) for r in all_results]
        max_len = max((len(v) for v in run_vls), default=0)
        if max_len == 0:
            ax.set(title=labels[arm], xlabel="Optimization Round")
            continue
        padded = [v + [float("nan")] * (max_len - len(v)) for v in run_vls]
        arr    = np.array(padded, dtype=float)
        xs     = np.arange(max_len)

        for row in padded:
            ax.plot(xs, row, lw=1, alpha=0.25, color=colors[arm])

        mean = np.nanmean(arr, axis=0)
        ax.plot(xs, mean, lw=2.5, color=colors[arm], label=f"Mean ({len(all_results)} runs)")
        ax.set_xticks(xs)
        ax.set(xlabel="Optimization Round", title=labels[arm])
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Validation Loss")
    fig.suptitle("HP Search Dynamics — Val Loss per Optimization Round", fontsize=13)
    path = config.output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("HP search dynamics plot saved -> %s", path)


# Keep alias so any external code using the old name still works
def plot_cross_run_summary(
    config: Config,
    all_results: List[Dict[str, Any]],
    filename: str = "cross_run_summary.png",
) -> None:
    """Calls both separated plots. Kept for backwards compatibility."""
    plot_training_dynamics(config, all_results,
                           filename=filename.replace(".png", "_training.png"))
    plot_hp_search_dynamics(config, all_results,
                            filename=filename.replace(".png", "_hp_search.png"))
