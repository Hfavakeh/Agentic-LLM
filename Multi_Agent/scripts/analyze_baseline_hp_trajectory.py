"""Reconstruct the HP-change trajectory anchored on the baseline (initial) setup
for the earliest warm-loop runs, with the val RMSE at each step.

Round 0 = the baseline / initial_hyperparameters (no change yet); its loss is the
first entry of best_curve. Each subsequent round's `changes_applied` holds the new
ABSOLUTE value(s) for the HP(s) the proposer edited, so we carry values forward to
get the full per-round config. RMSE = sqrt(val_loss) (scaled-space val MSE, the
metric these early runs logged).
"""
import json
import os
import math
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = "outputs-0413-qwen2.5-coder"
OUT = os.path.join("analysis", "baseline_hp_trajectory")
os.makedirs(OUT, exist_ok=True)

HP_KEYS = ["learning_rate", "weight_decay", "dropout", "batch_size",
           "lstm_hidden", "lstm_layers", "patience", "optimizer_choice"]


def load_seed(path):
    j = json.load(open(path))
    init = dict(j["initial_hyperparameters"])
    curve = j.get("best_curve", [])
    rows = []
    # round 0 = baseline
    base_loss = curve[0] if curve else None
    rows.append({"round": 0, "outcome": "baseline", "val_loss": base_loss,
                 "val_mae": None, **init})
    cur = dict(init)
    for r in j["rounds"]:
        cur.update(r.get("changes_applied", {}))
        rows.append({
            "round": r["round"],
            "outcome": r.get("outcome"),
            "val_loss": r.get("val_loss"),
            "val_mae": r.get("val_mae"),
            **{k: cur.get(k) for k in HP_KEYS},
        })
    return rows


def main():
    seed_dirs = sorted(d for d in os.listdir(RUN_DIR) if d.startswith("seed_"))
    all_seeds = {}
    for sd in seed_dirs:
        logs = glob.glob(os.path.join(RUN_DIR, sd, "optimization_log_run*.json"))
        if not logs:
            continue
        all_seeds[sd] = load_seed(logs[0])

    # ---- write CSV ----
    csv_path = os.path.join(OUT, "baseline_hp_trajectory.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["seed", "round", "outcome", "val_loss", "val_rmse", "val_mae"] + HP_KEYS
        f.write(",".join(cols) + "\n")
        for sd, rows in all_seeds.items():
            for row in rows:
                vl = row.get("val_loss")
                rmse = math.sqrt(vl) if vl is not None else ""
                vals = [sd, row["round"], row["outcome"],
                        "" if vl is None else f"{vl:.6f}",
                        "" if rmse == "" else f"{rmse:.6f}",
                        "" if row.get("val_mae") is None else f"{row['val_mae']:.6f}"]
                vals += [row.get(k, "") for k in HP_KEYS]
                f.write(",".join(str(v) for v in vals) + "\n")
    print("Wrote", csv_path)

    # ---- print a readable table for the primary seed ----
    primary = "seed_42" if "seed_42" in all_seeds else next(iter(all_seeds))
    print(f"\n=== {RUN_DIR} / {primary} : baseline -> final trajectory ===")
    rows = all_seeds[primary]
    hdr = f"{'rnd':>3} {'outcome':<10} {'val_loss':>9} {'RMSE':>7} {'lr':>8} {'wd':>8} {'drop':>5} {'bs':>4} {'hid':>4} {'lay':>4} {'pat':>4} {'opt':>6}"
    print(hdr)
    for r in rows:
        vl = r.get("val_loss")
        rmse = f"{math.sqrt(vl):.4f}" if vl is not None else "-"
        vl_s = f"{vl:.4f}" if vl is not None else "-"
        print(f"{r['round']:>3} {str(r['outcome']):<10} {vl_s:>9} {rmse:>7} "
              f"{r.get('learning_rate'):>8} {r.get('weight_decay'):>8} "
              f"{r.get('dropout'):>5} {r.get('batch_size'):>4} {r.get('lstm_hidden'):>4} "
              f"{r.get('lstm_layers'):>4} {r.get('patience'):>4} {str(r.get('optimizer_choice')):>6}")

    # ================= PLOT 1: RMSE trajectory =================
    fig, ax = plt.subplots(figsize=(8, 5))
    for sd, rows in all_seeds.items():
        xs = [r["round"] for r in rows if r.get("val_loss") is not None]
        ys = [math.sqrt(r["val_loss"]) for r in rows if r.get("val_loss") is not None]
        ax.plot(xs, ys, marker="o", label=sd)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.text(0.02, 0.97, "round 0 = baseline (initial setup)", transform=ax.transAxes,
            va="top", fontsize=9, color="gray")
    ax.set_xlabel("optimization round (0 = baseline)")
    ax.set_ylabel("validation RMSE  (sqrt of scaled val MSE)")
    ax.set_title(f"RMSE from initial setup to last round — {RUN_DIR}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUT, "rmse_trajectory.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print("Wrote", p1)

    # ================= PLOT 2: HP small-multiples (primary seed) =================
    rows = all_seeds[primary]
    numeric = ["learning_rate", "weight_decay", "dropout", "batch_size",
               "lstm_hidden", "lstm_layers", "patience"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.ravel()
    xs = [r["round"] for r in rows]
    for i, k in enumerate(numeric):
        ax = axes[i]
        ys = [r.get(k) for r in rows]
        ax.plot(xs, ys, marker="o", color="darkorange")
        ax.scatter([0], [ys[0]], color="steelblue", zorder=5, s=70, label="baseline")
        ax.set_title(k)
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
        if k in ("learning_rate", "weight_decay"):
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    # optimizer (categorical) + RMSE in remaining panels
    ax = axes[7]
    opts = [r.get("optimizer_choice") for r in rows]
    ax.step(xs, opts, where="post", marker="o", color="seagreen")
    ax.set_title("optimizer_choice")
    ax.set_xlabel("round")
    ax.grid(alpha=0.3)
    ax = axes[8]
    rys = [math.sqrt(r["val_loss"]) if r.get("val_loss") is not None else None for r in rows]
    ax.plot(xs, rys, marker="o", color="crimson")
    ax.scatter([0], [rys[0]], color="steelblue", zorder=5, s=70)
    ax.set_title("val RMSE")
    ax.set_xlabel("round")
    ax.grid(alpha=0.3)
    fig.suptitle(f"Hyperparameter trajectory from baseline (round 0) — {RUN_DIR} / {primary}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p2 = os.path.join(OUT, "hp_trajectory_primary_seed.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print("Wrote", p2)


if __name__ == "__main__":
    main()
