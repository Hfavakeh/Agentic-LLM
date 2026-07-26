"""Thesis-quality model-behaviour figures: predictions, error distribution, curves.

RUN THIS ON YOUR MACHINE, not in the sandbox -- it needs torch to load the saved
checkpoints and run inference on the test split.

It rebuilds the network from the setting recorded in `final_evaluation_run1.json`,
loads the matching `*_best_run1.pt` checkpoint, evaluates it on the held-out test
split, and writes vector PDFs styled to match the other thesis figures:

  fig_predictions.pdf       -- predicted vs true X and Y, with the identity line
  fig_error_distribution.pdf-- Euclidean error histogram + CDF with the p95 mark
  fig_training_curves.pdf   -- train/val MSE and MAE per epoch (see note below)

NOTE on training curves. `Trainer.history` is never written to disk -- it lives in
memory and is only consumed by the PNG written at run time. So curves cannot be
recovered from past runs. Either
  (a) apply the two-line patch in `--emit-history-patch` so future runs save it, or
  (b) point --history at a JSON containing the per-epoch arrays.

Examples
--------
    python scripts/build_model_figures.py \
        --run results/history-use/history-none-llama3/seed_42 --arm llm

    python scripts/build_model_figures.py --emit-history-patch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys as _sys
if str(REPO) not in _sys.path:
    _sys.path.insert(0, str(REPO))

INK, GREY, LIGHT = "#1A1A1A", "#6B7280", "#D1D5DB"
NAVY, TEAL, CORAL, AMBER = "#243B53", "#1F7A6F", "#D9655B", "#E0A340"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

HISTORY_PATCH = r'''
Add to pipeline/trainer.py, at the end of Trainer.train(), just before it returns:

    # persist the per-epoch curves so figures can be rebuilt without re-training
    try:
        import json as _json
        _p = self.config.output_dir / f"training_history_{tag}.json"
        _p.write_text(_json.dumps(self.history), encoding="utf-8")
    except Exception:
        pass          # never let plotting bookkeeping break a run

`tag` should identify the arm and run (e.g. f"{arm}_run{run_id}"), matching the
naming already used for "{arm}_history_run{n}.png". After this, every new run
leaves a training_history_*.json next to its checkpoints and
build_model_figures.py can draw the curves for any seed.
'''


def load_predictions(run: Path, arm: str):
    """Rebuild the model from its recorded setting and predict on the test split."""
    import torch
    from pipeline import Config
    from pipeline.data import DataProcessor
    from pipeline.model import LSTM_Localizer

    fe = json.loads((run / "final_evaluation_run1.json").read_text(encoding="utf-8"))
    setting = (fe.get("per_arm", {}).get(arm) or {}).get("setting")
    if setting is None:
        raise SystemExit(f"no recorded setting for arm '{arm}' in {run}")

    ckpt = next((run / n for n in (f"{arm}_best_run1.pt", "eval_best.pt",
                                   "setting_best.pt") if (run / n).exists()), None)
    if ckpt is None:
        raise SystemExit(f"no checkpoint found in {run}")

    cfg = Config()
    for k, v in setting.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if "window_size" in setting:
        cfg.window_size = setting["window_size"]

    proc = DataProcessor(cfg)
    data = proc.prepare()                       # honours the protocol split
    X_test, y_test = data["X_test"], data["y_test"]

    model = LSTM_Localizer(
        input_features=X_test.shape[-1], target_dim=y_test.shape[-1],
        lstm_hidden=setting.get("lstm_hidden", 128),
        lstm_layers=setting.get("lstm_layers", 1),
        dropout=setting.get("dropout", 0.1),
    )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
    model.eval()

    with torch.no_grad():
        preds = model(torch.as_tensor(X_test, dtype=torch.float32)).numpy()
    # back to metres
    scaler = data.get("scaler_y")
    if scaler is not None:
        preds = scaler.inverse_transform(preds)
        y_test = scaler.inverse_transform(y_test)
    return preds, y_test


def fig_predictions(preds, targets, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
    for ax, dim, lab, col in [(axes[0], 0, "X", NAVY), (axes[1], 1, "Y", TEAL)]:
        lo = min(targets[:, dim].min(), preds[:, dim].min())
        hi = max(targets[:, dim].max(), preds[:, dim].max())
        ax.scatter(targets[:, dim], preds[:, dim], s=5, alpha=0.35, color=col,
                   linewidths=0)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=INK)
        r = np.corrcoef(targets[:, dim], preds[:, dim])[0, 1] ** 2
        ax.set_xlabel(f"True ${lab.lower()}$ (m)")
        ax.set_ylabel(f"Predicted ${lab.lower()}$ (m)")
        ax.set_title(f"${lab.lower()}$   ($R^2={r:.3f}$)")
        ax.set_aspect("equal", adjustable="box")
    fig.savefig(out / "fig_predictions.pdf")
    plt.close(fig)


def fig_error_distribution(preds, targets, out: Path) -> None:
    err = np.linalg.norm(preds - targets, axis=1)
    p50, p95 = np.percentile(err, 50), np.percentile(err, 95)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.5))
    ax1.hist(err, bins=45, color=NAVY, alpha=0.75)
    ax1.axvline(p50, color=INK, lw=1.0, ls="--")
    ax1.text(p50, ax1.get_ylim()[1] * 0.94, f" median {p50:.2f} m",
             fontsize=7.2, color=INK)
    ax1.set_xlabel("Euclidean error (m)")
    ax1.set_ylabel("Samples")

    s = np.sort(err)
    ax2.plot(s, np.arange(1, len(s) + 1) / len(s), color=TEAL, lw=1.3)
    ax2.axhline(0.95, color=GREY, lw=0.8, ls=":")
    ax2.axvline(p95, color=CORAL, lw=1.0, ls="--")
    ax2.text(p95, 0.35, f" p95 = {p95:.2f} m", fontsize=7.2, color=CORAL)
    ax2.set_xlabel("Euclidean error (m)")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_ylim(0, 1.02)
    for ax in (ax1, ax2):
        ax.grid(True, color=LIGHT, lw=0.6)
        ax.set_axisbelow(True)
    fig.savefig(out / "fig_error_distribution.pdf")
    plt.close(fig)


def fig_training_curves(hist: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.4))
    for ax, key, title in [(ax1, "loss", "MSE"), (ax2, "mae", "MAE")]:
        tr, vl = hist.get(f"train_{key}"), hist.get(f"val_{key}")
        if not tr:
            continue
        ax.plot(tr, color=NAVY, lw=1.2, label="train")
        ax.plot(vl, color=CORAL, lw=1.2, label="validation")
        if vl:
            b = int(np.argmin(vl))
            ax.axvline(b, color=GREY, lw=0.8, ls=":")
            ax.text(b, max(vl) * 0.95, f" best epoch {b}", fontsize=7, color=GREY)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, color=LIGHT, lw=0.6)
        ax.set_axisbelow(True)
    ax1.legend(frameon=False)
    fig.savefig(out / "fig_training_curves.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="a seed_<N> run directory")
    ap.add_argument("--arm", default="llm",
                    help="which arm's selected model to plot (default: llm)")
    ap.add_argument("--history", help="JSON with per-epoch arrays, if you have one")
    ap.add_argument("--out", default=str(REPO / "thesis" / "figures"))
    ap.add_argument("--emit-history-patch", action="store_true",
                    help="print the trainer patch that persists per-epoch curves")
    args = ap.parse_args()

    if args.emit_history_patch:
        print(HISTORY_PATCH)
        return
    if not args.run:
        ap.error("--run is required (or use --emit-history-patch)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Path(args.run)

    preds, targets = load_predictions(run, args.arm)
    fig_predictions(preds, targets, out)
    fig_error_distribution(preds, targets, out)
    print(f"  wrote fig_predictions.pdf and fig_error_distribution.pdf "
          f"({len(preds)} test samples, arm={args.arm})")

    hp = Path(args.history) if args.history else next(
        iter(sorted(run.glob("training_history*.json"))), None)
    if hp and Path(hp).exists():
        fig_training_curves(json.loads(Path(hp).read_text(encoding="utf-8")), out)
        print("  wrote fig_training_curves.pdf")
    else:
        print("  skipped training curves: no per-epoch history saved for this run.\n"
              "  Run with --emit-history-patch to see how to persist it going forward.")


if __name__ == "__main__":
    main()
