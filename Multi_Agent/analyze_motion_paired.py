"""Paired seed-level statistics for the motion loss-shaping experiment
(Email-7 point 2). All arms are final-eval'd on the SAME fixed seeds (201-230),
so the comparison is paired per seed.

For each final-eval seed it reports baseline / rule / LLM error, the paired
differences LLM-baseline and LLM-rule, the mean paired difference with a 95%
confidence interval, a paired t-test + Wilcoxon p-value, and the number of seeds
where the LLM wins / loses. The LLM per-seed value is the mean over the search
seeds' chosen lever vectors (the arm-level result); a single modal-vector view is
also printed as a robustness check.

Usage:
    python analyze_motion_paired.py --llm motion-qwen3 --refs outputs-motion-refs \
        --label qwen3 --out analysis/motion_qwen3_full
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False


def load_refs(refs_dir: Path):
    fs = sorted(glob.glob(str(refs_dir / "seed_*" / "motion_experiment_run1.json"))) \
        or [str(refs_dir / "motion_experiment_run1.json")]
    d = json.loads(Path(fs[0]).read_text(encoding="utf-8"))
    seeds = d["final_eval_seeds"]
    return (np.array(seeds),
            np.array(d["final_eval"]["baseline"]["per_seed"], dtype=float),
            np.array(d["final_eval"]["motion_rule"]["per_seed"], dtype=float))


def find_seed_runs(root: Path):
    """One motion_experiment_run1.json per search seed, searched recursively so a
    nested ``seed_N/seed_N/`` layout (a file-transfer artifact) is still found.
    When a seed has several candidates, keep the most complete (largest) one."""
    best_per_seed = {}
    for f in sorted(root.rglob("motion_experiment_run1.json")):
        seed = next((p for p in f.relative_to(root).parts if p.startswith("seed_")),
                    str(f.parent))
        cur = best_per_seed.get(seed)
        if cur is None or f.stat().st_size > cur.stat().st_size:
            best_per_seed[seed] = f
    return [best_per_seed[k] for k in sorted(best_per_seed)]


def load_llm(model_dir: Path, arm: str = "llm"):
    """Return (per_search_seed matrix [n_search, n_final], list of levers) for
    the named arm (``llm`` or ``random`` — both search the 6 knobs per seed)."""
    arrs, levers = [], []
    for f in find_seed_runs(model_dir):
        d = json.loads(f.read_text(encoding="utf-8"))
        arrs.append(d["final_eval"][arm]["per_seed"])
        best = d["arms"][arm].get("best_levers") or {}
        levers.append(tuple(sorted(best.items())))
    return np.array(arrs, dtype=float), levers


def paired(llm: np.ndarray, ref: np.ndarray):
    diff = llm - ref                       # negative => LLM lower (better)
    n = len(diff)
    md = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1) if HAVE_SCIPY else 2.045
    ci = (md - tcrit * se, md + tcrit * se)
    t_p = float(stats.ttest_rel(llm, ref).pvalue) if HAVE_SCIPY else None
    w_p = float(stats.wilcoxon(llm, ref).pvalue) if HAVE_SCIPY else None
    wins = int((llm < ref).sum())          # LLM error lower = win
    losses = int((llm > ref).sum())
    ties = int((llm == ref).sum())
    return dict(mean=md, ci=ci, t_p=t_p, w_p=w_p, wins=wins, losses=losses, ties=ties, n=n)


def fmt_p(p):
    return "n/a" if p is None else (f"{p:.4f}" if p >= 1e-4 else "<0.0001")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="LLM run dir (seed_*/...)")
    ap.add_argument("--refs", required=True, help="baseline+rule refs dir")
    ap.add_argument("--label", default="LLM")
    ap.add_argument("--arm", default="llm", choices=["llm", "random"],
                    help="which searched arm to paired-test (llm or random-over-knobs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    llm_dir, refs_dir = Path(args.llm), Path(args.refs)
    seeds, base, rule = load_refs(refs_dir)
    mat, levers = load_llm(llm_dir, arm=args.arm)
    llm = mat.mean(axis=0)                  # arm-level per-seed (mean over search seeds)

    st_b = paired(llm, base)
    st_r = paired(llm, rule)

    # Modal-vector robustness check: the single most-chosen lever vector.
    from collections import Counter
    modal = Counter(levers).most_common(1)[0][0]
    modal_rows = [i for i, lv in enumerate(levers) if lv == modal]
    modal_llm = mat[modal_rows].mean(axis=0)
    st_b_m = paired(modal_llm, base)
    st_r_m = paired(modal_llm, rule)

    L = []
    L.append(f"# Paired seed-level analysis — {args.label} (Email-7 point 2)\n")
    L.append(f"_{len(seeds)} paired final-eval seeds ({seeds.min()}-{seeds.max()}); "
             f"LLM error per seed = mean over {mat.shape[0]} search-seed vectors. "
             f"Lower error is better; a 'win' = LLM error below the comparator on that seed._\n")

    L.append("## Summary (paired differences, metres)\n")
    L.append("| Comparison | mean Δ | 95% CI | paired t p | Wilcoxon p | LLM wins | LLM losses |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for name, st in [("LLM − baseline", st_b), ("LLM − rule", st_r)]:
        L.append(f"| {name} | {st['mean']:+.4f} | [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}] | "
                 f"{fmt_p(st['t_p'])} | {fmt_p(st['w_p'])} | {st['wins']}/{st['n']} | {st['losses']}/{st['n']} |")
    L.append("")
    L.append(f"Arm means: baseline {base.mean():.4f}, rule {rule.mean():.4f}, "
             f"{args.label} {llm.mean():.4f}.\n")

    L.append("## Modal-vector robustness check\n")
    L.append(f"Most-chosen lever vector across search seeds ({len(modal_rows)}/{mat.shape[0]} seeds): "
             f"`{dict(modal)}`\n")
    L.append("| Comparison | mean Δ | 95% CI | paired t p | LLM wins |")
    L.append("|---|--:|--:|--:|--:|")
    for name, st in [("LLM − baseline", st_b_m), ("LLM − rule", st_r_m)]:
        L.append(f"| {name} | {st['mean']:+.4f} | [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}] | "
                 f"{fmt_p(st['t_p'])} | {st['wins']}/{st['n']} |")
    L.append("")

    L.append("## Per-seed table (metres)\n")
    L.append("| seed | baseline | rule | LLM | LLM−base | LLM−rule |")
    L.append("|--:|--:|--:|--:|--:|--:|")
    for i, s in enumerate(seeds):
        L.append(f"| {int(s)} | {base[i]:.4f} | {rule[i]:.4f} | {llm[i]:.4f} | "
                 f"{llm[i]-base[i]:+.4f} | {llm[i]-rule[i]:+.4f} |")
    L.append("")

    text = "\n".join(L)
    out_dir = Path(args.out) if args.out else Path("analysis") / f"motion_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_seed_analysis.md").write_text(text, encoding="utf-8")

    # Console summary
    print(f"=== {args.label}: paired seed-level (n={st_b['n']}) ===")
    print(f"arm means: baseline {base.mean():.4f} | rule {rule.mean():.4f} | {args.label} {llm.mean():.4f}")
    for name, st in [("LLM-baseline", st_b), ("LLM-rule", st_r)]:
        print(f"{name:14s} meanΔ {st['mean']:+.4f}  CI[{st['ci'][0]:+.4f},{st['ci'][1]:+.4f}]  "
              f"t_p {fmt_p(st['t_p'])}  wilcoxon {fmt_p(st['w_p'])}  wins {st['wins']}/{st['n']}  losses {st['losses']}/{st['n']}")
    print(f"wrote {out_dir/'paired_seed_analysis.md'}")


if __name__ == "__main__":
    main()
