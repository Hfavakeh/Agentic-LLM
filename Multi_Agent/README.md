# Thesis repository — LLM-guided hyperparameter optimization for indoor localization

Folder map for the thesis write-up phase. Runnable code and the three datasets
stay at the repo root; everything else is grouped by what it is.

## Root — code and data (do not move)

| Path | What |
|---|---|
| `main.py` | CLI entry point (see CLAUDE.md for flags) |
| `pipeline/` | training stack: `Config`, data, model, trainer, evaluator |
| `arms/` | the five comparison arms + shared search engine |
| `experiments/` | orchestration: single run, multi-seed runner, final eval |
| `reporting/` | plots, per-setting CSVs, final-eval reports |
| `model_pipeline.py`, `Agent.py` | back-compat shims re-exporting old names |
| `motion_descriptors.py`, `pareto.py` | standalone modules, also CLIs |
| `preprocessed-*.csv` | the three sensor datasets (radar / capacitive / IR) |

The CSVs are referenced by **bare filename** in `pipeline/config.py` and
`pipeline/datasets.py`. Moving them out of the root breaks every default run.

## `scripts/` — offline analysis and report builders

All `analyze_*.py`, `statistical_analysis.py`, `build_transcripts.py`,
`failure_taxonomy.py`, plus the `run_*.py` / `augment_optuna.py` helpers.
Run them from the repo root:

```bash
python scripts/analyze_logs.py --root results --out analysis
python scripts/statistical_analysis.py --dir results/archive/model-sweep
python scripts/analyze_history_ablation.py --root results/history-use --tag llama3
python scripts/analyze_curve_summaries.py --out analysis/curve_summaries
python scripts/analyze_exploration.py --out analysis/exploration
```

`analyze_history_ablation.py`, `analyze_curve_summaries.py` and
`analyze_exploration.py` hardcode which run dirs form each condition, in
`PAIRS` / `MODEL_FOLDERS` / `EXPLORE_FOLDERS` near the top of each file, or via
the `history-{cond}-{tag}` pattern. **Renaming a run dir means updating those
maps** — otherwise the script finds no seeds and reports an empty condition
rather than failing. All three take `--root` (default: the repo's `results/`).

Scripts that import `pipeline` / `arms` / `experiments` carry a `_REPO_ROOT`
`sys.path` bootstrap at the top, so they work from any working directory.

`scripts/report_builders/` holds the Node `build_*.js` document generators
(they need `docx` installed at `C:/Users/hfava/AppData/Roaming/npm/node_modules`).

## `results/` — experiment output directories

Group folders are named for the *question they answer*, not the email thread
they came from. Run dirs inside them drop the redundant `outputs-` prefix.

| Folder | Count | The question it answers |
|---|---|---|
| `motion/` | 27 | Does motion-aware loss / motion descriptors help? |
| `history-use/` | 15 | Does the LLM use the optimization history, or ignore it? (`none` / `empty` / `shuffled`) |
| `curve-summaries/` | 3 | Do per-epoch curve summaries improve the LLM or the rule-based arm? |
| `prompt-ablation/` | 5 | Is poor exploration the prompt, the model, or the search space? |
| `repair-ablation/` | 8 | Does semantic repair change the outcome vs hard validation? |
| `motion-knowledge/` | 6 | Can the LLM use human-motion knowledge? (the thesis core) |
| `cloud-models/` | 8 | Does a much larger model change the conclusion? |
| `baseline/` | 2 | fixed-reference runs and `outputs/default-run` |
| `archive/model-sweep/` | 67 | dated `MMDD-model` sweeps (superseded, kept) |
| `archive/misc/` | 6 | scratch dirs, smoke tests, tool-call experiments |

Former names: `q2-curves` → `curve-summaries`, `q3-history` → `history-use`,
`point3` → `motion-knowledge`. Q4 has no run dir of its own — it is computed
from the `history-none-*` runs plus `prompt-ablation/`.

### Two run layouts — read this before deleting anything

Most runs use `seed_<N>/` subdirs. A handful of **single-seed runs write their
logs flat** in the run dir instead, with no `seed_*` folder at all. These hold
complete, valid results (`protocol_log_run1.json`, `final_evaluation_run1.json`),
but a `seed_*` listing makes them look empty — which is exactly how they nearly
got discarded during the reorganization.

By convention they now carry a **`-pilot` suffix** so the layout is visible in
the name:

| Dir | Group |
|---|---|
| `history-{empty,none,shuffled}-llama3-pilot` | `history-use/` |
| `point3-minimax-m2.5-cloud-pilot` | `motion-knowledge/` |
| `2205-{llama3-8b,qwen3-8b}-pilot` | `archive/model-sweep/` |
| `{multi,new}-radar-tools-pilot` | `archive/misc/` |

`1505-*-motion-smoke` and `optuna-sweep-smoke` are also flat; `-smoke` already
marks them as throwaway. `baseline/baseline-rerun/` is flat by nature — it is a
baseline, not a search.

`analyze_history_ablation.py` handles both layouts. `analyze_logs.py` discovers
only the `seed_<N>/` form, so `-pilot` runs are **absent from its cross-run
tables** — that is why `history-use` reports 12 experiments, not 15.

Nothing was deleted. `archive/` means "superseded, still readable";
`_attic/aborted-runs/` holds three runs that died before writing any log.

**Discovery is recursive.** `analyze_logs.py` treats any directory containing a
`seed_<N>/` subdir with an optimisation log as an experiment, whatever it is
named, and searches `--max-depth` levels (default 3) below `--root`:

```bash
python scripts/analyze_logs.py --root results --out analysis          # all 113 experiments
python scripts/analyze_logs.py --root results/motion --out analysis/motion
```

A full `--root results` pass covers 113 experiments / 171 seeds / 16,190 rounds
and takes about a minute. Descriptively named dirs (`motion-qwen3/`,
`q3-empty-llama/`, `curve-phi/`) are now included — the old name-based
`outputs-<date>-<model>` filter silently skipped all of them.

## `reports/` — written deliverables

Word/PDF reports at the top level, `slides/` for the `.pptx` decks,
`writeups/` for the markdown sources and per-seed transcripts.
These are the drafts the thesis chapters will be assembled from.

## `logs/` — 326 archived run logs (~1.2 GB)

`Agent_optimization_<timestamp>.log`, grouped into `YYYY-MM` folders spanning
2026-02 through 2026-07:

| Month | Files | Size |
|---|---|---|
| 2026-02 | 39 | 8.2 MB |
| 2026-03 | 16 | 19 MB |
| 2026-04 | 83 | 59 MB |
| 2026-05 | 89 | 275 MB |
| 2026-06 | 38 | 604 MB |
| 2026-07 | 61 | 253 MB |

New runs write `Agent_optimization_<timestamp>.log` to the repo root
(`pipeline/logging_setup.py`); move them into the matching `YYYY-MM` folder
periodically.

Housekeeping already applied: 44 zero-byte logs (runs that died before writing
anything) were deleted, and byte-identical re-copies were set aside in
`_attic/duplicate-logs/` rather than deleted. Both are recorded in
`_attic/MOVES.csv`.

## `analysis/` — cross-run aggregates

Output of `analyze_logs.py`: `cross_run.csv`, `all_runs.csv`, `all_rounds.csv`,
and per-experiment subdirectories.

## `_attic/`

`MOVES.csv` records every file relocation from the reorganization
(`src,dest`, one per line) so any move can be reversed. Also holds loose
non-project files (VPN shortcut, console dumps, a `.rar` backup).
