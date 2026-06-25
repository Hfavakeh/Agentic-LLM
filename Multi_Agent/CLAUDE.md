# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Thesis experiment: Study whether an LLM can improve the training of a small NN for indoor localization, compared to conventional optimization methods. Moreover, helping the LLM leverage broad knowledge of human behavior to improve the model. Each run trains a baseline, then lets a local Ollama LLM (or a deterministic rule-based controller, or random search) iteratively propose hyperparameter changes across N rounds, and finally evaluates on test. Three sensor datasets are supported: radar, capacitive, and IR. No `requirements.txt` is checked in — dependencies (`torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pydantic`, `autogen-core`, `autogen-ext[ollama]`) are managed in the user's local env.

## Commands

```bash
# Full multi-agent experiment (uses Config defaults)
python main.py

# Smoke test (1 run x 2 rounds x 5 epochs)
python main.py --smoke-test

# Common overrides
python main.py --model qwen3:8b --rounds 10 --runs 3 --epochs 50
python main.py --data preprocessed-CapEXP3(in).csv --output outputs-cap-run1
python main.py --seeds 42 123 456            # explicit multi-seed sweep
python main.py --single-seed --seeds 42      # one seed, no multi-seed runner

# Ablations / arms
python main.py --semantic-repair   # SEPARATE LLM+repair arm (main run is hard validation)
python main.py --no-arch-changes   # forbid lstm_hidden/lstm_layers proposals
python main.py --no-motion         # disable motion-aware MSE, diagnostics, prompt

# Final evaluation (protocol point 8)
python main.py --final-eval-seeds 30   # fresh seeds for the headline result (default 30)
python main.py --no-final-eval         # search + reports only, skip final eval

# Offline analysis utilities (each is standalone, --help for flags)
python analyze_logs.py --root . --out analysis
python statistical_analysis.py --dir .                  # builds llm_report.html
python build_transcripts.py --root outputs-XXXX         # per-seed transcripts
python motion_descriptors.py --csv preprocessed-RadarEXP1(in).csv --out analysis/motion

# Word/HTML report generation requires a Node install with `docx` at:
#   C:/Users/hfava/AppData/Roaming/npm/node_modules
node build_report_docx.js
node build_transcripts_docx.js
```

The LLM agent talks to a local Ollama server via `autogen_ext.models.ollama.OllamaChatCompletionClient`. Make sure `ollama serve` is running and the model tag passed via `--model` is pulled.

## Architecture

The core is four packages plus a thin `main.py` CLI; everything else at the repo root is offline reporting/analysis. `model_pipeline.py` and `Agent.py` are **back-compat shims** that re-export the old public names from `pipeline/` and `arms/` — import from the packages in new code, and keep the shims until nothing references the old names.

**`pipeline/`** — self-contained training stack (was `model_pipeline.py`).
- `config.py` — `Config` (dataclass) is the single source of truth for all run settings. In `__post_init__` it consults `DATASET_SPECS` and overrides `hz` and `window_size` from the per-dataset spec, so just changing `csv_path` automatically rewires the windowing. It also injects the baseline window into `HP_GRID["window_size"]` (the fair-comparison fix) — `HP_GRID` is a shared mutable dict, so every module must import the same object, never copy it.
- `datasets.py` — `DATASET_SPECS` encodes the **non-contiguous train/val/test indices** mandated by the thesis protocol (e.g. radar val sits *after* test in the file; cap val is at the start). `DataProcessor.temporal_split` (in `data.py`) uses these splits — falling back to a contiguous 80/10/10 only when the CSV basename is unknown. Do not "fix" this fallback to be the default; the splits must match the spec for every supported dataset.
- `search_space.py` — `HP_GRID` is the single source of truth for the search space — the protocol's exact discrete value lists for the 9 conventional HPs (lr, weight_decay, dropout, batch_size, lstm_hidden, lstm_layers, window_size, optimizer adam/adamw, patience 8/12/16). It is imported by the agent validator, `sample_random_hparams`, and the Optuna sampler so all arms (LLM, rule-based, random, Optuna) draw from identical sets. `HP_BOUNDS` is *derived* (min/max per numeric param) only for range-normalising helpers. Motion loss-shaping levers (`LOSS_SHAPING_KEYS`) are deferred from the search space (machinery intact for a later motion experiment).
- `trainer.py` — `Trainer` owns the train loop, early stopping, hyperparameter hot-swap, and `apply_hyperparameter_update(..., resets_model=...)`. Architectural changes (`lstm_hidden`, `lstm_layers`) or `window_size` changes force `resets_model=True`; weight_decay/lr/dropout/optimizer can be swapped on a warm model. `data.py` holds `DataProcessor`/`TimeSeriesDataset`, `model.py` the `LSTM_Localizer`, `evaluator.py` the test-set `Evaluator` + per-run plots.
- `logging_setup.py` — configures the shared `logger` on package import; it writes to `Agent_optimization_<timestamp>.log` *in the repo root*. The hundreds of pre-existing `.log` files at root are old run logs, not source — leave them alone unless asked to clean up.

**`arms/`** — the five comparison arms plus the shared machinery that drives them (was split across the old `agents/` + `search/`). All five methods of the bake-off live side by side here:
- `baseline.py` — `run_baseline(config, run_id)`, the fixed-reference arm: trains the `Config` defaults once (no per-round adaptation), evaluates the best checkpoint on test, returns the `results["baseline"]` slot. Not a search method, so it has no proposer.
- `random_search.py` (`sample_random_hparams`, `run_random_search`), `optuna_search.py` (`run_optuna_search`; TPE, seed 1000, n_startup 5). Self-contained search loops over `HP_GRID`.
- `llm.py` — `SingleAgentOptimizer.propose_setting`, Ollama-backed. Renders the qualitative history context (`prompts.format_protocol_payload`: ANCHOR/best-5/last-5/already-tried/observed-patterns, all as labels, no raw numbers), sends the `prompts.protocol_system_prompt` (discrete grid, no motion, no Pareto), parses the 5-line reply (`parsing._parse_protocol_proposal`), then **hard-validates** the proposed delta via `validation.validate_protocol_changes` (grid membership + arch-frozen + not-already-tried). On failure it retries ONCE with specific feedback, else marks the attempt `rejected` (no silent repair). Tracks `valid_first_try / valid_after_retry / rejected / repeats_proposed / invalid_values`.
- `rule_based.py` — `RuleBasedOptimizer.propose_setting`, deterministic controller using the **same soft diagnosis labels** as the prompt (`_diagnose_protocol` on the last attempt's qualitative behavior; label helpers live in `labels.py`) and stepping one grid value vs the anchor (`_act_protocol`). Always in-grid; the only difference vs the LLM is *how* the label is acted on. Per diagnosis it walks a frozen `_PROTOCOL_MOVES_BASE` priority list (`+ _PROTOCOL_ARCH_MOVES` when `allow_arch_changes`), returning the first single-HP move that actually changes the anchor (not a grid boundary) and isn't already-tried via `context["is_tried"]`; when every priority move is a no-op or duplicate it returns `valid=False, failure_reason="exhausted"` so the engine rejects the attempt instead of training a duplicate. `MotionAwareRuleBasedOptimizer` (C2) lives here too.
- `engine.py` — shared from-scratch core: `evaluate_setting` (one HP setting trained 3× from scratch on seeds 101/102/103 = `TRAIN_SEEDS`, scored by mean validation RMSE **in metres**, evaluated at the early-stopping best epoch — same RMSE functional/units as the headline test RMSE), `train_and_test_setting` (test eval only AFTER a method selects its best setting), `setting_signature` (already-tried dedupe), `make_loaders` / `build_dataset_and_loaders`.
- `driver.py` — `run_proposer_search`, the generic 25-attempt loop the LLM and rule-based arms share (the concrete proposer is passed in, anchored on best-so-far; a rejected proposal still consumes the attempt with no training).
- `parsing.py` / `validation.py` / `prompts.py` / `labels.py` — proposer support shared by the LLM + rule-based arms (`PROTOCOL_DIAGNOSES` lives in `labels.py`).
- `legacy.py` — the old warm-loop machinery (`_format_payload_as_text`, `_format_history`, `_validate_proposal`, `SemanticRepairRequired`, `OptimizerTools`; plus the deprecated `suggest_hyperparameters` / `_call_with_retry` / `_attempt` / `_build_system_prompt` methods still on the classes). DEPRECATED and unused by the protocol runner; slated for removal after a confirmed full run.

**`reporting/`** — decoupled from execution: `plots.py` (trajectory / dynamics plots), `reports.py` (per-setting CSV, budget table, final-eval markdown+JSON), `cross_run.py` (`compute_cross_run_metrics`).

**`experiments/`** — orchestration: `single_run.py` (`run_experiment`: baseline → LLM → random → rule-based → Optuna → reports → final eval), `final_eval.py` (`run_final_evaluation` re-evaluates each best setting on fresh seeds 201+ = `FINAL_EVAL_SEEDS_POOL`, disjoint from train/optuna seeds, and reports paired differences vs the LLM), `runner.py` (`run_full_experiment`, `run_one_seed`, `run_multi_seed_experiment`, `EXPERIMENT_SEEDS`). The LLM arm defaults to HARD validation (`semantic_repair=False`); `--semantic-repair` runs the separate repair-on arm.

**`main.py`** — thin CLI only: `parse_args`, `build_config`, and dispatch to the single- or multi-seed runner.

**`motion_descriptors.py`** — extracts interpretable motion features (speed in m/s, dwell/stop-go) from the trajectory targets. Feeds the LLM payload, per-bin error breakdowns, and the optional motion-weighted MSE. Sampling rate matters: bins computed in units-per-sample are *not* comparable across cap (3 Hz), radar (4 Hz), and IR (5 Hz) — always convert to m/s via the dataset's `hz` before reasoning about thresholds. Stays at the repo root (also a standalone CLI); `pareto.py` (deferred multi-objective scalarizer) likewise.

## Outputs and analysis layout

Each run writes into the `output_dir` configured by `Config` / `--output` (default looks like `outputs-1205-qwen3.5-4b-motion/`). The repo already contains ~100 historical `outputs-*` directories from prior model sweeps (one per Ollama model / date). `analysis/` aggregates across them — `analyze_logs.py` reads every `outputs-*/` it finds under `--root` and writes `cross_run.csv`, `all_runs.csv`, `all_rounds.csv`, and per-experiment subdirs. `statistical_analysis.py` produces `llm_report.html` from conversation logs.

## Conventions specific to this codebase

- Diagnosis labels are the email's SOFT set: `{healthy, possible_overfitting_tendency, possible_underfitting_tendency, plateau, unstable, inconclusive}` (`PROTOCOL_DIAGNOSES` in `arms/labels.py`). The prompt and `RuleBasedOptimizer.propose_setting` share them so the only difference between arms is how a label is acted on.
- The LLM prompt is rendered as **qualitative labels** (best/good/poor, low/med/high, small/med/large, behavior labels), never raw numbers — small local models parse it more reliably and the protocol mandates it. Raw numbers stay in the per-setting CSV + JSON logs only.
- Validation is HARD by default (`semantic_repair=False`, the protocol main run): an out-of-grid value, unknown key, arch change while frozen, or already-tried setting is rejected (one retry, then the attempt is wasted — no silent clamp/snap). `--semantic-repair` enables the separate repair-on arm.
- Scoring is mean validation RMSE over the 3 trainings; "best" is the lowest. The score is in METRES (sqrt of the position MSE on inverse-transformed preds/targets, `val_position_loss_m`) read at the early-stopping best epoch — the same RMSE functional and units as `Evaluator.compute_metrics`' test RMSE — so search selection and the headline final eval agree. (Earlier code scored sqrt of the scaled-space `val_position_loss` over the min of all epochs; that caused an objective mismatch and is superseded.) Pareto/multi-objective and motion loss-shaping levers are DEFERRED from the search space (machinery intact for a later experiment).
- Early stopping is ENABLED for every arm (protocol): `Trainer.train()` monitors the validation position MSE **in metres** (`val_position_loss_m` — the same metric the search score and headline test RMSE use, so the kept epoch, the score, and the test metric all agree; falls back to scaled `val_loss` only when the metres metric is unavailable that epoch) and stops after `patience` non-improving epochs, then restores the best-epoch weights. `patience` (8/12/16) is a searchable HP that overrides `Config.early_stopping_patience`. Max 100 epochs per training, 25 attempts per method. (Earlier code/notes saying early stopping was disabled and patience removed are superseded by the professor's protocol.)
- File and dir names that look like dates (`outputs-0413-qwen2.5-coder`) follow `MMDD-modeltag` from the run date, not ISO format.
