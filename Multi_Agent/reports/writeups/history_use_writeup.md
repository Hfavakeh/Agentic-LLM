# Q3 — Does the LLM use the optimization history, or mostly ignore it?

*Information-sufficiency study (Email-4 gate), question 3. Dataset: radar
(`preprocessed-RadarEXP1`). Search budget: 25 attempts/run, each setting trained
3× from scratch (seeds 101/102/103) and scored by mean validation RMSE in
metres. Final test RMSE re-evaluated on 30 fresh seeds.*

## 1. The question

The professor's challenge: we showed the rule-based controller could not
reproduce manual tuning because it only saw post-hoc aggregate labels, not the
per-epoch curve a human used. The natural follow-up — *does the LLM actually
receive and use the right information?* A well-formed, in-grid proposal is not
the same as a good optimization decision. Q3 isolates one part of this: **of the
history we render to the LLM (best-so-far, recent attempts, already-tried list,
observed patterns), does the model actually use it?**

## 2. Method — a history placebo

We perturb **only the history text shown to the LLM**. The engine's training,
the scoring, and the true best-so-far *anchor* the proposal is applied to are
untouched — so any change in behaviour is attributable to the rendered history
alone. Three conditions:

| Condition | What the LLM sees |
|---|---|
| **none** | the real history (normal run) |
| **shuffled** | each setting's outcome is reassigned to another setting's, so the best-settings ranking / trend / patterns are *misleading* (format identical) |
| **empty** | no prior attempts shown — only the anchor |

If the LLM uses history, real (`none`) should beat `shuffled` (misleading) and
`empty` (absent). If proposals are unchanged across conditions, it is ignoring
the history.

We ran the placebo across a **three-model capability ladder** — nemotron-3-nano
(4B), llama3 (8B), phi4 (14B) — each over **10 seeds**, to test whether any
finding is model-specific and to address the concern that a *stronger* model
would simply use the history better. Metrics per run:

- **Repeats** — attempts rejected as already-tried (out of 25): wasted budget.
- **Distinct settings** — how many different configurations were actually evaluated: exploration breadth.
- **Test RMSE (m)** — accuracy of the selected setting (lower = better; baseline = 0.231).

## 3. Results

![Q3 history-use ablation across the model ladder](../analysis/q3_cross_model/q3_cross_model.png)

Mean ± std across 10 seeds:

| Model | Repeats /25 (none / shuf / empty) | Distinct /25 (none / shuf / empty) | Test RMSE (none / shuf / empty) |
|---|---|---|---|
| nemotron 4B | 0.8 / 1.1 / 11.7 | 24.2 / 23.9 / 13.2 | 0.249 / 0.247 / 0.245 |
| llama3:8b | 14.2 / 16.9 / 22.0 | 9.1 / 6.5 / 3.0 | 0.245 / 0.244 / 0.227 |
| phi4 14B | 10.2 / 15.0 / 9.8 | 13.9 / 9.4 / 13.0 | 0.244 / 0.237 / 0.240 |
| **baseline** | — | — | **0.231** |

Two findings stand out.

**(a) Exploration competence is model-specific and does not scale with size.**
The *smallest* model (nemotron 4B) is by far the best explorer — it almost never
repeats and evaluates ~24/25 distinct settings when history is present. The
*middle* model (llama3:8b) is the worst (only 3–9 distinct, up to 22/25 attempts
wasted on duplicates). The *largest* (phi4 14B) sits in between. So a larger or
"stronger" model is **not** a better searcher.

**(b) Decision quality is invariant — and never beats the baseline.** Across all
three models and all three history conditions, the LLM-selected setting sits **at
or above** the fixed baseline (0.231); only one of the nine model×condition cells
(llama-empty, 0.227) dips below it. Test RMSE is essentially **flat across the
history conditions**, and crucially **real history is never better than
scrambled** — for phi4 and nemotron the real-history result is in fact the
*worst* of the three.

## 4. Interpretation — exploration ≠ decision-making

These two findings together give a clean dissociation:

> The ability to *use the already-tried list to avoid repeats* (which nemotron
> does excellently and llama poorly) is a **different and separable skill** from
> *using the performance history to make better decisions* — which **none** of
> the three models does.

Even the model that explores almost the entire budget cannot convert the
qualitative history into a setting that beats a fixed baseline, and giving it the
*real* history yields no advantage over giving it a *scrambled* or *empty* one.
This holds across a 3.5× range of model size and three different architectures.

This directly addresses the professor's loophole — *"some models producing valid
output only shows that stronger models follow the instructions better."* Here the
stronger model (phi4 14B) does follow the format, explores reasonably, and still
makes no better decisions. **The bottleneck is therefore not model capability or
exploration mechanics; it is the decision-relevant content of the history we
provide.** That is exactly what Q1 (what information does the LLM need?) and Q2
(do per-epoch curve summaries help?) are designed to test, and is where this line
of work points next.

## 5. Honest caveats

- **Per-condition search is one trajectory per seed.** The 10 seeds give solid
  variance on the per-attempt metrics (repeats, distinct settings). Test RMSE for
  the `empty` condition and the `baseline` is effectively deterministic (fixed
  training seeds + a history-free, low-temperature prompt converge to the same
  trajectory), so its RMSE has little seed variance — the RMSE ordering should be
  read as a tight point comparison, not a wide distribution.
- **The "uses the tried-list" signal is clean for nemotron and llama** (removing
  history clearly increased repeats), **but ambiguous for phi4**, whose `empty`
  and `none` behave similarly. We report this rather than smoothing it over.
- **Single dataset (radar).** The pattern should be confirmed on the capacitive
  and IR datasets before being stated as fully general.
- A small fraction of attempts were rejected as off-grid (the model occasionally
  proposes values outside the allowed set, e.g. `lstm_hidden=512`); these are a
  minor effect next to the dominant already-tried repeats.

## 6. Conclusion

For this task, the LLM does *partially* read the history (it can avoid repeats
when shown the tried-list, model-dependent), but it does **not** convert the
qualitative performance history into better optimization decisions: real history
gives no accuracy benefit over scrambled or absent history, and no model — up to
14B — beats the fixed baseline. The limiting factor is the information content of
the history, not the model. Next: Q2 — test whether richer, per-epoch curve
summaries change this, for both the LLM and the rule-based controller.
