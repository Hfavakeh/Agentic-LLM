# Preview report — from LLM hyperparameter tuning to motion-aware guidance

*Prepared for Prof. Mihai Lazarescu (reply to Email-6). Dataset: radar
(`preprocessed-RadarEXP1`). Two parts, as requested: (A) a short conclusion from
the stronger-model diagnostic, and (B) the first motion-aware guidance
experiment, structured around hypotheses.*

---

## Part A — Stronger-model diagnostic: conclusion

**Setup (unchanged).** The protocol was frozen and only the LLM was swapped: same
dataset, same 9-hyperparameter discrete grid, same 25 attempts, same 3 trainings
per setting (seeds 101/102/103), same fresh-seed final evaluation, same five
comparison arms (baseline, LLM, random, Optuna, curve-aware rule-based). We ran
two stronger models — **Gemma-4** and **Qwen3-14B** — and, on Qwen3-14B, the SoTA
optimizer-prompting recipe **OPRO**.

**Headline result — final test RMSE (metres, lower is better).** Reference arms
are model-invariant: baseline **0.2308**, rule-based **0.2371**, random ≈ 0.242,
Optuna 0.2614.

| Cell | n | baseline | LLM | rule-based | LLM beats rule |
|---|--:|--:|--:|--:|:--:|
| Gemma-4 — real history | 9 | 0.2308 | 0.2437 | 0.2371 | 0 / 9 |
| Gemma-4 — shuffled history | 8 | 0.2308 | 0.2465 | 0.2371 | 0 / 8 |
| Gemma-4 — empty history | 8 | 0.2308 | 0.2486 | 0.2371 | 0 / 8 |
| Gemma-4 — explore prompt | 10 | 0.2308 | 0.2504 | 0.2371 | 0 / 10 |
| Qwen3-14B — **OPRO** prompt | 8 | 0.2308 | 0.2451 | 0.2371 | 0 / 8 |
| Qwen3-14B — explore prompt | 9 | 0.2308 | 0.2500 | 0.2371 | 0 / 9 |

**Answers to the six diagnostic questions:**

1. **What improved with Gemma-4 and Qwen3-14B?** Proposal *quality*. Qwen3-14B
   proposes ~25/25 distinct settings, essentially **zero repeats, zero invalid
   outputs, no timeouts**, and reaches **0.85–0.95 grid coverage** — matching
   random/Optuna and far above the small models (≤ 0.61). The small-model hallmark
   (constant repetition) is gone. OPRO gave the single cleanest run of all.

2. **What did not improve?** The thing that matters — the **final model**. No LLM
   cell beats the rule-based controller (**0 of 52 seed-cells**) and none beats the
   fixed baseline. The best LLM result anywhere (Gemma-4 real history 0.2437;
   Qwen3-14B OPRO 0.2451) still sits above both references.

3. **Did the stronger models use history better?** **Yes**, shown cleanly on
   Gemma-4: degrading the rendered history hurts monotonically (RMSE 0.2437 real →
   0.2465 shuffled → 0.2486 empty) and repeats explode (3 → 7 → **57**). Unlike the
   small models, it genuinely reads and depends on the history.

4. **Did they explore better?** **Yes** — Qwen3-14B reaches near-random grid
   coverage. Exploration breadth is no longer a limitation at this scale.

5. **Did they select configurations that improved validation but failed on test?**
   **Exactly — this is the crux.** On the metric the search optimizes, the LLM and
   the controller are near-tied (**val RMSE ≈ 0.193 vs 0.1927**) and both crush the
   baseline (≈ 0.29). But those low-validation settings **overfit the small
   validation set**: the same settings land at 0.244–0.250 on held-out test, *worse*
   than the baseline's 0.231. The LLM's proposals are not poor on their own metric;
   the whole search overfits validation, and the LLM overfits it slightly more.

6. **Why does the curve-aware rule-based controller stay competitive/better?** It
   uses the same signals but converts them into a **stable, low-variance
   selection**. The LLM chases the validation minimum more aggressively and pays
   for it on test. Capability, prompt style, exploration breadth and history use are
   now all ruled out as the binding constraint — what remains is *selection under a
   noisy validation signal*, where a conservative deterministic rule wins.

**Conclusion (not "the LLM failed").** In generic hyperparameter tuning on this
small, well-behaved search space, a stronger LLM **improves proposal validity,
diversity and history use — but this does not translate into better
generalization.** Better proposals did not produce a better optimum. This closes
the "it's just a weak model" loophole and satisfies the stop condition for generic
LLM HP tuning, motivating the pivot to the thesis core in Part B.

*(Full numbers, behaviour tables, best-so-far curve and caveats:
`docs/email5_stronger_models_report.md`.)*

---

## Part B — Motion-aware guidance: first experiment

The question that defines the thesis: **can an LLM use human-motion knowledge to
improve a small localization network?** Instead of tuning generic hyperparameters,
we now freeze the network and let the proposer **reshape the training objective
(the loss)** from interpretable summaries of *how the tracked person moves* and
*where the model's error concentrates by motion regime*.

**Headline result (radar, paired over 30 seeds, with controls).** Searching the six
loss knobs beats plain MSE (0.2319): every arm that touches them lands in a tight
**0.2285–0.2301** band. But the controls show that **motion knowledge is not what
does the work**. Undirected **random** search over the same knobs — with *zero*
motion knowledge — reaches **0.2293**, statistically indistinguishable from the
motion-aware LLM (0.2285; Δ = −0.0008, p = 0.23), and stripping the motion summary
from the LLM changes nothing (0.2294; p = 0.32). The LLM also does not beat the
deterministic motion rule (0.2301; p = 0.37). A behavioural analysis explains it: the
LLM emits a near-fixed recipe under a near-constant diagnosis label. **Net: the gain
over plain MSE comes from the extra loss flexibility, not from the LLM's motion
knowledge or reasoning.**

### B.1 Experimental design

The 9 conventional hyperparameters are **frozen at the baseline setting**. The only
thing that varies is a six-lever **loss-shaping vector**:

| Lever | Meaning | Human-motion rationale |
|---|---|---|
| `v_max` | plausible top walking speed (m/s); faster predicted steps penalised | set just above the observed p95 speed |
| `lambda_vel` | strength of the speed-plausibility penalty (0 = off) | raise when predictions look noisy / motion is smooth |
| `lambda_smooth` | penalty on implausible acceleration / jerk (0 = off) | raise when the trajectory is smooth with frequent dwells |
| `bin_weight_slow / medium / fast` | per-speed-regime error weights (1.0 = neutral) | up-weight the regime the model fits worst |

Neutral levers (`lambda_vel=0, lambda_smooth=0`, all weights 1.0) = plain MSE.
Same protocol structure as the HP bake-off: each lever vector is trained 3× from
scratch, scored by mean validation RMSE in metres; the winner is evaluated on
fresh seeds. **Four arms**, mapping directly onto your request:

- **baseline** — plain MSE (the floor).
- **C2 — motion heuristic (the "simple rule")** — a fixed, deterministic
  motion-to-lever mapping (v_max ≈ 1.1× p95 speed, gentle penalties, fast-regime
  up-weight).
- **C3 — LLM motion** — the LLM reads the motion summaries + per-regime error and
  proposes lever vectors.
- **random** — 25 random lever vectors (undirected reference).

The central comparison is **C3 vs C2**: does the LLM's motion *interpretation* beat
a fixed motion *rule*?

### B.2 Evidence — motion profile of the tracked person (radar, real)

Computed from the target trajectory only (no training), converted to physical
units at 4 Hz:

| Feature | Value | Reading |
|---|---|---|
| speed mean / p95 / max | **0.31 / 0.59 / 3.34 m/s** | slow walker with occasional fast bursts |
| acceleration mean / p95 | 0.58 / 1.86 m/s² | moderate start-stop dynamics |
| turning mean / p95 | 26° / 107° per step | frequent direction changes |
| sharp-turn share (\|turn\| > 45°) | **19.1 %** | ~1 in 5 moving steps is a sharp turn |
| stop share / dwell episodes | 2.6 % / 5.8 per min | rarely fully stationary; brief pauses |

**Motion reading:** the person mostly walks slowly and turns a lot, with short
fast bursts. Prior HP-tuning results already showed the model's error is **worst in
the fast regime** (per-regime error spread ≈ 1.1×, worst = fast). This is the
qualitative summary the LLM is given, e.g. *"the person walks slowly with frequent
turns and occasional fast bursts; the model's error concentrates in the fast
regime."*

### B.3 Hypothesis-structured worked example

> **Hypothesis.** The model fails mainly during fast movement (and, relatedly,
> turning). Up-weighting the fast regime and penalising implausibly fast/jerky
> predicted steps should reduce fast-regime error without harming the rest.

**Evidence — per-regime error table (baseline).** Mean Euclidean position error by
target-speed tercile, at the baseline model's best epoch:

| Regime | Baseline error (m) | |
|---|--:|--|
| slow | 0.2650 | |
| medium | 0.2765 | |
| **fast** | **0.2781** | ← worst regime, as hypothesised |
| spread (worst / best) | 1.049 | |

**LLM interpretation (C3).** Shown the profile + this table, qwen3 consistently
diagnosed *"possible underfitting tendency"* (334 of 339 accepted proposals) and
reasoned in motion terms — setting `v_max` just above the p95 speed and up-weighting
the harder regimes. Verbatim: *"raising v_max to match p95 (0.6)… aligns with the
motion's high jerk and outlier speeds"* and *"medium-speed regime has highest error
(0.2737), so increasing its weight while lowering lambda_smooth to accommodate
jerky motion."*

**Proposed change — C3 (LLM) vs C2 (rule):**

| Lever | C2 rule (deterministic) | C3 LLM (qwen3, modal over seeds) |
|---|--:|--:|
| v_max | 1.0 | 1.0 |
| lambda_vel | 0.1 | **0.2** |
| lambda_smooth | 0.1 | 0.1 |
| bin_weight_slow | 1.0 | 1.0 |
| bin_weight_medium | 1.0 | **3.0** |
| bin_weight_fast | **1.5** | **2.0** |

*(Both key off the same p95 speed for `v_max`=1.0. The rule applies one gentle
fast-regime up-weight (×1.5); the LLM up-weights **both** medium (×3) and fast (×2)
and doubles the velocity-plausibility penalty — a more aggressive, motion-reasoned
loss.)*

**Result** (all arms final-eval'd on the same 30 seeds 201–230):

| Arm | test RMSE (m) | Δ vs baseline | fast-regime err |
|---|--:|--:|--:|
| baseline (plain MSE) | 0.2319 ± 0.0085 | — | 0.2781 |
| C2 motion rule | 0.2301 ± 0.0095 | −0.0018 | 0.2635 |
| **C3 LLM motion** | **0.2285** | **−0.0033** | 0.2668 |

*(Baseline/C2 are single deterministic vectors, ± over the 30 final-eval seeds. C3
is the mean over 10 LLM search seeds, each final-eval'd on the same 30 seeds. The
arm means order C3 < C2 < baseline, but a proper per-seed paired test — B.4 — shows
only the C3-vs-baseline gap is significant, not C3-vs-rule.)*

### B.4 Paired seed-level significance (qwen3)

All arms are final-eval'd on the same 30 seeds (201–230), so the comparison is
paired per seed. LLM error per seed = mean over the 10 search-seed lever vectors;
a win = LLM error below the comparator on that seed.

| Comparison | mean Δ (m) | 95% CI | paired t p | Wilcoxon p | LLM wins |
|---|--:|--:|--:|--:|--:|
| **LLM − baseline** | **−0.0033** | [−0.0059, −0.0008] | **0.012** | 0.011 | 23 / 30 |
| **LLM − rule** | −0.0015 | [−0.0050, +0.0019] | 0.37 | 0.44 | 17 / 30 |

**Read:** the LLM **significantly beats the plain-MSE baseline** (CI excludes 0,
23/30 wins) but **does not beat the C2 motion rule** (CI spans 0, p = 0.37, 17/30
wins — a coin flip). gemma4 gives the same pattern (vs baseline p = 0.033, 22/30;
vs rule p = 0.57, 16/30). Full per-seed table:
`analysis/motion_qwen3_full/paired_seed_analysis.md`.

### B.5 Depth of LLM reasoning (protocol logs)

Clean, in-grid, non-repeating proposals show the LLM *follows the protocol*, not
that it reasons well. Across the accepted proposals (222 qwen3 / 176 gemma4):

- **Diagnosis is a near-constant generic label** — qwen3 "possible underfitting
  tendency" on **90%**, gemma4 "plateau" on **74%** (confirms the professor's
  suspicion).
- **The diagnosis does not drive the action** — whatever the diagnosis, both models
  emit the same recipe: add a velocity-plausibility penalty (220/222 qwen3), add a
  smoothness penalty, up-weight the fast/medium regime.
- **Reason matches the change** only 68% (qwen3) / 51% (gemma4) of the time.
- **Only 18% (qwen3) / 15% (gemma4) of proposals improve the running best** — the
  search works by keeping the best of many near-identical tries, not by reasoning.

So the LLM applies a sensible but near-fixed motion prior with a constant label —
essentially what C2 encodes, which explains B.4: it does not beat the rule because
it is, in effect, reproducing it.

### B.6 Controls — is it motion knowledge, or just the six extra knobs?

The six-knob loss is strictly more flexible than plain MSE, so some of the gain could
come from the extra tunable parameters rather than motion knowledge. Two controls
isolate this, both over the **same six knobs** and the same protocol: (a) undirected
**random** search (zero motion knowledge), and (b) the LLM given the **per-regime
error only**, with the motion-summary block removed.

| Arm | test RMSE (m) | motion knowledge | vs baseline p |
|---|--:|:--:|--:|
| baseline (plain MSE) | 0.2319 | — | — |
| C2 motion rule | 0.2301 | yes (fixed) | — |
| **random over 6 knobs** | **0.2293** | **NONE** | 0.049 |
| **qwen3 — per-regime error only** | **0.2294** | no summary | 0.052 |
| gemma4 — full motion | 0.2291 | yes | 0.033 |
| qwen3 — full motion | 0.2285 | yes | 0.012 |

Direct paired contrasts (30 seeds):

- **qwen3-full vs random:** Δ = −0.0008, 95% CI [−0.0021, +0.0005], p = 0.228, 19/30
  wins — **not significant**. An undirected search knowing nothing about motion
  matches the motion-aware LLM.
- **qwen3-full vs no-profile:** Δ = −0.0009, CI [−0.0027, +0.0009], p = 0.322 —
  **not significant**. Removing the motion interpretation changes nothing; the
  per-regime error signal alone is enough.
- **random vs baseline:** Δ = −0.0026, p = 0.049 — random alone already beats plain
  MSE (borderline).
- **random vs rule:** p = 0.659 — not significant.

**Conclusion: the gain over plain MSE is attributable to the six extra loss knobs,
not to motion knowledge or LLM reasoning.**

### B.7 Interpretation — did the LLM add anything beyond the rule?

- **Over plain MSE: yes, but not because of motion.** Every arm that tunes the six
  knobs beats MSE by ~2–3 mm — including random search with no motion knowledge. The
  flexibility, not the knowledge, is doing the work.
- **Over the motion rule: no.** The LLM's edge over C2 is within seed noise (p=0.37).
- **Over random search: no.** The LLM is statistically indistinguishable from
  undirected random search over the same knobs (p = 0.23), and removing its motion
  summary entirely costs nothing (p = 0.32).
- **Why: the reasoning is shallow.** A near-constant diagnosis label, an action that
  does not depend on the diagnosis, a reason matching the change only ~half to
  two-thirds of the time, and only 18% of proposals improving anything (B.5).
- **Net.** On this dataset, as operationalised through these six loss knobs, the
  LLM's human-motion knowledge does not measurably improve the localization network
  beyond a one-line rule or random search. This is a negative result of the same kind
  as Part A — and, like Part A, it localises the failure to the *operationalisation*,
  not to the model.
- **Honest caveats.** random-vs-baseline is only borderline (p = 0.049);
  qwen3-vs-random is directionally in the LLM's favour but underpowered (p = 0.23,
  19/30) — more search seeds could resolve it, though the effect would be small
  either way. And this rests on a **single trajectory** (see B.10).

### B.8 Cross-model check (gemma4)

To guard against reporting only the better-looking model, a second strong model
(gemma4) was run under the identical protocol and is reported in full:

| Arm | test RMSE (m) | chosen loss (modal) |
|---|--:|---|
| baseline (plain MSE) | 0.2319 | — |
| C2 motion rule | 0.2301 | fast ×1.5 |
| C3 LLM — **qwen3** | **0.2285** | medium ×3 + fast ×2, λ_vel 0.2 |
| C3 LLM — **gemma4** | **0.2291** | fast ×3, λ_vel 0.2, λ_smooth 0.2 |

**Both models tell the same story**: each significantly beats the plain-MSE baseline
but neither beats the C2 rule (gemma4 vs baseline p = 0.033, 22/30; vs rule p = 0.57,
16/30). They reached it via *different* recipes (qwen3 spread weight across medium +
fast; gemma4 leaned on fast ×3) yet landed within ~0.6 mm of each other. gemma4's
decisions were clean (1 repeat in 250) but it is a **slow** model — 73 of its 74
rejections were 300 s serving timeouts. So the conclusion does not depend on which
model we picked.

### B.9 Validity checks (leakage & loss)

- **No test leakage.** Motion profile (speed/accel/turning/stop-go/roughness) from
  **train** targets; loss speed-regime edges from **train** terciles (the controller
  sets only the per-bin weights, never the edges); per-regime error shown to the LLM
  from **val**. Selection is on val RMSE; the **test** set is touched only at final
  evaluation.
- **Loss is temporally sound.** `total = base + λ_vel·mean(relu(pred_speed − v_max)²)
  + λ_smooth·mean(‖accel‖²)`, with `pred_speed = ‖pred − prev_y‖·scale·hz` and
  `accel = (pred − prev_y) − (prev_y − prev_prev_y)`. Consecutive positions are
  carried as **per-sample** fields (t−1, t−2), so shuffling training batches does
  **not** break the velocity/smoothness terms.

### B.10 Status & next steps

**Complete on radar:** qwen3 + gemma4 full motion, the C2 rule, the plain-MSE
baseline, and both point-5 controls (random over the six knobs; LLM with per-regime
error only). The main remaining threat to the conclusion is that it rests on a
**single trajectory / motion profile**. Next: (1) at least one more subject or motion
profile (capacitive 3 Hz or IR 5 Hz, or a distinct radar segment — ideally
contrasting profiles such as mostly-stationary vs fast-bursts vs many-turns), to test
whether the negative result holds where the motion prior should matter more;
(2) symmetric Part A cells for Qwen3 (real / shuffled / empty history) to remove the
Gemma-4-vs-Qwen3 asymmetry. No further generic-HP experiments.

---

## Appendix — how to reproduce / extend

The motion profile is consumed **only by the LLM arm**. Per model, run the LLM arm
on the server (GPU):

```bash
docker compose run --rm app --motion-experiment \
  --motion-arms llm --model <MODEL_TAG> \
  --seeds 17 42 73 128 256 314 451 512 666 777 \
  --rounds 25 --final-eval-seeds 30 \
  --output outputs/motion-<MODEL_TAG>
```

The **baseline** and **C2 rule** are deterministic and model-independent, so they
are computed **once** (no LLM needed) and reused across models:

```bash
python main.py --motion-experiment --motion-arms baseline motion_rule \
  --seeds 42 --final-eval-seeds 30 --output outputs-motion-refs
```

Then aggregate the LLM run **together with** the shared refs into one comparison
table + figures (the aggregator merges multiple `--root` dirs by arm):

```bash
python analyze_motion_experiment.py \
  --root outputs/motion-<MODEL_TAG> outputs-motion-refs \
  --out analysis/motion_<MODEL_TAG>
```

Notes:

- `baseline` / C2 need only one search seed (deterministic); the 30 final-eval seeds
  give the number. Re-run only if the dataset or split changes.
- Each `seed_*/motion_protocol_log_run1.json` holds the LLM's rendered payload, raw
  reply and parsed levers — the source for B.3's "LLM interpretation" line.
- The `qwen3` numbers in this report come from `outputs/motion-qwen3` +
  `outputs-motion-refs`, aggregated to `analysis/motion_qwen3_full`.
