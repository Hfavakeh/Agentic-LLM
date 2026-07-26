// Email-6 preview report: (A) stronger-model conclusion + (B) first motion-aware
// experiment, hypothesis-structured. Content mirrors docs/email6_preview_report.md.
// Reuses the house docx style from build_email5_report_docx.js.
const fs = require('fs');
const path = require('path');

const NODE_MODULES = 'C:/Users/hfava/AppData/Roaming/npm/node_modules';
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
} = require(path.join(NODE_MODULES, 'docx'));

const FONT = 'Calibri', MONO = 'Consolas';
const HEAD_FILL = '1F3864', ALT_FILL = 'F2F2F2', TODO = 'B25000';
const border = { style: BorderStyle.SINGLE, size: 8, color: '808080' };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function P(text, opts = {}) {
  return new Paragraph({ spacing: { before: 60, after: 60 }, ...opts,
    children: [new TextRun({ text, font: FONT, size: 22, ...(opts.run || {}) })] });
}
function PR(runs, opts = {}) {
  return new Paragraph({ spacing: { before: 60, after: 60 }, ...opts,
    children: runs.map(r => new TextRun({ text: r.text, font: r.mono ? MONO : FONT,
      size: r.mono ? 20 : 22, bold: !!r.bold, italics: !!r.italic, color: r.color || '000000' })) });
}
function H1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 100 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: true, color: HEAD_FILL })] });
}
function H2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 60 },
    children: [new TextRun({ text, font: FONT, size: 24, bold: true, color: HEAD_FILL })] });
}
function cap(text) {
  return new Paragraph({ spacing: { before: 120, after: 40 },
    children: [new TextRun({ text, font: FONT, size: 20, italics: true, color: '595959' })] });
}
function bulletR(runs) {
  return new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { before: 40, after: 40 },
    children: runs.map(r => new TextRun({ text: r.text, font: r.mono ? MONO : FONT,
      size: r.mono ? 20 : 22, bold: !!r.bold, italics: !!r.italic, color: r.color || '000000' })) });
}
function numItem(runs) {
  return new Paragraph({ numbering: { reference: 'nums', level: 0 }, spacing: { before: 40, after: 40 },
    children: runs.map(r => new TextRun({ text: r.text, font: FONT, size: 22, bold: !!r.bold, italics: !!r.italic })) });
}
function code(lines) {
  return new Paragraph({ spacing: { before: 40, after: 40 }, shading: { fill: 'F2F2F2', type: ShadingType.CLEAR },
    children: lines.flatMap((ln, i) => {
      const run = new TextRun({ text: ln, font: MONO, size: 18 });
      return i === 0 ? [run] : [new TextRun({ text: ln, font: MONO, size: 18, break: 1 })];
    }) });
}
function cell(content, opts = {}) {
  const para = new Paragraph({ alignment: opts.align || AlignmentType.LEFT, spacing: { before: 20, after: 20 },
    children: (typeof content === 'string' ? [{ text: content }] : content).map(r => new TextRun({
      text: r.text, font: r.mono || opts.mono ? MONO : FONT, size: opts.mono ? 18 : 20,
      bold: r.bold !== undefined ? r.bold : !!opts.bold, italics: !!r.italic,
      color: r.color || opts.color || '000000' })) });
  return new TableCell({ borders: cellBorders, width: { size: opts.width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 }, children: [para] });
}
function mkTable({ widths, header, rows }) {
  const total = widths.reduce((a, b) => a + b, 0);
  const head = new TableRow({ tableHeader: true,
    children: header.map((h, i) => cell(h, { width: widths[i], fill: HEAD_FILL, bold: true, color: 'FFFFFF', align: AlignmentType.CENTER })) });
  const body = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => {
      const fill = ri % 2 === 0 ? undefined : ALT_FILL;
      if (typeof c === 'string') return cell(c, { width: widths[i], fill });
      return cell([c], { width: widths[i], fill, bold: c.bold, align: c.align });
    }) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}
const R = (a) => a === 'r' ? AlignmentType.RIGHT : a === 'c' ? AlignmentType.CENTER : AlignmentType.LEFT;
const FR = { text: '— (from run)', align: R('c'), italic: true, color: TODO }; // placeholder cell


// content ────────────────────────────────────────────────────────────────────
const content = [];
content.push(H1('From LLM hyperparameter tuning to motion-aware guidance'));
content.push(cap('Revised report for Prof. Mihai Lazarescu, addressing the eight points in your reply. Datasets: radar (preprocessed-RadarEXP1) and, for the replication in B.6, IR (preprocessed-IR-EXP2).'));

content.push(PR([{ text: 'What changed since the version you read. ', bold: true }, { text: 'The main Part B claim is withdrawn. I previously wrote that the motion-aware LLM “beats both the plain-MSE baseline and the rule-based, the first LLM in this project to beat the baseline”. Your point 2 was correct: that ordering sat inside the seed noise. On the paired test the LLM does beat plain MSE (p = 0.009) but ' }, { text: 'does not beat the motion rule', bold: true }, { text: ' (p = 0.29); and the motion rule does not beat plain MSE either (p = 0.35). Controls then show undirected random search over the same six levers matches the LLM (p = 0.64). A second motion profile (IR) reverses the gain entirely. I also found and fixed a grid bug that had pinned one lever to a constant, and re-ran every Part B number on the corrected grid.' }]));

content.push(mkTable({
  widths: [620, 5180, 3210],
  header: ['#', 'Your point', 'Where'],
  rows: [
    [{ text: '1', align: R('c') }, 'Part A Gemma-4 / Qwen3 asymmetry', 'Part A table — Qwen3 real/shuffled/empty added'],
    [{ text: '2', align: R('c') }, 'Paired seed-level results', 'B.3'],
    [{ text: '3', align: R('c') }, 'Report Gemma-4 in Part B too', 'B.3, B.4'],
    [{ text: '4', align: R('c') }, 'Depth of LLM reasoning, Part A and Part B', 'A.7 and B.4'],
    [{ text: '5', align: R('c') }, 'Separate motion knowledge from the extra knobs', 'B.5'],
    [{ text: '6', align: R('c') }, 'No test leakage — which split for each quantity', 'B.7'],
    [{ text: '7', align: R('c') }, 'Exact loss formula and temporal alignment', 'B.7'],
    [{ text: '8', align: R('c') }, 'At least one more subject / motion profile', 'B.6 — full IR replication'],
  ],
}));

// ---- Part A ----
content.push(H1('Part A — Stronger-model diagnostic: conclusion'));
content.push(P('The protocol was frozen and only the LLM was swapped: same dataset, same 9-hyperparameter discrete grid, same 25 attempts, same 3 trainings per setting (seeds 101/102/103), same fresh-seed final evaluation, same five comparison arms. I ran two stronger models, Gemma-4 12b and Qwen3-14B. Following your point 1, Qwen3-14B now has the same real / shuffled / empty history cells as Gemma-4, so the two models are directly comparable.'));
content.push(P('Final test RMSE (metres, lower is better). Reference arms are model-invariant: baseline 0.2308, rule-based 0.2371, random 0.2412, Optuna 0.2614.'));
content.push(mkTable({
  widths: [3300, 700, 1500, 1500, 1500, 1510],
  header: ['Cell', 'n', 'baseline', 'LLM', 'rule-based', 'seeds LLM < rule'],
  rows: [
    ['Gemma-4 — real history', { text: '9', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2437', align: R('c'), bold: true }, { text: '0.2371', align: R('c') }, { text: '0 / 9', align: R('c') }],
    ['Gemma-4 — shuffled history', { text: '8', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2465', align: R('c') }, { text: '0.2371', align: R('c') }, { text: '0 / 8', align: R('c') }],
    ['Gemma-4 — empty history', { text: '8', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2486', align: R('c') }, { text: '0.2371', align: R('c') }, { text: '0 / 8', align: R('c') }],
    ['Gemma-4 — explore prompt', { text: '10', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2504', align: R('c') }, { text: '0.2371', align: R('c') }, { text: '0 / 10', align: R('c') }],
    ['Qwen3-14B — real history', { text: '10', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2512', align: R('c'), bold: true }, { text: '0.2371', align: R('c') }, { text: '0 / 10', align: R('c') }],
    ['Qwen3-14B — shuffled history', { text: '10', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2406', align: R('c'), bold: true }, { text: '0.2371', align: R('c') }, { text: '2 / 10', align: R('c'), bold: true }],
    ['Qwen3-14B — empty history', { text: '10', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2476', align: R('c') }, { text: '0.2371', align: R('c') }, { text: '0 / 10', align: R('c') }],
    ['Qwen3-14B — OPRO prompt', { text: '8', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2451', align: R('c') }, { text: '0.2371', align: R('c') }, { text: '0 / 8', align: R('c') }],
    ['Qwen3-14B — explore prompt', { text: '9', align: R('c') }, { text: '0.2308', align: R('c') }, { text: '0.2500', align: R('c') }, { text: '0.2371', align: R('c') }, { text: '0 / 9', align: R('c') }],
  ],
}));
content.push(cap('The rule-based controller is deterministic, so its RMSE is the same 0.2371 on every seed. The last column is therefore a per-seed tally against that fixed line, not a verdict: the only non-zero entry (Qwen3 shuffled) still has a worse mean than the rule (0.2406 vs 0.2371) — two seeds cleared it by 1.2 mm and 0.6 mm while the other eight fell above, one by 16 mm. Compare cell means, not the tally.'));

content.push(P('Answers to the six questions:', { spacing: { before: 120, after: 40 }, run: { bold: true } }));
content.push(numItem([{ text: 'What improved with Gemma-4 and Qwen3-14B? Proposal quality.', bold: true }]));
content.push(P('Qwen3-14B proposes ~25/25 distinct settings, essentially zero repeats, zero invalid outputs, no timeouts, and reaches 0.85–0.95 grid coverage — matching random/Optuna and far above the small models (≤ 0.61). The small-model hallmark of constant repetition is gone. OPRO gave the single cleanest run of all.'));
content.push(numItem([{ text: 'What did not improve? The final model.', bold: true }]));
content.push(P('No LLM cell mean beats the rule-based controller (0 of 9 cells) and none beats the fixed baseline. Across all 82 seed-cells only 2 fall below the rule, and 0 below the baseline. The best LLM result anywhere is now Qwen3-14B with SHUFFLED history (0.2406) — a placebo condition; the best faithful-history result (Gemma-4, 0.2437) is worse. Both sit above the baseline (0.2308) and the rule (0.2371).'));
content.push(numItem([{ text: 'Did the stronger models use history better? Both depend on it, but only Gemma-4 profits from it.', bold: true }]));
content.push(P('Behaviourally both clearly read the history: degrading it collapses proposal diversity in the same direction. For Qwen3-14B distinct settings fall 24.3 → 19.7 → 15.3 across real / shuffled / empty and the repeat rate rises 0.028 → 0.084 → 0.388, a 14× increase with no history at all; Gemma-4 shows the same collapse (19.4 → 15.4 → 8.9 distinct, repeats 3 → 7 → 57).'));
content.push(P('But the two models diverge on what that history buys. Gemma-4 degrades monotonically, as expected if history is useful: 0.2437 real → 0.2465 shuffled → 0.2486 empty. Qwen3-14B inverts it — real history is its worst cell (0.2512), shuffled its best (0.2406). Paired over the same 10 seeds, shuffled−real is Δ = −0.0105, p = 0.0057 (Wilcoxon 0.0098), better on 8/10 seeds; empty−real is not significant (p = 0.24).'));
content.push(PR([{ text: 'History use and history benefit are different things. ', bold: true }, { text: 'Qwen3-14B demonstrably consumes the history, yet a truthful history produces a worse-generalizing model than a corrupted one. This is question 5 in miniature: the search overfits the small validation set, so the model that follows a faithful history chases the validation minimum hardest and pays on test. My earlier “stronger models use history better” claim was Gemma-4-specific — which is exactly why the symmetric cells were needed.' }]));
content.push(numItem([{ text: 'Did they explore better? Yes — Qwen3-14B reaches near-random grid coverage.', bold: true }]));
content.push(P('Exploration breadth is no longer a limitation at this scale.'));
content.push(numItem([{ text: 'Did they select configs that improved validation but failed on test? Yes — this is the crux.', bold: true }]));
content.push(P('On the metric the search optimizes, the LLM and the rule-based are near-tied (val RMSE ≈ 0.193 vs 0.1927) and both crush the baseline (≈ 0.29). But those low-validation settings overfit the small validation set: the same settings land at 0.244–0.250 on held-out test, worse than the baseline’s 0.231. The LLM’s proposals are not poor on their own metric — the whole search overfits validation, and the LLM overfits it slightly more.'));
content.push(numItem([{ text: 'Why does the rule-based controller stay competitive? Stable, low-variance selection.', bold: true }]));
content.push(P('It uses the same signals but converts them into a conservative choice. The LLM chases the validation minimum more aggressively and pays for it on test. Capability, prompt style, exploration breadth and history use are now all ruled out as the binding constraint. What remains is selection under a noisy validation signal, where a deterministic rule wins.'));

content.push(H2('A.7  Depth of LLM reasoning in Part A (your point 4)'));
content.push(P('Clean, non-repeating, in-grid proposals show only that the LLM follows the protocol. Applying the same five-part analysis to Part A and Part B protocol logs gives sharply different pictures:'));
content.push(mkTable({
  widths: [3600, 2900, 2510],
  header: ['Reasoning metric', 'Part A (generic HP)', 'Part B (motion loss)'],
  rows: [
    ['accepted proposals (qwen3 / gemma4)', { text: '243 / 175', align: R('c') }, { text: '173 / 209', align: R('c') }],
    ['most common diagnosis', { text: 'underfit 41% / plateau 42%', align: R('c'), bold: true }, { text: 'underfit 79% / plateau 80%', align: R('c') }],
    ['distinct diagnosis labels used', { text: '6 / 5', align: R('c'), bold: true }, { text: 'near-single label', align: R('c') }],
    ['reason names the knob it moved', { text: '96% / 93%', align: R('c'), bold: true }, { text: '46% / 61%', align: R('c') }],
    ['proposals improving running best', { text: '26% / 31%', align: R('c') }, { text: '24% / 16%', align: R('c') }],
  ],
}));
content.push(PR([{ text: 'Your suspicion holds for Part B, not for Part A. ', bold: true }, { text: 'In the motion task the diagnosis is a near-constant generic label that does not drive the action. In generic HP tuning the same models use the full label set, and for qwen3 the diagnosis genuinely selects the action in the textbook direction: overfitting → raise dropout (29/67), underfitting → raise LSTM capacity (18), plateau → raise patience (3/5). The stated reason names the hyperparameter actually moved in 96% of proposals versus 46% in Part B. Gemma-4’s coupling is weaker — dropout_up is its top action under 4 of its 5 diagnoses.' }]));
content.push(PR([{ text: 'This rules out shallow reasoning as the explanation for Part A. ', bold: true }, { text: 'The LLM diagnoses sensibly, acts consistently with the diagnosis, and explains itself accurately — and still never beats the baseline (0 of 82 seed-cells). Part A is not a reasoning failure; it is a selection failure under a noisy validation signal.' }]));
content.push(PR([{ text: 'Conclusion for Part A. ', bold: true }, { text: 'In generic hyperparameter tuning on this small, well-behaved search space, a stronger LLM improves proposal validity, diversity and history use, and reasons substantively — but none of it translates into better generalization. Better proposals and sound reasoning did not produce a better optimum. This closes both the “it is just a weak model” and the “it never really reasoned” loopholes.' }]));

// ---- Part B ----
content.push(H1('Part B — Motion-aware guidance experiment'));
content.push(PR([{ text: 'Headline (revised). ', bold: true }, { text: 'On radar the motion-aware LLM reaches 0.2286 against a plain-MSE baseline of 0.2319 — a significant 3.3 mm gain (p = 0.009). But it does not beat the deterministic motion rule (p = 0.29), the rule does not beat plain MSE (p = 0.35), and undirected random search over the same six levers matches the LLM (p = 0.64). On a second motion profile (IR) the whole effect reverses: every motion arm is significantly WORSE than plain MSE. ' }, { text: 'What survives both datasets is that the LLM’s motion knowledge is indistinguishable from random search over the same levers.', bold: true }]));

content.push(H2('B.1  Experimental design'));
content.push(P('The 9 conventional hyperparameters are frozen at the baseline setting. The only thing that varies is a six-lever loss-shaping vector:'));
content.push(mkTable({
  widths: [2600, 3400, 3010],
  header: ['Lever', 'Meaning', 'Human-motion rationale'],
  rows: [
    [{ text: 'v_max', mono: true }, 'plausible top walking speed (m/s); faster predicted steps penalised', 'set just above the observed p95 speed'],
    [{ text: 'lambda_vel', mono: true }, 'strength of the speed-plausibility penalty (0 = off)', 'raise when predictions look noisy / motion is smooth'],
    [{ text: 'lambda_smooth', mono: true }, 'penalty on implausible acceleration / jerk (0 = off)', 'raise when trajectory is smooth with frequent dwells'],
    [{ text: 'bin_weight slow / medium / fast', mono: true }, 'per-speed-regime error weights (1.0 = neutral)', 'up-weight the regime the model fits worst'],
  ],
}));
content.push(P('Neutral levers (lambda_vel=0, lambda_smooth=0, all weights 1.0) = plain MSE. Each lever vector is trained 3x from scratch, scored by mean validation RMSE in metres; the winner is evaluated on 30 fresh seeds (201-230). Four arms: plain-MSE baseline; the deterministic motion rule (v_max about 1.1x p95 speed, gentle penalties, fast-regime up-weight); the motion-aware LLM; and random search over the same six levers.'));
content.push(PR([{ text: 'Grid correction. ', bold: true }, { text: 'The original v_max grid had a floor of 1.0 m/s, but every walker here has a p95 speed of 0.41-0.60 m/s, so the rule’s v_max always snapped to that floor — the lever was a constant, not adaptive, and the velocity penalty almost never fired. The grid was re-ranged to [0.5, 0.75, 1.0, 1.5, 2.0] (same size, neutral value retained). v_max now adapts: 0.75 on radar, 0.5 on IR and capacitive. Every Part B number below is on the corrected grid.' }]));

content.push(H2('B.2  Motion profile of the tracked person (radar)'));
content.push(P('Computed from the TRAIN targets only (no training, no test data), converted to physical units at 4 Hz:'));
content.push(mkTable({
  widths: [3400, 2000, 3610],
  header: ['Feature', 'Value', 'Reading'],
  rows: [
    ['speed mean / p95 / max', { text: '0.31 / 0.59 / 3.34 m/s', align: R('c'), bold: true }, 'slow walker with occasional fast bursts'],
    ['acceleration mean / p95', { text: '0.58 / 1.86 m/s2', align: R('c') }, 'moderate start-stop dynamics'],
    ['turning mean / p95', { text: '26 / 107 deg per step', align: R('c') }, 'frequent direction changes'],
    ['sharp-turn share (turn > 45 deg)', { text: '19.1 %', align: R('c'), bold: true }, 'about 1 in 5 moving steps is a sharp turn'],
    ['stop share / dwell episodes', { text: '2.6 % / 5.8 per min', align: R('c') }, 'rarely fully stationary; brief pauses'],
  ],
}));

content.push(H2('B.3  Hypothesis, proposal, and paired result (your point 2)'));
content.push(PR([{ text: 'Hypothesis. ', bold: true }, { text: 'The model fails mainly during fast movement. Up-weighting the fast regime and penalising implausibly fast or jerky predicted steps should reduce fast-regime error without harming the rest.', italic: true }]));
content.push(P('Per-regime baseline error (mean Euclidean error by target-speed tercile, at the best epoch): slow 0.2650, medium 0.2765, fast 0.2781 — worst regime is fast, as hypothesised, spread 1.049.'));
content.push(P('Shown the profile and that table, qwen3 set v_max just above the p95 speed and up-weighted the harder regimes. Modal lever vectors:'));
content.push(mkTable({
  widths: [3600, 2700, 2710],
  header: ['Lever', 'motion rule (deterministic)', 'LLM (qwen3, modal)'],
  rows: [
    [{ text: 'v_max', mono: true }, { text: '0.75', align: R('c') }, { text: '0.75', align: R('c') }],
    [{ text: 'lambda_vel', mono: true }, { text: '0.1', align: R('c') }, { text: '0.3', align: R('c'), bold: true }],
    [{ text: 'lambda_smooth', mono: true }, { text: '0.1', align: R('c') }, { text: '0.05', align: R('c') }],
    [{ text: 'bin_weight_medium', mono: true }, { text: '1.0', align: R('c') }, { text: '3.0', align: R('c'), bold: true }],
    [{ text: 'bin_weight_fast', mono: true }, { text: '1.5', align: R('c') }, { text: '2.0', align: R('c'), bold: true }],
  ],
}));
content.push(P('Result on radar (all arms evaluated on the same 30 fresh seeds):'));
content.push(mkTable({
  widths: [3300, 2200, 1800, 1710],
  header: ['Arm', 'test RMSE (m)', 'delta vs baseline', 'fast-regime err'],
  rows: [
    ['baseline (plain MSE)', { text: '0.2319 +/- 0.0085', align: R('c') }, { text: '—', align: R('c') }, { text: '0.2781', align: R('c') }],
    ['motion rule', { text: '0.2303 +/- 0.0088', align: R('c') }, { text: '-0.0016', align: R('c') }, { text: '0.2616', align: R('c') }],
    [{ text: 'LLM motion-aware', bold: true }, { text: '0.2286', align: R('c'), bold: true }, { text: '-0.0033', align: R('c'), bold: true }, { text: '0.2637', align: R('c') }],
  ],
}));
content.push(cap('Correction: the previous version listed the baseline here as 0.2308, which is Part A’s baseline, while the delta column was computed against 0.2319. Part A and Part B have different baselines — same nine hyperparameters and same 30 seeds, but Part B trains through the motion pipeline (which carries the previous two target positions per sample), so all 30 per-seed values differ, by 1.1 mm on average. Within Part B every arm shares that pipeline, so the comparisons are like-for-like; only cross-part arithmetic is invalid.'));
content.push(P('Paired seed-level results. All arms share the same 30 final-eval seeds, so every difference is paired per seed:'));
content.push(mkTable({
  widths: [2900, 1600, 2400, 1300, 1310],
  header: ['Comparison', 'mean delta', '95% CI', 'paired p', 'wins'],
  rows: [
    [{ text: 'LLM - baseline', bold: true }, { text: '-0.0033', align: R('c'), bold: true }, { text: '[-0.0057, -0.0009]', align: R('c') }, { text: '0.009', align: R('c'), bold: true }, { text: '22 / 30', align: R('c') }],
    [{ text: 'LLM - motion rule', bold: true }, { text: '-0.0018', align: R('c') }, { text: '[-0.0051, +0.0016]', align: R('c') }, { text: '0.29', align: R('c') }, { text: '17 / 30', align: R('c') }],
    [{ text: 'motion rule - baseline', bold: true }, { text: '-0.0015', align: R('c') }, { text: '[-0.0048, +0.0018]', align: R('c') }, { text: '0.35', align: R('c') }, { text: '15 / 30', align: R('c') }],
    ['gemma4 - baseline', { text: '-0.0030', align: R('c') }, { text: '[-0.0056, -0.0004]', align: R('c') }, { text: '0.026', align: R('c') }, { text: '20 / 30', align: R('c') }],
    ['gemma4 - motion rule', { text: '-0.0015', align: R('c') }, { text: '[-0.0048, +0.0019]', align: R('c') }, { text: '0.38', align: R('c') }, { text: '17 / 30', align: R('c') }],
  ],
}));
content.push(PR([{ text: '"Beats" means statistically beats. ', bold: true }, { text: 'The LLM IS numerically below the rule (0.2286 vs 0.2303, by 1.8 mm), but the per-seed differences span -18.4 to +21.9 mm with a standard deviation of 8.9 mm — about five times the mean gap — so the sign is not reproducible. Both models (your point 3) tell the same story: each beats plain MSE, neither beats the rule. gemma4 was run on the identical corrected grid and refs; it reached 0.2289 via a different recipe (fast x3 plus slow x2) yet landed within 0.3 mm of qwen3.' }]));
content.push(PR([{ text: 'Note the third row — the comparator itself does not work. ', bold: true }, { text: 'The motion rule sits 1.5 mm below plain MSE, but that is not significant (p = 0.35, lower on exactly 15/30 seeds — a coin flip). So the -0.0016 in the table above must not be read as the motion rule beating the baseline.' }]));

content.push(H2('B.4  Depth of LLM reasoning in Part B (your point 4)'));
content.push(P('Across accepted proposals (173 qwen3 / 209 gemma4):'));
content.push(bulletR([{ text: 'The diagnosis is a dominant generic label. ', bold: true }, { text: 'qwen3 says "possible underfitting tendency" on 79% of proposals; gemma4 says "plateau" on 80%. Your suspicion was right for this task.' }]));
content.push(bulletR([{ text: 'The diagnosis does not drive the action. ', bold: true }, { text: 'Whatever the label, both models emit the same recipe — a velocity-plausibility penalty on 173/173 qwen3 proposals (every single one), a smoothness penalty, and a fast/medium up-weight.' }]));
content.push(bulletR([{ text: 'The reason only partly matches the change: ', bold: true }, { text: '46% (qwen3) / 61% (gemma4), against 96% / 93% in Part A.' }]));
content.push(bulletR([{ text: 'Most proposals do not help: ', bold: true }, { text: 'only 24% (qwen3) / 16% (gemma4) beat the running best. The search works by keeping the best of many near-identical tries, not by reasoning toward it.' }]));
content.push(P('So in the motion task the LLM applies a near-fixed prior under a dominant label — essentially what the deterministic rule already encodes, which explains why it does not beat it.'));

content.push(H2('B.5  Controls — motion knowledge, or just six extra knobs? (your point 5)'));
content.push(P('The six-lever loss is strictly more flexible than plain MSE, so part of any gain could come from the extra tunable parameters rather than motion knowledge. Two controls isolate this, both over the same six levers and the same protocol: undirected random search (zero motion knowledge), and the LLM given only the per-regime error with the motion summary removed.'));
content.push(mkTable({
  widths: [3300, 2000, 2200, 1860],
  header: ['Arm (radar)', 'test RMSE', 'motion knowledge', 'vs baseline p'],
  rows: [
    ['baseline (plain MSE)', { text: '0.2319', align: R('c') }, { text: '—', align: R('c') }, { text: '—', align: R('c') }],
    ['motion rule', { text: '0.2303', align: R('c') }, { text: 'yes (fixed)', align: R('c') }, { text: '0.35', align: R('c') }],
    [{ text: 'random over 6 levers', bold: true }, { text: '0.2289', align: R('c'), bold: true }, { text: 'NONE', align: R('c'), bold: true }, { text: '0.017', align: R('c') }],
    [{ text: 'LLM, per-regime error only', bold: true }, { text: '0.2294', align: R('c'), bold: true }, { text: 'no summary', align: R('c') }, { text: '0.044', align: R('c') }],
    ['LLM, full motion (gemma4)', { text: '0.2289', align: R('c') }, { text: 'yes', align: R('c') }, { text: '0.026', align: R('c') }],
    ['LLM, full motion (qwen3)', { text: '0.2286', align: R('c') }, { text: 'yes', align: R('c') }, { text: '0.009', align: R('c') }],
  ],
}));
content.push(bulletR([{ text: 'LLM vs random: ', bold: true }, { text: 'delta = -0.0003, p = 0.64, 17/30 — not significant. An undirected search knowing nothing about motion matches the motion-aware LLM.' }]));
content.push(bulletR([{ text: 'LLM vs no-profile: ', bold: true }, { text: 'delta = -0.0008, p = 0.31, 18/30 — not significant. Removing the motion interpretation changes nothing; the per-regime error signal alone is enough.' }]));
content.push(PR([{ text: 'On radar, then, the gain over plain MSE is attributable to the six extra levers, not to motion knowledge or LLM reasoning.', bold: true }]));

content.push(H2('B.6  Replication on a second motion profile — IR (your point 8)'));
content.push(P('Everything above rests on one trajectory. I repeated the full experiment on the IR dataset (5 Hz), whose walker is markedly different: smoother, straighter paths with 3x fewer sharp turns (0.061 vs 0.201 sharp-turn share) and a lower top speed. This is the profile where a speed-plausibility prior ought to help most, so it is a demanding test rather than a convenient one. Same protocol, same 10 search seeds, same 30 final-eval seeds, same corrected grid.'));
content.push(mkTable({
  widths: [3100, 1700, 1700, 1500, 1010],
  header: ['Arm', 'radar', 'IR', 'IR delta vs base', 'IR p'],
  rows: [
    [{ text: 'baseline (plain MSE)', bold: true }, { text: '0.2319', align: R('c') }, { text: '0.2139', align: R('c'), bold: true }, { text: '—', align: R('c') }, { text: '—', align: R('c') }],
    ['motion rule', { text: '0.2303', align: R('c') }, { text: '0.2160', align: R('c') }, { text: '+0.0021', align: R('c') }, { text: '0.045', align: R('c') }],
    ['random over 6 levers', { text: '0.2289', align: R('c') }, { text: '0.2178', align: R('c') }, { text: '+0.0039', align: R('c') }, { text: '<0.0001', align: R('c') }],
    [{ text: 'LLM motion-aware', bold: true }, { text: '0.2286', align: R('c') }, { text: '0.2194', align: R('c'), bold: true }, { text: '+0.0055', align: R('c'), bold: true }, { text: '0.0002', align: R('c'), bold: true }],
  ],
}));
content.push(cap('On radar the ordering is LLM < random < rule < baseline (loss-shaping helps). On IR it is exactly inverted: baseline < rule < random < LLM (loss-shaping hurts). All IR differences against the baseline are significant, and all 10 of 10 LLM search seeds landed above the IR baseline — a consistent reversal, not an outlier.'));
content.push(P('Two results replicate across both datasets, and one does not:'));
content.push(bulletR([{ text: 'REPLICATES — the LLM never separates from random search. ', bold: true }, { text: 'radar delta = -0.0003, p = 0.64; IR delta = +0.0016, p = 0.17. On both profiles undirected search over the same levers matches the motion-aware LLM.' }]));
content.push(bulletR([{ text: 'REPLICATES — no motion arm meaningfully beats the deterministic rule. ', bold: true }, { text: 'radar p = 0.29 (ns); on IR the LLM is significantly worse than the rule (p = 0.027).' }]));
content.push(bulletR([{ text: 'DOES NOT REPLICATE — the gain over plain MSE. ', bold: true }, { text: 'On radar every lever-touching arm beat plain MSE by 2-3 mm; on IR every one of them is significantly worse (the LLM by 5.5 mm, p = 0.0002).' }]));
content.push(PR([{ text: 'I report this as a replication failure of my own earlier positive finding. ', bold: true }, { text: 'The claim that "the six extra levers produce the gain" was itself radar-specific. A third profile (capacitive) is the natural next step: its references are already regenerated on the corrected grid and show the same direction (rule +0.0028 vs baseline, p = 0.066), so only the LLM and random arms remain.' }]));

content.push(H2('B.7  Validity — leakage and the loss formula (your points 6 and 7)'));
content.push(bulletR([{ text: 'No test leakage. ', bold: true }, { text: 'Every motion quantity comes from train or validation, never test: the motion profile (speed, acceleration, turning, stop-go, roughness) from the TRAIN targets; the loss speed-regime edges from TRAIN terciles (the controller sets only the per-bin weights, never the edges); the per-regime error shown to the LLM from VALIDATION. Selection is on validation RMSE; the test set is touched only at the final evaluation.' }]));
content.push(bulletR([{ text: 'Exact loss. ', bold: true }, { text: 'total = base_MSE + lambda_vel * mean(relu(pred_speed - v_max)^2) + lambda_smooth * mean(||accel||^2), with pred_speed = ||pred - prev_y|| * scale * hz and accel = (pred - prev_y) - (prev_y - prev_prev_y). It uses the predicted position together with the two previous TARGET positions.' }]));
content.push(bulletR([{ text: 'Temporal alignment is safe under shuffling. ', bold: true }, { text: 'The two previous positions are carried as per-sample fields (t-1, t-2) attached to each training example, not inferred from neighbouring rows in the batch. The loss is therefore computed per sample across independent samples, and shuffling the training batches does NOT break the velocity or smoothness terms.' }]));

content.push(H2('B.8  Did the LLM add anything beyond the rule?'));
content.push(bulletR([{ text: 'Over the motion rule: no. ', bold: true }, { text: 'The 1.8 mm edge on radar is inside seed noise (p = 0.29), and on IR the LLM is significantly worse than the rule.' }]));
content.push(bulletR([{ text: 'Over random search: no. ', bold: true }, { text: 'Statistically indistinguishable on both datasets (p = 0.64 radar, p = 0.17 IR), and removing the motion summary entirely costs nothing (p = 0.31).' }]));
content.push(bulletR([{ text: 'Over plain MSE: on radar yes, but not because of motion, and it does not replicate. ', bold: true }, { text: 'Random search with zero motion knowledge achieves the same gain, and on IR the same flexibility makes every arm significantly worse.' }]));
content.push(bulletR([{ text: 'Why: the reasoning is shallow in this task specifically. ', bold: true }, { text: 'A dominant label, an action independent of it, a reason matching the change only 46% of the time — against 96% in Part A. The LLM reasons well about hyperparameters and poorly about motion loss shaping.' }]));
content.push(PR([{ text: 'Net. ', bold: true }, { text: 'On these two datasets, as operationalised through these six loss levers, the LLM’s human-motion knowledge does not measurably improve the localization network beyond a one-line rule or random search. This is a negative result of the same kind as Part A, and like Part A it is informative: it localises the failure to the operationalisation rather than to the model. The LLM demonstrably reasons well in Part A, so the binding constraint is where motion knowledge is being injected — six scalar loss levers may simply be too blunt a channel for it — not the LLM’s ability to reason about motion.' }]));

content.push(H1('Status and next steps'));
content.push(P('All eight points are addressed. The Part A asymmetry is resolved, the paired statistics are reported, both models are shown, reasoning depth is measured in both parts, the controls separate motion knowledge from lever flexibility, leakage and the loss formula are documented, and the single-trajectory threat is retired by the IR replication — which overturned my own earlier positive claim.'));
content.push(P('Remaining work, in order of value: (1) the capacitive profile — its references are already regenerated on the corrected grid, so only the LLM and random arms are outstanding — to establish whether the radar gain or the IR harm is the anomaly; (2) a mostly-stationary or stop-go profile, which requires segmenting a trajectory rather than a new dataset, since none of the three datasets has a stop share above about 3%; (3) if the negative result holds on a third profile, the operationalisation itself should be reconsidered rather than the model. No further generic-hyperparameter experiments.'));

// document ───────────────────────────────────────────────────────────────────
const doc = new Document({
  creator: 'H. Favakeh',
  title: 'From LLM hyperparameter tuning to motion-aware guidance (revised)',
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: HEAD_FILL }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: HEAD_FILL }, paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: 'nums', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 320 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } } },
    children: content,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const target = process.argv[2] || path.resolve('reports/LLM-motion-aware-revised.docx');
  try {
    fs.writeFileSync(target, buf);
    console.log('Wrote:', target, '(' + buf.length + ' bytes)');
  } catch (err) {
    if (err.code === 'EBUSY') {
      const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 13);
      const ext = path.extname(target), stem = target.slice(0, -ext.length);
      const fallback = `${stem}_${stamp}${ext}`;
      fs.writeFileSync(fallback, buf);
      console.log('Original locked; wrote fallback:', fallback);
    } else { throw err; }
  }
});
