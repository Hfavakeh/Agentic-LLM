// Email-5 stronger-model diagnostic report. Self-contained (content hardcoded
// from docs/email5_stronger_models_report.md); matches the house docx style.
const fs = require('fs');
const path = require('path');

const NODE_MODULES = 'C:/Users/hfava/AppData/Roaming/npm/node_modules';
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  ImageRun,
} = require(path.join(NODE_MODULES, 'docx'));

function imageP(relPath, w, h) {
  const abs = path.resolve(relPath);
  if (!fs.existsSync(abs)) {
    return new Paragraph({ children: [new TextRun({ text: `[missing image: ${relPath}]`, italics: true, color: '999999' })] });
  }
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 },
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(abs),
      transformation: { width: w, height: h },
      altText: { title: 'Best-so-far validation RMSE', description: 'best-so-far curve', name: 'bestsofar' } })],
  });
}

const FONT = 'Calibri', MONO = 'Consolas';
const HEAD_FILL = '1F3864', ALT_FILL = 'F2F2F2';
const border = { style: BorderStyle.SINGLE, size: 8, color: '808080' };
const cellBorders = { top: border, bottom: border, left: border, right: border };

// helpers ────────────────────────────────────────────────────────────────────
function P(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 }, ...opts,
    children: [new TextRun({ text, font: FONT, size: 22, ...(opts.run || {}) })],
  });
}
function PR(runs, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 }, ...opts,
    children: runs.map(r => new TextRun({
      text: r.text, font: r.mono ? MONO : FONT, size: r.mono ? 20 : 22,
      bold: !!r.bold, italics: !!r.italic, color: r.color || '000000',
    })),
  });
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
      size: r.mono ? 20 : 22, bold: !!r.bold, italics: !!r.italic })) });
}
function numItem(runs) {
  return new Paragraph({ numbering: { reference: 'nums', level: 0 }, spacing: { before: 40, after: 40 },
    children: runs.map(r => new TextRun({ text: r.text, font: FONT, size: 22, bold: !!r.bold, italics: !!r.italic })) });
}
function cell(content, opts = {}) {
  const para = new Paragraph({ alignment: opts.align || AlignmentType.LEFT, spacing: { before: 20, after: 20 },
    children: (typeof content === 'string' ? [{ text: content }] : content).map(r => new TextRun({
      text: r.text, font: r.mono || opts.mono ? MONO : FONT, size: opts.mono ? 18 : 20,
      bold: r.bold !== undefined ? r.bold : !!opts.bold, color: r.color || opts.color || '000000' })) });
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
const R = (align) => align === 'r' ? AlignmentType.RIGHT : align === 'c' ? AlignmentType.CENTER : AlignmentType.LEFT;

// content ──────────────────────────────────────────────────────────────────
const content = [];
content.push(H1('Email-5 — Stronger-model diagnostic: is capability the missing piece?'));
content.push(cap('Controlled diagnostic (Email-5 directive), not model shopping. The protocol is frozen — same dataset (radar, preprocessed-RadarEXP1), same 9-HP discrete grid, 25 attempts/run, 3 trainings/valid setting (seeds 101/102/103), same final-evaluation seeds (201+), same five comparison arms. The only thing that changes is the LLM: two stronger models, and for one of them the SoTA optimizer-prompting recipe (OPRO).'));

content.push(H2('1. The question'));
content.push(P('Q2–Q4 (Email-4) showed the small local LLMs (3B–8B) fail to beat a deterministic curve-aware rule-based controller, and that the bottleneck is selection quality, not information, prompt, or search breadth. Email-5 asks the obvious follow-up: is the failure just capability? If small models fail because they are weak, a stronger model should repeat less, explore more, use history better, and — the question that matters — beat the curve-aware rule-based controller. We ran two of the professor’s candidates: Gemma-4 and Qwen3-14B.'));

content.push(H2('2. Method'));
content.push(P('Every run contains five arms drawing from the same grid: the fixed baseline, the LLM, random, Optuna, and the curve-aware rule-based controller. We re-ran the LLM arm under the frozen protocol with the two stronger models across these cells:'));
content.push(bulletR([{ text: 'Gemma-4', bold: true }, { text: ' — real history (none), plus the Q3 history placebo (shuffled, empty) and the Q4 exploration prompt (--explore-prompt).' }]));
content.push(bulletR([{ text: 'Qwen3-14B', bold: true }, { text: ' — the OPRO prompt (--opro-prompt: past (setting, numeric score) pairs sorted worst→best, asking for a complete new candidate expected to score lower) and the exploration prompt.' }]));
content.push(P('Metrics per cell: valid / rejected / repeated proposals, distinct trained settings out of 25, grid coverage (fraction of each HP’s allowed values tried, averaged over the 9 HPs; random/Optuna ≈ 1.0), best-so-far behaviour, final validation + test RMSE, and the comparison against random, Optuna, and especially the rule-based controller. RMSE is metres on the held-out test set, averaged over the final-eval seeds.'));

content.push(H2('3. Results — final test RMSE (mean over seeds, lower is better)'));
content.push(P('Reference arms are model-invariant: baseline 0.2308, rule-based 0.2371, random ≈ 0.242, Optuna 0.2614.'));
content.push(mkTable({
  widths: [2500, 500, 1150, 1050, 1150, 1050, 1050, 1150],
  header: ['Cell', 'n', 'baseline', 'LLM', 'rule-based', 'random', 'Optuna', 'LLM beats rule'],
  rows: [
    ['Gemma-4 — real history', { text: '9', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2437', align: R('r'), bold: true }, { text: '0.2371', align: R('r') }, { text: '0.2416', align: R('r') }, { text: '0.2614', align: R('r') }, { text: '0/9', align: R('c'), bold: true }],
    ['Gemma-4 — shuffled history', { text: '8', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2465', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0.2419', align: R('r') }, { text: '0.2614', align: R('r') }, { text: '0/8', align: R('c') }],
    ['Gemma-4 — empty history', { text: '8', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2486', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0.2419', align: R('r') }, { text: '0.2614', align: R('r') }, { text: '0/8', align: R('c') }],
    ['Gemma-4 — explore prompt', { text: '10', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2504', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0.2412', align: R('r') }, { text: '0.2614', align: R('r') }, { text: '0/10', align: R('c') }],
    ['Qwen3-14B — OPRO prompt', { text: '8', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2451', align: R('r'), bold: true }, { text: '0.2371', align: R('r') }, { text: '0.2419', align: R('r') }, { text: '0.2614', align: R('r') }, { text: '0/8', align: R('c'), bold: true }],
    ['Qwen3-14B — explore prompt', { text: '9', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2500', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0.2416', align: R('r') }, { text: '0.2614', align: R('r') }, { text: '0/9', align: R('c') }],
  ],
}));
content.push(PR([{ text: 'No LLM cell beats the rule-based controller — 0 wins across all 52 seed-cells — and none beats the fixed baseline. ', bold: true }, { text: 'The best LLM result anywhere (Gemma-4 real history 0.2437; Qwen3 OPRO 0.2451) still sits above both references.' }]));

content.push(H2('3b. Final validation RMSE (the metric the search actually selects on)'));
content.push(P('The professor asked for validation and test RMSE. The validation picture is the mirror image of the test one, and it matters.'));
content.push(mkTable({
  widths: [3400, 2000, 2000, 1960],
  header: ['Cell', 'LLM val RMSE', 'rule-based val', 'baseline val'],
  rows: [
    ['Gemma-4 — real history', { text: '0.1935', align: R('r') }, { text: '0.1927', align: R('r') }, { text: '0.294', align: R('r') }],
    ['Gemma-4 — shuffled', { text: '0.1953', align: R('r') }, { text: '0.1927', align: R('r') }, { text: '0.294', align: R('r') }],
    ['Gemma-4 — empty', { text: '0.1944', align: R('r') }, { text: '0.1927', align: R('r') }, { text: '0.294', align: R('r') }],
    ['Gemma-4 — explore', { text: '0.1930', align: R('r') }, { text: '0.1927', align: R('r') }, { text: '0.295', align: R('r') }],
    ['Qwen3-14B — OPRO', { text: '0.1943', align: R('r') }, { text: '0.1927', align: R('r') }, { text: '0.294', align: R('r') }],
    ['Qwen3-14B — explore', { text: '0.1972', align: R('r') }, { text: '0.1927', align: R('r') }, { text: '0.294', align: R('r') }],
  ],
}));
content.push(PR([{ text: 'On the objective the search optimizes, the LLM and the controller are near-tied (~0.193 vs 0.1927) and both crush the baseline (~0.29)', bold: true }, { text: ' — the LLM does find low-validation settings, roughly as well as the controller. But those settings overfit the small validation set: on held-out test the same settings land at 0.244–0.250, worse than the baseline’s 0.231. So the LLM’s proposals are not poor on their own metric — the whole search (LLM and rule-based) overfits validation, and the LLM overfits slightly more, which is exactly the test gap. This sharpens Q5: the failure is not bad proposals, it is that better validation selection does not transfer to test.' }]));

content.push(H2('4. Results — proposal behaviour and grid coverage'));
content.push(mkTable({
  widths: [2500, 1150, 1150, 1200, 1150, 2300],
  header: ['Cell', 'valid /25', 'rejected /25', 'distinct /25', 'coverage', 'dominant reject cause'],
  rows: [
    ['Gemma-4 — real history', { text: '19.4', align: R('c') }, { text: '5.6', align: R('c') }, { text: '19.4', align: R('c') }, { text: '0.60', align: R('c') }, 'LLM timeouts (47), 3 repeats'],
    ['Gemma-4 — shuffled', { text: '15.4', align: R('c') }, { text: '9.6', align: R('c') }, { text: '15.4', align: R('c') }, { text: '0.51', align: R('c') }, 'timeouts (70), 7 repeats'],
    ['Gemma-4 — empty', { text: '8.9', align: R('c') }, { text: '16.1', align: R('c') }, { text: '8.9', align: R('c') }, { text: '0.45', align: R('c') }, 'timeouts (72), 57 repeats'],
    ['Gemma-4 — explore', { text: '20.7', align: R('c') }, { text: '4.3', align: R('c') }, { text: '20.7', align: R('c') }, { text: '0.63', align: R('c') }, 'timeouts (43)'],
    ['Qwen3-14B — OPRO', { text: '24.9', align: R('c') }, { text: '0.1', align: R('c') }, { text: '24.9', align: R('c') }, { text: '0.95', align: R('c'), bold: true }, '1 repeat total, no timeouts'],
    ['Qwen3-14B — explore', { text: '24.9', align: R('c') }, { text: '0.1', align: R('c') }, { text: '24.9', align: R('c') }, { text: '0.85', align: R('c'), bold: true }, '1 repeat total, no timeouts'],
  ],
}));
content.push(P('Two qualitatively different proposers: Gemma-4 is slow (many 300 s timeouts — a serving artifact, not a decision failure) and explores moderately (~0.45–0.63 coverage). Qwen3-14B is fast, essentially never times out, never repeats, and reaches near-random grid coverage (0.85–0.95) with ~25/25 distinct settings.'));
content.push(imageP('analysis/email5/best_so_far.png', 560, 350));
content.push(P('Best-so-far validation RMSE (running minimum over the 25 attempts, averaged over seeds) improves fast for the first few attempts, then plateaus at ~0.193 — just above the rule-based controller’s 0.1927 line — for the rest of the budget, in every cell and for both models. The extra attempts (and Qwen3’s broader coverage) buy essentially nothing after the first handful: the LLM converges to about the controller’s validation level and stops improving, never crossing below it.'));

content.push(H2('5. The six questions, answered'));
content.push(numItem([{ text: 'Fewer repeated / invalid proposals? Yes, decisively — for Qwen3.', bold: true }]));
content.push(P('~25/25 distinct settings, ~0 repeats, ~0 invalid with real history / clean serving. Gemma-4 also rarely repeats with real history (3 in 9 seeds); most of its rejections are 300 s timeouts (serving), not bad decisions. The small-model hallmark — constant repetition — is gone.'));
content.push(numItem([{ text: 'Uses history better? Yes (shown cleanly on Gemma-4).', bold: true }]));
content.push(P('Degrading the rendered history hurts monotonically: RMSE 0.2437 (real) → 0.2465 (shuffled) → 0.2486 (empty), and repeats explode 3 → 7 → 57 as history is scrambled/removed. The model genuinely reads and depends on the history — unlike the small models, which ignored it.'));
content.push(numItem([{ text: 'Explores the grid more broadly? Yes — Qwen3 reaches 0.85–0.95 coverage.', bold: true }]));
content.push(P('Essentially matching random/Optuna (~1.0) and far above the small models (≤0.61 in Q4). Exploration breadth is no longer a limitation at this model scale.'));
content.push(numItem([{ text: 'Beats the curve-aware rule-based controller? No — 0 of 52 seed-cells.', bold: true }]));
content.push(P('rule-based (0.2371) beats every LLM cell, and the baseline (0.2308) beats them all. This is the decisive result.'));
content.push(numItem([{ text: 'If not, what does it imply for LLMs in generic HP tuning?', bold: true }]));
content.push(P('Capability is not the binding constraint. Qwen3-14B is fast, non-repeating, near-full-coverage, history-sensitive, and even uses the SoTA OPRO prompt — it does everything the “weak small model” failed to do — and still cannot turn any of that into better selection than a trivial deterministic rule using the same signals. Better proposals did not produce a better optimum. Generic hyperparameter tuning on this small, well-behaved search space is simply not where an LLM adds value.'));
content.push(numItem([{ text: 'What motion-aware experiment next?', bold: true }]));
content.push(P('The exit condition is met (Gemma-4 + one stronger model both lose to the controller), so we pivot to the thesis core: give the LLM interpretable human-motion summaries (speed, acceleration, turning, stop-go, trajectory roughness, per-regime errors) and test whether it can propose useful changes to the loss, windowing, or training objective — a task where broad knowledge of how people move could plausibly matter, unlike generic HP tuning. The Q5 payload-motion plumbing is already in place as the starting point.'));

content.push(H2('6. OPRO specifically'));
content.push(P('OPRO was the professor’s named benchmark (LLMs-as-optimizers; candidates + scores in the prompt). On Qwen3-14B it produced the best proposal behaviour of any cell — 0.95 coverage, 24.9/25 distinct, no repeats — and the best LLM RMSE among the Qwen3 cells (0.2451 vs 0.2500 for the exploration prompt). So the SoTA recipe does help relative to a naive exploration prompt. But it still loses to the rule-based controller and the baseline. The literature’s claim (small LLMs have limited optimization ability and need clear, direct instructions) is consistent with our OPRO cell being the cleanest LLM run — yet clearer instructions raised proposal quality without raising decision quality.'));

content.push(H2('7. Honest caveats'));
content.push(bulletR([{ text: 'Incomplete seed coverage. ', bold: true }, { text: 'Gemma-4 shuffled/empty and both Qwen3 cells are missing seed 777; Qwen3-OPRO seed_666 has no final-eval (OPRO n=8, explore n=9). Backfilling to 10 seeds would tighten the means (the qualitative story is already unambiguous — 0 wins everywhere).' }]));
content.push(bulletR([{ text: 'No default-prompt Qwen3 cell. ', bold: true }, { text: 'For the cleanest OPRO A/B we want Qwen3 with the plain protocol prompt (history none, no variant). We have OPRO and explore for Qwen3 but not the default reference.' }]));
content.push(bulletR([{ text: 'Thinking mode not A/B’d. ', bold: true }, { text: 'Qwen3-14B produced clean output with no timeouts, but a thinking-on vs thinking-off comparison was not run — an open item.' }]));
content.push(bulletR([{ text: 'Gemma-4 tag/size not recorded ', bold: true }, { text: 'in the run artifacts; confirm the exact quantized tag for the write-up.' }]));
content.push(bulletR([{ text: 'Single dataset (radar).', bold: false }]));

content.push(H2('8. Conclusion'));
content.push(P('A stronger model was the natural hypothesis for why the LLM fails, and Email-5 tests it directly. Qwen3-14B removes every symptom the small models showed — it does not repeat, it explores the whole grid, it uses history, it runs cleanly, and it even uses the OPRO optimizer prompt — and it still does not beat the curve-aware rule-based controller or the fixed baseline, in any seed. Gemma-4 tells the same story. Capability, prompt style, exploration breadth, and history use are all ruled out as the binding constraint. What remains is the finding Q2–Q4 already pointed to: on this small, navigable search space the LLM cannot convert good proposals into better selection than a trivial rule. That satisfies the professor’s stop condition for generic LLM hyperparameter tuning and moves the work to the motion-aware thesis core.'));

content.push(H2('Appendix — Motion-aware descriptors: definitions and computation'));
content.push(P('For the motion-aware experiment (Q6), the LLM is shown interpretable summaries of how the tracked person moves, computed from the target trajectory only (ground-truth positions p_t = (x_t, y_t) in metres). All rates use the dataset sampling rate hz (radar 4 Hz, cap 3 Hz, IR 5 Hz), so thresholds are comparable across datasets. Δp_t = p_t − p_{t−1}.'));
content.push(mkTable({
  widths: [1550, 3200, 2400, 2210],
  header: ['Term', 'Definition (per sample)', 'Reported summaries', 'What it captures'],
  rows: [
    [{ text: 'Speed', bold: true }, 'speed_t = ‖Δp_t‖ · hz  (m/s)', 'mean, std, median, IQR, p95, min, max', 'how fast the person walks'],
    [{ text: 'Acceleration', bold: true }, 'accel_t = (speed_t − speed_{t−1}) · hz  (m/s²)', 'mean|accel|, std, p95|accel|', 'how sharply speed changes (starts/stops)'],
    [{ text: 'Turning', bold: true }, 'heading θ_t = atan2(Δy_t, Δx_t); turn = wrap(θ_t − θ_{t−1}) in deg, masked to moving samples (speed ≥ 0.05 m/s)', 'mean|turn|, p95|turn|, sharp-turn share (|turn| > 45°)', 'how much the path bends; sharp direction changes'],
    [{ text: 'Roughness', bold: true }, '(a) sinuosity = Σ‖Δp_t‖ / ‖p_last − p_first‖; (b) jerk = (accel_t − accel_{t−1}) · hz  (m/s³)', 'sinuosity (1.0 = straight), mean|jerk|', 'jagged / winding vs smooth path'],
    [{ text: 'Stop-go / dwell', bold: true }, 'stop if speed_t < 0.05 m/s; a dwell = contiguous run of stop samples', 'stop share, #episodes, stop→go transitions, dwell duration (mean/p95/max s), episodes/min', 'hesitation / pausing behaviour'],
    [{ text: 'Per-regime error', bold: true }, 'samples bucketed slow/medium/fast by the terciles (1/3, 2/3) of target speed; per bin mean ‖pred − target‖ (m)', 'error per regime, worst regime, spread ratio (worst/best)', 'where the model error concentrates by regime'],
  ],
}));
content.push(bulletR([{ text: 'Speed, acceleration and jerk are successive finite differences of position scaled by hz, so the first one/two samples are undefined and dropped.' }]));
content.push(bulletR([{ text: 'Turning is masked to moving samples because heading is ill-defined (jitter) when the person is essentially stationary.' }]));
content.push(bulletR([{ text: 'Per-regime error is the only descriptor from the model’s predictions (the rest are trajectory properties); it is recomputed on validation at each setting’s best epoch and tells the LLM which regime to up-weight in the loss.' }]));
content.push(bulletR([{ text: 'All produced by motion_descriptors.py (extract_motion_features, summarize_dynamics, summarize_dwell, per_regime_error) and shown to the LLM as raw numbers in the motion payload.' }]));

// document ───────────────────────────────────────────────────────────────────
const doc = new Document({
  creator: 'H. Favakeh',
  title: 'Email-5 — Stronger-model diagnostic',
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
  const target = process.argv[2] || path.resolve('Email5_Stronger_Models_Report.docx');
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
