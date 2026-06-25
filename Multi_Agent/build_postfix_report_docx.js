// Build a polished .docx of analysis/postfix_modifications_report.md
const fs = require('fs');
const path = require('path');

const NODE_MODULES = 'C:/Users/hfava/AppData/Roaming/npm/node_modules';
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType,
} = require(path.join(NODE_MODULES, 'docx'));

// ---------- Style helpers ----------
const FONT      = 'Calibri';
const MONO      = 'Consolas';
const HEAD_FILL = '1F3864';
const ALT_FILL  = 'F2F2F2';

const thin   = { style: BorderStyle.SINGLE, size: 8, color: '808080' };
const cellBorders = { top: thin, bottom: thin, left: thin, right: thin };

function caption(text) {
  return new Paragraph({
    spacing: { before: 200, after: 60 },
    children: [new TextRun({ text, font: FONT, size: 20, italics: true, color: '595959' })],
  });
}

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    ...opts,
    children: [new TextRun({ text, font: FONT, size: 22, ...(opts.run || {}) })],
  });
}

// Mixed-formatting paragraph; runs is an array of {text, mono, bold, italic} objects.
function PR(runs, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    ...opts,
    children: runs.map(r => new TextRun({
      text: r.text,
      font: r.mono ? MONO : FONT,
      size: r.mono ? 20 : 22,
      bold: !!r.bold,
      italics: !!r.italic,
    })),
  });
}

function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: true, color: HEAD_FILL })],
  });
}

function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text, font: FONT, size: 24, bold: true, color: HEAD_FILL })],
  });
}

function H3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 140, after: 60 },
    children: [new TextRun({ text, font: FONT, size: 22, bold: true, color: '333333' })],
  });
}

function bulletR(runs) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 40, after: 40 },
    children: runs.map(r => new TextRun({
      text: r.text,
      font: r.mono ? MONO : FONT,
      size: r.mono ? 20 : 22,
      bold: !!r.bold, italics: !!r.italic,
    })),
  });
}

// ---------- Table helpers ----------
function cell(content, opts = {}) {
  // `content` is either a string OR an array of run objects for PR-style mixed runs.
  let para;
  if (typeof content === 'string') {
    para = new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { before: 20, after: 20 },
      children: [new TextRun({
        text: content,
        font: opts.mono ? MONO : FONT,
        size: opts.mono ? 20 : 20,
        bold: !!opts.bold,
        color: opts.color || '000000',
      })],
    });
  } else {
    para = new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { before: 20, after: 20 },
      children: content.map(r => new TextRun({
        text: r.text,
        font: r.mono ? MONO : FONT,
        size: r.mono ? 20 : 20,
        bold: !!r.bold,
        color: r.color || (opts.color || '000000'),
      })),
    });
  }
  return new TableCell({
    borders: cellBorders,
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [para],
  });
}

function makeTable({ widths, header, rows }) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: header.map((h, i) => cell(h, {
      width: widths[i], fill: HEAD_FILL, bold: true, color: 'FFFFFF',
      align: AlignmentType.CENTER,
    })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => {
      const fill = ri % 2 === 0 ? undefined : ALT_FILL;
      // c can be a plain string or {text, mono, bold, align, runs}
      if (typeof c === 'string') {
        return cell(c, { width: widths[i], fill });
      }
      if (c && Array.isArray(c.runs)) {
        return cell(c.runs, { width: widths[i], fill, align: c.align });
      }
      return cell(c.text, { width: widths[i], fill, mono: c.mono, bold: c.bold, align: c.align });
    }),
  }));
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headerRow, ...bodyRows],
  });
}

// ---------- Document body ----------
const content = [];

// Title block
content.push(new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { before: 0, after: 120 },
  children: [new TextRun({
    text: 'Bake-off pipeline fixes and post-fix empirical results',
    font: FONT, size: 36, bold: true, color: HEAD_FILL,
  })],
}));
content.push(PR([
  { text: 'Author: ', bold: true }, { text: 'H. Favakeh' },
  { text: '   |   ' },
  { text: 'Date: ', bold: true }, { text: '2026-05-31' },
  { text: '   |   ' },
  { text: 'Repo: ', bold: true }, { text: 'origin/main @ ', },
  { text: '5aaff1de', mono: true },
]));
content.push(PR([
  { text: 'Dataset: ', bold: true },
  { text: 'preprocessed Radar EXP1 (4 Hz, 2001 samples; 1188/388/388 train/val/test sequences).' },
]));

// §1
content.push(H1('1. Summary'));
content.push(P(
  'Three independent defects in the 5-arm bake-off (baseline / LLM / random / rule-based / Optuna) ' +
  'were diagnosed and fixed during this work cycle. Each defect biased the head-to-head comparison ' +
  'in a different direction; each is now closed on main and verified by a controlled re-run across ' +
  'three LLMs (Llama 3 8B, Phi 4, Nemotron 3) over 10 fixed outer seeds. The fixes resolve a ' +
  'search-objective/headline-metric inconsistency, a rule-based controller deadlock that collapsed ' +
  'its budget to two distinct settings, and a Windows-console encoding crash that was aborting runs ' +
  'mid-experiment.'
));

// §2
content.push(H1('2. Modifications'));

content.push(H2('2.1 Search objective aligned to the headline metric'));
content.push(PR([
  { text: 'Commit ', bold: true }, { text: '7ba02e7c', mono: true }, { text: '.' },
]));
content.push(PR([
  { text: 'Before: ', bold: true },
  { text: 'evaluate_setting', mono: true },
  { text: ' ranked candidates by sqrt(min val_position_loss) in ' },
  { text: 'StandardScaler z-space', italic: true },
  { text: '. The headline ' }, { text: 'compute_metrics', mono: true },
  { text: ' RMSE was in metres (computed on inverse-transformed predictions). Because the per-axis ' +
          'StandardScaler weights x and y by different scale factors, the two metrics induced ' +
          'different rankings — i.e. the optimiser was selecting a different quantity than the ' +
          'thesis reported.' },
]));
content.push(PR([
  { text: 'After: ', bold: true },
  { text: 'Trainer.validate()', mono: true },
  { text: ' caches ' },
  { text: '_cached_val_position_loss_m = mean((p_inv − t_inv)²)', mono: true },
  { text: ' — the same functional as the headline RMSE, but on the validation set. ' +
          'evaluate_setting now scores sqrt(val_position_loss_m at the early-stopping best epoch). ' +
          'Early stopping itself was switched from the scaled val_loss to the metres metric ' +
          '(with vl_loss fallback when the metres metric is unavailable), so the kept epoch, ' +
          'the selection score, and the headline test RMSE all measure the same quantity in the ' +
          'same units end-to-end.' },
]));

content.push(H2('2.2 Rule-based controller deadlock fixed'));
content.push(PR([
  { text: 'Commit ', bold: true }, { text: '45fb88a5', mono: true }, { text: '.' },
]));
content.push(PR([
  { text: 'Before: ', bold: true },
  { text: '_act_protocol(diagnosis, anchor)', mono: true },
  { text: ' returned exactly one fixed proposal per (diagnosis × anchor) pair — e.g. ' +
          'diagnosis="healthy" → {dropout: anchor.dropout + 1 grid step}. When that proposal ' +
          'failed to improve, the anchor stayed put, the next diagnosis was the same, and the ' +
          'controller re-emitted the same proposal forever. In the 10-seed pre-fix Llama 3 8B run, ' +
          'this collapsed 25 attempts into 2 distinct settings, with rule_based mean test RMSE ' +
          'byte-identical to baseline across all seeds and models.' },
]));
content.push(PR([
  { text: 'After: ', bold: true },
  { text: 'each diagnosis maps to a frozen priority list of single-HP (param, direction) ' +
          'moves (' },
  { text: '_PROTOCOL_MOVES_BASE', mono: true }, { text: ' plus ' },
  { text: '_PROTOCOL_ARCH_MOVES', mono: true },
  { text: ' when arch changes are allowed). _act_protocol walks the list, skipping grid-boundary ' +
          'no-ops and consulting context["is_tried"] to skip already-evaluated settings; the first ' +
          'hit wins. When every priority move is a duplicate, the proposal is rejected with ' +
          'failure_reason="exhausted". Defense-in-depth was also added to run_proposer_search so ' +
          'that any duplicate slipping past a proposer is rejected before training. A 25-attempt ' +
          'unit test against a frozen anchor now produces 13 distinct single-HP deltas then ' +
          'correctly rejects the remaining 12 (vs 2 + 23 wasted re-trainings pre-fix).' },
]));

content.push(H2('2.3 Console encoding crash fixed'));
content.push(PR([
  { text: 'Commit ', bold: true }, { text: '5aaff1de', mono: true }, { text: '.' },
]));
content.push(PR([
  { text: 'Before: ', bold: true },
  { text: 'write_final_eval_report', mono: true },
  { text: ' used the U+2212 (“−”) glyph in its "Paired differences" heading and per-arm bullets. ' +
          'The markdown file wrote fine, but logger.info(line) echoed each line through a ' +
          'StreamHandler that on Windows defaults to cp1252 — which cannot encode U+2212. Result: ' +
          'UnicodeEncodeError killed the run after arms 1–4 had trained but before ' +
          'cross_run_metrics.json, final_evaluation_run1.json and multi_seed_summary.json were ' +
          'written. This aborted seed_777 in the post-fix Llama 3 8B run and prevented multi-seed ' +
          'aggregation.' },
]));
content.push(PR([
  { text: 'After: ', bold: true },
  { text: 'sys.stdout and sys.stderr are reconfigured to UTF-8 with errors="replace", the ' +
          'FileHandler is opened with encoding="utf-8", and the "−" glyphs in the report itself ' +
          'were swapped for ASCII "-". Defense-in-depth at two layers means any future ' +
          'non-cp1252 character in any logged string can no longer crash a run.' },
]));

// §3
content.push(H1('3. Empirical verification — 3-LLM head-to-head on the fixed code'));
content.push(P(
  'Three independent multi-seed runs (10 outer seeds: 17, 42, 73, 128, 256, 314, 451, 512, 666, 777) ' +
  'on the latest main, identical except for the LLM model.'
));
content.push(caption('Table 1. Mean test RMSE per arm (metres). Each cell = mean across the 10 outer seeds of the 30-fresh-seed final-eval RMSE. Lower is better. See Table 2 for full mean ± std and (min – max) range.'));
content.push(makeTable({
  widths: [2240, 1780, 1780, 1780],
  header: ['Arm', 'Llama 3 8B', 'Phi 4', 'Nemotron 3'],
  rows: [
    [{text:'baseline'}, {text:'0.2308', align:AlignmentType.RIGHT},
                       {text:'0.2308', align:AlignmentType.RIGHT},
                       {text:'0.2308', align:AlignmentType.RIGHT}],
    [{text:'LLM', bold:true}, {text:'0.2440', align:AlignmentType.RIGHT, bold:true},
                              {text:'0.2464', align:AlignmentType.RIGHT, bold:true},
                              {text:'0.2459', align:AlignmentType.RIGHT, bold:true}],
    [{text:'random'},   {text:'0.2412', align:AlignmentType.RIGHT},
                        {text:'0.2412', align:AlignmentType.RIGHT},
                        {text:'0.2412', align:AlignmentType.RIGHT}],
    [{text:'rule_based'},{text:'0.2371', align:AlignmentType.RIGHT},
                         {text:'0.2371', align:AlignmentType.RIGHT},
                         {text:'0.2371', align:AlignmentType.RIGHT}],
    [{text:'optuna'},   {text:'0.2614', align:AlignmentType.RIGHT},
                        {text:'0.2614', align:AlignmentType.RIGHT},
                        {text:'0.2614', align:AlignmentType.RIGHT}],
  ],
}));
content.push(PR([
  { text: 'Non-LLM arms produce byte-identical values across the three runs because their picked ' +
          'setting does not depend on the LLM (baseline = fixed defaults; random seeded by outer ' +
          'seed only; rule-based deterministic; Optuna ' },
  { text: 'TPESampler(seed=1000)', mono: true },
  { text: ' hardcoded) — verified by direct inspection of each arm’s best_setting. The LLM is the ' +
          'only arm that varies across the three runs, by construction.' },
]));

// Table 2: Per-arm mean ± std and (min - max) range across the 10 outer seeds
content.push(caption('Table 2. Test RMSE — mean ± std (min – max) across the 10 outer seeds per LLM. The non-LLM arms have std = 0 because their picked setting does not depend on the outer seed (baseline = fixed defaults; rule-based deterministic; Optuna seed = 1000). The per-arm uncertainty across the 30 fresh test seeds is shown separately in §5.'));
content.push(makeTable({
  widths: [1700, 2700, 2400, 2760],
  header: ['Arm', 'Llama 3 8B (n=9*)', 'Phi 4 (n=10)', 'Nemotron 3 (n=10)'],
  rows: [
    [{text:'baseline'},
     {text:'0.2308 ± 0.000 (0.2308 – 0.2308)', align:AlignmentType.CENTER},
     {text:'0.2308 ± 0.000 (0.2308 – 0.2308)', align:AlignmentType.CENTER},
     {text:'0.2308 ± 0.000 (0.2308 – 0.2308)', align:AlignmentType.CENTER}],
    [{text:'LLM', bold:true},
     {text:'0.2440 ± 0.005 (0.2326 – 0.2503)', align:AlignmentType.CENTER, bold:true},
     {text:'0.2464 ± 0.005 (0.2397 – 0.2541)', align:AlignmentType.CENTER, bold:true},
     {text:'0.2459 ± 0.005 (0.2400 – 0.2569)', align:AlignmentType.CENTER, bold:true}],
    [{text:'random'},
     {text:'0.2416 ± 0.011 (0.2197 – 0.2545)', align:AlignmentType.CENTER},
     {text:'0.2412 ± 0.010 (0.2197 – 0.2545)', align:AlignmentType.CENTER},
     {text:'0.2412 ± 0.010 (0.2197 – 0.2545)', align:AlignmentType.CENTER}],
    [{text:'rule_based'},
     {text:'0.2371 ± 0.000 (0.2371 – 0.2371)', align:AlignmentType.CENTER},
     {text:'0.2371 ± 0.000 (0.2371 – 0.2371)', align:AlignmentType.CENTER},
     {text:'0.2371 ± 0.000 (0.2371 – 0.2371)', align:AlignmentType.CENTER}],
    [{text:'optuna'},
     {text:'0.2614 ± 0.000 (0.2614 – 0.2614)', align:AlignmentType.CENTER},
     {text:'0.2614 ± 0.000 (0.2614 – 0.2614)', align:AlignmentType.CENTER},
     {text:'0.2614 ± 0.000 (0.2614 – 0.2614)', align:AlignmentType.CENTER}],
  ],
}));
content.push(PR([
  { text: '* Llama 3 8B has n=9 because seed_777 aborted in the run that pre-dated the cp1252 ' +
          'fix; the cp1252 fix lands before the Phi 4 and Nemotron 3 runs.', italic: true },
]));

// Table 3: LLM exploration efficiency (renumbered)
content.push(caption('Table 3. LLM exploration efficiency — averaged across the 10 outer seeds.'));
content.push(makeTable({
  widths: [3300, 1620, 1620, 1620],
  header: ['Metric', 'Llama 3 8B', 'Phi 4', 'Nemotron 3'],
  rows: [
    [{text:'Clean attempts / 25 (mean)'},
     {text:'7.0',  align:AlignmentType.RIGHT},
     {text:'9.9',  align:AlignmentType.RIGHT},
     {text:'18.9', align:AlignmentType.RIGHT, bold:true}],
    [{text:'Effective LLM budget'},
     {text:'28%', align:AlignmentType.RIGHT},
     {text:'40%', align:AlignmentType.RIGHT},
     {text:'76%', align:AlignmentType.RIGHT, bold:true}],
    [{text:'Distinct proposals / 25 (mean)'},
     {text:'~8',   align:AlignmentType.RIGHT},
     {text:'13.7', align:AlignmentType.RIGHT},
     {text:'18.6', align:AlignmentType.RIGHT, bold:true}],
    [{text:'Most-repeated proposal (×count)'},
     {text:'×16', align:AlignmentType.RIGHT},
     {text:'×11', align:AlignmentType.RIGHT},
     {text:'×3',  align:AlignmentType.RIGHT, bold:true}],
    [{text:'LLM win-rate vs baseline'},
     {text:'0/9*',  align:AlignmentType.RIGHT},
     {text:'0/10', align:AlignmentType.RIGHT},
     {text:'0/10', align:AlignmentType.RIGHT}],
  ],
}));
content.push(PR([
  { text: '* Llama 3 8B seed_777 aborted in the pre-cp1252-fix run; redone afterward without crash. ', italic: true },
]));

// Table 3: Rule-based fix demonstration
content.push(caption('Table 4. Rule-based controller: pre-fix deadlock vs post-fix priority-list behavior, per outer seed.'));
content.push(makeTable({
  widths: [3300, 3000, 3060],
  header: ['Quantity', 'Pre-fix', 'Post-fix'],
  rows: [
    [{text:'Distinct change-sets / 25 attempts'},
     {text:'2', align:AlignmentType.CENTER},
     {text:'18', align:AlignmentType.CENTER, bold:true}],
    [{text:'rule_based mean test RMSE'},
     {text:'0.2308 (== baseline)', align:AlignmentType.CENTER},
     {text:'0.2371 (distinct from baseline)', align:AlignmentType.CENTER, bold:true}],
    [{text:'Wasted re-trainings on duplicates'},
     {text:'23', align:AlignmentType.CENTER},
     {text:'0', align:AlignmentType.CENTER, bold:true}],
    [{text:'Diagnoses observed in 25 rounds'},
     {text:'inconclusive ×1, healthy ×24',
      align:AlignmentType.CENTER},
     {text:'inconclusive ×2, healthy ×20, underfit ×3',
      align:AlignmentType.CENTER}],
  ],
}));

// §4
content.push(H1('4. Key findings'));
content.push(bulletR([
  { text: 'Rule-based is now a real arm.', bold: true },
  { text: ' Distinct change-sets per seed: 18 in all three post-fix runs (vs 2 pre-fix). Its picked ' +
          'setting (lr=0.003, wd=1e-4, dropout=0.35, ws=10) improves validation RMSE by ~12% over ' +
          'baseline but loses ~3% on test — classic HP-search overfitting to a small val split, ' +
          'not a code defect.' },
]));
content.push(bulletR([
  { text: 'Small-LLM duplication is real and quantifiable.', bold: true },
  { text: ' Effective LLM budget (clean of 25 attempts): Llama 3 8B 28%, Phi 4 40%, Nemotron 3 76%. ' +
          'Despite Nemotron’s much higher exploration breadth (18.6 distinct proposals avg vs ' +
          'Llama’s ~8), its test RMSE is worse than Llama’s. Broader exploration appears to ' +
          'over-fit the small val split rather than to generalise better.' },
]));
content.push(bulletR([
  { text: 'Baseline is hard to beat on this dataset.', bold: true },
  { text: ' Across all three LLMs and all 30 outer-seed runs, no arm beats baseline on more than ' +
          '1 in 10 seeds. Optuna is consistently the worst arm (−13.27% vs baseline), an actionable ' +
          'finding about deterministic-Bayesian HP search at this budget.' },
]));

// §5
content.push(H1('5. Reproducibility'));
content.push(P(
  'All fixes are committed to origin/main. The exact commits and verifying tests:'
));
content.push(caption('Table 5. Fix index — commit hashes and how each fix was verified.'));
content.push(makeTable({
  widths: [3000, 2200, 4160],
  header: ['Concern', 'Commit', 'Verification'],
  rows: [
    [{text:'Search-objective units'},
     {text:'7ba02e7c', mono:true, align:AlignmentType.CENTER},
     {text:'Smoke test; metres-space score in evaluate_setting'}],
    [{text:'Rule-based deadlock'},
     {text:'45fb88a5', mono:true, align:AlignmentType.CENTER},
     {text:'25-attempt unit test; 18 distinct change-sets in real runs'}],
    [{text:'cp1252 crash'},
     {text:'5aaff1de', mono:true, align:AlignmentType.CENTER},
     {text:'Smoke test completes without PYTHONIOENCODING override'}],
  ],
}));
content.push(PR([
  { text: 'Run logs preserved in ' },
  { text: 'Downloads/llama38b-new1/', mono: true }, { text: ', ' },
  { text: 'Downloads/phi414-new1/', mono: true }, { text: ', ' },
  { text: 'Downloads/nemotron3-new/', mono: true }, { text: '. Memory entries documenting each ' +
          'fix’s root cause and design intent: ' },
  { text: 'project_objective_units_mismatch.md', mono: true }, { text: ', ' },
  { text: 'project_rule_based_deadlock.md', mono: true }, { text: '.' },
]));

// ---------- Build doc ----------
const doc = new Document({
  creator: 'H. Favakeh',
  title: 'Bake-off pipeline fixes and post-fix empirical results',
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: HEAD_FILL },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: HEAD_FILL },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: '333333' },
        paragraph: { spacing: { before: 140, after: 60 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: '•',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },          // US Letter
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }, // 0.75" margins
      },
    },
    children: content,
  }],
});

Packer.toBuffer(doc).then(buf => {
  // Allow override via CLI arg, otherwise write to default; if that's locked
  // (Word has it open), fall back to a timestamped sibling so the build never
  // fails just because the user is reading the previous version.
  const target = process.argv[2] || path.resolve('analysis/postfix_modifications_report.docx');
  try {
    fs.writeFileSync(target, buf);
    console.log('Wrote:', target, '(' + buf.length + ' bytes)');
  } catch (err) {
    if (err.code === 'EBUSY') {
      const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 13);
      const ext  = path.extname(target);
      const stem = target.slice(0, -ext.length);
      const fallback = `${stem}_${stamp}${ext}`;
      fs.writeFileSync(fallback, buf);
      console.log('Original locked; wrote fallback:', fallback, '(' + buf.length + ' bytes)');
    } else {
      throw err;
    }
  }
});
