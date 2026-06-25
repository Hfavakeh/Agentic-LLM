const fs = require('fs');
const path = require('path');

const NODE_MODULES = 'C:/Users/hfava/AppData/Roaming/npm/node_modules';
const {
  Document, Packer, Paragraph, TextRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageBreak,
} = require(path.join(NODE_MODULES, 'docx'));

const ROOT = 'C:/Users/hfava/Documents/Thesis/Hossein_NN/Multi_Agent';
const SEEDS = [
  { name: 'seed_8532', summary: 'No retries, no fallbacks — clean LLM trajectory.' },
  { name: 'seed_8934', summary: 'Validator catches a bad diagnosis at round 4 and forces a fallback.' },
];
const EXP = 'outputs-0503-llama3-8b';

function fmt(v, d = 4) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return v.toFixed(d);
  return String(v);
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    ...opts,
    children: [new TextRun({ text, ...(opts.run || {}) })],
  });
}

function pRich(runs, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    ...opts,
    children: runs.map(r => new TextRun(r)),
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 30, after: 30 },
    children: [new TextRun({ text, size: 20 })],
  });
}

function bulletRich(runs) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 30, after: 30 },
    children: runs.map(r => new TextRun(r)),
  });
}

function code(text) {
  const lines = String(text).split('\n');
  const children = [];
  lines.forEach((ln, i) => {
    if (i > 0) children.push(new TextRun({ break: 1 }));
    children.push(new TextRun({ text: ln, font: 'Consolas', size: 18 }));
  });
  return new Paragraph({
    spacing: { before: 80, after: 120 },
    shading: { fill: 'F5F5F5', type: ShadingType.CLEAR },
    children,
  });
}

function quote(text) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    indent: { left: 360 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: '2E75B6', space: 8 } },
    children: [new TextRun({ text, italics: true, size: 20 })],
  });
}

function h1(t) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text: t })],
  });
}
function h2(t) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text: t })],
  });
}
function h3(t) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 140, after: 60 },
    children: [new TextRun({ text: t })],
  });
}

function renderUserPayload(p) {
  const lines = [];
  const m = p.metrics || {};
  lines.push({ k: 'val_loss', v: fmt(m.val_loss) });
  lines.push({ k: 'train_loss', v: fmt(m.train_loss) });
  lines.push({ k: 'loss_ratio', v: fmt(m.loss_ratio) });
  lines.push({ k: 'val_mae', v: fmt(m.val_mae) });
  lines.push({ k: 'mean_euclid_m', v: fmt(m.mean_euclidean_distance_m) });

  const blocks = [];
  blocks.push(h3('Metrics'));
  blocks.push(bullet(`val_loss = ${fmt(m.val_loss)},  train_loss = ${fmt(m.train_loss)},  loss_ratio = ${fmt(m.loss_ratio)}`));
  blocks.push(bullet(`val_mae = ${fmt(m.val_mae)},  mean_euclid_m = ${fmt(m.mean_euclidean_distance_m)}`));

  const rs = p.round_summary || {};
  if (Object.values(rs).some(v => v !== null && v !== undefined)) {
    blocks.push(h3('Round summary'));
    blocks.push(bullet(`best_val_loss = ${fmt(rs.round_best_val_loss)},  avg_val_loss = ${fmt(rs.round_avg_val_loss)},  epochs_trained = ${rs.round_epochs_trained ?? 0}`));
  }

  const t = p.trends || {};
  if (Object.keys(t).length) {
    blocks.push(h3('Trends'));
    blocks.push(bullet(`validation = ${t.validation ?? '—'},  training = ${t.training ?? '—'},  epochs_since_improvement = ${t.epochs_since_improvement ?? '—'}`));
  }

  const tp = p.training_progress || {};
  if (Object.keys(tp).length) {
    blocks.push(h3('Training progress'));
    blocks.push(bullet(`total_epochs = ${tp.total_epochs},  current_round = ${tp.current_round},  best_val_loss = ${fmt(tp.best_val_loss)}`));
  }

  const tools = p.tool_results || {};
  if (Object.keys(tools).length) {
    blocks.push(h3('Diagnostic tools'));
    for (const [name, res] of Object.entries(tools)) {
      if (!res || typeof res !== 'object') continue;
      const parts = [];
      for (const [k, v] of Object.entries(res)) {
        if (v && typeof v === 'object') continue;
        parts.push(`${k}=${typeof v === 'number' ? fmt(v) : v}`);
      }
      blocks.push(bulletRich([
        { text: name + ': ', bold: true, font: 'Consolas', size: 20 },
        { text: parts.join(', '), size: 20 },
      ]));
    }
  }

  const hp = p.current_hyperparameters || {};
  if (Object.keys(hp).length) {
    blocks.push(h3('Current hyperparameters'));
    const parts = Object.entries(hp).map(([k, v]) => `${k}=${v}`);
    blocks.push(bullet(parts.join(', ')));
  }

  const oh = p.optimization_history || [];
  if (oh.length) {
    blocks.push(h3('Optimization history (compact)'));
    for (const e of oh) {
      const ch = e.changes_applied || {};
      const chs = Object.keys(ch).length
        ? Object.entries(ch).map(([k, v]) => `${k}=${v}`).join(', ')
        : 'no change';
      blocks.push(bullet(`round ${e.round}: val_loss=${fmt(e.val_loss)}, outcome=${e.outcome}, changes=[${chs}]`));
    }
  }

  return blocks;
}

function renderSeed(seedDir, summaryNote) {
  const conv = JSON.parse(fs.readFileSync(path.join(seedDir, 'conversation_log_run1.json'), 'utf-8'));
  const opt = JSON.parse(fs.readFileSync(path.join(seedDir, 'optimization_log_run1.json'), 'utf-8'));

  const optByRound = {};
  for (const r of (opt.rounds || [])) {
    const k = r.round;
    if (!optByRound[k]) optByRound[k] = [];
    optByRound[k].push(r);
  }

  const final = opt.final_summary || {};
  const rs = final.retry_stats || {};

  const children = [];

  const seedName = path.basename(seedDir);
  children.push(new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({
      text: `LLM conversation transcript — ${EXP} / ${seedName}`,
      bold: true, size: 32,
    })],
  }));

  if (summaryNote) {
    children.push(p(summaryNote, { run: { italics: true, size: 22, color: '404040' } }));
  }

  children.push(pRich([
    { text: 'Final summary: ', bold: true, size: 20 },
    { text: `best_val_loss = ${fmt(final.best_val_loss)} at round ${final.best_round}, ` +
            `rounds_completed = ${final.rounds_completed}, rounds_skipped = ${final.rounds_skipped}, ` +
            `retries = ${rs.retries ?? 0}, fallbacks = ${rs.fallbacks ?? 0}.`, size: 20 },
  ]));

  // System prompt — once
  if (conv.length > 0) {
    children.push(h1('System prompt (identical for every round)'));
    children.push(code(conv[0].llm_input.system_prompt.trim()));
  }

  // Per-round
  for (const entry of conv) {
    const rnum = entry.round;
    children.push(new Paragraph({ pageBreakBefore: true, children: [] }));
    children.push(h1(`Round ${rnum}`));
    children.push(p(`Timestamp: ${entry.timestamp || '—'}`, { run: { italics: true, size: 18, color: '606060' } }));

    const rows = optByRound[rnum] || [];
    const skipped = rows.find(r => r.skipped);
    const applied = rows.find(r => !r.skipped);

    const rawOut = (entry.llm_raw_output || '').trim();
    const payload = entry.llm_input?.user_payload || {};
    const isFallbackRecord = rawOut === '' && Object.keys(payload).length === 0;

    if (isFallbackRecord && skipped) {
      children.push(quote(
        'This round was rejected by the validator. The conversation log entry is the ' +
        'post-rejection fallback (empty input + empty output by design). The rejected ' +
        'proposal is captured in the optimisation log below.'
      ));
    } else {
      children.push(h2('LLM input (user message)'));
      for (const block of renderUserPayload(payload)) children.push(block);

      children.push(h2('LLM raw output'));
      children.push(code(rawOut));
    }

    children.push(h2('Validator + applied result'));
    if (skipped) {
      children.push(bulletRich([
        { text: 'REJECTED by validator', bold: true, color: 'C00000' },
        { text: ` — ${skipped.failure_reason || '—'}` },
      ]));
      children.push(bullet('→ fallback: round skipped, no hyperparameter change applied.'));
    }
    if (applied) {
      const ch = applied.changes_applied || {};
      const chs = Object.keys(ch).length
        ? Object.entries(ch).map(([k, v]) => `${k}=${v}`).join(', ')
        : 'no change';
      children.push(bullet(`changes_applied: ${chs}`));
      children.push(bullet(
        `val_loss after this round: ${fmt(applied.val_loss)} ` +
        `(outcome = ${applied.outcome}, Δ best_val_loss = ${fmt(applied.delta_best_val_loss)})`
      ));
    }
  }

  return children;
}

function buildDoc(children) {
  return new Document({
    styles: {
      default: { document: { run: { font: 'Calibri', size: 20 } } },
      paragraphStyles: [
        { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { size: 26, bold: true, font: 'Calibri', color: '1F3864' },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
        { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { size: 22, bold: true, font: 'Calibri', color: '2E75B6' },
          paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 } },
        { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { size: 20, bold: true, font: 'Calibri', color: '404040' },
          paragraph: { spacing: { before: 120, after: 40 }, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: 'bullets',
          levels: [{ level: 0, format: LevelFormat.BULLET, text: '•',
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
        },
      },
      children,
    }],
  });
}

(async () => {
  for (const s of SEEDS) {
    const seedDir = path.join(ROOT, EXP, s.name);
    const children = renderSeed(seedDir, s.summary);
    const doc = buildDoc(children);
    const out = path.join(ROOT, `transcript_${EXP}_${s.name}.docx`);
    const buf = await Packer.toBuffer(doc);
    fs.writeFileSync(out, buf);
    console.log('WROTE', out, buf.length, 'bytes');
  }
})();
