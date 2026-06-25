const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";

const DARK = "21295C", BLUE = "065A82", TEAL = "1C7293";
const INK = "1F2937", MUTE = "6B7280", CARD = "EEF3F6";
const WHITE = "FFFFFF", ICE = "CADCFC";
const RED = "B5524B", GOOD = "2C7A6B";
const HF = "Georgia", BF = "Calibri";

function title(s, t) {
  s.addText(t, { x: 0.6, y: 0.42, w: 12.13, h: 0.85, fontFace: HF, fontSize: 30,
    bold: true, color: DARK, align: "left", valign: "middle" });
}

// ---- Slide 1 — Title (dark) ----------------------------------------------
let s = p.addSlide();
s.background = { color: DARK };
s.addText("THESIS PROGRESS REPORT  ·  SCIENTIFIC CORE", {
  x: 0.75, y: 1.25, w: 11, h: 0.4, fontFace: BF, fontSize: 14, bold: true,
  color: TEAL, charSpacing: 3 });
s.addText("Can an LLM use human-motion knowledge\nto improve neural-network training?", {
  x: 0.75, y: 1.85, w: 11.8, h: 2.5, fontFace: HF, fontSize: 38, bold: true,
  color: WHITE, lineSpacing: 46 });
s.addText("Testing whether an LLM can analyse interpretable motion features and "
  + "shape the training objective itself — and whether that beats a "
  + "deterministic rule-based controller.", {
  x: 0.75, y: 4.5, w: 10.6, h: 1.1, fontFace: BF, fontSize: 18, color: ICE,
  lineSpacing: 26 });
s.addText("Indoor localization  ·  radar dataset  ·  20 paired seeds  ·  "
  + "models: qwen3:8b, llama3.1:8b, phi4-mini:3.8b", {
  x: 0.75, y: 6.45, w: 12, h: 0.4, fontFace: BF, fontSize: 13, color: "8893B0" });

// ---- Slide 2 — What was built --------------------------------------------
s = p.addSlide();
s.background = { color: WHITE };
title(s, "What was built: motion-aware loss shaping");
s.addText([
  { text: "L  =  speed-bin-weighted MSE   +   ", options: {} },
  { text: "λ_vel", options: { bold: true, color: BLUE } },
  { text: " · velocity-plausibility penalty   +   ", options: {} },
  { text: "λ_smooth", options: { bold: true, color: BLUE } },
  { text: " · smoothness penalty", options: {} },
], { x: 0.6, y: 1.4, w: 12.13, h: 0.7, fill: { color: CARD }, fontFace: BF,
  fontSize: 15, italic: true, color: INK, align: "center", valign: "middle" });

const levers = [
  ["Velocity prior", "Penalises predicted speeds above a physical human "
    + "ceiling v_max, evaluated in real metres."],
  ["Smoothness prior", "Penalises implausible acceleration / jerk across "
    + "three consecutive trajectory points."],
  ["Speed-bin weighting", "Re-weights position error by the sample's motion "
    + "regime — slow / medium / fast terciles."],
];
levers.forEach((lv, i) => {
  const x = 0.6 + i * 4.12;
  s.addShape(p.ShapeType.rect, { x, y: 2.35, w: 3.89, h: 1.95, fill: { color: CARD },
    line: { color: CARD } });
  s.addShape(p.ShapeType.rect, { x, y: 2.35, w: 0.12, h: 1.95, fill: { color: TEAL },
    line: { color: TEAL } });
  s.addText(lv[0], { x: x + 0.25, y: 2.5, w: 3.5, h: 0.5, fontFace: HF,
    fontSize: 16, bold: true, color: BLUE });
  s.addText(lv[1], { x: x + 0.25, y: 2.95, w: 3.5, h: 1.25, fontFace: BF,
    fontSize: 12.5, color: INK, lineSpacing: 16 });
});
s.addText("Three controllers compared over the SAME six levers:", {
  x: 0.6, y: 4.55, w: 12, h: 0.4, fontFace: BF, fontSize: 14, bold: true,
  color: INK });
const ctrls = [
  ["C1", "Metric-only rule-based", "No motion access — the control."],
  ["C2", "Motion-aware rule-based", "Sets levers from fixed heuristics (v_max = p95 × 1.1)."],
  ["C3", "The LLM", "Sets the same levers by reasoning over the motion profile."],
];
ctrls.forEach((c, i) => {
  const x = 0.6 + i * 4.12;
  s.addShape(p.ShapeType.rect, { x, y: 5.05, w: 3.89, h: 1.55, fill: { color: WHITE },
    line: { color: BLUE, width: 1 } });
  s.addText(c[0], { x: x + 0.2, y: 5.18, w: 1.2, h: 0.55, fontFace: HF,
    fontSize: 22, bold: true, color: BLUE });
  s.addText(c[1], { x: x + 0.2, y: 5.68, w: 3.5, h: 0.4, fontFace: BF,
    fontSize: 13, bold: true, color: INK });
  s.addText(c[2], { x: x + 0.2, y: 6.02, w: 3.55, h: 0.5, fontFace: BF,
    fontSize: 11, color: MUTE, lineSpacing: 13 });
});

// ---- Slide 3 — Engagement ------------------------------------------------
s = p.addSlide();
s.background = { color: WHITE };
title(s, "Does the LLM actually use the motion levers?");
s.addText("Share of optimisation rounds in which the LLM proposed at least "
  + "one motion-aware loss-shaping change (radar, 100 rounds per model):", {
  x: 0.6, y: 1.35, w: 12.13, h: 0.55, fontFace: BF, fontSize: 14, color: MUTE });
const eng = [
  ["phi4-mini 3.8B", "0%", "Never engages — narrates the motion profile but only tunes lr / dropout.", RED],
  ["llama3.1:8b", "~2%", "Reads the data but defaults to generic values instead of acting on it.", "C08A2E"],
  ["qwen3:8b", "85%", "Engages and acts — reads observed speed, sets v_max from it.", GOOD],
];
eng.forEach((e, i) => {
  const x = 0.6 + i * 4.12;
  s.addShape(p.ShapeType.rect, { x, y: 2.05, w: 3.89, h: 2.45, fill: { color: CARD },
    line: { color: CARD } });
  s.addText(e[0], { x: x + 0.2, y: 2.2, w: 3.5, h: 0.4, fontFace: BF,
    fontSize: 13, bold: true, color: INK });
  s.addText(e[1], { x: x + 0.2, y: 2.5, w: 3.5, h: 1.0, fontFace: HF,
    fontSize: 54, bold: true, color: e[3] });
  s.addText(e[2], { x: x + 0.2, y: 3.6, w: 3.5, h: 0.8, fontFace: BF,
    fontSize: 11.5, color: MUTE, lineSpacing: 14 });
});
s.addShape(p.ShapeType.rect, { x: 0.6, y: 4.85, w: 12.13, h: 1.65, fill: { color: DARK },
  line: { color: DARK } });
s.addText("Data-grounded reasoning — when it engages", {
  x: 0.9, y: 5.0, w: 11.5, h: 0.45, fontFace: HF, fontSize: 16, bold: true,
  color: TEAL });
s.addText("qwen3:8b reads the dataset's observed speed (p95 = 0.60 m/s) and "
  + "sets v_max ≈ 0.63 — the physically correct ceiling for this slow indoor "
  + "motion, not a generic default. Engagement quality, however, is "
  + "model-dependent: llama3.1 cites the same number yet still picks a "
  + "generic value.", {
  x: 0.9, y: 5.45, w: 11.5, h: 0.95, fontFace: BF, fontSize: 13, color: ICE,
  lineSpacing: 17 });

// ---- Slide 4 — Results ---------------------------------------------------
s = p.addSlide();
s.background = { color: WHITE };
title(s, "Results: test RMSE across every controller");
const head = ["Controller", "Test RMSE (mean ± std)", "vs baseline"];
const rows = [
  ["Baseline — no optimisation", "0.2460 ± 0.012", "best", GOOD],
  ["C1 — metric-only rule-based", "0.2601 ± 0.007", "worse", RED],
  ["C2 — motion-aware rule-based", "0.2666 ± 0.010", "worse", RED],
  ["C3 — LLM qwen3:8b, motion-ON", "0.2891 ± 0.024", "worse", RED],
  ["C3 — LLM qwen3:8b, motion-OFF", "0.2899 ± 0.033", "worse", RED],
  ["C3 — LLM llama3.1:8b", "0.2466 ± 0.006", "≈ baseline", "C08A2E"],
  ["Random search", "~0.37", "much worse", RED],
];
const tbl = [[
  { text: "Controller", options: { bold: true, color: WHITE, fill: { color: DARK }, fontFace: BF } },
  { text: "Test RMSE (mean ± std)", options: { bold: true, color: WHITE, fill: { color: DARK }, align: "center", fontFace: BF } },
  { text: "vs baseline", options: { bold: true, color: WHITE, fill: { color: DARK }, align: "center", fontFace: BF } },
]];
rows.forEach((r, i) => {
  const bg = i % 2 ? "FFFFFF" : "EEF3F6";
  tbl.push([
    { text: r[0], options: { color: INK, fill: { color: bg }, fontFace: BF } },
    { text: r[1], options: { color: INK, fill: { color: bg }, align: "center", fontFace: BF } },
    { text: r[2], options: { color: r[3], bold: true, fill: { color: bg }, align: "center", fontFace: BF } },
  ]);
});
s.addTable(tbl, { x: 0.6, y: 1.45, w: 12.13, colW: [5.93, 3.6, 2.6],
  rowH: 0.52, fontSize: 13, valign: "middle", border: { type: "solid", color: "D5DEE5", pt: 1 } });
s.addText("Lower is better. n = 20 paired seeds per controller.", {
  x: 0.6, y: 5.7, w: 7, h: 0.35, fontFace: BF, fontSize: 11, italic: true, color: MUTE });
s.addText("No controller beats the no-optimisation baseline (0.246).", {
  x: 0.6, y: 6.1, w: 12.13, h: 0.6, fill: { color: CARD }, fontFace: HF,
  fontSize: 16, bold: true, color: BLUE, align: "center", valign: "middle" });

// ---- Slide 5 — Findings --------------------------------------------------
s = p.addSlide();
s.background = { color: WHITE };
title(s, "Four findings from the isolated ablations");
const finds = [
  ["1", "Motion loss-shaping is inert", "LLM scores 0.2891 with shaping ON and "
    + "0.2899 with it OFF — a 0.0008 gap, far inside the noise."],
  ["2", "A fixed motion heuristic fails too", "C2 (0.2666) is marginally worse "
    + "than C1 (0.2601). Motion shaping does not help the rule-based arm either."],
  ["3", "LLM reasoning ≤ fixed heuristic", "qwen3's reasoned motion shaping "
    + "(0.289) does not beat the fixed-formula C2 (0.267)."],
  ["4", "Nothing beats the baseline", "The no-optimisation baseline (0.246) "
    + "leads; the strongest LLM merely matches it."],
];
finds.forEach((f, i) => {
  const col = i % 2, rowi = Math.floor(i / 2);
  const x = 0.6 + col * 6.18, y = 1.5 + rowi * 2.5;
  s.addShape(p.ShapeType.rect, { x, y, w: 5.95, h: 2.25, fill: { color: CARD },
    line: { color: CARD } });
  s.addShape(p.ShapeType.ellipse, { x: x + 0.28, y: y + 0.28, w: 0.7, h: 0.7,
    fill: { color: BLUE }, line: { color: BLUE } });
  s.addText(f[0], { x: x + 0.28, y: y + 0.28, w: 0.7, h: 0.7, fontFace: HF,
    fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
  s.addText(f[1], { x: x + 1.2, y: y + 0.3, w: 4.5, h: 0.7, fontFace: HF,
    fontSize: 16, bold: true, color: BLUE, valign: "middle" });
  s.addText(f[2], { x: x + 0.35, y: y + 1.1, w: 5.3, h: 1.0, fontFace: BF,
    fontSize: 12.5, color: INK, lineSpacing: 16 });
});
s.addText("Final models are selected on prior-free real-metre position error, "
  + "so reported RMSE is never the shaped loss — loss-shaping can only help "
  + "indirectly, and here it did not.", {
  x: 0.6, y: 6.55, w: 12.13, h: 0.5, fontFace: BF, fontSize: 11.5, italic: true,
  color: MUTE, align: "center" });

// ---- Slide 6 — Conclusion (dark) -----------------------------------------
s = p.addSlide();
s.background = { color: DARK };
s.addText("Conclusion: a clean, isolated negative result", {
  x: 0.75, y: 0.6, w: 11.8, h: 0.8, fontFace: HF, fontSize: 30, bold: true,
  color: WHITE });
s.addText("The LLM can analyse motion features and propose data-grounded loss "
  + "shaping — qwen3:8b demonstrably reads the speed distribution and sets a "
  + "physically correct v_max. But neither the LLM's motion reasoning nor an "
  + "equivalent fixed heuristic improves localization, and the LLM as an "
  + "optimiser does not beat the fixed baseline. The contribution is the "
  + "rigour: a full C1 / C2 / C3 × motion-on/off × baseline / random "
  + "comparison that cleanly separates capability from effect.", {
  x: 0.75, y: 1.5, w: 11.8, h: 2.2, fontFace: BF, fontSize: 15, color: ICE,
  lineSpacing: 23 });
const closing = [
  ["Capability ≠ effect", "The LLM uses motion data; it just doesn't change the outcome."],
  ["Caveat", "Radar only — cap / IR datasets and the cross-dataset adaptivity test are pending."],
  ["Next step", "Run cap / IR; test selecting checkpoints on the shaped objective."],
];
closing.forEach((c, i) => {
  const y = 4.05 + i * 1.02;
  s.addShape(p.ShapeType.rect, { x: 0.75, y, w: 0.12, h: 0.8, fill: { color: TEAL },
    line: { color: TEAL } });
  s.addText(c[0], { x: 1.05, y, w: 3.0, h: 0.8, fontFace: HF, fontSize: 15,
    bold: true, color: TEAL, valign: "middle" });
  s.addText(c[1], { x: 4.1, y, w: 8.45, h: 0.8, fontFace: BF, fontSize: 13.5,
    color: ICE, valign: "middle", lineSpacing: 17 });
});

p.writeFile({ fileName: "Point3_Presentation.pptx" })
  .then(() => console.log("wrote Point3_Presentation.pptx"));
