import { evaluate, spectrum, thetaInGlass } from "./model.js";
import { xyzFromR, labFromXyz, deltaE00, srgbFromXyz } from "./color.js";
import { el, text, frame, path, niceTicks, scale } from "./charts.js";

const ANGLES = [0, 15, 30, 45, 60];
const SWEEP = Array.from({ length: 16 }, (_, i) => i * 5); // 0..75
const GRAYS = { 0: "#16181a", 15: "#454a51", 30: "#6b7179", 45: "#9aa0a7", 60: "#bfc4c9" };
const MATS = ["TiO2", "SiO2", "Si3N4"];

const $ = (id) => document.getElementById(id);
const m = (v, d = 1) => fmt(v, d).replace("-", "−");
const fmt = (v, d = 1) => (Object.is(Math.round(v * 10 ** d) / 10 ** d, -0) ? 0 : v).toFixed(d);

const data = await (await fetch("data/optics.json")).json();
const lam = data.lambda;
const iVis0 = data.color.visStart, iVis1 = iVis0 + data.color.visCount - 1;

const state = { preset: 0, layers: clone(data.presets[0].layers), angle: 30 };
function clone(x) { return JSON.parse(JSON.stringify(x)); }
const nAt550 = (mat) => data.nk[mat][lam.indexOf(550)][0];

// ---------------------------------------------------------------- 입력부
function renderPresets() {
  const box = $("presets");
  box.replaceChildren();
  data.presets.forEach((p, i) => {
    const s = p.summary;
    const chip = srgbFromXyz(labToXyzApprox(s.lab[0])).hex;
    const lab = document.createElement("label");
    lab.innerHTML = `<input type="radio" name="preset" value="${i}" ${i === state.preset ? "checked" : ""}>
      <span class="chip" style="background:${chip}"></span>
      <span>${p.name}</span>
      <span class="meta">L* ${fmt(s.lab[0][0], 0)}  광전류 ${fmt(s.jsc_rel * 100, 1)}%</span>`;
    lab.querySelector("input").addEventListener("change", () => {
      state.preset = i; state.layers = clone(p.layers); renderStack(); schedule();
    });
    box.appendChild(lab);
  });
}

// 프리셋 칩 색은 요약 Lab 에서 역산한다(스펙트럼 재계산 없이).
function labToXyzApprox([L, a, b]) {
  const w = data.color.white;
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const inv = (t) => (t ** 3 > (6 / 29) ** 3 ? t ** 3 : 3 * (6 / 29) ** 2 * (t - 4 / 29));
  return [w[0] * inv(fx), w[1] * inv(fy), w[2] * inv(fz)];
}

function renderStack() {
  const tb = $("stack-rows");
  tb.replaceChildren();
  const fixedRow = (label, d, n) => {
    const tr = document.createElement("tr");
    tr.className = "fixed";
    tr.innerHTML = `<td></td><td>${label}</td><td class="n">${d}</td><td class="n">${n}</td><td></td>`;
    return tr;
  };
  tb.appendChild(fixedRow("커버유리", "3.2 mm", data.module.n_glass.toFixed(2)));
  state.layers.forEach((layer, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="n">${i + 1}</td>
      <td><select aria-label="${i + 1}층 재료">${MATS.map((m) => `<option value="${m}" ${m === layer.mat ? "selected" : ""}>${data.materials[m].label}</option>`).join("")}</select></td>
      <td class="n"><input type="number" min="0" max="600" step="0.5" value="${layer.d}" aria-label="${i + 1}층 두께 nm"></td>
      <td class="n">${nAt550(layer.mat).toFixed(2)}</td>
      <td><button class="del" type="button" aria-label="${i + 1}층 삭제">삭제</button></td>`;
    tr.querySelector("select").addEventListener("change", (e) => { layer.mat = e.target.value; renderStack(); schedule(); });
    tr.querySelector("input").addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      if (Number.isFinite(v) && v >= 0) { layer.d = v; schedule(); }
    });
    tr.querySelector("button").addEventListener("click", () => { state.layers.splice(i, 1); renderStack(); schedule(); });
    tb.appendChild(tr);
  });
  tb.appendChild(fixedRow("EVA", "0.45 mm", data.module.n_eva.toFixed(2)));
  tb.appendChild(fixedRow("SiNₓ 반사방지막", "75", nAt550("Si3N4").toFixed(2)));
  tb.appendChild(fixedRow("c-Si 셀", "∞", nAt550("Si").toFixed(2)));
}

$("add-layer").addEventListener("click", () => {
  const last = state.layers.at(-1);
  state.layers.push({ mat: last?.mat === "TiO2" ? "SiO2" : "TiO2", d: 80 });
  renderStack(); schedule();
});
$("reset").addEventListener("click", () => { state.layers = clone(data.presets[state.preset].layers); renderStack(); schedule(); });
$("angle").addEventListener("input", (e) => { state.angle = +e.target.value; $("angle-out").value = `${state.angle}°`; schedule(); });

let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; update(); });
}

// ---------------------------------------------------------------- 계산과 그림
function update() {
  const angles = [...new Set([...ANGLES, state.angle])].sort((a, b) => a - b);
  const ev = evaluate(state.layers, angles, data);
  const byAngle = Object.fromEntries(ev.angles.map((a) => [a.thetaAir, a]));
  const sweep = SWEEP.map((a) => {
    const R = a in byAngle ? byAngle[a].R : spectrum(state.layers, a, data).R;
    const lab = labFromXyz(xyzFromR(R, data), data.color.white);
    return { a, lab, dE: deltaE00(byAngle[0].lab, lab) };
  });
  drawSpectrum(ev, byAngle);
  drawSwatches(byAngle);
  drawAB(sweep);
  drawDE(sweep);
  drawSection();
  drawTrade(ev);
  drawRefract();
  renderReadout(ev, byAngle, sweep);
}

function drawSpectrum(ev, byAngle) {
  const f = frame($("fig-spectrum"), {
    width: 760, height: 280, margin: { l: 44, r: 64, t: 22, b: 40 },
    x: [lam[0], lam.at(-1)], y: [0, 0.4],
    xTicks: [400, 500, 600, 700, 800, 900, 1000, 1100], yTicks: [0, 0.1, 0.2, 0.3, 0.4],
    xLabel: "파장 λ (nm)", yLabel: "반사율 R", yFmt: (v) => v.toFixed(1),
  });
  let yMax = 0.4;
  for (const a of ev.angles) for (const r of a.R) yMax = Math.max(yMax, r);
  if (yMax > 0.4) {
    const top = Math.ceil(yMax * 10) / 10;
    return drawSpectrumScaled(ev, byAngle, top);
  }
  paintSpectrum(f, ev, byAngle);
}
function drawSpectrumScaled(ev, byAngle, top) {
  const f = frame($("fig-spectrum"), {
    width: 760, height: 280, margin: { l: 44, r: 64, t: 22, b: 40 },
    x: [lam[0], lam.at(-1)], y: [0, top],
    xTicks: [400, 500, 600, 700, 800, 900, 1000, 1100], yTicks: niceTicks(0, top, 5),
    xLabel: "파장 λ (nm)", yLabel: "반사율 R", yFmt: (v) => v.toFixed(1),
  });
  paintSpectrum(f, ev, byAngle);
}
function paintSpectrum(f, ev, byAngle) {
  const { g, x, y, H } = f;
  const top = y.domain ? y.domain[1] : 0.4;
  // 배경: AM1.5G 광자속, 가시역 표시, V(λ)
  const ph = data.photon, phMax = Math.max(...ph);
  const area = [[x(lam[0]), H], ...lam.map((l, j) => [x(l), y((ph[j] / phMax) * top * 0.92)]), [x(lam.at(-1)), H]];
  el("path", { d: path(area) + "Z", fill: "#eef0f2" }, g.firstChild ? g.insertBefore(el("g"), g.firstChild) : g);
  el("rect", { x: x(380), y: H + 0.5, width: x(780) - x(380), height: 3, fill: "var(--ink-3)", opacity: 0.35 }, g);
  const vl = data.display.vlambda_d65;
  el("path", {
    d: path(vl.map((v, k) => [x(lam[iVis0 + k]), y(v * top * 0.92)])),
    fill: "none", stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "2 3",
  }, g);
  text(g, x(640), y(top * 0.5), "D65×ȳ", { class: "label" });
  text(g, x(1000), y(top * 0.62), "AM1.5G 광자속", { class: "label", "text-anchor": "middle", fill: "var(--ink-3)" });
  const ends = [];
  for (const a of ANGLES) {
    if (a === state.angle) continue;
    const R = byAngle[a].R;
    el("path", { d: path(lam.map((l, j) => [x(l), y(R[j])])), fill: "none", stroke: GRAYS[a], "stroke-width": 1 }, g);
    ends.push({ y: y(R.at(-1)), label: `${a}°`, fill: GRAYS[a] });
  }
  const sel = byAngle[state.angle];
  el("path", { d: path(lam.map((l, j) => [x(l), y(sel.R[j])])), fill: "none", stroke: "var(--focus)", "stroke-width": 1.8 }, g);
  ends.push({ y: y(sel.R.at(-1)), label: `${state.angle}°`, fill: "var(--focus)", sel: true });
  ends.sort((p, q) => p.y - q.y);
  for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 11);
  const shift = Math.max(0, ends.at(-1).y - (H - 2));
  for (const e of ends) {
    e.y -= shift;
    text(g, x(lam.at(-1)) + 6, e.y + 3.5, e.label, { class: "num", fill: e.fill, "font-weight": e.sel ? 500 : null });
  }
  // 가시역 평균 반사율 표시
  let s = 0;
  for (let j = iVis0; j <= iVis1; j++) s += sel.R[j];
  text(g, f.W, -10, `가시역 평균 R ${(100 * s / (iVis1 - iVis0 + 1)).toFixed(1)}% (${state.angle}°)`, { class: "label", fill: "var(--ink)", "text-anchor": "end" });
}

function drawSwatches(byAngle) {
  const box = $("swatches");
  box.replaceChildren();
  for (const a of ANGLES) {
    const r = byAngle[a];
    const d = document.createElement("div");
    d.className = "sw";
    d.innerHTML = `<div class="patch${a === state.angle ? " sel" : ""}" style="background:${r.swatch.hex}">${r.swatch.mapped ? '<span class="flag">*</span>' : ""}</div>
      <div class="cap">${a}° <span>유리 ${fmt(r.thetaGlass, 1)}°</span><br>L* ${m(r.lab[0])}<br>a* ${m(r.lab[1])}  b* ${m(r.lab[2])}<br>ΔE00 ${fmt(r.dE, 2)}</div>`;
    box.appendChild(d);
  }
}

function drawAB(sweep) {
  // 데이터 중심, 가로세로 같은 축척(색차 거리가 왜곡되지 않게)
  const as = sweep.map((s) => s.lab[1]), bs = sweep.map((s) => s.lab[2]);
  const cxv = (Math.min(...as) + Math.max(...as)) / 2, cyv = (Math.min(...bs) + Math.max(...bs)) / 2;
  const half = Math.max(8, (Math.max(Math.max(...as) - Math.min(...as), Math.max(...bs) - Math.min(...bs)) / 2) * 1.25);
  const step = half > 20 ? 10 : 5;
  const xr = [Math.floor((cxv - half) / step) * step, Math.ceil((cxv + half) / step) * step];
  const yr = [Math.floor((cyv - half) / step) * step, Math.ceil((cyv + half) / step) * step];
  const span = Math.max(xr[1] - xr[0], yr[1] - yr[0]);
  xr[1] = xr[0] + span; yr[1] = yr[0] + span;
  const f = frame($("fig-ab"), {
    width: 360, height: 340, margin: { l: 40, r: 16, t: 22, b: 40 },
    x: xr, y: yr, xTicks: niceTicks(xr[0], xr[1], 4), yTicks: niceTicks(yr[0], yr[1], 4), xLabel: "a*", yLabel: "b*", grid: false,
  });
  const { g, x, y } = f;
  const L = span;
  if (xr[0] < 0 && xr[1] > 0) el("line", { x1: x(0), x2: x(0), y1: y(yr[0]), y2: y(yr[1]), stroke: "var(--rule)" }, g);
  if (yr[0] < 0 && yr[1] > 0) el("line", { x1: x(xr[0]), x2: x(xr[1]), y1: y(0), y2: y(0), stroke: "var(--rule)" }, g);
  el("path", { d: path(sweep.map((s) => [x(s.lab[1]), y(s.lab[2])])), fill: "none", stroke: "var(--ink-2)", "stroke-width": 1 }, g);
  const placed = [];
  for (const s of sweep) {
    const labelled = s.a % 15 === 0;
    const isSel = s.a === state.angle;
    const hex = srgbFromXyz(labToXyzApprox(s.lab)).hex;
    el("circle", { cx: x(s.lab[1]), cy: y(s.lab[2]), r: isSel ? 6 : labelled ? 4.5 : 2, fill: labelled || isSel ? hex : "var(--ink-2)", stroke: isSel ? "var(--focus)" : "var(--ink)", "stroke-width": isSel ? 2 : 0.75 }, g);
    const lx = x(s.lab[1]) + 8, ly = y(s.lab[2]) + 4;
    if ((labelled || isSel) && placed.every(([px, py]) => Math.abs(px - lx) > 26 || Math.abs(py - ly) > 11)) {
      placed.push([lx, ly]);
      text(g, lx, ly, `${s.a}°`, { class: "num", fill: isSel ? "var(--focus)" : null });
    }
  }
}

function drawDE(sweep) {
  const top = Math.max(6, Math.ceil(Math.max(...sweep.map((s) => s.dE)) / 2) * 2);
  const f = frame($("fig-de"), {
    width: 360, height: 340, margin: { l: 36, r: 16, t: 40, b: 40 },
    x: [0, 75], y: [0, top], xTicks: [0, 15, 30, 45, 60, 75], yTicks: niceTicks(0, top, 5),
    xLabel: "공기 기준 각도 θ (°)", yLabel: "ΔE00",
  });
  const { g, x, y, W } = f;
  for (const v of [1, 3, 5]) if (v < top) {
    el("line", { x1: 0, x2: W, y1: y(v), y2: y(v), stroke: "var(--ink-3)", "stroke-dasharray": "2 3", "stroke-width": 0.75 }, g);
  }
  // 위 축: 유리 내부 각도
  for (const tg of [10, 20, 30, 35]) {
    const ta = Math.asin(Math.min(1, Math.sin((tg * Math.PI) / 180) * data.module.n_glass)) * 180 / Math.PI;
    if (ta > 75) continue;
    el("line", { x1: x(ta), x2: x(ta), y1: 0, y2: 4, stroke: "var(--ink-3)", "stroke-width": 0.75 }, g);
    text(g, x(ta), -4, `${tg}`, { class: "num", "text-anchor": "middle" });
  }
  el("line", { x1: 0, x2: W, y1: 0, y2: 0, stroke: "var(--ink-3)", "stroke-width": 0.75 }, g);
  text(g, W, -20, "유리 내부 각도 (°)", { class: "label", "text-anchor": "end" });
  el("path", { d: path(sweep.map((s) => [x(s.a), y(s.dE)])), fill: "none", stroke: "var(--ink)", "stroke-width": 1.4 }, g);
  const sel = sweep.find((s) => s.a === state.angle);
  el("circle", { cx: x(sel.a), cy: y(sel.dE), r: 4, fill: "var(--paper)", stroke: "var(--focus)", "stroke-width": 2 }, g);
  text(g, x(sel.a) + 8, y(sel.dE) - 8, `${fmt(sel.dE, 2)}`, { class: "num", fill: "var(--focus)" });
}

function drawSection() {
  const svg = $("fig-section");
  svg.replaceChildren();
  const Wd = 360, labelX = 150;
  const total = state.layers.reduce((s, l) => s + l.d, 0);
  const pxPerNm = Math.min(0.55, 170 / Math.max(total, 1));
  const breakH = 26;
  let yCur = 10;
  const bands = [];
  const nToGray = (n) => { const t = Math.min(1, Math.max(0, (n - 1.4) / 1.2)); const v = Math.round(236 - t * 150); return `rgb(${v},${v},${v + 2})`; };
  bands.push({ label: "커버유리 3.2 mm", h: breakH, fill: "#eef2f4", brk: true, n: data.module.n_glass });
  state.layers.forEach((l, i) => bands.push({ label: `${data.materials[l.mat].label}  ${fmt(l.d, 1)} nm`, h: Math.max(1.5, l.d * pxPerNm), fill: nToGray(nAt550(l.mat)), n: nAt550(l.mat), idx: i + 1 }));
  bands.push({ label: "EVA 0.45 mm", h: breakH, fill: "#f3f3f1", brk: true, n: data.module.n_eva });
  bands.push({ label: "SiNₓ 75 nm", h: 75 * pxPerNm, fill: nToGray(nAt550("Si3N4")), n: nAt550("Si3N4") });
  bands.push({ label: "c-Si", h: 22, fill: "#3a3f45", n: nAt550("Si") });
  const H = 20 + bands.reduce((s, b) => s + b.h, 0) + 24;
  svg.setAttribute("viewBox", `0 0 ${Wd} ${H}`);
  const x0 = 12, bw = 120;
  for (const b of bands) {
    el("rect", { x: x0, y: yCur, width: bw, height: b.h, fill: b.fill, stroke: "var(--ink-3)", "stroke-width": 0.5 }, svg);
    if (b.brk) {
      const my = yCur + b.h / 2;
      el("path", { d: `M${x0 - 4},${my + 3} l8,-6 M${x0 + bw - 4},${my + 3} l8,-6`, stroke: "var(--ink-2)", "stroke-width": 1 }, svg);
    }
    const ly = yCur + b.h / 2 + 3.5;
    el("line", { x1: x0 + bw, x2: labelX - 4, y1: yCur + b.h / 2, y2: yCur + b.h / 2, stroke: "var(--rule)", "stroke-width": 0.75 }, svg);
    text(svg, labelX, ly, (b.idx ? `${b.idx}  ` : "") + b.label, { class: b.idx ? "num" : "label", fill: "var(--ink)" });
    text(svg, Wd - 4, ly, `n ${b.n.toFixed(2)}`, { class: "num", "text-anchor": "end" });
    yCur += b.h;
  }
  // 스케일 막대
  const sb = 100 * pxPerNm;
  el("line", { x1: x0, x2: x0, y1: H - 16 - sb, y2: H - 16, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
  text(svg, x0 + 6, H - 16, "100 nm (코팅 축척)", { class: "label" });
}

function drawTrade(ev) {
  const cloud = data.cloud ?? [];
  $("cloud-n").textContent = cloud.length;
  const L0 = ev.angles[0].lab[0], J = ev.jscRel;
  const xs = [L0, ...cloud.map((c) => c.L), ...data.presets.map((p) => p.summary.lab[0][0])];
  const ysv = [J, ...cloud.map((c) => c.j), ...data.presets.map((p) => p.summary.jsc_rel)];
  const x0 = Math.floor(Math.min(...xs) / 10) * 10, x1 = Math.ceil(Math.max(...xs) / 10) * 10;
  const y0 = Math.floor(Math.min(...ysv) * 20) / 20, y1 = 1.0;
  const f = frame($("fig-trade"), {
    width: 360, height: 300, margin: { l: 44, r: 16, t: 22, b: 40 },
    x: [x0, x1], y: [y0, y1], xTicks: niceTicks(x0, x1, 5), yTicks: niceTicks(y0, y1, 5),
    xLabel: "0° 반사색 명도 L*", yLabel: "상대 광전류", yFmt: (v) => v.toFixed(2),
  });
  const { g, x, y } = f;
  for (const c of cloud) el("circle", { cx: x(c.L), cy: y(c.j), r: 1.8, fill: "var(--ink-3)", opacity: 0.45 }, g);
  data.presets.forEach((p) => {
    const px = x(p.summary.lab[0][0]), py = y(p.summary.jsc_rel);
    el("rect", { x: px - 4.5, y: py - 4.5, width: 9, height: 9, fill: srgbFromXyz(labToXyzApprox(p.summary.lab[0])).hex, stroke: "var(--ink)", "stroke-width": 0.75 }, g);
    text(g, px + 8, py + 3.5, p.name, { class: "label" });
  });
  el("circle", { cx: x(L0), cy: y(J), r: 5, fill: "none", stroke: "var(--focus)", "stroke-width": 2 }, g);
}

function drawRefract() {
  const svg = $("fig-refract");
  svg.replaceChildren();
  const W = 360, H = 260, cx = 180, cy = 120;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el("rect", { x: 0, y: cy, width: W, height: 90, fill: "#eef2f4" }, svg);
  el("rect", { x: 0, y: cy + 90, width: W, height: 6, fill: "#9aa1a8" }, svg);
  el("line", { x1: 0, x2: W, y1: cy, y2: cy, stroke: "var(--ink-2)", "stroke-width": 1 }, svg);
  el("line", { x1: cx, x2: cx, y1: 10, y2: cy + 96, stroke: "var(--ink-3)", "stroke-dasharray": "2 3", "stroke-width": 0.75 }, svg);
  const ta = (state.angle * Math.PI) / 180, tg = (thetaInGlass(state.angle, data) * Math.PI) / 180;
  const Lin = 105, Lg = 90 / Math.cos(tg);
  const x1 = cx - Lin * Math.sin(ta), y1 = cy - Lin * Math.cos(ta);
  const x2 = cx + Lg * Math.sin(tg), y2 = cy + 90;
  el("line", { x1, y1, x2: cx, y2: cy, stroke: "var(--ink)", "stroke-width": 1.4 }, svg);
  el("line", { x1: cx, y1: cy, x2, y2, stroke: "var(--focus)", "stroke-width": 1.6 }, svg);
  const arc = (r, a0, a1, up) => {
    const s = up ? -1 : 1;
    const p0 = [cx + r * Math.sin(a0) * (up ? -1 : 1), cy + s * r * Math.cos(a0)];
    const p1 = [cx + r * Math.sin(a1) * (up ? -1 : 1), cy + s * r * Math.cos(a1)];
    return `M${p0[0]},${p0[1]} A${r},${r} 0 0 ${up ? 0 : 0} ${p1[0]},${p1[1]}`;
  };
  if (state.angle > 0) {
    el("path", { d: arc(34, 0, ta, true), fill: "none", stroke: "var(--ink-2)", "stroke-width": 0.75 }, svg);
    el("path", { d: arc(40, 0, tg, false), fill: "none", stroke: "var(--focus)", "stroke-width": 0.75 }, svg);
  }
  text(svg, cx + 10, cy - 30, `θ공기 ${state.angle}°`, { class: "num", fill: "var(--ink)" });
  text(svg, cx - 10, cy + 58, `θ유리 ${fmt(thetaInGlass(state.angle, data), 1)}°`, { class: "num", fill: "var(--focus)", "text-anchor": "end" });
  text(svg, 8, cy - 8, "공기 n = 1.00", { class: "label" });
  text(svg, 8, cy + 18, `커버유리 n = ${data.module.n_glass.toFixed(2)}`, { class: "label" });
  text(svg, 8, cy + 86, "코팅 (유리 뒷면)", { class: "label", fill: "var(--ink)" });
}

function renderReadout(ev, byAngle, sweep) {
  const a0 = byAngle[0], as = byAngle[state.angle];
  const maxDe = Math.max(...sweep.filter((s) => s.a <= 60).map((s) => s.dE));
  const C0 = Math.hypot(a0.lab[1], a0.lab[2]);
  const h0 = ((Math.atan2(a0.lab[2], a0.lab[1]) * 180) / Math.PI + 360) % 360;
  const rows = [
    ["0° L* C* h", `${fmt(a0.lab[0])}  ${fmt(C0)}  ${fmt(h0, 0)}°`],
    [`${state.angle}° 대비 ΔE00`, fmt(as.dE, 2)],
    ["ΔE00 최대 (0–60°)", fmt(maxDe, 2)],
    ["상대 광전류", `${fmt(ev.jscRel * 100, 1)}%`],
    ["코팅 총두께", `${fmt(state.layers.reduce((s, l) => s + l.d, 0), 1)} nm`],
  ];
  $("readout").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

// ---------------------------------------------------------------- 정적 부분
function renderLead() {
  const ps = data.presets;
  const js = ps.map((p) => p.summary.jsc_rel * 100);
  const de60 = ps.map((p) => p.summary.dE00.at(-1));
  $("lead").innerHTML = `세 프리셋(${ps.map((p) => p.name).join(", ")})은 코팅 없는 모듈 대비 광전류를 <strong class="num">${fmt(Math.min(...js), 1)}–${fmt(Math.max(...js), 1)}%</strong> 유지한다. 60°에서 본 색은 0° 대비 ΔE00 <strong class="num">${fmt(Math.min(...de60), 1)}–${fmt(Math.max(...de60), 1)}</strong>만큼 달라진다. 평면 유리에서는 밝은 색일수록 광전류 손실이 크고(그림 6), 각도에 따른 색 이동은 층 구성에 따라 크게 다르다(그림 4). 관찰 각도는 공기 기준으로 입력하며, 유리 뒷면 코팅이 실제로 받는 각도는 3절에서 설명한다.`;
}

function renderCond() {
  const m = data.materials;
  const rows = [
    ["적층", "공기 | 커버유리 3.2 mm | 코팅 | EVA 0.45 mm | SiNₓ 75 nm | c-Si"],
    ["인코히어런트 층", "커버유리, EVA (두께가 가간섭 길이보다 커서 강도로 합산)"],
    ["편광", "s, p 평균 (비편광 조명)"],
    ["파장", `${lam[0]}–${lam.at(-1)} nm, 5 nm 간격. 색은 380–780 nm`],
    ["광원과 관찰자", "CIE D65, CIE 1931 2° (colour-science 적분 가중치)"],
    ["색차", "CIEDE2000, 0° 반사색 기준"],
    ["광전류", "AM1.5G(ASTM G173-03) 광자속 × Si 유입 투과율, 1180 nm까지, EQE = 1"],
    ["굴절률", `유리 1.52, EVA 1.48 (분산 무시). ${Object.values(m).map((v) => `${v.label}: ${v.source}`).join(". ")}`],
    ["검증", `<span id="verify-cell">웹 계산을 Python tmm/colour-science 결과와 대조 (tests/tmm.test.mjs)</span>`],
  ];
  $("cond-table").innerHTML = rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("");
}

renderLead();
renderCond();
renderPresets();
renderStack();
update();
document.documentElement.dataset.ready = "1";
