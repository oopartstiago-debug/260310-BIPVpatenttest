// 의존성 없는 SVG 그림 도구. 축, 눈금, 선, 직접 라벨만 제공한다.
const NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
export function text(parent, x, y, str, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}

export const scale = (d0, d1, r0, r1) => {
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
};

export function niceTicks(a, b, n = 5) {
  const span = b - a;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? mag * 10;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** 빈 그림 틀. 반환: { svg, g(플롯 영역), x, y, W, H } */
export function frame(svg, { width, height, margin, x, y, xTicks, yTicks, xLabel, yLabel, xFmt = String, yFmt = String, grid = true }) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const m = margin;
  const W = width - m.l - m.r, H = height - m.t - m.b;
  const xs = scale(x[0], x[1], 0, W);
  const ys = scale(y[0], y[1], H, 0);
  const g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
  if (grid) {
    const gg = el("g", { class: "grid" }, g);
    for (const v of yTicks) el("line", { x1: 0, x2: W, y1: ys(v), y2: ys(v) }, gg);
  }
  const ax = el("g", { class: "axis" }, g);
  el("line", { x1: 0, x2: W, y1: H, y2: H }, ax);
  el("line", { x1: 0, x2: 0, y1: 0, y2: H }, ax);
  const tx = el("g", { class: "tick" }, g);
  for (const v of xTicks) {
    el("line", { x1: xs(v), x2: xs(v), y1: H, y2: H - 4, stroke: "var(--ink-3)", "stroke-width": 0.75 }, tx);
    text(tx, xs(v), H + 14, xFmt(v), { "text-anchor": "middle" });
  }
  for (const v of yTicks) {
    el("line", { x1: 0, x2: 4, y1: ys(v), y2: ys(v), stroke: "var(--ink-3)", "stroke-width": 0.75 }, tx);
    text(tx, -6, ys(v) + 3.5, yFmt(v), { "text-anchor": "end" });
  }
  if (xLabel) text(g, W, H + 30, xLabel, { "text-anchor": "end", class: "label" });
  if (yLabel) text(g, -m.l + 2, -10, yLabel, { class: "label" });
  return { svg, g, x: xs, y: ys, W, H };
}

export function path(points) {
  let d = "";
  points.forEach(([x, y], i) => { d += (i ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2); });
  return d;
}
