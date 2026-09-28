// physics/optics.py 의 layer_lists / spectrum / evaluate 를 웹에서 같은 규약으로 계산한다.
import { incTmm, complex } from "./tmm.js";
import { xyzFromR, labFromXyz, deltaE00, srgbFromXyz } from "./color.js";

const { C } = complex;
const deg = Math.PI / 180;

export function thetaInGlass(thetaAir, data) {
  return Math.asin(Math.sin(thetaAir * deg) / data.module.n_glass) / deg;
}

function layerLists(coating, j, data) {
  const m = data.module, nk = data.nk;
  const n = [C(1), C(m.n_glass)];
  const d = [Infinity, m.glass_d];
  const c = ["i", "i"];
  for (const layer of coating) {
    const [re, im] = nk[layer.mat][j];
    n.push(C(re, im)); d.push(layer.d); c.push("c");
  }
  n.push(C(m.n_eva), C(...nk.Si3N4[j]), C(...nk.Si[j]));
  d.push(m.eva_d, m.arc_d, Infinity);
  c.push("i", "c", "i");
  return [n, d, c];
}

export function spectrum(coating, thetaAir, data) {
  const lam = data.lambda;
  const R = new Float64Array(lam.length), T = new Float64Array(lam.length);
  const th = C(thetaAir * deg);
  for (let j = 0; j < lam.length; j++) {
    const [n, d, c] = layerLists(coating, j, data);
    const s = incTmm("s", n, d, c, th, lam[j]);
    const p = incTmm("p", n, d, c, th, lam[j]);
    R[j] = 0.5 * (s.R + p.R);
    T[j] = 0.5 * (s.T + p.T);
  }
  return { R, T };
}

export function jscOf(T, data) {
  let s = 0;
  const w = data.photon;
  for (let j = 0; j < w.length; j++) s += w[j] * T[j];
  return s;
}

export function evaluate(coating, angles, data) {
  const refJ = data.reference_jsc;
  const out = { angles: [], jscRel: null };
  let lab0 = null;
  for (const a of angles) {
    const { R, T } = spectrum(coating, a, data);
    const xyz = xyzFromR(R, data);
    const lab = labFromXyz(xyz, data.color.white);
    if (!lab0) { lab0 = lab; out.jscRel = jscOf(T, data) / refJ; }
    out.angles.push({
      thetaAir: a, thetaGlass: thetaInGlass(a, data), R, T, xyz, lab,
      dE: deltaE00(lab0, lab), swatch: srgbFromXyz(xyz),
    });
  }
  return out;
}
