// 전달행렬법(TMM). Byrnes `tmm` 패키지의 coh_tmm / inc_tmm 과 같은 규약으로 포팅했다.
// physics/optics.py 가 기준 구현이고, tests/tmm.test.mjs 가 golden 값과 대조한다.

const EPS = 1e-12;

// ---------------------------------------------------------------- 복소수
const C = (re, im = 0) => ({ re, im });
const add = (a, b) => C(a.re + b.re, a.im + b.im);
const sub = (a, b) => C(a.re - b.re, a.im - b.im);
const mul = (a, b) => C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const div = (a, b) => {
  const s = b.re * b.re + b.im * b.im;
  return C((a.re * b.re + a.im * b.im) / s, (a.im * b.re - a.re * b.im) / s);
};
const conj = (a) => C(a.re, -a.im);
const abs2 = (a) => a.re * a.re + a.im * a.im;
const cexp = (a) => {
  const e = Math.exp(a.re);
  return C(e * Math.cos(a.im), e * Math.sin(a.im));
};
const csqrt = (a) => {
  const r = Math.hypot(a.re, a.im);
  const re = Math.sqrt((r + a.re) / 2);
  const im = Math.sqrt(Math.max(0, (r - a.re) / 2));
  return C(re, a.im < 0 ? -im : im);
};
const clog = (a) => C(Math.log(Math.hypot(a.re, a.im)), Math.atan2(a.im, a.re));
const csin = (a) => C(Math.sin(a.re) * Math.cosh(a.im), Math.cos(a.re) * Math.sinh(a.im));
const ccos = (a) => C(Math.cos(a.re) * Math.cosh(a.im), -Math.sin(a.re) * Math.sinh(a.im));
// asin z = -i ln(i z + sqrt(1 - z^2))
const casin = (z) => {
  const iz = C(-z.im, z.re);
  const w = clog(add(iz, csqrt(sub(C(1), mul(z, z)))));
  return C(w.im, -w.re);
};

// 전진파(감쇠 또는 에너지가 +z로 흐르는) 각도인지. Byrnes is_forward_angle 과 같다.
function isForward(n, th) {
  const ncos = mul(n, ccos(th));
  if (Math.abs(ncos.im) > 100 * EPS) return ncos.im > 0;
  return ncos.re > 0;
}

function snellList(nList, th0) {
  const s0 = mul(nList[0], csin(th0));
  // Byrnes list_snell 과 같이 양 끝 매질만 전진파 방향으로 고정한다(중간층은 주값 그대로).
  return nList.map((n, i) => {
    let th = casin(div(s0, n));
    if ((i === 0 || i === nList.length - 1) && !isForward(n, th)) th = sub(C(Math.PI), th);
    return th;
  });
}

function interfaceR(pol, ni, nf, thi, thf) {
  const ci = ccos(thi), cf = ccos(thf);
  if (pol === "s") return div(sub(mul(ni, ci), mul(nf, cf)), add(mul(ni, ci), mul(nf, cf)));
  return div(sub(mul(nf, ci), mul(ni, cf)), add(mul(nf, ci), mul(ni, cf)));
}
function interfaceT(pol, ni, nf, thi, thf) {
  const ci = ccos(thi), cf = ccos(thf);
  const num = mul(C(2), mul(ni, ci));
  if (pol === "s") return div(num, add(mul(ni, ci), mul(nf, cf)));
  return div(num, add(mul(nf, ci), mul(ni, cf)));
}

const m2 = (a, b) => [
  [add(mul(a[0][0], b[0][0]), mul(a[0][1], b[1][0])), add(mul(a[0][0], b[0][1]), mul(a[0][1], b[1][1]))],
  [add(mul(a[1][0], b[0][0]), mul(a[1][1], b[1][0])), add(mul(a[1][0], b[0][1]), mul(a[1][1], b[1][1]))],
];

/** 코히어런트 스택. nList: 복소 굴절률 배열, dList: 두께(nm, 양 끝 Infinity), th0: 복소 입사각(rad). */
export function cohTmm(pol, nList, dList, th0, lam) {
  const N = nList.length;
  const th = snellList(nList, th0);
  const kz = nList.map((n, i) => div(mul(C(2 * Math.PI), mul(n, ccos(th[i]))), C(lam)));
  const delta = kz.map((k, i) => {
    if (i === 0 || i === N - 1) return C(0);
    const dl = mul(k, C(dList[i]));
    return dl.im > 35 ? C(dl.re, 35) : dl; // Byrnes와 같은 불투명층 상한
  });
  const r01 = interfaceR(pol, nList[0], nList[1], th[0], th[1]);
  const t01 = interfaceT(pol, nList[0], nList[1], th[0], th[1]);
  let M = [[div(C(1), t01), div(r01, t01)], [div(r01, t01), div(C(1), t01)]];
  for (let i = 1; i < N - 1; i++) {
    const r = interfaceR(pol, nList[i], nList[i + 1], th[i], th[i + 1]);
    const t = interfaceT(pol, nList[i], nList[i + 1], th[i], th[i + 1]);
    const em = cexp(mul(C(0, -1), delta[i]));
    const ep = cexp(mul(C(0, 1), delta[i]));
    const A = [[em, C(0)], [C(0), ep]];
    const B = [[div(C(1), t), div(r, t)], [div(r, t), div(C(1), t)]];
    M = m2(M, m2(A, B));
  }
  const r = div(M[1][0], M[0][0]);
  const t = div(C(1), M[0][0]);
  const ni = nList[0], nf = nList[N - 1];
  const ci = ccos(th[0]), cf = ccos(th[N - 1]);
  let T;
  if (pol === "s") T = (abs2(t) * mul(nf, cf).re) / mul(ni, ci).re;
  else T = (abs2(t) * mul(nf, conj(cf)).re) / mul(ni, conj(ci)).re;
  return { R: abs2(r), T, th };
}

/**
 * 코히어런트/인코히어런트 혼합 스택 (Byrnes inc_tmm 과 같은 결과).
 * cList: 'i' | 'c'. 처음과 끝은 'i' 여야 한다.
 */
export function incTmm(pol, nList, dList, cList, th0, lam) {
  const thAll = snellList(nList, th0);
  const inc = [];
  cList.forEach((c, i) => { if (c === "i") inc.push(i); });
  // 인코히어런트 층 사이의 부분 스택마다 전방/후방 R, T
  const Rf = [], Tf = [], Rb = [], Tb = [];
  for (let k = 0; k < inc.length - 1; k++) {
    const a = inc[k], b = inc[k + 1];
    const ns = nList.slice(a, b + 1);
    const ds = [Infinity, ...dList.slice(a + 1, b), Infinity];
    const f = cohTmm(pol, ns, ds, thAll[a], lam);
    const bk = cohTmm(pol, [...ns].reverse(), [...ds].reverse(), thAll[b], lam);
    Rf.push(f.R); Tf.push(f.T); Rb.push(bk.R); Tb.push(bk.T);
  }
  // 인코히어런트 내부층 1회 통과 강도 감쇠
  const P = inc.map((i, k) => {
    if (k === 0 || k === inc.length - 1) return 1;
    const kzIm = mul(nList[i], ccos(thAll[i])).im;
    return Math.max(1e-30, Math.exp((-4 * Math.PI * dList[i] * kzIm) / lam));
  });
  const Lint = (k) => [
    [1 / Tf[k], -Rb[k] / Tf[k]],
    [Rf[k] / Tf[k], (Tf[k] * Tb[k] - Rf[k] * Rb[k]) / Tf[k]],
  ];
  const mr = (a, b) => [
    [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
    [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
  ];
  let L = Lint(0);
  for (let k = 1; k < inc.length - 1; k++) {
    L = mr(L, mr([[1 / P[k], 0], [0, P[k]]], Lint(k)));
  }
  return { R: L[1][0] / L[0][0], T: 1 / L[0][0] };
}

export const complex = { C };
