// 반사 스펙트럼 → XYZ(D65, CIE 1931 2°) → CIELAB → ΔE00, 그리고 화면 표시용 sRGB.
// 가중치 W는 physics/make_data.py 가 colour-science 로 만든 값을 그대로 쓴다(상수 하드코딩 금지).

export function xyzFromR(R, data) {
  const { W, visStart, visCount } = data.color;
  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let j = 0; j < visCount; j++) s += W[c][j] * R[visStart + j];
    out[c] = s;
  }
  return out;
}

export function labFromXyz(xyz, white) {
  const e = (6 / 29) ** 3;
  const f = (t) => (t > e ? Math.cbrt(t) : t / (3 * (6 / 29) ** 2) + 4 / 29);
  const fx = f(xyz[0] / white[0]), fy = f(xyz[1] / white[1]), fz = f(xyz[2] / white[2]);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// CIEDE2000 (Sharma, Wu, Dalal 2005)
export function deltaE00(l1, l2) {
  const [L1, a1, b1] = l1, [L2, a2, b2] = l2;
  const rad = Math.PI / 180;
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2);
  const Cb = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)));
  const a1p = (1 + G) * a1, a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const hp = (b, a) => (b === 0 && a === 0 ? 0 : ((Math.atan2(b, a) / rad) + 360) % 360);
  const h1p = hp(b1, a1p), h2p = hp(b2, a2p);
  const dLp = L2 - L1, dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) {
    dhp = h2p - h1p;
    if (dhp > 180) dhp -= 360;
    else if (dhp < -180) dhp += 360;
  }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * rad);
  const Lbp = (L1 + L2) / 2, Cbp = (C1p + C2p) / 2;
  let hbp = h1p + h2p;
  if (C1p * C2p !== 0) {
    if (Math.abs(h1p - h2p) > 180) hbp = h1p + h2p < 360 ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2;
    else hbp = (h1p + h2p) / 2;
  }
  const T = 1 - 0.17 * Math.cos((hbp - 30) * rad) + 0.24 * Math.cos(2 * hbp * rad)
    + 0.32 * Math.cos((3 * hbp + 6) * rad) - 0.2 * Math.cos((4 * hbp - 63) * rad);
  const dTheta = 30 * Math.exp(-(((hbp - 275) / 25) ** 2));
  const Rc = 2 * Math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7));
  const Sl = 1 + (0.015 * (Lbp - 50) ** 2) / Math.sqrt(20 + (Lbp - 50) ** 2);
  const Sc = 1 + 0.045 * Cbp, Sh = 1 + 0.015 * Cbp * T;
  const Rt = -Math.sin(2 * dTheta * rad) * Rc;
  return Math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh));
}

// ---------------------------------------------------------------- 표시용 sRGB
const XYZ_TO_LSRGB = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.969266, 1.8760108, 0.041556],
  [0.0556434, -0.2040259, 1.0572252],
];
const mv = (m, v) => m.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);

// OKLab (Björn Ottosson)
function lsrgbToOklab([r, g, b]) {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}
function oklabToLsrgb([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}
const inGamut = (c) => c.every((v) => v >= -1e-6 && v <= 1 + 1e-6);
const encode = (v) => {
  const x = Math.min(1, Math.max(0, v));
  return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
};

/** XYZ(Y=100 기준) → sRGB hex. 색역 밖이면 OKLCh 에서 L, h 를 고정하고 C 만 줄인다. */
export function srgbFromXyz(xyz) {
  let lin = mv(XYZ_TO_LSRGB, xyz.map((v) => v / 100));
  let mapped = false;
  if (!inGamut(lin)) {
    mapped = true;
    const [L, a, b] = lsrgbToOklab(lin.map((v) => Math.max(v, 0)));
    const ok = lsrgbToOklab(lin.map((v) => v)); // 음수 채널도 cbrt 로 처리 가능
    const Lc = Math.min(1, Math.max(0, ok[0] || L));
    const h = Math.atan2(ok[2], ok[1]);
    let lo = 0, hi = Math.hypot(ok[1], ok[2]);
    for (let i = 0; i < 30; i++) {
      const mid = (lo + hi) / 2;
      const c = oklabToLsrgb([Lc, mid * Math.cos(h), mid * Math.sin(h)]);
      if (inGamut(c)) lo = mid; else hi = mid;
    }
    lin = oklabToLsrgb([Lc, lo * Math.cos(h), lo * Math.sin(h)]);
  }
  const rgb = lin.map((v) => Math.round(encode(v) * 255));
  const hex = "#" + rgb.map((v) => v.toString(16).padStart(2, "0")).join("");
  return { hex, mapped };
}
