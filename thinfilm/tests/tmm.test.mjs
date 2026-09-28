// JS 시뮬레이터 ↔ Python 기준 구현(Byrnes tmm + colour-science) 교차검증.
// node thinfilm/tests/tmm.test.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { evaluate, spectrum, thetaInGlass } from "../web/src/model.js";
import { deltaE00, labFromXyz } from "../web/src/color.js";

const here = dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(readFileSync(join(here, "../web/data/optics.json"), "utf8"));
const golden = JSON.parse(readFileSync(join(here, "golden.json"), "utf8"));

const TOL = { R: 1e-6, T: 1e-6, lab: 1e-4, dE: 1e-4, jsc: 1e-7, thg: 1e-3 };
let fails = 0, checks = 0;
const worst = { R: 0, T: 0, lab: 0, dE: 0, jsc: 0 };
const check = (name, err, tol, ctx) => {
  checks++;
  worst[name] = Math.max(worst[name] ?? 0, err);
  if (!(err <= tol)) { fails++; console.log(`FAIL ${name} err=${err.toExponential(2)} tol=${tol} ${ctx}`); }
};

for (const [ci, g] of golden.entries()) {
  const ev = evaluate(g.layers, g.angles.map((a) => a.theta_air), data);
  check("jsc", Math.abs(ev.jscRel - g.jsc_rel), TOL.jsc, `case ${ci}`);
  g.angles.forEach((ga, k) => {
    const ja = ev.angles[k];
    const ctx = `case ${ci} θ=${ga.theta_air}`;
    let eR = 0, eT = 0;
    for (let j = 0; j < ga.R.length; j++) {
      eR = Math.max(eR, Math.abs(ja.R[j] - ga.R[j]));
      eT = Math.max(eT, Math.abs(ja.T[j] - ga.T[j]));
    }
    check("R", eR, TOL.R, ctx);
    check("T", eT, TOL.T, ctx);
    check("lab", Math.max(...ja.lab.map((v, i) => Math.abs(v - ga.lab[i]))), TOL.lab, ctx);
    check("dE", Math.abs(ja.dE - ga.dE00), TOL.dE, ctx);
    check("thg", Math.abs(ja.thetaGlass - ga.theta_glass), TOL.thg, ctx);
  });
}

// 물리 불변식
const noLoss = spectrum([], 0, data);
check("R", Math.abs(labFromXyz(data.color.white, data.color.white)[0] - 100), 1e-9, "white L*=100");
check("thg", Math.abs(thetaInGlass(30, data) - 19.205), 1e-3, "공기 30° → 유리 19.2°");
// Sharma et al. 2005 CIEDE2000 표준 테스트 쌍 일부
const sharma = [
  [[50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425],
  [[50, -1.3802, -84.2814], [50, 0, -82.7485], 1.0],
  [[50, 2.5, 0], [73, 25, -18], 27.1492],
  [[2.0776, 0.0795, -1.135], [0.9033, -0.0636, -0.5514], 0.9082],
];
for (const [a, b, want] of sharma) check("dE", Math.abs(deltaE00(a, b) - want), 1e-4, `Sharma ${want}`);
// 코팅 없는 모듈: R + T ≤ 1 (흡수는 Si 로 들어간 T 에 포함되지 않음 → 등호는 비흡수일 때만)
let maxSum = 0;
for (let j = 0; j < noLoss.R.length; j++) maxSum = Math.max(maxSum, noLoss.R[j] + noLoss.T[j]);
check("R", Math.max(0, maxSum - 1), 1e-9, "R+T<=1");

console.log(`checks=${checks} fails=${fails}`);
console.log("worst:", Object.fromEntries(Object.entries(worst).map(([k, v]) => [k, v.toExponential(2)])));
process.exit(fails ? 1 : 0);
