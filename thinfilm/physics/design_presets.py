"""목표색에 맞춘 TiO2/SiO2 코팅 프리셋 설계 (한 번 돌려 stacks.json 에 저장).

목적함수 = ΔE00(목표, 0°) + 0.25·ΔE00(0°→45°) + 20·(1 − 상대광전류)
탐색은 fastspec(벡터화, Byrnes inc_tmm 과 1e-14 이내 일치)으로 5 nm 전체 격자에서 한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

import fastspec
import optics

TARGETS = [
    {"id": "slate", "name": "청회색", "lab": [45.0, -4.0, -18.0]},
    {"id": "green", "name": "녹색", "lab": [50.0, -20.0, 10.0]},
    {"id": "bronze", "name": "브론즈", "lab": [52.0, 6.0, 20.0]},
]
MATS = ["TiO2", "SiO2", "TiO2", "SiO2", "TiO2"]


def coarse_eval(coating, angle):
    return fastspec.spectrum(coating, angle)


REF_J = optics.jsc_rel(coarse_eval([], 0)[1])


def objective(x, target):
    coating = [{"mat": m, "d": float(d)} for m, d in zip(MATS, x)]
    R0, T0 = coarse_eval(coating, 0)
    R45, _ = coarse_eval(coating, 45)
    lab0 = optics.lab_from_xyz(optics.xyz_from_R(R0))
    lab45 = optics.lab_from_xyz(optics.xyz_from_R(R45))
    j = optics.jsc_rel(T0) / REF_J
    return optics.delta_e00(target, lab0) + 0.25 * optics.delta_e00(lab0, lab45) + 20 * (1 - j)


def main():
    presets = []
    for t in TARGETS:
        res = differential_evolution(
            objective, [(10, 250)] * len(MATS), args=(np.array(t["lab"]),),
            seed=1, popsize=20, maxiter=400, tol=1e-8, polish=True,
        )
        layers = [{"mat": m, "d": round(float(d), 1)} for m, d in zip(MATS, res.x)]
        print(t["id"], round(res.fun, 2), layers)
        presets.append({**t, "layers": layers})
    Path(optics.HERE / "stacks.json").write_text(json.dumps(presets, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
