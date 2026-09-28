"""웹이 쓰는 데이터(web/data/optics.json)와 교차검증용 golden(tests/golden.json)을 만든다.

python3 make_data.py
"""
import json
import random

import numpy as np

import fastspec
import optics

ROOT = optics.HERE.parent


def r6(a):
    return [round(float(x), 12) for x in a]


def main():
    presets = json.loads((optics.HERE / "stacks.json").read_text())
    vis_idx = np.nonzero(optics.VIS)[0]
    ref_T0 = optics.spectrum([], 0)[1]
    y_bar = optics.W_XYZ[1] / optics.W_XYZ[1].max()  # 표시용 D65×ȳ (정규화)

    evaluated = []
    for p in presets:
        ev = optics.evaluate(p["layers"])
        evaluated.append({
            **p,
            "summary": {
                "jsc_rel": round(ev["jsc_rel"], 4),
                "lab": [[round(v, 2) for v in a["lab"]] for a in ev["angles"]],
                "dE00": [round(a["dE00_vs_0"], 2) for a in ev["angles"]],
            },
        })

    # 설계 공간 표본: 무작위 TiO2/SiO2 3–7층, 0° 명도와 상대 광전류 (그림 6)
    rng_c = random.Random(3)
    cloud = []
    for _ in range(600):
        k = rng_c.randint(3, 7)
        layers = [{"mat": "TiO2" if i % 2 == 0 else "SiO2", "d": rng_c.uniform(10, 250)} for i in range(k)]
        R, T = fastspec.spectrum(layers, 0)
        lab = optics.lab_from_xyz(optics.xyz_from_R(R))
        cloud.append({"L": round(float(lab[0]), 2), "C": round(float(np.hypot(lab[1], lab[2])), 2),
                      "j": round(optics.jsc_rel(T) / optics.jsc_rel(ref_T0), 4)})

    data = {
        "cloud": cloud,
        "lambda": optics.LAMBDA.tolist(),
        "module": {
            "n_glass": optics.N_GLASS, "n_eva": optics.N_EVA,
            "glass_d": optics.GLASS_D, "eva_d": optics.EVA_D, "arc_d": optics.ARC_D,
        },
        "materials": {k: {"label": v["label"], "source": v["source"]} for k, v in optics.MATERIALS.items()},
        "nk": optics.nk_table(),
        "color": {
            "W": [r6(row) for row in optics.W_XYZ],
            "white": r6(optics.WHITE_D65),
            "visStart": int(vis_idx[0]), "visCount": int(len(vis_idx)),
            "observer": "CIE 1931 2°", "illuminant": "D65",
        },
        "photon": r6(optics.W_PHOTON),
        "reference_jsc": optics.jsc_rel(ref_T0),
        "display": {"vlambda_d65": r6(y_bar)},
        "presets": evaluated,
    }
    out = ROOT / "web" / "data" / "optics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False))
    print("wrote", out, round(out.stat().st_size / 1024), "KB")

    # golden: 무작위 스택 + 프리셋, 여러 각도
    rng = random.Random(7)
    cases = [p["layers"] for p in presets] + [[]]
    for _ in range(5):
        k = rng.randint(1, 7)
        cases.append([{"mat": rng.choice(["TiO2", "SiO2", "Si3N4"]), "d": round(rng.uniform(5, 300), 1)} for _ in range(k)])
    golden = []
    for layers in cases:
        ev = optics.evaluate(layers, angles=(0, 30, 60, 75))
        golden.append({
            "layers": layers,
            "jsc_rel": ev["jsc_rel"],
            "angles": [{"theta_air": a["theta_air"], "theta_glass": a["theta_glass"], "R": r6(a["R"]), "T": r6(a["T"]),
                        "xyz": a["xyz"], "lab": a["lab"], "dE00": a["dE00_vs_0"]} for a in ev["angles"]],
        })
    g = ROOT / "tests" / "golden.json"
    g.parent.mkdir(parents=True, exist_ok=True)
    g.write_text(json.dumps(golden))
    print("wrote", g, len(golden), "cases")


if __name__ == "__main__":
    main()
