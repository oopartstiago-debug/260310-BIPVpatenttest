"""기준 구현: 컬러 BIPV 커버유리 뒷면 간섭 코팅의 반사색과 상대 광전류.

구조 (빛이 들어오는 순서):
    공기 | 커버유리 3.2 mm (incoherent) | 코팅 층들 (coherent) | EVA 0.45 mm (incoherent)
         | SiNx ARC 75 nm (coherent) | c-Si (반무한)

- 스펙트럼 계산: tmm.inc_tmm (Byrnes), s/p 평균(비편광)
- 색: CIE 1931 2°, D65, 5 nm 적분 가중치 → XYZ → CIELAB(D65) → ΔE00
- 상대 광전류: AM1.5G(ASTM G173-03) 광자속 × Si로 들어간 투과율, 이상 EQE(=1) 가정
  기준 모듈(코팅 없음) 대비 비율. 셀 전기 손실은 포함하지 않는다.

웹 시뮬레이터(src/tmm.js)는 이 파일과 같은 입력을 받아 같은 값을 내야 한다.
make_data.py 가 golden 테스트 벡터를 만든다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tmm
import yaml

HERE = Path(__file__).parent
NK_DIR = HERE / "nk"

LAMBDA = np.arange(310.0, 1200.0 + 1e-9, 5.0)  # nm, 계산 격자
VIS = (LAMBDA >= 380) & (LAMBDA <= 780)

GLASS_D = 3.2e6  # nm
EVA_D = 0.45e6  # nm
ARC_D = 75.0  # nm
N_GLASS = 1.52  # 저철분 소다라임 근사(분산 무시). 출처 표기용 상수.
N_EVA = 1.48


# ---------------------------------------------------------------- n,k 로더
def _load_yml(path: Path):
    doc = yaml.safe_load(path.read_text())
    entry = doc["DATA"][0]
    kind = entry["type"]
    if kind == "tabulated nk":
        arr = np.array([[float(x) for x in line.split()] for line in entry["data"].strip().splitlines()])
        wl_nm = arr[:, 0] * 1000.0
        return lambda lam: np.interp(lam, wl_nm, arr[:, 1]) + 1j * np.interp(lam, wl_nm, arr[:, 2])
    if kind == "formula 1":  # Sellmeier: n^2 - 1 = C0 + Σ Bi λ²/(λ² - Ci²), λ in µm
        c = [float(x) for x in entry["coefficients"].split()]

        def f(lam):
            um2 = (np.asarray(lam) / 1000.0) ** 2
            n2 = 1 + c[0]
            for i in range(1, len(c), 2):
                n2 = n2 + c[i] * um2 / (um2 - c[i + 1] ** 2)
            return np.sqrt(n2) + 0j

        return f
    raise ValueError(f"unsupported n,k type {kind} in {path.name}")


MATERIALS = {
    "TiO2": {"file": "TiO2_Siefke.yml", "label": "TiO₂", "source": "Siefke et al. 2016, ALD 박막 (refractiveindex.info)"},
    "SiO2": {"file": "SiO2_Malitson.yml", "label": "SiO₂", "source": "Malitson 1965, 용융 실리카 (refractiveindex.info)"},
    "Si3N4": {"file": "Si3N4_Luke.yml", "label": "Si₃N₄", "source": "Luke et al. 2015 (refractiveindex.info)"},
    "Si": {"file": "Si_Green2008.yml", "label": "c-Si", "source": "Green 2008 (refractiveindex.info)"},
}
_NK = {k: _load_yml(NK_DIR / v["file"]) for k, v in MATERIALS.items()}


def nk_table() -> dict[str, list[list[float]]]:
    """격자 위의 n,k 표. 웹이 같은 값을 쓰도록 JSON으로 내보낸다."""
    out = {}
    for name, f in _NK.items():
        v = f(LAMBDA)
        out[name] = [[round(float(z.real), 10), round(float(z.imag), 10)] for z in v]
    return out


# ---------------------------------------------------------------- 광학
def layer_lists(coating: list[dict], lam: float):
    """coating: [{"mat": "TiO2", "d": 60.0}, ...] 유리 쪽에서 EVA 쪽 순서."""
    n = [1.0, N_GLASS]
    d = [np.inf, GLASS_D]
    c = ["i", "i"]
    for layer in coating:
        n.append(complex(_NK[layer["mat"]](lam)))
        d.append(float(layer["d"]))
        c.append("c")
    n += [N_EVA, complex(_NK["Si3N4"](lam)), complex(_NK["Si"](lam))]
    d += [EVA_D, ARC_D, np.inf]
    c += ["i", "c", "i"]
    return n, d, c


def spectrum(coating: list[dict], theta_air_deg: float):
    """반사율 R(λ)와 Si 유입 투과율 T(λ), 비편광."""
    th = math.radians(theta_air_deg)
    R = np.empty_like(LAMBDA)
    T = np.empty_like(LAMBDA)
    for i, lam in enumerate(LAMBDA):
        n, d, c = layer_lists(coating, lam)
        rs = tmm.inc_tmm("s", n, d, c, th, lam)
        rp = tmm.inc_tmm("p", n, d, c, th, lam)
        R[i] = 0.5 * (rs["R"] + rp["R"])
        T[i] = 0.5 * (rs["T"] + rp["T"])
    return R, T


def theta_in_glass(theta_air_deg: float) -> float:
    return math.degrees(math.asin(math.sin(math.radians(theta_air_deg)) / N_GLASS))


# ---------------------------------------------------------------- 색
def _colour_weights():
    import colour

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    d65 = colour.SDS_ILLUMINANTS["D65"]
    lam = LAMBDA[VIS]
    xyz_bar = np.array([cmfs[l] for l in lam])  # (N,3)
    s = np.array([d65[l] for l in lam])
    k = 100.0 / np.sum(s * xyz_bar[:, 1])
    return (xyz_bar * s[:, None] * k).T  # (3,N): R(λ)에 곱해 더하면 XYZ(Y=100 기준)


W_XYZ = _colour_weights()
WHITE_D65 = W_XYZ.sum(axis=1)  # R=1일 때 XYZ


def xyz_from_R(R):
    return W_XYZ @ R[VIS]


def lab_from_xyz(xyz):
    def f(t):
        return np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)

    fx, fy, fz = f(xyz / WHITE_D65)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def delta_e00(lab1, lab2) -> float:
    import colour

    return float(colour.delta_E(lab1, lab2, method="CIE 2000"))


# ---------------------------------------------------------------- 광전류
def _am15g_photon_weights():
    import pvlib

    spec = pvlib.spectrum.get_reference_spectra(standard="ASTM G173-03")["global"]
    e = np.interp(LAMBDA, spec.index.values.astype(float), spec.values)  # W m-2 nm-1
    photons = e * LAMBDA  # ∝ 광자수 (h c 상수 생략)
    photons[LAMBDA > 1180] = 0.0  # c-Si 흡수 한계 근처 이후 제외
    return photons / photons.sum()


W_PHOTON = _am15g_photon_weights()


def jsc_rel(T) -> float:
    return float(W_PHOTON @ T)


REFERENCE = []  # 코팅 없는 기준 모듈


def evaluate(coating: list[dict], angles=(0, 15, 30, 45, 60)) -> dict:
    ref_T0 = spectrum(REFERENCE, 0)[1]
    out = {"angles": [], "jsc_rel": None}
    lab0 = None
    for a in angles:
        R, T = spectrum(coating, a)
        lab = lab_from_xyz(xyz_from_R(R))
        if lab0 is None:
            lab0 = lab
            out["jsc_rel"] = jsc_rel(T) / jsc_rel(ref_T0)
        out["angles"].append(
            {
                "theta_air": a,
                "theta_glass": round(theta_in_glass(a), 3),
                "R": R.tolist(),
                "T": T.tolist(),
                "xyz": xyz_from_R(R).tolist(),
                "lab": lab.tolist(),
                "dE00_vs_0": delta_e00(lab0, lab),
            }
        )
    return out


if __name__ == "__main__":
    r = evaluate([{"mat": "TiO2", "d": 60}, {"mat": "SiO2", "d": 90}, {"mat": "TiO2", "d": 60}])
    for a in r["angles"]:
        print(a["theta_air"], a["theta_glass"], [round(x, 2) for x in a["lab"]], round(a["dE00_vs_0"], 2))
    print("Jsc rel", round(r["jsc_rel"], 4))
    print(json.dumps({"white": WHITE_D65.tolist()}))
