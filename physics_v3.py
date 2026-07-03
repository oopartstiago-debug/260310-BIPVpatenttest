# ==============================================================================
# AI Tilt 물리 v3 — 단일 소스 (적대검토 2026-06-11 반영, ADVERSARIAL_REVIEW_RESULTS.md)
#   beam·IAM·(1−sf_strip) : PV 띠 유효음영(중앙 배치, 그림자=피벗쪽부터 — 광선추적 확정)
#                           + Martin-Ruiz IAM (고입사각 Fresnel, 검토 §G)
#   dhi·F_sky·IAM_sky      : MC 시야계수 — ★현/피치 의미론 교정(구 2배 버그, 검토 §A)
#   ghi_est·albedo·F_grd·IAM_grd : 알베도 = 현장 파라미터 (기본 0.15, 검토 §H)
# 시야계수 테이블: view_factors_v17.json (gen: train_v17.py, ratio = 현/피치 그대로)
# panel_sf 는 v2 그대로 (독립 광선추적 30/30 일치로 무죄 판정)
# ==============================================================================
import json
import os
import numpy as np
import pvlib

from physics_v2 import panel_sf  # 검증 완료(무변경) — 단일 소스 유지

CHORD, PITCH = 114.0, 97.5         # 실기하 (도면 1043A: 현114·피치97.5 겹침16.5mm 밀폐 → gcr=현/피치=1.169)
ALBEDO = 0.10                       # 허공 위 파사드 주변값(통제불가·도시지면반사 ≈0.10~0.15, 0.15=낙관단). v17라벨은0.15산출이나 평탄고원서 최적각차~1°=무시 (verify_void_baseline.py 2026-07-03)
A_R = 0.16                          # Martin-Ruiz (일반 유리 라미네이트)
STRIP_FRAC = 83.0 / 114.0          # PV 띠 폭/현 (M6 half-cut 83mm)
STRIP_LO = 24.0 / 114.0            # 상단마진 24mm (도면 실측 상24/셀83/하7 = 하단 치우침, 그림자=상단부터라 유도)
_VF_FN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "view_factors_v17.json")
_VF = None


def _load_vf():
    global _VF
    if _VF is None:
        with open(_VF_FN, encoding="utf-8") as f:
            d = json.load(f)
        _VF = {"tilts": np.array(d["tilts"], dtype=float),
               "ratios": np.array(d["ratios"], dtype=float),   # ★현/피치 그대로 (v16 반의미 교정)
               "f_sky": np.array(d["f_sky"], dtype=float),
               "f_grd": np.array(d["f_grd"], dtype=float)}
    return _VF


def view_factor(tilt, c=CHORD, p=PITCH, kind="f_sky"):
    """ratio(현/피치)·tilt 이중선형 보간."""
    vf = _load_vf()
    t = np.asarray(tilt, dtype=float)
    r = float(np.clip(c / p, vf["ratios"][0], vf["ratios"][-1]))
    i = int(np.clip(np.searchsorted(vf["ratios"], r) - 1, 0, len(vf["ratios"]) - 2))
    w = (r - vf["ratios"][i]) / (vf["ratios"][i + 1] - vf["ratios"][i])
    row = (1 - w) * vf[kind][i] + w * vf[kind][i + 1]
    return np.interp(t.ravel(), vf["tilts"], row).reshape(t.shape)


def strip_shade(sf, lo=STRIP_LO, frac=STRIP_FRAC):
    """현 전체 음영률 sf → PV 띠 유효 음영률. 그림자 띠=[0, sf] (블레이드 상단부터, 광선추적 확정 2026-07-02).
    PV 띠=[lo, lo+frac], lo=상단마진/현. 셀 하단 치우침이면 상단마진이 그림자를 먼저 흡수(음영 유도)."""
    return np.clip(np.minimum(np.asarray(sf, dtype=float), lo + frac) - lo, 0, None) / frac


def eff_poa(tilt, elev, az, dni, dhi, c=CHORD, p=PITCH, albedo=ALBEDO,
            a_r=A_R, strip_lo=STRIP_LO, strip_frac=STRIP_FRAC, sa=180.0):
    """유효 POA v3. 모든 가정이 파라미터 = 민감도 후크 (loop_attack.py 가 사용)."""
    tilt = np.asarray(tilt, dtype=float); elev = np.asarray(elev, dtype=float); az = np.asarray(az, dtype=float)
    sf = panel_sf(tilt, elev, az, hd=c, p=p, sa=sa)
    ov = strip_shade(sf, strip_lo, strip_frac)
    aoi = pvlib.irradiance.aoi(tilt, sa, 90 - elev, az)
    bp = np.maximum(np.asarray(dni) * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=a_r), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(tilt.ravel(), a_r=a_r)
    iam_sky = np.asarray(sky, dtype=float).reshape(tilt.shape)
    iam_grd = np.asarray(grd, dtype=float).reshape(tilt.shape)
    ghi_est = np.asarray(dni) * np.maximum(np.sin(np.radians(elev)), 0) + np.asarray(dhi)
    return np.maximum(bp * iam_b * (1 - ov)
                      + np.asarray(dhi) * view_factor(tilt, c, p, "f_sky") * iam_sky
                      + ghi_est * albedo * view_factor(tilt, c, p, "f_grd") * iam_grd, 0)
