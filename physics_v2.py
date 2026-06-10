# ==============================================================================
# AI Tilt 물리 v2 — 단일 소스 (app.py 와 train_v16.py 가 공유)
# 근거: FABLE_REVIEW.md §C (2026-06-10 적대검토)
#   beam·(1−sf)            : 선형 retain — 자기음영=폭방향 균일밴드 + 셀 1열 = 매치드 셀
#   dhi·F_sky(tilt)        : MC 시야계수 (후방 벽 + 이웃 블레이드 ±4 차폐)
#   ghi_est·ALBEDO·F_grd   : 지면반사 (ghi_est = dni·sin(el) + dhi)
# F_sky/F_grd 테이블: view_factors_v16.json (gen: train_v16.py, MC seed 고정)
# ==============================================================================
import json
import os
import numpy as np
import pvlib

HD, DP = 57.0, 114.0
ALBEDO = 0.2
_VF_FN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "view_factors_v16.json")
_VF = None


def panel_sf(tilt, elev, az, hd=HD, p=DP, sa=180.0):
    """V15 선분교차 음영률 (불변)"""
    tilt = np.asarray(tilt, dtype=float); elev = np.asarray(elev, dtype=float); az = np.asarray(az, dtype=float)
    tr = np.radians(tilt); er = np.radians(np.clip(elev, 0.1, 89.9))
    ct, st2 = np.cos(tr), np.sin(tr)
    rx, ry = hd * ct, p - hd * st2
    fx, fy = hd * ct, -hd * st2
    adr = np.radians(az - sa); rdx = -np.cos(er) * np.cos(adr); rdy = -np.sin(er)
    dn = fx * rdy - fy * rdx; dns = np.where(np.abs(dn) < 1e-12, 1e-12, dn)
    tb = (rx * rdy - ry * rdx) / dns; ta = (rx * fy - ry * fx) / dns
    sf = np.where((ta > 0) & (tb > 1), 1.0, np.where((ta > 0) & (tb > 0) & (tb <= 1), tb, 0.0))
    return np.clip(np.where(elev <= 0, 0.0, sf), 0, 1)


def svf(tilt, hd=HD, p=DP):
    """구 휴리스틱 — 호환용으로 유지 (eff_poa v2 는 미사용, main.html JS 게이지용)"""
    return np.clip(1 - hd * np.cos(np.radians(np.asarray(tilt, dtype=float))) / p, 0.05, 1)


def _load_vf():
    global _VF
    if _VF is None:
        with open(_VF_FN, encoding="utf-8") as f:
            d = json.load(f)
        _VF = {"tilts": np.array(d["tilts"], dtype=float),
               "ratios": np.array(d["ratios"], dtype=float),
               "f_sky": np.array(d["f_sky"], dtype=float),   # (n_ratio, n_tilt)
               "f_grd": np.array(d["f_grd"], dtype=float)}
    return _VF


def view_factor(tilt, hd=HD, p=DP, kind="f_sky"):
    """ratio(hd/p)·tilt 이중선형 보간. tilt 는 임의 shape ndarray."""
    vf = _load_vf()
    t = np.asarray(tilt, dtype=float)
    r = float(np.clip(hd / p, vf["ratios"][0], vf["ratios"][-1]))
    i = int(np.clip(np.searchsorted(vf["ratios"], r) - 1, 0, len(vf["ratios"]) - 2))
    w = (r - vf["ratios"][i]) / (vf["ratios"][i + 1] - vf["ratios"][i])
    row = (1 - w) * vf[kind][i] + w * vf[kind][i + 1]
    return np.interp(t.ravel(), vf["tilts"], row).reshape(t.shape)


def eff_poa(tilt, elev, az, dni, dhi, hd=HD, p=DP, albedo=ALBEDO):
    """유효 POA v2 — FABLE_REVIEW.md §C"""
    tilt = np.asarray(tilt, dtype=float); elev = np.asarray(elev, dtype=float); az = np.asarray(az, dtype=float)
    sf = panel_sf(tilt, elev, az, hd, p)
    pd2 = np.maximum(pvlib.irradiance.beam_component(surface_tilt=tilt, surface_azimuth=180.0,
        solar_zenith=90 - elev, solar_azimuth=az, dni=dni), 0)
    ghi_est = dni * np.maximum(np.sin(np.radians(elev)), 0) + dhi
    return np.maximum(pd2 * (1 - sf)
                      + dhi * view_factor(tilt, hd, p, "f_sky")
                      + ghi_est * albedo * view_factor(tilt, hd, p, "f_grd"), 0)
