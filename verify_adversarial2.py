# ==============================================================================
# 적대검토 검증 2 — 동서향(체크 ⑤) + Perez 이방성(체크 ⑦) 민감도
# 실행: .venv/bin/python verify_adversarial2.py
# 전제: verify_adversarial.py 의 교정 VF(현/피치 1.0) + PV 띠 중앙 배치(도면 판독)
# ==============================================================================
import json
import numpy as np
import pandas as pd
import pvlib

from physics_v2 import panel_sf


def vf_mc_clean(tilt_deg, c_over_p, n_pts=64, n_dir=4000, n_nb=4, seed=42):
    """verify_adversarial.vf_mc_clean 복사 (import 시 전체 재실행 부작용 방지)"""
    half, p = c_over_p / 2.0, 1.0
    rng = np.random.default_rng(seed)
    tr = np.radians(tilt_deg)
    n = np.array([np.sin(tr), np.cos(tr), 0.0])
    t1 = np.array([np.cos(tr), -np.sin(tr), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    u = (rng.random(n_pts) * 2 - 1) * half
    px, py = u * np.cos(tr), -u * np.sin(tr)
    ct_ = np.sqrt(rng.random((n_pts, n_dir))); st_ = np.sqrt(1 - ct_**2)
    ph = rng.random((n_pts, n_dir)) * 2 * np.pi
    d = (ct_[..., None] * n + (st_ * np.cos(ph))[..., None] * t1 + (st_ * np.sin(ph))[..., None] * t2)
    dx, dy = d[..., 0], d[..., 1]
    blocked = np.zeros((n_pts, n_dir), dtype=bool)
    for k in list(range(1, n_nb + 1)) + list(range(-n_nb, 0)):
        ax_, ay_ = -half * np.cos(tr), k * p + half * np.sin(tr)
        bx_, by_ = half * np.cos(tr), k * p - half * np.sin(tr)
        ex, ey = bx_ - ax_, by_ - ay_
        qx, qy = ax_ - px[:, None], ay_ - py[:, None]
        den = dx * ey - dy * ex
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        t_ray = (qx * ey - qy * ex) / den
        s_seg = (qx * dy - qy * dx) / den
        blocked |= (t_ray > 1e-9) & (s_seg >= 0.0) & (s_seg <= 1.0)
    fwd = dx > 1e-9
    return (float(((dy > 1e-9) & fwd & ~blocked).mean()),
            float(((dy < -1e-9) & fwd & ~blocked).mean()))

C, P = 97.5, 97.5
ANGLES = np.arange(15, 91, 1.0)
TILT_GRID = np.arange(0.0, 90.1, 2.5)
ALB = 0.2
STRIP = 83.0 / 97.5
STRIP_LO = (1 - STRIP) / 2   # 중앙 배치 (도면: Rotation Pin 중앙, 양끝 Support/인터록 마진)

df = pd.read_csv("bipv_ai_master_data_v15.csv")
day = df[df["ghi_w_m2"] >= 10]
el = day["solar_elevation"].to_numpy(float); az = day["solar_azimuth"].to_numpy(float)
dni = day["dni"].to_numpy(float); dhi = day["dhi"].to_numpy(float)

FS = np.array([vf_mc_clean(t, 1.0)[0] for t in TILT_GRID])
FG = np.array([vf_mc_clean(t, 1.0)[1] for t in TILT_GRID])
fs_i = np.interp(ANGLES, TILT_GRID, FS)
fg_i = np.interp(ANGLES, TILT_GRID, FG)

R = {}

def annual(sa, strip, circumsolar=False, label=""):
    """sa: 파사드 방위. strip: None=현 전체 sf / float=띠 하단 s. circumsolar: dhi 의
    Perez 회절(태양주변) 성분을 빔처럼 취급(음영+입사각 적용) — 이방성 민감도 상한."""
    T = ANGLES[:, None]
    sf = panel_sf(T, el[None, :], az[None, :], hd=C, p=P, sa=sa)
    if strip is None:
        ov = sf
    else:
        ov = np.clip(np.minimum(sf, strip + STRIP) - strip, 0, None) / STRIP
    bp = np.maximum(pvlib.irradiance.beam_component(surface_tilt=T, surface_azimuth=sa,
        solar_zenith=90 - el[None, :], solar_azimuth=az[None, :], dni=dni[None, :]), 0)
    ghi_e = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    if circumsolar:
        # Perez 성분 분해를 무차폐 기준으로 추정: 회절성분비 = AI(이방성지수) ≈ dni/ext
        dni_ext = np.asarray(pvlib.irradiance.get_extra_radiation(pd.to_datetime(day["timestamp"]).dt.dayofyear), dtype=float)
        ai = np.clip(dni / dni_ext, 0, 1)               # Hay-Davies 이방성 지수 (Perez 근사)
        dhi_iso = dhi * (1 - ai)
        # 회절 성분은 빔과 같은 방향 → 같은 음영/입사각 (cosθ/sinα 스케일)
        rb = np.maximum(pvlib.irradiance.beam_component(surface_tilt=T, surface_azimuth=sa,
            solar_zenith=90 - el[None, :], solar_azimuth=az[None, :], dni=np.ones_like(dni)[None, :]), 0)
        cs = dhi[None, :] * ai[None, :] * rb / np.maximum(np.sin(np.radians(el))[None, :], 0.087)
        E = np.maximum(bp * (1 - ov) + cs * (1 - ov)
                       + dhi_iso[None, :] * fs_i[:, None] + ghi_e[None, :] * ALB * fg_i[:, None], 0)
    else:
        E = np.maximum(bp * (1 - ov) + dhi[None, :] * fs_i[:, None] + ghi_e[None, :] * ALB * fg_i[:, None], 0)
    idx = np.argmax(E, axis=0); opt = ANGLES[idx]
    e_ai = E[idx, np.arange(E.shape[1])].sum()
    fixed = {fa: E[int(np.where(ANGLES == fa)[0][0])].sum() for fa in [45, 60, 75, 90]}
    grid_best = max(range(len(ANGLES)), key=lambda j: E[j].sum())
    best_fa = int(ANGLES[grid_best]); e_best = E[grid_best].sum()
    out = {"median_opt": float(np.median(opt)), "p10": float(np.percentile(opt, 10)),
           "p90": float(np.percentile(opt, 90)), "best_fixed": best_fa,
           "ai_vs_45": round(float(e_ai / fixed[45] - 1) * 100, 2),
           "ai_vs_60": round(float(e_ai / fixed[60] - 1) * 100, 2),
           "ai_vs_90": round(float(e_ai / fixed[90] - 1) * 100, 2),
           "ai_vs_best": round(float(e_ai / e_best - 1) * 100, 2)}
    print(f" {label}: 중앙최적 {out['median_opt']:.0f}° (p10–90 {out['p10']:.0f}–{out['p90']:.0f}°) | 최고고정 {best_fa}° | "
          f"AI vs45 +{out['ai_vs_45']}% vs60 +{out['ai_vs_60']}% vs90 +{out['ai_vs_90']}% vs최고 +{out['ai_vs_best']}%")
    return out

print("=" * 78)
print("[E] 방위 민감도 (교정 VF + PV 띠 중앙)")
print("=" * 78)
R["S_180_strip"] = annual(180.0, STRIP_LO, label="남향 180° · 띠 중앙")
R["E_090_strip"] = annual(90.0, STRIP_LO, label="동향  90° · 띠 중앙")
R["W_270_strip"] = annual(270.0, STRIP_LO, label="서향 270° · 띠 중앙")
R["S_180_full"] = annual(180.0, None, label="남향 180° · 현 전체 sf(보수)")
R["E_090_full"] = annual(90.0, None, label="동향  90° · 현 전체 sf(보수)")

print()
print("=" * 78)
print("[F] Perez/Hay-Davies 이방성 민감도 — 회절성분을 빔처럼 취급(이방성 상한)")
print("=" * 78)
R["S_180_strip_aniso"] = annual(180.0, STRIP_LO, circumsolar=True, label="남향 · 띠 중앙 · 이방성")
R["S_180_full_aniso"] = annual(180.0, None, circumsolar=True, label="남향 · 현 전체 · 이방성")

with open("verify_adversarial2_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\nsaved → verify_adversarial2_results.json")
