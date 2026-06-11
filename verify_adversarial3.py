# ==============================================================================
# 적대검토 검증 3 — "90°대가 아직도 높다" 잔여 용의자 2건
#   G. IAM(고입사각 Fresnel 반사손실) — 현재 모델 미반영, 90° 일방 유리 누락
#   H. 지면반사 가정 (albedo·도시 차폐) — 90° 우위의 다른 축
# 베이스라인: 교정 VF(현/피치 1.0) + PV 띠 중앙(사용자 확정) + 남향
# 실행: .venv/bin/python verify_adversarial3.py
# ==============================================================================
import json
import numpy as np
import pandas as pd
import pvlib

from physics_v2 import panel_sf


def vf_mc_clean(tilt_deg, c_over_p, n_pts=64, n_dir=4000, n_nb=4, seed=42):
    """verify_adversarial.vf_mc_clean 복사 (top-level 스크립트 import 부작용 방지)"""
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
STRIP = 83.0 / 97.5
STRIP_LO = (1 - STRIP) / 2

df = pd.read_csv("bipv_ai_master_data_v15.csv")
day = df[df["ghi_w_m2"] >= 10]
el = day["solar_elevation"].to_numpy(float); az = day["solar_azimuth"].to_numpy(float)
dni = day["dni"].to_numpy(float); dhi = day["dhi"].to_numpy(float)

FS = np.array([vf_mc_clean(t, 1.0)[0] for t in TILT_GRID])
FG = np.array([vf_mc_clean(t, 1.0)[1] for t in TILT_GRID])
fs_i = np.interp(ANGLES, TILT_GRID, FS)
fg_i = np.interp(ANGLES, TILT_GRID, FG)

R = {}

def annual(iam=False, albedo=0.2, fgrd_scale=1.0, label=""):
    T = ANGLES[:, None]
    sf = panel_sf(T, el[None, :], az[None, :], hd=C, p=P)
    ov = np.clip(np.minimum(sf, STRIP_LO + STRIP) - STRIP_LO, 0, None) / STRIP
    aoi = pvlib.irradiance.aoi(T, 180.0, 90 - el[None, :], az[None, :])
    bp = np.maximum(dni[None, :] * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    ghi_e = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    if iam:
        iam_b = pvlib.iam.martin_ruiz(aoi, a_r=0.16)
        iam_d = pvlib.iam.martin_ruiz_diffuse(ANGLES, a_r=0.16)   # (sky, ground)
        iam_sky = np.asarray(iam_d[0], dtype=float)[:, None]
        iam_grd = np.asarray(iam_d[1], dtype=float)[:, None]
    else:
        iam_b = 1.0; iam_sky = 1.0; iam_grd = 1.0
    E = np.maximum(bp * iam_b * (1 - ov)
                   + dhi[None, :] * fs_i[:, None] * iam_sky
                   + ghi_e[None, :] * albedo * fg_i[:, None] * fgrd_scale * iam_grd, 0)
    idx = np.argmax(E, axis=0); opt = ANGLES[idx]
    e_ai = E[idx, np.arange(E.shape[1])].sum()
    fixed = {fa: E[int(np.where(ANGLES == fa)[0][0])].sum() for fa in [45, 60, 75, 90]}
    grid_best = max(range(len(ANGLES)), key=lambda j: E[j].sum())
    best_fa = int(ANGLES[grid_best]); e_best = E[grid_best].sum()
    out = {"median_opt": float(np.median(opt)), "p10": float(np.percentile(opt, 10)),
           "p90": float(np.percentile(opt, 90)), "best_fixed": best_fa,
           "ai_vs_45": round(float(e_ai / fixed[45] - 1) * 100, 2),
           "ai_vs_60": round(float(e_ai / fixed[60] - 1) * 100, 2),
           "ai_vs_75": round(float(e_ai / fixed[75] - 1) * 100, 2),
           "ai_vs_90": round(float(e_ai / fixed[90] - 1) * 100, 2),
           "ai_vs_best": round(float(e_ai / e_best - 1) * 100, 2)}
    print(f" {label}: 중앙최적 {out['median_opt']:.0f}° (p10–90 {out['p10']:.0f}–{out['p90']:.0f}°) | 최고고정 {best_fa}° | "
          f"AI vs45 +{out['ai_vs_45']}% vs60 +{out['ai_vs_60']}% vs75 +{out['ai_vs_75']}% vs90 +{out['ai_vs_90']}% vs최고 +{out['ai_vs_best']}%")
    return out

print("=" * 78)
print("[G] IAM(Martin-Ruiz a_r=0.16) — 고입사각 반사손실 반영")
print("=" * 78)
R["base_noIAM"] = annual(iam=False, label="① 기준(IAM 없음, alb 0.2)      ")
R["iam"] = annual(iam=True, label="② +IAM                          ")
print()
print("=" * 78)
print("[H] 지면반사 가정")
print("=" * 78)
R["iam_alb0.1"] = annual(iam=True, albedo=0.1, label="③ +IAM · alb 0.1               ")
R["iam_alb0.3"] = annual(iam=True, albedo=0.3, label="④ +IAM · alb 0.3               ")
R["iam_urban"] = annual(iam=True, albedo=0.15, fgrd_scale=0.5, label="⑤ +IAM · 도시차폐(alb.15·F_grd×½)")
R["noiam_alb0.1"] = annual(iam=False, albedo=0.1, label="⑥ IAM없음 · alb 0.1            ")

# 메커니즘 분해: 겨울/여름 정오, IAM 반영
print()
print(" [분해] IAM 반영 (alb 0.2, 띠 중앙) — 겨울 정오(elev30, dni927, dhi85) / 여름 정오(74, 743, 180):")
iam_d = pvlib.iam.martin_ruiz_diffuse(np.array([0.0, 30.0, 45.0, 60.0, 75.0, 90.0]), a_r=0.16)
for nm, e_, d_, h_ in [("겨울", 30.0, 927.0, 85.0), ("여름", 74.0, 743.0, 180.0)]:
    ghe = d_ * np.sin(np.radians(e_)) + h_
    for j, t in enumerate([0.0, 30.0, 45.0, 60.0, 75.0, 90.0]):
        sfv = float(panel_sf(t, e_, 180.0, hd=C, p=P))
        ovv = max(min(sfv, STRIP_LO + STRIP) - STRIP_LO, 0) / STRIP
        aoi = float(pvlib.irradiance.aoi(t, 180.0, 90 - e_, 180.0))
        bp = max(d_ * np.cos(np.radians(min(aoi, 90))), 0)
        ib = float(pvlib.iam.martin_ruiz(aoi, a_r=0.16)) if aoi < 90 else 0.0
        fs, fg = vf_mc_clean(t, 1.0)
        beam = bp * ib * (1 - ovv)
        dif = h_ * fs * float(np.asarray(iam_d[0])[j])
        grd = ghe * 0.2 * fg * float(np.asarray(iam_d[1])[j])
        print(f"   {nm} tilt {t:>4.0f}°: AOI={aoi:5.1f}° IAM={ib:.3f} | 빔(순)={beam:6.0f} 확산={dif:5.0f} 지면={grd:5.0f} 합={beam + dif + grd:6.0f}")

with open("verify_adversarial3_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\nsaved → verify_adversarial3_results.json")
