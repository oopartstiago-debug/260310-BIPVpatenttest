# ==============================================================================
# Fable 검증 2 — 확산광/지면반사 시야계수 Monte Carlo (교차판정 수정판)
#  + 교정된 diffuse·albedo로 최적각 재계산 (eff_poa v2 후보)
# 실행: .venv/bin/python verify_fable2.py
# ==============================================================================
import json
import numpy as np
import pandas as pd
import pvlib

HD, DP = 57.0, 114.0
ANGLES = np.arange(15, 91, 1.0)

def panel_sf(tilt, elev, az, hd=HD, p=DP, sa=180.0):
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

def svf_model(tilt, hd=HD, p=DP):
    return np.clip(1 - hd * np.cos(np.radians(np.asarray(tilt, dtype=float))) / p, 0.05, 1)

def beam_poa(tilt, elev, az, dni, sa=180.0):
    return np.maximum(pvlib.irradiance.beam_component(
        surface_tilt=tilt, surface_azimuth=sa, solar_zenith=90 - elev, solar_azimuth=az, dni=dni), 0)

# ── Monte Carlo 시야계수 (cosine-weighted irradiance fraction) ────────────────
def view_factors(tilt_deg, hd=HD, p=DP, n_pts=96, n_dir=6000, n_nb=4, seed=42, wall=True):
    """발전면(블레이드 앞면)에서 하늘/지면 도달 비율. 2D 단면, 이웃 ±n_nb 차폐.
    wall=True: 루버 뒤는 실내(실외기실) — 후방(dx<0) 광선은 하늘/지면 미도달."""
    rng = np.random.default_rng(seed)
    tr = np.radians(tilt_deg)
    n = np.array([np.sin(tr), np.cos(tr), 0.0])
    t1 = np.array([np.cos(tr), -np.sin(tr), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    u = (rng.random(n_pts) * 2 - 1) * hd
    px, py = u * np.cos(tr), -u * np.sin(tr)
    ct_ = np.sqrt(rng.random((n_pts, n_dir))); st_ = np.sqrt(1 - ct_**2)
    ph = rng.random((n_pts, n_dir)) * 2 * np.pi
    d = (ct_[..., None] * n + (st_ * np.cos(ph))[..., None] * t1 + (st_ * np.sin(ph))[..., None] * t2)
    dx, dy = d[..., 0], d[..., 1]
    blocked = np.zeros((n_pts, n_dir), dtype=bool)
    for k in list(range(1, n_nb + 1)) + list(range(-n_nb, 0)):
        ax_, ay_ = -hd * np.cos(tr), k * p + hd * np.sin(tr)   # 이웃 블레이드 뒤끝
        bx_, by_ = hd * np.cos(tr), k * p - hd * np.sin(tr)    # 이웃 블레이드 앞끝
        ex, ey = bx_ - ax_, by_ - ay_
        qx, qy = ax_ - px[:, None], ay_ - py[:, None]          # q = A − P
        den = dx * ey - dy * ex
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        t_ray = (qx * ey - qy * ex) / den
        s_seg = (qx * dy - qy * dx) / den
        blocked |= (t_ray > 1e-9) & (s_seg >= 0.0) & (s_seg <= 1.0)
    fwd = (dx > 1e-9) if wall else np.ones_like(dx, dtype=bool)
    f_sky = float(((dy > 1e-9) & fwd & ~blocked).mean())
    f_grd = float(((dy < -1e-9) & fwd & ~blocked).mean())
    return f_sky, f_grd

print("[5'] 시야계수 MC (수정판) vs 모델 (1+cos)/2·svf :")
tilts = np.arange(0.0, 90.1, 2.5)
FS, FG = [], []
for t in tilts:
    fs, fg = view_factors(t)
    FS.append(fs); FG.append(fg)
FS, FG = np.array(FS), np.array(FG)
model_dd = (1 + np.cos(np.radians(tilts))) / 2 * svf_model(tilts)
tab = {}
for t, fs, fg, md in zip(tilts, FS, FG, model_dd):
    if t % 15 == 0:
        print(f"  tilt {t:>4.0f}°: F_sky={fs:.3f} F_grd={fg:.3f} | model_dd={md:.3f} (model/MC={md/max(fs,1e-9):.2f})")
    tab[float(t)] = {"f_sky": round(fs, 4), "f_grd": round(fg, 4), "model_dd": round(float(md), 4)}

# 수렴/이웃수 sanity: n_nb=8로 60° 재계산
fs8, fg8 = view_factors(60.0, n_nb=8, n_dir=12000, seed=7)
print(f"  sanity tilt 60° n_nb=8: F_sky={fs8:.3f} (n_nb=4: {tab[60.0]['f_sky']})")

# ── 교정 diffuse/albedo + 선형 retain으로 최적각 재계산 ──────────────────────
df = pd.read_csv("bipv_ai_master_data_v15.csv")
day = df[df["ghi_w_m2"] >= 10].copy()
el = day["solar_elevation"].to_numpy(float)
az = day["solar_azimuth"].to_numpy(float)
dni = day["dni"].to_numpy(float)
dhi = day["dhi"].to_numpy(float)
ghi = day["ghi_w_m2"].to_numpy(float)
tgt = day["target_angle_v15"].to_numpy(float)

fsky_i = np.interp(ANGLES, tilts, FS)   # (A,)
fgrd_i = np.interp(ANGLES, tilts, FG)

def eff_v2_grid(retain, albedo=0.2):
    T = ANGLES[:, None]
    sf = panel_sf(T, el[None, :], az[None, :])
    bp = beam_poa(T, el[None, :], az[None, :], dni[None, :])
    out = bp * retain(sf) + dhi[None, :] * fsky_i[:, None] + ghi[None, :] * albedo * fgrd_i[:, None]
    return np.maximum(out, 0)

def eff_v2_at(angles, retain, albedo=0.2):
    sf = panel_sf(angles, el, az)
    bp = beam_poa(angles, el, az, dni)
    fs = np.interp(angles, tilts, FS); fg = np.interp(angles, tilts, FG)
    return np.maximum(bp * retain(sf) + dhi * fs + ghi * albedo * fg, 0)

R = {"view_factors": tab}
print("\n[9] eff_poa v2 (선형 retain + MC diffuse(벽포함) + albedo 민감도):")
lin = lambda sf: 1 - sf
cur = lambda sf: 1 - 0.7 * sf
for name, retain, alb in [("v2 linear alb=0.2", lin, 0.2),
                          ("v2 linear alb=0.1", lin, 0.1),
                          ("v2 linear alb=0.3", lin, 0.3),
                          ("v2 linear alb=0",   lin, 0.0),
                          ("(참고) current retain alb=0.2", cur, 0.2)]:
    E = eff_v2_grid(retain, albedo=alb)
    idx = np.argmax(E, axis=0)
    opt = ANGLES[idx]
    e_ai = E[idx, np.arange(len(el))].sum()
    e60 = eff_v2_at(np.full_like(el, 60.0), retain, alb).sum()
    e45 = eff_v2_at(np.full_like(el, 45.0), retain, alb).sum()
    e90 = eff_v2_at(np.full_like(el, 90.0), retain, alb).sum()
    eold = eff_v2_at(tgt, retain, alb).sum()   # 기존 라벨 각도를 v2 물리로 평가
    row = {"median_opt": float(np.median(opt)), "p10": float(np.percentile(opt, 10)), "p90": float(np.percentile(opt, 90)),
           "adv_vs_f60": round(float(e_ai/e60-1)*100, 2), "adv_vs_f45": round(float(e_ai/e45-1)*100, 2),
           "adv_vs_f90": round(float(e_ai/e90-1)*100, 2), "adv_vs_oldlabels": round(float(e_ai/eold-1)*100, 2)}
    R[name] = row
    print(f"  {name}: 중앙최적 {row['median_opt']:.0f}° (p10-p90 {row['p10']:.0f}–{row['p90']:.0f}°) | vs F60 {row['adv_vs_f60']}% F45 {row['adv_vs_f45']}% F90 {row['adv_vs_f90']}% | vs 구라벨 {row['adv_vs_oldlabels']}%")

with open("verify_fable2_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\nsaved → verify_fable2_results.json")
