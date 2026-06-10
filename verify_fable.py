# ==============================================================================
# Fable 5 적대검토 검증 스크립트 — SHADING_POWER_AUDIT.md 주장 재현/반박
# 실행: .venv/bin/python verify_fable.py
# 산출: verify_fable_results.json + stdout 리포트
# ==============================================================================
import json
import numpy as np
import pandas as pd
import pvlib

HD, DP = 57.0, 114.0
AMIN, AMAX = 15, 90
ANGLES = np.arange(AMIN, AMAX + 1, 1.0)  # 76 angles

# ── app.py 물리 복제 (retain 함수만 플러그블) ────────────────────────────────
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

def svf(tilt, hd=HD, p=DP):
    return np.clip(1 - hd * np.cos(np.radians(np.asarray(tilt, dtype=float))) / p, 0.05, 1)

def beam_poa(tilt, elev, az, dni, sa=180.0):
    return np.maximum(pvlib.irradiance.beam_component(
        surface_tilt=tilt, surface_azimuth=sa, solar_zenith=90 - elev, solar_azimuth=az, dni=dni), 0)

# retain variants: f(sf) = 유지되는 beam 비율
RETAINS = {
    "(a) current 1-0.7sf": lambda sf: 1 - 0.7 * sf,
    "(b) linear 1-sf":     lambda sf: 1 - sf,
    "(c) cliff 3-substring": lambda sf: 1 - np.ceil(sf * 3 - 1e-9) / 3.0,
    "(d) harsh current-limit": lambda sf: np.where(sf > 0.05, np.maximum(0, 1 - sf) * 0.4, 1 - sf),
}

def eff_poa_v(tilt, elev, az, dni, dhi, retain, sa=180.0, albedo=0.0, ghi=None, gvf_mode="none"):
    """tilt: (A,) 또는 (A,N) / 나머지 (N,)"""
    sf = panel_sf(tilt, elev, az, sa=sa)
    s = svf(tilt)
    pd2 = beam_poa(tilt, elev, az, dni, sa=sa)
    dd = dhi * (1 + np.cos(np.radians(tilt))) / 2
    out = pd2 * retain(sf) + dd * s
    if albedo > 0 and ghi is not None:
        grd = ghi * albedo * (1 - np.cos(np.radians(tilt))) / 2
        if gvf_mode == "svf":   # 아래쪽 슬롯 차폐도 svf 동형으로 근사
            grd = grd * s
        out = out + grd
    return np.maximum(out, 0)

def grid_optimal(elev, az, dni, dhi, retain, sa=180.0, albedo=0.0, ghi=None, gvf_mode="none"):
    """각 행의 argmax 각도와 그때의 eff (벡터화: A×N)"""
    T = ANGLES[:, None]
    E = eff_poa_v(T, elev[None, :], az[None, :], dni[None, :], dhi[None, :],
                  retain, sa=sa, albedo=albedo, ghi=None if ghi is None else ghi[None, :], gvf_mode=gvf_mode)
    idx = np.argmax(E, axis=0)
    return ANGLES[idx], E, idx

def energy_at(angles, elev, az, dni, dhi, retain, sa=180.0):
    return eff_poa_v(angles, elev, az, dni, dhi, retain, sa=sa).sum()

R = {}

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
df = pd.read_csv("bipv_ai_master_data_v15.csv")
day = df[df["ghi_w_m2"] >= 10].copy()
el = day["solar_elevation"].to_numpy(float)
az = day["solar_azimuth"].to_numpy(float)
dni = day["dni"].to_numpy(float)
dhi = day["dhi"].to_numpy(float)
ghi = day["ghi_w_m2"].to_numpy(float)
tgt = day["target_angle_v15"].to_numpy(float)
print(f"rows total={len(df)} daytime={len(day)} | target range {tgt.min()}..{tgt.max()} unique={len(np.unique(tgt))}")
R["data"] = {"total": len(df), "daytime": len(day), "tgt_min": float(tgt.min()), "tgt_max": float(tgt.max())}

# ── [1] 오라클 재현: argmax(eff_poa, current retain) vs target_angle_v15 ─────
opt_a, _, _ = grid_optimal(el, az, dni, dhi, RETAINS["(a) current 1-0.7sf"])
diff = np.abs(opt_a - tgt)
R["oracle"] = {"match_pm1_pct": float((diff <= 1).mean() * 100), "mae_deg": float(diff.mean()),
               "match_exact_pct": float((diff == 0).mean() * 100)}
print(f"\n[1] 오라클: ±1° 일치 {R['oracle']['match_pm1_pct']:.2f}% | MAE {R['oracle']['mae_deg']:.3f}° (audit 주장: 100%, 0.42°)")

# ── [2] 정오 예시 재현 (§1.1: elev 74°에서 90/60/22° → 256/492/550) ─────────
i_noon = np.argmin(np.abs(el - 74.0) + np.abs(az - 180) * 0.1)
noon = {}
for a in (90.0, 60.0, 22.0):
    sf_n = float(panel_sf(np.array([a]), el[i_noon:i_noon+1], az[i_noon:i_noon+1])[0])
    e_n = float(eff_poa_v(np.array([a]), el[i_noon:i_noon+1], az[i_noon:i_noon+1],
                          dni[i_noon:i_noon+1], dhi[i_noon:i_noon+1], RETAINS["(a) current 1-0.7sf"])[0])
    noon[a] = {"sf_pct": round(sf_n * 100, 1), "eff": round(e_n, 0)}
R["noon_example"] = {"elev": float(el[i_noon]), "rows": noon}
print(f"[2] 정오(elev={el[i_noon]:.1f}°): " + " | ".join(f"{a:.0f}°→SF {v['sf_pct']}%, eff {v['eff']:.0f}" for a, v in noon.items()))

# ── [3] retain 변형별 재최적화 (§1.2 표 재현) ────────────────────────────────
print("\n[3] retain 변형별 (audit §1.2 재현):")
print(f"{'variant':<26}{'중앙최적각':>9}{'E가중SF%':>9}{'vs고정60':>9}{'vs고정45':>9}{'SF50%유지':>9}")
R["retain_variants"] = {}
for name, fn in RETAINS.items():
    opt_v, E_v, idx_v = grid_optimal(el, az, dni, dhi, fn)
    e_ai = E_v[idx_v, np.arange(len(el))].sum()
    e_60 = energy_at(np.full_like(el, 60.0), el, az, dni, dhi, fn)
    e_45 = energy_at(np.full_like(el, 45.0), el, az, dni, dhi, fn)
    sf_at = panel_sf(opt_v, el, az)
    bp = beam_poa(opt_v, el, az, dni)
    ewsf = float((sf_at * bp).sum() / max(bp.sum(), 1e-9) * 100)
    row = {"median_opt": float(np.median(opt_v)), "energy_weighted_sf_pct": round(ewsf, 1),
           "adv_vs_f60_pct": round(float(e_ai / e_60 - 1) * 100, 2),
           "adv_vs_f45_pct": round(float(e_ai / e_45 - 1) * 100, 2),
           "retain_at_sf50_pct": round(float(fn(np.array([0.5]))[0]) * 100, 0)}
    R["retain_variants"][name] = row
    print(f"{name:<26}{row['median_opt']:>8.0f}°{row['energy_weighted_sf_pct']:>9}{row['adv_vs_f60_pct']:>8}%{row['adv_vs_f45_pct']:>8}%{row['retain_at_sf50_pct']:>8.0f}%")

# ── [4] albedo 추가 시 최적각 이동 (§2.2: −5.8° 주장) ────────────────────────
print("\n[4] albedo(0.2) 추가 시 최적각 이동:")
R["albedo"] = {}
for name in ["(a) current 1-0.7sf", "(b) linear 1-sf"]:
    fn = RETAINS[name]
    base, _, _ = grid_optimal(el, az, dni, dhi, fn)
    for gvf in ["none", "svf"]:
        alb, _, _ = grid_optimal(el, az, dni, dhi, fn, albedo=0.2, ghi=ghi, gvf_mode=gvf)
        key = f"{name}|gvf={gvf}"
        R["albedo"][key] = {"mean_shift_deg": round(float((alb - base).mean()), 2),
                            "median_opt_with": float(np.median(alb))}
        print(f"  {key}: 평균이동 {R['albedo'][key]['mean_shift_deg']:+.2f}° → 중앙최적 {R['albedo'][key]['median_opt_with']:.0f}°")

# ── [5] svf 휴리스틱 검증: Monte Carlo 시야계수 vs (1+cos t)/2 × svf(t) ─────
print("\n[5] svf 휴리스틱 vs Monte Carlo 시야계수 (cosine-weighted, 이웃 ±2 차폐):")
rng = np.random.default_rng(42)
NP_, ND = 64, 4000
tilts_mc = np.arange(15, 91, 5, dtype=float)
mc_rows = {}
for t in tilts_mc:
    tr = np.radians(t)
    n = np.array([np.sin(tr), np.cos(tr), 0.0])        # 발전면 법선 (위/바깥)
    t1 = np.array([np.cos(tr), -np.sin(tr), 0.0])      # 단면 방향
    t2 = np.array([0.0, 0.0, 1.0])                     # 블레이드 축
    u = (rng.random(NP_) * 2 - 1) * HD                 # 표면 점
    px, py = u * np.cos(tr), -u * np.sin(tr)
    ct_ = np.sqrt(rng.random((NP_, ND))); st_ = np.sqrt(1 - ct_**2)
    ph = rng.random((NP_, ND)) * 2 * np.pi
    d = (ct_[..., None] * n + (st_ * np.cos(ph))[..., None] * t1 + (st_ * np.sin(ph))[..., None] * t2)
    dx, dy = d[..., 0], d[..., 1]
    sky = dy > 1e-9                                    # 위로 가야 하늘
    blocked = np.zeros((NP_, ND), dtype=bool)
    for k in (1, 2, -1, -2):                           # 이웃 블레이드 (위2, 아래2)
        ax_, ay_ = -HD * np.cos(tr), k * DP + HD * np.sin(tr)
        bx_, by_ = HD * np.cos(tr), k * DP - HD * np.sin(tr)
        ox, oy = px[:, None] - ax_, py[:, None] - ay_
        ex, ey = bx_ - ax_, by_ - ay_
        den = dx * ey - dy * ex
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        s_hit = (ox * dy - oy * dx) / -den * -1        # param along segment
        # 표준 2D ray-segment: t_ray = (o × e)/(d × e), s_seg = (o × d)/(d × e)
        t_ray = (ox * ey - oy * ex) / den
        s_seg = (ox * dy - oy * dx) / den
        blocked |= (t_ray > 1e-9) & (s_seg >= 0) & (s_seg <= 1)
    mc = float((sky & ~blocked).mean())                # cosine-weighted 도달 비율
    model = float((1 + np.cos(tr)) / 2 * svf(np.array([t]))[0])
    mc_rows[int(t)] = {"mc": round(mc, 4), "model": round(model, 4),
                       "ratio_model_over_mc": round(model / max(mc, 1e-9), 2)}
R["svf_mc"] = mc_rows
for t in (15, 30, 45, 60, 75, 90):
    r_ = mc_rows[t]
    print(f"  tilt {t:>2}°: MC {r_['mc']:.3f} vs model {r_['model']:.3f} (model/MC = {r_['ratio_model_over_mc']})")

# ── [6] 동~서향 (az=90/270) 정량화 ───────────────────────────────────────────
print("\n[6] 동/서향 설치 정량화 (retain=(b) linear):")
fnb = RETAINS["(b) linear 1-sf"]
opt_s, _, _ = grid_optimal(el, az, dni, dhi, fnb)  # 남향 최적각(라벨에 해당)
R["east_west"] = {}
for sa_, nm in ((90.0, "동향"), (270.0, "서향")):
    opt_e, E_e, idx_e = grid_optimal(el, az, dni, dhi, fnb, sa=sa_)
    e_true = E_e[idx_e, np.arange(len(el))].sum()
    e_south_labels = energy_at(opt_s, el, az, dni, dhi, fnb, sa=sa_)
    e_60e = energy_at(np.full_like(el, 60.0), el, az, dni, dhi, fnb, sa=sa_)
    R["east_west"][nm] = {"median_opt": float(np.median(opt_e)),
                          "loss_using_south_labels_pct": round(float(1 - e_south_labels / e_true) * 100, 2),
                          "true_adv_vs_f60_pct": round(float(e_true / e_60e - 1) * 100, 2)}
    r_ = R["east_west"][nm]
    print(f"  {nm}(sa={sa_:.0f}): 중앙최적 {r_['median_opt']:.0f}° | 남향라벨 사용 시 손실 {r_['loss_using_south_labels_pct']}% | 진짜 AI vs F60 {r_['true_adv_vs_f60_pct']}%")

# ── [7] full-shade floor: sf=1인데 beam의 30%가 남는 유령 에너지 ─────────────
sf60 = panel_sf(np.full_like(el, 22.0), el, az)  # 눕힌 각에서 sf 분포 확인용
opt_a_sf = panel_sf(opt_a, el, az)
bp_a = beam_poa(opt_a, el, az, dni)
phantom = (0.3 * bp_a * (opt_a_sf >= 0.999)).sum()
total_a = eff_poa_v(opt_a, el, az, dni, dhi, RETAINS["(a) current 1-0.7sf"]).sum()
R["phantom_floor"] = {"hours_full_shade_with_beam": int(((opt_a_sf >= 0.999) & (bp_a > 1)).sum()),
                      "phantom_energy_pct_of_total": round(float(phantom / total_a) * 100, 3)}
print(f"\n[7] full-shade(sf=1)인데 beam 30% 잔존하는 유령항: 해당시간 {R['phantom_floor']['hours_full_shade_with_beam']}h, 총에너지의 {R['phantom_floor']['phantom_energy_pct_of_total']}%")

# ── [8] 온도 가짜중요도: temp ~ 계절/시각/GHI 설명가능성 ─────────────────────
h = pd.to_datetime(day["timestamp"]).dt.hour.to_numpy(float)
d_ = pd.to_datetime(day["timestamp"]).dt.dayofyear.to_numpy(float)
X = np.column_stack([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                     np.sin(2*np.pi*d_/365), np.cos(2*np.pi*d_/365), ghi, np.ones_like(h)])
y = day["temp_actual"].to_numpy(float)
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
r2_temp = 1 - ((y - X @ coef)**2).sum() / ((y - y.mean())**2).sum()
R["temp_spurious"] = {"r2_temp_from_doy_hour_ghi": round(float(r2_temp), 3)}
print(f"[8] temp_actual을 doy/hour/ghi로 설명: R²={r2_temp:.3f} → 높을수록 '12% 중요도=가짜상관' 지지")

with open("verify_fable_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\nsaved → verify_fable_results.json")
