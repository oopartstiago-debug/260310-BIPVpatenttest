# ==============================================================================
# 적대검토 검증 — "90° 닫힘 최적" 결론 (ADVERSARIAL_REVIEW_BRIEF.md §3)
# 실행: .venv/bin/python verify_adversarial.py
#   A. 시야계수 MC 기하 감사 — panel_sf(현=hd) vs view_factors_mc(현=2·hd) 2배 불일치 의심
#   B. 독립 전방 광선추적(panel_sf 미사용) — sf 교차검증 + 개구부 항등식(브리프 §2)
#   C. 교정 시야계수로 연간 최적각 재계산 — corner solution(90°) 생존 여부 + c/p 스윕
#   D. PV 띠 위치 — 그림자 띠가 현의 어느 쪽부터 드는지 + 마진 흡수 효과
# ==============================================================================
import json
import numpy as np
import pandas as pd
import pvlib

from physics_v2 import panel_sf

C_REAL, P_REAL = 97.5, 97.5   # 실기하: 현=피치 (도면 1043A)
ANGLES = np.arange(15, 91, 1.0)
R = {}

# ──────────────────────────────────────────────────────────────────────────────
# A. 시야계수 MC 기하 감사
# ──────────────────────────────────────────────────────────────────────────────
def vf_mc_orig(tilt_deg, ratio, n_pts=64, n_dir=4000, n_nb=4, seed=42):
    """train_v16.view_factors_mc 원본 그대로 복사 (import 시 재학습 실행 방지)"""
    hd, p = ratio, 1.0
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
        ax_, ay_ = -hd * np.cos(tr), k * p + hd * np.sin(tr)
        bx_, by_ = hd * np.cos(tr), k * p - hd * np.sin(tr)
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


def vf_mc_clean(tilt_deg, c_over_p, n_pts=64, n_dir=4000, n_nb=4, seed=42):
    """클린 파라미터화: 현 = c_over_p × 피치 (모호성 제거). 그 외 동일 기하/가정."""
    half, p = c_over_p / 2.0, 1.0   # ★유일한 차이: 블레이드 반폭 = 현/2
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


print("=" * 78)
print("[A] 시야계수 MC 기하 감사 — 현 폭 2배 불일치 검증")
print("=" * 78)
TILTS_CHK = [0.0, 30.0, 60.0, 90.0]
rows = []
for t in TILTS_CHK:
    o05 = vf_mc_orig(t, 0.5); o10 = vf_mc_orig(t, 1.0)
    c10 = vf_mc_clean(t, 1.0); c20 = vf_mc_clean(t, 2.0); c05 = vf_mc_clean(t, 0.5)
    rows.append({"tilt": t, "orig(r=0.5)": o05, "clean(c/p=1.0)": c10,
                 "orig(r=1.0)": o10, "clean(c/p=2.0)": c20, "clean(c/p=0.5)": c05})
    print(f" tilt {t:>4.0f}°  F_sky: orig(0.5)={o05[0]:.3f} ?= clean(c/p=1.0)={c10[0]:.3f} | "
          f"orig(1.0)={o10[0]:.3f} ?= clean(c/p=2.0)={c20[0]:.3f} | clean(c/p=0.5)={c05[0]:.3f}")
match_1 = all(abs(r["orig(r=0.5)"][0] - r["clean(c/p=1.0)"][0]) < 0.02 for r in rows)
match_2 = all(abs(r["orig(r=1.0)"][0] - r["clean(c/p=2.0)"][0]) < 0.02 for r in rows)
print(f" → orig 'ratio 0.5' ≡ 현/피치 1.0 : {match_1} | orig 'ratio 1.0' ≡ 현/피치 2.0 : {match_2}")
if match_1 and match_2:
    print(" ★ 확정: view_factors_mc 의 ratio 인자는 현/피치의 절반 의미 = 세션의 'ratio 1.0' 계산은")
    print("   물리적으로 불가능한 현/피치=2.0(블레이드 겹침)을 모델링했음 → 90° corner 의심 근거")
R["A_geometry_audit"] = {"rows": [{k: (v if not isinstance(v, tuple) else list(v)) for k, v in r.items()} for r in rows],
                         "orig0.5==clean1.0": match_1, "orig1.0==clean2.0": match_2}

# 무차폐 sanity: c/p→0 에서 (벽 포함) 해석값과 비교 — tilt 90° = 0.5 정확
fs_t90, fg_t90 = vf_mc_clean(90.0, 0.01)
print(f" sanity c/p→0, tilt 90°: F_sky={fs_t90:.3f} F_grd={fg_t90:.3f} (해석값 0.5/0.5)")
R["A_sanity_unobstructed_90"] = [fs_t90, fg_t90]

# ──────────────────────────────────────────────────────────────────────────────
# B. 독립 전방 광선추적 (panel_sf 미사용) — sf 교차검증 + 개구부 항등식
# ──────────────────────────────────────────────────────────────────────────────
def trace_beam(tilt, elev, az, c=C_REAL, p=P_REAL, sa=180.0, n_rays=40000):
    """태양 평행광선을 개구부 1주기에 균일 샘플 → 첫 충돌 블레이드/면/위치 추적.
    panel_sf 와 완전히 다른 방법(전방 추적 vs 점-그림자 교차).
    반환: sf_true(앞면 PV 음영률), capture(개구부 통과 빔 중 앞면 도달 비율),
          illum(s∈[0,1] 조명 구간 마스크 — 그림자 띠 위치 판별용)"""
    tr = np.radians(tilt); er = np.radians(np.clip(elev, 0.1, 89.9))
    adr = np.radians(az - sa)
    rdx, rdy = -np.cos(er) * np.cos(adr), -np.sin(er)   # in-plane 광선 방향 (panel_sf 동일 정의)
    if rdx >= -1e-9:   # 태양이 파사드 뒤 → 빔 0
        return 1.0, 0.0, np.zeros(200, dtype=bool)
    nbx, nby = np.sin(tr), np.cos(tr)                    # 블레이드 앞면 법선
    cos_aoi = -(rdx * nbx + rdy * nby)                   # in-plane 입사 (3D 공통 z성분은 약분)
    # 개구부(파사드 x=0, 1주기 y∈[0,p))를 지나는 광선 샘플: 시작점을 x=+L 평면에서 역산
    L = c * 2 + 10.0
    y_at_ap = (np.arange(n_rays) + 0.5) / n_rays * p     # 개구부 통과 y (균일 = 개구부 플럭스 균일)
    x0 = np.full(n_rays, L)
    y0 = y_at_ap - rdy / rdx * (0.0 - L)                 # x=L 에서의 y (개구부 도달점 기준 역산)
    # 충돌 대상 블레이드 범위: 광선이 x∈[0,L] 구간에서 지나는 y 범위
    y_min = min(y_at_ap.min(), y0.min()) - c; y_max = max(y_at_ap.max(), y0.max()) + c
    k_lo = int(np.floor((y_min - c) / p)) - 1; k_hi = int(np.ceil((y_max + c) / p)) + 1
    best_t = np.full(n_rays, np.inf); best_k = np.full(n_rays, 10**9); best_s = np.full(n_rays, np.nan)
    best_front = np.zeros(n_rays, dtype=bool)
    ex, ey = c * np.cos(tr), -c * np.sin(tr)             # 블레이드 벡터 (피벗→바깥팁, panel_sf 기하)
    for k in range(k_lo, k_hi + 1):
        ax_, ay_ = 0.0, k * p                            # 피벗 (파사드 면)
        qx, qy = ax_ - x0, ay_ - y0
        den = rdx * ey - rdy * ex
        if abs(den) < 1e-12:
            continue
        t_ray = (qx * ey - qy * ex) / den
        s_seg = (qx * rdy - qy * rdx) / den
        hit = (t_ray > 1e-9) & (s_seg >= 0.0) & (s_seg <= 1.0) & (t_ray < best_t)
        best_t = np.where(hit, t_ray, best_t)
        best_k = np.where(hit, k, best_k)
        best_s = np.where(hit, s_seg, best_s)
        best_front = np.where(hit, cos_aoi > 0, best_front)
    hit_any = np.isfinite(best_t)
    # 주기성: 모든 블레이드 동일 → 어느 블레이드든 앞면 히트의 s 분포 = 조명 구간
    front_hit = hit_any & best_front
    n_bins = 200
    illum = np.zeros(n_bins, dtype=bool)
    if front_hit.any():
        b = np.clip((best_s[front_hit] * n_bins).astype(int), 0, n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins)
        illum = cnt > 0
    # sf_true: 앞면 조명 현 길이 비율 (광선밀도→길이 변환: 1주기 개구부 플럭스 = 전체 가로채기)
    #   앞면 도달 파워 = front_hit 수 × (개구부 플럭스/n_rays) ; 무음영 앞면 파워 = cos_aoi·c
    #   개구부 플럭스(in-plane) = |rdx|·p
    P_ap = abs(rdx) * p
    P_front = front_hit.sum() / n_rays * P_ap
    P_noshade = max(cos_aoi, 0) * c
    sf_true = 1.0 - P_front / P_noshade if P_noshade > 1e-12 else 1.0
    capture = P_front / P_ap                              # 개구부 통과 빔 중 PV 앞면 도달 비율
    return float(np.clip(sf_true, 0, 1)), float(capture), illum


print()
print("=" * 78)
print("[B] 독립 광선추적 vs panel_sf + 개구부 항등식 (현=피치=97.5)")
print("=" * 78)
CASES = [("겨울 정오", 30.0, 180.0), ("여름 정오", 74.0, 180.0), ("봄 정오", 52.0, 180.0),
         ("겨울 오후", 20.0, 220.0), ("여름 오전", 45.0, 110.0)]
b_rows = []
for name, e_, a_ in CASES:
    print(f" {name} (elev {e_:.0f}°, az {a_:.0f}°):")
    for t in [0.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
        sf_m = float(panel_sf(t, e_, a_, hd=C_REAL, p=P_REAL))
        sf_t, cap, _ = trace_beam(t, e_, a_)
        ok = "✓" if abs(sf_m - sf_t) < 0.02 else "✗"
        print(f"   tilt {t:>4.0f}°: panel_sf={sf_m:.3f} vs 광선추적={sf_t:.3f} {ok} | 개구부 capture={cap:.3f}")
        b_rows.append({"case": name, "tilt": t, "sf_model": round(sf_m, 4), "sf_trace": round(sf_t, 4),
                       "capture": round(cap, 4), "match": bool(abs(sf_m - sf_t) < 0.02)})
R["B_raytrace_vs_panel_sf"] = b_rows
n_bad = sum(not r["match"] for r in b_rows)
print(f" → panel_sf 불일치 {n_bad}/{len(b_rows)}건")

# ──────────────────────────────────────────────────────────────────────────────
# C. 연간 재계산 — (a) 세션 재현(현/피치 2.0 VF) vs (b) 교정(현/피치 1.0 VF)
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("[C] 연간 최적각 재계산 — corner solution(90°) 생존 여부")
print("=" * 78)
df = pd.read_csv("bipv_ai_master_data_v15.csv")
day = df[df["ghi_w_m2"] >= 10]
el = day["solar_elevation"].to_numpy(float); az = day["solar_azimuth"].to_numpy(float)
dni = day["dni"].to_numpy(float); dhi = day["dhi"].to_numpy(float)
ghi = day["ghi_w_m2"].to_numpy(float)
ALB = 0.2
TILT_GRID = np.arange(0.0, 90.1, 2.5)

def beam_poa(tilt, elev, azim):
    return np.maximum(pvlib.irradiance.beam_component(surface_tilt=tilt, surface_azimuth=180.0,
        solar_zenith=90 - elev, solar_azimuth=azim, dni=(dni[None, :] if np.ndim(tilt) else dni)), 0)

def annual(vf_fn, label, cp):
    FS = np.array([vf_fn(t, cp)[0] for t in TILT_GRID])
    FG = np.array([vf_fn(t, cp)[1] for t in TILT_GRID])
    fs_i = np.interp(ANGLES, TILT_GRID, FS); fg_i = np.interp(ANGLES, TILT_GRID, FG)
    T = ANGLES[:, None]
    sf = panel_sf(T, el[None, :], az[None, :], hd=C_REAL, p=P_REAL)
    bp = np.maximum(pvlib.irradiance.beam_component(surface_tilt=T, surface_azimuth=180.0,
        solar_zenith=90 - el[None, :], solar_azimuth=az[None, :], dni=dni[None, :]), 0)
    ghi_e = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    E = np.maximum(bp * (1 - sf) + dhi[None, :] * fs_i[:, None] + ghi_e[None, :] * ALB * fg_i[:, None], 0)
    idx = np.argmax(E, axis=0)
    opt = ANGLES[idx]
    e_ai = E[idx, np.arange(E.shape[1])].sum()
    fixed = {}
    for fa in [37, 45, 60, 66, 70, 75, 80, 85, 90]:
        j = int(np.where(ANGLES == fa)[0][0]); fixed[fa] = E[j].sum()
    best_fa = max(fixed, key=fixed.get)
    out = {"median_opt": float(np.median(opt)), "p10": float(np.percentile(opt, 10)),
           "p90": float(np.percentile(opt, 90)), "best_fixed": int(best_fa),
           "ai_vs_60": round(float(e_ai / fixed[60] - 1) * 100, 2),
           "ai_vs_90": round(float(e_ai / fixed[90] - 1) * 100, 2),
           "ai_vs_best_fixed": round(float(e_ai / fixed[best_fa] - 1) * 100, 2),
           "fixed_rel_90": {str(k): round(float(v / fixed[90] - 1) * 100, 1) for k, v in fixed.items()}}
    print(f" {label}: 중앙최적 {out['median_opt']:.0f}° (p10–p90 {out['p10']:.0f}–{out['p90']:.0f}°) | "
          f"최고고정 {best_fa}° | AI vs60 +{out['ai_vs_60']}% vs90 {out['ai_vs_90']:+}% vs최고 +{out['ai_vs_best_fixed']}%")
    return out

R["C_session_repro_cp2.0"] = annual(vf_mc_clean, "(a) 세션 재현 = VF 현/피치 2.0 (버그)", 2.0)
R["C_corrected_cp1.0"] = annual(vf_mc_clean, "(b) 교정       = VF 현/피치 1.0 (실기하)", 1.0)
for cp in [0.85, 0.90, 0.95]:
    R[f"C_corrected_cp{cp}"] = annual(vf_mc_clean, f"(스윕) VF 현/피치 {cp}", cp)

# 겨울/여름 정오 분해 (브리프 §1 표 교정판)
print("\n [분해] 교정 VF(현/피치 1.0) 겨울 정오 (elev 30, dni 927, dhi 85) / 여름 정오 (74, 743, 180):")
for nm, e_, d_, h_ in [("겨울", 30.0, 927.0, 85.0), ("여름", 74.0, 743.0, 180.0)]:
    ghe = d_ * np.sin(np.radians(e_)) + h_
    for t in [0.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
        sfv = float(panel_sf(t, e_, 180.0, hd=C_REAL, p=P_REAL))
        bp = max(float(pvlib.irradiance.beam_component(surface_tilt=t, surface_azimuth=180.0,
                 solar_zenith=90 - e_, solar_azimuth=180.0, dni=d_)), 0)
        fs, fg = vf_mc_clean(t, 1.0)
        tot = bp * (1 - sfv) + h_ * fs + ghe * ALB * fg
        print(f"   {nm} tilt {t:>4.0f}°: 빔(순)={bp * (1 - sfv):6.0f} 확산={h_ * fs:5.0f} 지면={ghe * ALB * fg:5.0f} 합={tot:6.0f}")

# ──────────────────────────────────────────────────────────────────────────────
# D. PV 띠 위치 — 그림자 띠가 현의 어느 쪽부터?
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("[D] PV 띠 위치 — 그림자 띠 방향 + 마진 흡수 시 유효 음영")
print("=" * 78)
# 대표 케이스에서 조명 구간(s∈[0,1], 0=피벗/안쪽, 1=바깥팁) 확인
for name, e_, t_ in [("겨울 60°", 30.0, 60.0), ("여름 30°", 74.0, 30.0), ("봄 45°", 52.0, 45.0)]:
    sf_t, cap, illum = trace_beam(t_, e_, 180.0)
    s_lit = np.where(illum)[0]
    if len(s_lit):
        print(f" {name}: sf={sf_t:.3f} | 조명 s=[{s_lit.min() / 200:.2f}, {s_lit.max() / 200:.2f}] "
              f"→ 그림자는 s={'0(피벗/안쪽)' if s_lit.min() > 10 else '1(바깥팁)'} 쪽")
    else:
        print(f" {name}: 완전 음영")
# PV 띠 83/97.5 = 0.851 — 배치 3안 × 연간 (그림자=피벗쪽 가정은 위 결과로 검증)
STRIP = 83.0 / 97.5
def annual_strip(strip_lo, label):
    """sf(현 전체) → PV 띠 [strip_lo, strip_lo+STRIP] 유효 음영. 그림자 띠=[0, sf] (피벗쪽) 가정."""
    strip_hi = strip_lo + STRIP
    T = ANGLES[:, None]
    sf = panel_sf(T, el[None, :], az[None, :], hd=C_REAL, p=P_REAL)
    ov = np.clip(np.minimum(sf, strip_hi) - strip_lo, 0, None) / STRIP   # 띠와 그림자 겹침
    bp = np.maximum(pvlib.irradiance.beam_component(surface_tilt=T, surface_azimuth=180.0,
        solar_zenith=90 - el[None, :], solar_azimuth=az[None, :], dni=dni[None, :]), 0)
    FS = np.array([vf_mc_clean(t, 1.0)[0] for t in TILT_GRID])
    FG = np.array([vf_mc_clean(t, 1.0)[1] for t in TILT_GRID])
    fs_i = np.interp(ANGLES, TILT_GRID, FS); fg_i = np.interp(ANGLES, TILT_GRID, FG)
    ghi_e = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    E = np.maximum(bp * (1 - ov) + dhi[None, :] * fs_i[:, None] + ghi_e[None, :] * ALB * fg_i[:, None], 0)
    idx = np.argmax(E, axis=0); opt = ANGLES[idx]
    e_ai = E[idx, np.arange(E.shape[1])].sum()
    fixed = {fa: E[int(np.where(ANGLES == fa)[0][0])].sum() for fa in [60, 75, 90]}
    best = max(fixed, key=fixed.get)
    print(f" {label}: 중앙최적 {np.median(opt):.0f}° | AI vs60 +{(e_ai / fixed[60] - 1) * 100:.1f}% "
          f"vs90 {(e_ai / fixed[90] - 1) * 100:+.1f}%")
    return {"median_opt": float(np.median(opt)),
            "ai_vs_60": round(float(e_ai / fixed[60] - 1) * 100, 2),
            "ai_vs_90": round(float(e_ai / fixed[90] - 1) * 100, 2)}

R["D_strip_inner_margin"] = annual_strip(1 - STRIP, "PV 띠 바깥쪽 배치(마진=피벗쪽이 그림자 먼저 흡수)")
R["D_strip_outer_margin"] = annual_strip(0.0, "PV 띠 안쪽 배치(마진=바깥팁쪽)")
R["D_strip_center"] = annual_strip((1 - STRIP) / 2, "PV 띠 중앙 배치")

with open("verify_adversarial_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\nsaved → verify_adversarial_results.json")
