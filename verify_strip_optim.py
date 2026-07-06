# ==============================================================================
# 셀 스트립 최적화 (2026-07-03) — "음영 회피 × 커버리지" 결합 최적점
#   손실예산(verify_loss_budget): 자기음영 6.1%p(작음) < 상단여백 17.1%p(큼).
#   상단여백이 존재하는 이유 = 그림자 흡수(하단배치). → 여백↓(커버리지↑)와 음영노출 트레이드오프.
#   ① 운영각서 그림자밴드 sf(mm) 실제 분포 — 여백을 얼마나 남겨야 하나
#   ② (상단여백 lo, 셀폭 w) 스윕 → 연간 발전(=POA×커버리지) 최대점
#      제약: lo+w ≤ 114(현), w ≤ 조립가능 최대. 현 설계 lo24/w83 대비 이득?
#   에너지 기준 = per 파사드면적 = poa(유효음영 ov) × (w/현)
# 실행: .venv/bin/python verify_strip_optim.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P
from physics_v2 import panel_sf

df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df.solar_elevation > 0]
el = dd.solar_elevation.to_numpy(float); az = dd.solar_azimuth.to_numpy(float)
dni = dd.dni.to_numpy(float); dhi = dd.dhi.to_numpy(float)
N = len(dd); C = P.CHORD; SA = 180.0
T = np.arange(0, 90.1, 1.0)


def poa_at(t, lo_frac, w_frac):
    """운영각 t 고정, 상단여백 lo_frac·셀폭 w_frac (현 대비). 연간 POA 합(셀면적당, 유효음영 반영)."""
    tt = np.full(N, t, float); b = np.radians(tt)
    sf = panel_sf(tt, el, az, hd=C, p=P.PITCH, sa=SA)
    ov = P.strip_shade(sf, lo_frac, w_frac)
    aoi = pvlib.irradiance.aoi(tt, SA, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(tt.ravel(), a_r=P.A_R)
    ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    poa = np.maximum(bp * iam_b * (1 - ov)
                     + dhi * P.view_factor(tt, C, P.PITCH, "f_sky") * np.asarray(sky, float)
                     + ghi * P.ALBEDO * P.view_factor(tt, C, P.PITCH, "f_grd") * np.asarray(grd, float), 0)
    return poa.sum()


# 운영각 = 현 설계에서 최적
def E_design(t):
    return poa_at(t, P.STRIP_LO, P.STRIP_FRAC) * P.STRIP_FRAC
t_op = T[int(np.argmax([E_design(t) for t in T]))]

# ── ① 운영각서 그림자밴드 sf(mm) 분포 (빔 있는 시간만, 가중) ────────────────
tt = np.full(N, t_op, float)
sf = panel_sf(tt, el, az, hd=C, p=P.PITCH, sa=SA)          # 현 전체 음영률
aoi = pvlib.irradiance.aoi(tt, SA, 90 - el, az)
beam = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
w = beam / beam.sum()                                       # 빔에너지 가중
sf_mm = sf * C
print(f"주간 {N}행 · 현{C:.0f}/피치{P.PITCH:.0f} · 운영각 {t_op:.0f}°")
print(f"현 설계: 상단여백 {P.STRIP_LO*C:.0f}mm / 셀 {P.STRIP_FRAC*C:.0f}mm / 하단 {C-P.STRIP_LO*C-P.STRIP_FRAC*C:.0f}mm\n")
print("── ① 운영각 그림자밴드 sf (블레이드 상단부터, 빔에너지 가중) ──")
for q in [50, 75, 90, 95, 99]:
    # 빔가중 분위수
    idx = np.argsort(sf_mm); cs = np.cumsum(w[idx])
    v = sf_mm[idx][np.searchsorted(cs, q / 100)]
    print(f"  {q}th pct: {v:5.1f}mm")
print(f"  빔가중 평균: {(sf_mm*w).sum():.1f}mm   최대: {sf_mm.max():.1f}mm")
print(f"  → 현 상단여백 24mm이 흡수하는 그림자 범위 대비 판단\n")

# ── ② (상단여백 lo, 셀폭 w) 스윕 ──────────────────────────────────────────
print("── ② 스트립 배치 스윕 (운영각 고정, per 파사드면적 발전) ──")
print(f"{'상단여백mm':>9}{'셀폭mm':>8}{'하단mm':>8}{'커버%':>7}{'상대발전%':>10}")
base = E_design(t_op)
best = (None, -1)
# 셀폭 후보 (83=현, 확대), 상단여백 후보
for wmm in [83, 90, 95, 100, 107]:
    for lomm in [0, 7, 14, 24]:
        if lomm + wmm > C:      # 물리 제약: 현 안에 들어와야
            continue
        wf = wmm / C; lf = lomm / C
        E = poa_at(t_op, lf, wf) * wf     # per 파사드 = POA(셀면적당) × 커버리지
        rel = E / base * 100
        tag = "  ← 현 설계" if (wmm == 83 and lomm == 24) else ""
        print(f"{lomm:>9}{wmm:>8}{C-lomm-wmm:>8.0f}{wf*100:>6.0f}%{rel:>9.1f}%{tag}")
        if E > best[1]:
            best = ((lomm, wmm), E)
print("─" * 44)
(blo, bw), bE = best
print(f"최적: 상단여백 {blo}mm · 셀 {bw}mm · 하단 {C-blo-bw:.0f}mm → {bE/base*100:.1f}% (현 대비 +{bE/base*100-100:.1f}%)")
print(f"\n※ 셀폭 확대는 M6 하프셀 물리치수(83mm)·조립·배선 human-gate 있음 — 이 수치는 '기하 상한'")
