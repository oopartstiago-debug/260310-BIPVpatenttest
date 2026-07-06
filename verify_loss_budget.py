# ==============================================================================
# 광 손실 예산 (2026-07-03) — "음영 회피 새 아이디어" 발굴의 물리적 상한 판별
#   운영각(루버 자체 최적)에서, 자유패널 천장(=100%) 대비 손실을 버킷 분해:
#     ① 각도 페널티  : 자유패널을 루버 운영각으로 강제(35°→~78°, 단일패널 IAM+기하)
#     ② 자기음영     : 윗블레이드 그림자 (ov, 회수: 하단배치·스태거·리다이렉트)
#     ③ 하늘 시야차단: 이웃 블레이드가 확산광 가림 (F_sky<1, 회수: 피치확대=밀폐파괴)
#     ④ 상단여백     : 비PV 마진 (1-strip_frac, 회수: 셀 커버리지)
#     (지면반사·IAM은 term에 포함, 별도 표기)
#   각 버킷 = 회수 가능성이 다름 → 어떤 새 기구가 이론상 이득 가능한지 판별.
# 실행: .venv/bin/python verify_loss_budget.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P
from physics_v2 import panel_sf

df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df.solar_elevation > 0]
el = dd.solar_elevation.to_numpy(float); az = dd.solar_azimuth.to_numpy(float)
dni = dd.dni.to_numpy(float); dhi = dd.dhi.to_numpy(float)
N = len(dd)
T = np.arange(0, 90.1, 1.0)
SA = 180.0


def poa_free(t, ov_on=False, fsky_on=False, cover_on=False):
    """자유패널 기준에서 손실 토글을 하나씩 켠다. 반환=연간 합(단위 패널면적당).
    ov_on: 자기음영 · fsky_on: 하늘 시야차단 · cover_on: 상단여백(커버리지)."""
    t = np.full(N, t, float); b = np.radians(t)
    aoi = pvlib.irradiance.aoi(t, SA, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(t.ravel(), a_r=P.A_R)
    iam_sky = np.asarray(sky, float); iam_grd = np.asarray(grd, float)
    ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi

    # 빔: 자기음영 토글
    ov = P.strip_shade(panel_sf(t, el, az, hd=P.CHORD, p=P.PITCH, sa=SA),
                       P.STRIP_LO, P.STRIP_FRAC) if ov_on else 0.0
    beam = bp * iam_b * (1 - ov)
    # 하늘 확산: 시야계수 토글 (자유=완전천장 (1+cos)/2, 루버=F_sky)
    if fsky_on:
        sky_term = dhi * P.view_factor(t, P.CHORD, P.PITCH, "f_sky") * iam_sky
        grd_term = ghi * P.ALBEDO * P.view_factor(t, P.CHORD, P.PITCH, "f_grd") * iam_grd
    else:
        sky_term = dhi * (1 + np.cos(b)) / 2 * iam_sky
        grd_term = ghi * P.ALBEDO * (1 - np.cos(b)) / 2 * iam_grd
    poa = np.maximum(beam + sky_term + grd_term, 0)
    cover = P.STRIP_FRAC if cover_on else 1.0
    return poa.sum() * cover


# 자유패널 천장 (자체 최적각)
Efree = max(poa_free(t) for t in T)
tfree = T[int(np.argmax([poa_free(t) for t in T]))]

# 루버 운영각 = 풀 물리(모든 손실 on)에서 최적
def E_louver_full(t):
    return P.eff_poa(np.full(N, t), el, az, dni, dhi).sum() * P.STRIP_FRAC
Elou = np.array([E_louver_full(t) for t in T])
top = int(np.argmax(Elou)); t_op = T[top]

# t_op 에서 손실 누적 peel
A = poa_free(t_op)                                   # 자유패널을 운영각으로 (각도 페널티만)
B = poa_free(t_op, ov_on=True)                       # +자기음영
C = poa_free(t_op, ov_on=True, fsky_on=True)         # +하늘 시야차단
D = poa_free(t_op, ov_on=True, fsky_on=True, cover_on=True)  # +상단여백 = 실제 루버

pct = lambda x: x / Efree * 100
print(f"주간 {N}행 · 기하 현{P.CHORD:.0f}/피치{P.PITCH:.0f} gcr{P.CHORD/P.PITCH:.3f} · 셀커버 {P.STRIP_FRAC*100:.0f}% · 알베도 {P.ALBEDO}")
print(f"자유패널 천장: {tfree:.0f}° = {Efree/1e6:.3f} (=100%)")
print(f"루버 운영각(자체 최적): {t_op:.0f}°\n")
print("═" * 66)
print(f"{'단계 (운영각 '+f'{t_op:.0f}°'+' 고정)':<34}{'상대%':>9}{'이 손실':>11}")
print("─" * 66)
print(f"{'자유패널 천장 (최적각 '+f'{tfree:.0f}°)':<34}{100.0:>8.1f}%{'':>11}")
print(f"{'① 각도 페널티 (→운영각)':<34}{pct(A):>8.1f}%{pct(Efree-A):>10.1f}%")
print(f"{'② +자기음영 (ov)':<34}{pct(B):>8.1f}%{pct(A-B):>10.1f}%  ← 하단배치·스태거·리다이렉트")
print(f"{'③ +하늘 시야차단 (F_sky)':<34}{pct(C):>8.1f}%{pct(B-C):>10.1f}%  ← 피치확대만(밀폐파괴)")
print(f"{'④ +상단여백 (커버리지)':<34}{pct(D):>8.1f}%{pct(C-D):>10.1f}%  ← 셀 커버리지 확대")
print("─" * 66)
print(f"{'= 실제 루버 (per 파사드면적)':<34}{pct(D):>8.1f}%")
print("═" * 66)
print(f"\n총 손실 {100-pct(D):.1f}%p 중 회수 가능성별:")
print(f"  자기음영(②) {pct(A-B):.1f}%p — 부분 회수(하단배치 이미 일부, 스태거/리다이렉트 신규)")
print(f"  상단여백(④) {pct(C-D):.1f}%p — 회수 가능(커버리지, 배포레버)")
print(f"  하늘차단(③) {pct(B-C):.1f}%p — 회수 불가(밀폐가 피치 강제)")
print(f"  각도(①)     {pct(Efree-A):.1f}%p — 회수 불가(루버=near-vertical 숙명, 트래킹 상한 이미 규명)")
