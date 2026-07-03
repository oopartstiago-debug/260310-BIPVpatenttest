# ==============================================================================
# 축/방위 분해 (2026-06-23) — "동서 트래킹이 크면 동서형 BIPV가 효율 좋은 거 아냐?"
#   두 질문을 분리:
#   Q1. 고정 방위: 남향 vs 동향 vs 서향, 절대 발전량 누가 큰가? (트래킹 아님)
#   Q2. 트래킹 축: 어느 축이 이득 큰가? 동서스윕(방위/수직축) vs 상하(고도/E-W축) vs 2축
#   Q3. 동서 트래킹 축에도 밀집 음영을 넣으면 이득이 남는가? (infinite_sheds 근사)
#   전부 자유패널(등방성 하늘, 음영없음) 기준, 밀집은 별도 표기.
# ==============================================================================
import numpy as np, pandas as pd, pvlib
df = pd.read_csv("bipv_ai_master_data_v15.csv")
d = df[df.ghi_w_m2 >= 10].copy()
el = d.solar_elevation.to_numpy(float); zen = 90 - el
saz = d.solar_azimuth.to_numpy(float)
dni = d.dni.to_numpy(float); dhi = d.dhi.to_numpy(float)
ghi = dni * np.sin(np.radians(np.clip(el, 0, 90))) + dhi
ALB = 0.15; cosd = lambda x: np.cos(np.radians(x))
TIL = np.arange(0, 90.1, 2.0); AZ = np.arange(60, 300.1, 5.0)


def poa(tilt, sazi):
    aoi = pvlib.irradiance.aoi(tilt, sazi, zen, saz)
    return (np.maximum(dni * cosd(np.clip(aoi, 0, 90)), 0)
            + dhi * (1 + cosd(tilt)) / 2 + ghi * ALB * (1 - cosd(tilt)) / 2)


# ── Q1. 고정 방위 절대 발전량 (각 방위에서 최적 틸트) ────────────────────────
print("Q1. 고정 방위별 절대 발전량 (트래킹 X, 각 방위 최적 틸트) — 남향 기준 100")
base = max(poa(t, 180.0).sum() for t in TIL)
for sa, nm in [(180, "남향"), (135, "남동"), (90, "동향"), (270, "서향"), (225, "남서")]:
    e = max(poa(t, sa).sum() for t in TIL)
    print(f"   {nm}({sa}°): {100*e/base:5.1f}   (최적틸트 {TIL[np.argmax([poa(t,sa).sum() for t in TIL])]:.0f}°)")
print("   → 고정 동/서향은 남향보다 적게 받음. '동서형 고정 BIPV가 더 좋다'는 성립 안 함.\n")

# ── Q2. 트래킹 축별 이득 (자유패널, 오라클 vs 남향 최고고정) ──────────────────
print("Q2. 트래킹 축별 이득 (자유패널, vs 남향 최고고정)")
E_fix = base
# 2축: AOI=0
E_dual = (dni + dhi*(1+cosd(zen))/2 + ghi*ALB*(1-cosd(zen))/2).sum()
# 상하 축(E-W축, 남향고정·틸트만) = 우리 루버
E_updown = np.stack([poa(t, 180.0) for t in TIL]).max(0).sum()
# 동서 스윕(수직축, 틸트 고정·방위 추종): 각 틸트별 매시각 최적 방위, 그 중 최고 틸트
best_v = 0
for t in TIL:
    P = np.stack([poa(t, a) for a in AZ]).max(0).sum()
    best_v = max(best_v, P)
E_azim = best_v
print(f"   2축 (방위+상하, 태양정조준)        = {(E_dual/E_fix-1)*100:+6.1f}%")
print(f"   동서스윕 단축 (수직축, 방위추종)    = {(E_azim/E_fix-1)*100:+6.1f}%   ← 사용자가 주목한 축")
print(f"   상하 단축 (E-W축, 우리 루버의 축)   = {(E_updown/E_fix-1)*100:+6.1f}%")
print("   → 맞다: 동서(방위) 추종 축이 상하 축보다 이득 큼. 하지만 둘 다 '움직이는' 트래커.\n")

# ── Q3. 동서스윕 축에 밀집 음영을 넣으면? (infinite_sheds, 수직핀 밀집행 근사) ──
print("Q3. 동서스윕(수직핀) 축도 밀집하면 음영이 이득을 먹나 (infinite_sheds 근사)")
import warnings; warnings.filterwarnings("ignore")
from pvlib.bifacial import infinite_sheds
# 수직핀이 방위로 회전 = 행 방위가 시각마다 변하는 트래커. 근사로 단축 트래커+gcr 음영.
# pvlib 단축 트래커(N-S 수평축=동서틸트) + gcr 자기음영(backtrack 없음)으로 상한 근사.
for gcr in (0.99, 0.7, 0.5):
    tr = pvlib.tracking.singleaxis(np.clip(zen,0,89.9), saz, axis_tilt=0, axis_azimuth=180,
                                   max_angle=90, backtrack=True, gcr=gcr)
    st = np.asarray(tr["surface_tilt"], float); sz = np.asarray(tr["surface_azimuth"], float)
    st = np.nan_to_num(st); sz = np.nan_to_num(sz, nan=180.0)
    r = infinite_sheds.get_irradiance(surface_tilt=st, surface_azimuth=sz,
        solar_zenith=np.clip(zen,0,89.9), solar_azimuth=saz, gcr=min(gcr,0.999),
        height=1.0, pitch=1.0, ghi=ghi, dhi=dhi, dni=dni, albedo=ALB, model="isotropic")
    e_track = np.nansum(np.asarray(r["poa_front"], float))
    # 같은 gcr 최고 고정(남향 틸트스캔)
    bestfix = 0
    for t in TIL:
        rf = infinite_sheds.get_irradiance(surface_tilt=float(t), surface_azimuth=180.0,
            solar_zenith=np.clip(zen,0,89.9), solar_azimuth=saz, gcr=min(gcr,0.999),
            height=1.0, pitch=1.0, ghi=ghi, dhi=dhi, dni=dni, albedo=ALB, model="isotropic")
        bestfix = max(bestfix, np.nansum(np.asarray(rf["poa_front"], float)))
    print(f"   gcr={gcr:.2f}: 동서 단축트래커(backtrack) vs 최고고정 = {(e_track/bestfix-1)*100:+6.1f}%")
print("   → 밀집(gcr≈1)에선 동서 트래킹도 음영/백트래킹으로 이득 크게 깎임. 성기면 회복.")
