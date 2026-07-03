# ==============================================================================
# 트래킹 이득 폭포 분해 (2026-06-23) — "수직으로 맞춰주는데 왜 +0.5%뿐인가"
#   사용자 직관: 태양 향해 수직 정렬하는 트래커는 +25~35% 먹어야 함.
#   검증: 이 루버가 실제로 태양을 향할 수 있는가? 손실이 어디서 나는가?
#   (A) 자유패널 2축 트래커(태양 정조준, AOI=0)      ← 사용자가 상상하는 것
#   (B) 자유패널 단축(남향 고정·틸트만, 자오선 평면)  ← 루버의 회전 자유도
#   (C) 밀집 루버 단축(실제, 이웃 음영+뒷벽)          ← 실제 제품 = +0.5%
#   전부 "오라클(매시각 최적) vs 최고 고정각" = 트래킹 이득. open=등방성·음영없음.
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P

df = pd.read_csv("bipv_ai_master_data_v15.csv")
d = df[df.ghi_w_m2 >= 10].copy()
el = d.solar_elevation.to_numpy(float); zen = 90 - el
saz = d.solar_azimuth.to_numpy(float)
dni = d.dni.to_numpy(float); dhi = d.dhi.to_numpy(float)
ghi = dni * np.sin(np.radians(np.clip(el, 0, 90))) + dhi
A = np.arange(0, 90.1, 1.0); ALB = 0.15
cosd = lambda x: np.cos(np.radians(x))


def poa_open(tilt, sazi):
    """자유 경사면 POA (등방성 하늘, 음영·이웃 없음). tilt,sazi 스칼라 or 배열."""
    aoi = pvlib.irradiance.aoi(tilt, sazi, zen, saz)
    beam = np.maximum(dni * cosd(np.clip(aoi, 0, 90)), 0)
    sky = dhi * (1 + cosd(tilt)) / 2
    grd = ghi * ALB * (1 - cosd(tilt)) / 2
    return beam + sky + grd


# (A) 2축 트래커: surface_tilt=zenith, surface_az=sun_az → AOI=0 (태양 정조준)
E_track_dual = (dni                                   # 빔 전량 (AOI=0)
                + dhi * (1 + cosd(zen)) / 2
                + ghi * ALB * (1 - cosd(zen)) / 2).sum()
# 자유패널 최고 고정 (남향, 틸트 스캔) = 공통 baseline
ann_fix_free = np.array([poa_open(t, 180.0).sum() for t in A])
E_fix_free = ann_fix_free.max(); best_free = A[ann_fix_free.argmax()]
gain_A = (E_track_dual / E_fix_free - 1) * 100

# (B) 단축 트래커 (남향 고정, 틸트만 매시각 최적) — 자오선 평면
E_open = np.stack([poa_open(t, 180.0) for t in A])
E_track_1ax = E_open.max(0).sum()
gain_B = (E_track_1ax / E_fix_free - 1) * 100

# (C) 밀집 루버 단축 (실제 physics_v3)
E_dense = np.stack([np.asarray(P.eff_poa(np.full(len(el), t), el, saz, dni, dhi, sa=180.0)).ravel() for t in A])
ann_dense = E_dense.sum(1); E_fix_dense = ann_dense.max(); best_dense = A[ann_dense.argmax()]
gain_C = (E_dense.max(0).sum() / E_fix_dense - 1) * 100

print("트래킹 이득 폭포 (오라클 vs 최고 고정각)\n")
print(f"(A) 자유패널 2축 트래커 (태양 정조준 AOI=0)      = {gain_A:+6.1f}%   ← 사용자 직관(교과서 트래커)")
print(f"(B) 자유패널 단축 (남향고정·틸트만, 자오선평면)  = {gain_B:+6.1f}%   ← 루버의 회전 자유도로 제한")
print(f"(C) 밀집 루버 단축 (실제: 이웃음영+뒷벽)         = {gain_C:+6.1f}%   ← 실제 제품")
print(f"\n손실 분해:")
print(f"  A→B  방위 추적 불가(동/서 해를 못 향함)로 잃는 이득 = {gain_A-gain_B:5.1f}%p")
print(f"  B→C  밀집 자기음영+뒷벽으로 잃는 이득               = {gain_B-gain_C:5.1f}%p")
print(f"  → 교과서 트래커 이득 {gain_A:.0f}% 중 실제로 남는 건 {gain_C:.1f}% 뿐.")

# ── "수직으로 맞춰준다"의 실체: 오라클이 실제로 달성하는 AOI ──────────────────
print("\n'태양을 수직으로 맞춰준다'가 실제로 되는가 — 오라클이 달성하는 입사각(AOI)")
# 실제 루버 오라클 틸트
orac_tilt = A[E_dense.argmax(0)]
aoi_orac = pvlib.irradiance.aoi(orac_tilt, 180.0, zen, saz)
aoi_suntrack = np.zeros_like(aoi_orac)  # 2축은 0
hr = pd.to_datetime(d.timestamp).dt.hour.to_numpy()
print(f"  실제 루버 오라클의 AOI: 중앙값 {np.median(aoi_orac):.0f}°  (0°=완전정조준)")
print(f"    정오(11-13시) AOI 중앙값 = {np.median(aoi_orac[(hr>=11)&(hr<=13)]):.0f}°")
print(f"    아침(7-9시)   AOI 중앙값 = {np.median(aoi_orac[(hr>=7)&(hr<=9)]):.0f}°  ← 동쪽 해, 남향면이라 못 향함")
print(f"    저녁(16-18시) AOI 중앙값 = {np.median(aoi_orac[(hr>=16)&(hr<=18)]):.0f}°  ← 서쪽 해")
print(f"  오라클 틸트 분포: 중앙 {np.median(orac_tilt):.0f}° / p10-90 {np.percentile(orac_tilt,10):.0f}-{np.percentile(orac_tilt,90):.0f}°")
print(f"    → '맞춰준다'지만 AOI를 0으로 못 만든다(중앙 {np.median(aoi_orac):.0f}°). 빔의 cos({np.median(aoi_orac):.0f}°)={cosd(np.median(aoi_orac)):.2f}만 받음.")
print(f"    그나마 최고 고정각도 비슷한 AOI라 둘 차이가 작다 = 평탄 고원.")
