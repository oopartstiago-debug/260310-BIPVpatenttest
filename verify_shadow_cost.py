# ==============================================================================
# 자기음영의 "진짜" 비용 (2026-07-03) — 사용자 지적 반영
#   앞 verify_loss_budget 오류: 운영각 78° 고정 후 '잔여' 그림자만 6.1%p로 잼.
#   → 자기음영이 38°→78°로 각도를 민 '원인'인데, 그 각도이동 비용을 별도 버킷으로 떼어
#     자기음영이 작아 보였음. 제대로 = 음영 ON/OFF 각각 '자기 최적각'에서 비교.
#
#   비교 (셀커버리지 factor 제외 = 단위 PV면적당, 순수 각도/음영/시야 물리):
#     (a) 자유패널 (음영×·시야차단× — 완전천장)         : 최적각 ≈ 위도
#     (b) 루버, 자기음영 OFF (시야차단 ON = 블레이드뒤 반공간): 최적각·에너지
#     (c) 루버, 풀 물리 (자기음영 ON + 시야차단 ON)        : 최적각 78°·에너지
#   → (a)-(b) = 하늘시야차단(반공간) 비용 · (b)-(c) = 자기음영 비용(각도이동 포함)
# 실행: .venv/bin/python verify_shadow_cost.py
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


def annual(t, shadow, vf):
    """단위 PV면적당 연간 POA(커버리지 factor 제외). shadow/vf = 물리 토글."""
    tt = np.full(N, t, float); b = np.radians(tt)
    aoi = pvlib.irradiance.aoi(tt, SA, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(tt.ravel(), a_r=P.A_R)
    ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    # 자기음영: strip 전체(frac=1, lo=0)로 켜고/끄기 = 순수 음영효과
    ov = P.strip_shade(panel_sf(tt, el, az, hd=C, p=P.PITCH, sa=SA), 0.0, 1.0) if shadow else 0.0
    beam = bp * iam_b * (1 - ov)
    if vf:   # 블레이드가 하늘/지면 시야 가림 (반공간 효과)
        sky_t = dhi * P.view_factor(tt, C, P.PITCH, "f_sky") * np.asarray(sky, float)
        grd_t = ghi * P.ALBEDO * P.view_factor(tt, C, P.PITCH, "f_grd") * np.asarray(grd, float)
    else:    # 완전천장 (자유패널)
        sky_t = dhi * (1 + np.cos(b)) / 2 * np.asarray(sky, float)
        grd_t = ghi * P.ALBEDO * (1 - np.cos(b)) / 2 * np.asarray(grd, float)
    return np.maximum(beam + sky_t + grd_t, 0).sum()


def opt(shadow, vf):
    E = np.array([annual(t, shadow, vf) for t in T])
    i = int(E.argmax()); return T[i], E[i]


ta, Ea = opt(False, False)   # (a) 자유패널
tb, Eb = opt(False, True)    # (b) 루버, 음영OFF·시야ON
tc, Ec = opt(True, True)     # (c) 루버, 풀물리

# 참고: 만약 음영ON인데 억지로 38°에 세우면?
E_c38 = annual(ta, True, True)

print(f"주간 {N}행 · 현{C:.0f}/피치{P.PITCH:.0f} gcr{C/P.PITCH:.3f} · 단위 PV면적당(커버리지 제외)\n")
print("═" * 62)
print(f"{'시나리오':<38}{'최적각':>7}{'연간(상대%)':>15}")
print("─" * 62)
print(f"{'(a) 자유패널 (음영× 시야차단×)':<38}{ta:>6.0f}°{100.0:>13.1f}%")
print(f"{'(b) 루버, 자기음영 OFF (시야차단만)':<38}{tb:>6.0f}°{Eb/Ea*100:>13.1f}%")
print(f"{'(c) 루버, 풀물리 (자기음영 ON)':<38}{tc:>6.0f}°{Ec/Ea*100:>13.1f}%")
print("═" * 62)
print(f"\n▶ 자기음영이 각도를 민 크기: {tb:.0f}° → {tc:.0f}°  (+{tc-tb:.0f}°)")
print(f"▶ 하늘시야차단(반공간) 비용  (a)-(b): {(Ea-Eb)/Ea*100:5.1f}%p")
print(f"▶ ★자기음영 진짜 비용 (b)-(c): {(Eb-Ec)/Ea*100:5.1f}%p   ← 각도이동 포함")
print(f"   (앞 예산의 '6.1%p'는 78°서 남은 잔여만 — 각도이동 비용을 뺀 값이라 과소)")
print(f"\n※ 참고: 자기음영 ON인데 억지로 {ta:.0f}°(자유최적)에 세우면 {E_c38/Ea*100:.1f}% — ")
print(f"   즉 음영 때문에 {ta:.0f}°는 재앙({E_c38/Ec*100-100:+.1f}% vs 78°), 그래서 78°로 도망친 것.")
