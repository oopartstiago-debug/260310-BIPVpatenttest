# ==============================================================================
# "왜 최적각이 교과서 위도각(~37°) 아니라 81°인가" — 요소별 분해
#   (2026-07-01, 사용자 질문: 진짜 위아래 루버 자기음영 때문에 81°까지 가나?)
#
# 각도 규약: tilt=0°=수평(누움), 90°=수직(섬)  [pvlib surface_tilt]
# 서울 위도 37.5°N. 자유 단일 패널이면 남향 최적 tilt ≈ 위도.
#
# eff_poa = 직달[bp·IAM·(1−ov)] + 확산[dhi·F_sky·IAM] + 지면[ghi·alb·F_grd·IAM]
#   요소 스위치:
#     · 자기음영 ov : ON=위 블레이드가 아래 가림 / OFF=자유(그림자 없음)
#     · 시야계수 F  : DENSE=밀집슬랫+후방벽(view_factors_v17) / FREE=자유패널 (1±cosβ)/2
#   → 4조합의 최고 고정각을 비교해 37°→81° 경로를 정량 분해
# 실행: .venv/bin/python verify_angle_decomposition.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P
from physics_v2 import panel_sf

df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df.solar_elevation > 0].copy()
el = dd.solar_elevation.to_numpy(float); az = dd.solar_azimuth.to_numpy(float)
dni = dd.dni.to_numpy(float); dhi = dd.dhi.to_numpy(float)
N = len(dd)
TILTS = np.arange(0, 90.1, 1.0)
C = P.CHORD; PIT = P.PITCH
print(f"주간 {N}행 · 서울 위도 37.5° · 각도규약 0=수평 90=수직\n")


def poa(tilt, shading=True, dense_vf=True):
    tilt = np.asarray(tilt, float)
    beta = np.radians(tilt)
    # ── 직달 ──
    aoi = pvlib.irradiance.aoi(tilt, 180.0, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    if shading:
        sf = panel_sf(tilt, el, az, hd=C, p=PIT)
        ov = P.strip_shade(sf)
    else:
        ov = 0.0
    direct = bp * iam_b * (1 - ov)
    # ── 확산 IAM (tilt만의 함수, 자유/밀집 공통) ──
    sky, grd = pvlib.iam.martin_ruiz_diffuse(np.atleast_1d(tilt).ravel(), a_r=P.A_R)
    iam_sky = np.asarray(sky, float).reshape(np.shape(tilt))
    iam_grd = np.asarray(grd, float).reshape(np.shape(tilt))
    # ── 시야계수 ──
    if dense_vf:
        F_sky = P.view_factor(tilt, C, PIT, "f_sky")
        F_grd = P.view_factor(tilt, C, PIT, "f_grd")
    else:  # 자유 단일 패널 (지평선 트인 입면)
        F_sky = (1 + np.cos(beta)) / 2
        F_grd = (1 - np.cos(beta)) / 2
    ghi_est = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    diffuse = dhi * F_sky * iam_sky
    ground = ghi_est * P.ALBEDO * F_grd * iam_grd
    return np.maximum(direct + diffuse + ground, 0)


def best(shading, dense_vf):
    E = np.array([poa(np.full(N, t), shading, dense_vf).sum() for t in TILTS])
    bi = int(E.argmax())
    within1 = TILTS[(E[bi] - E) / E[bi] <= 0.01]
    return TILTS[bi], within1.min(), within1.max(), E


print("═" * 78)
print("요소별 최고 고정각 (0=수평 … 90=수직)")
print("─" * 78)
cases = [
    ("A. 자유 단일패널 (음영X·벽X) ← 교과서 위도각 나와야", False, False),
    ("B. + 자기음영만 (벽X)",                              True,  False),
    ("C. + 후방벽·밀집슬랫만 (음영X)",                      False, True),
    ("D. 자기음영 + 후방벽 = ★현 모델",                     True,  True),
]
res = {}
for name, sh, dv in cases:
    b, lo, hi, E = best(sh, dv)
    res[name] = (b, lo, hi)
    print(f"  {name:<40} 최고={b:>4.0f}°  (고원1% {lo:.0f}-{hi:.0f}°)")

print()
print("═" * 78)
print("37°→81° 를 무엇이 미는가 (각 요소 단독 기여)")
print("─" * 78)
a = res["A. 자유 단일패널 (음영X·벽X) ← 교과서 위도각 나와야"][0]
b = res["B. + 자기음영만 (벽X)"][0]
c = res["C. + 후방벽·밀집슬랫만 (음영X)"][0]
d = res["D. 자기음영 + 후방벽 = ★현 모델"][0]
print(f"  기준 자유패널        {a:>4.0f}°   (교과서 위도 37.5° 와 대조)")
print(f"  자기음영 단독이 밀어올림  {a:>3.0f}° → {b:>3.0f}°   (Δ {b-a:+.0f}°)")
print(f"  후방벽 단독이 밀어올림    {a:>3.0f}° → {c:>3.0f}°   (Δ {c-a:+.0f}°)")
print(f"  둘 다(현 모델)          {a:>3.0f}° → {d:>3.0f}°   (Δ {d-a:+.0f}°)")
print()

# ── 특정 각도들의 상대 발전량 (현 모델 기준) ────────────────────────────────
print("═" * 78)
print("현 모델(D)에서 각도별 연 발전량 (최고각 대비)")
print("─" * 78)
_, _, _, ED = best(True, True)
mx = ED.max()
for t in (0, 15, 30, 37, 45, 60, 75, 81, 90):
    e = poa(np.full(N, float(t)), True, True).sum()
    bar = "█" * int((e / mx) * 40)
    print(f"  {t:>2.0f}° {'(위도)' if t==37 else '(수직)' if t==90 else '(수평)' if t==0 else '      '}  "
          f"{(e/mx-1)*100:+6.2f}%  {bar}")
print()
print("해석: A가 ~37°(위도)면 → 모델 물리는 정상. B/C의 Δ가 37→81 의 진짜 원인.")
print("      자유패널이면 위도각, 밀집 루버라서 자기음영+후방벽이 저각을 죽여 고각으로 민다.")
