# ==============================================================================
# HDC 도8 프로파일이 자기음영을 반영하나? (2026-07-13)
#   질문(사용자): "HDC는 그림자 반영을 안 하는거지?"
#   방법: 시간별로 세 각도를 나란히 —
#     ① 그림자무시 최적각 = 자기음영 끈 상태서 발전최대각 (eff_poa strip_lo=10 → ov=0)
#     ② 그림자반영 최적각 = 우리 정식 physics (자기음영 on)
#     ③ HDC 도8 = 8시30°/10시45°/12시90°/14시75°/16시45°/18시30°
#   도8이 ①(무시)을 따라가면 = HDC 프로파일은 자기음영 미반영. ②를 따라가면 반영.
#   실행: .venv/bin/python verify_hdc_shadow.py
# ==============================================================================
import numpy as np, pandas as pd
import physics_v3 as P
from physics_v2 import panel_sf

df = pd.read_csv("bipv_ai_master_data_v17.csv")
df["timestamp"] = pd.to_datetime(df.timestamp)
df["month"] = df.timestamp.dt.month; df["hour"] = df.timestamp.dt.hour
dd = df[df.solar_elevation > 0].copy()
T = np.arange(0, 90.1, 1.0)

H8 = np.array([8, 10, 12, 14, 16, 18], float)
A8 = np.array([30, 45, 90, 75, 45, 30], float)


def opt_angle(sub, no_shadow):
    """그 부분집합 발전최대각. no_shadow=True면 자기음영 off(strip_lo=10)."""
    el = sub.solar_elevation.to_numpy(float); az = sub.solar_azimuth.to_numpy(float)
    dni = sub.dni.to_numpy(float); dhi = sub.dhi.to_numpy(float); N = len(sub)
    lo = 10.0 if no_shadow else P.STRIP_LO
    E = np.array([P.eff_poa(np.full(N, t), el, az, dni, dhi, strip_lo=lo).sum() for t in T])
    return T[E.argmax()]


for mn, mlabel in [(7, "7월 여름 (HDC 도7B/도8 대응)"), (4, "4월 (HDC 도7A)")]:
    sub = dd[dd.month == mn]
    print(f"\n{'═'*62}\n{mlabel}")
    print(f"  {'시각':>4}{'태양고도':>7} | {'①그림자무시최적':>13}{'②그림자반영최적':>13}{'③HDC도8':>9} | 도8이 따라가는 쪽")
    for h in sorted(sub.hour.unique()):
        s = sub[sub.hour == h]
        if len(s) < 5:
            continue
        el = s.solar_elevation.mean()
        a_no = opt_angle(s, True)
        a_sh = opt_angle(s, False)
        a8 = np.interp(h, H8, A8)
        # 도8이 어느 쪽에 더 가까운가
        near = "무시①" if abs(a8 - a_no) < abs(a8 - a_sh) else ("반영②" if abs(a8 - a_sh) < abs(a8 - a_no) else "중간")
        flag = ""
        if h <= 9 or h >= 16:
            # 저각 구간: 자기음영률 표시
            sf = panel_sf(np.full(len(s), a8), s.solar_elevation.to_numpy(float),
                          s.solar_azimuth.to_numpy(float), hd=P.CHORD, p=P.PITCH).mean()
            flag = f"  (도8각 자기음영 {sf*100:.0f}%)"
        print(f"  {h:>3}시{el:>6.1f}° | {a_no:>11.0f}°{a_sh:>12.0f}°{a8:>8.0f}° | {near}{flag}")

print(f"\n{'═'*62}")
print("해석: ①(그림자무시)은 태양을 그대로 좇아 저각으로 내려감. ②(그림자반영)은 자기음영")
print("  때문에 저각서도 고각 유지. HDC 도8이 아침·저녁 30°로 내려가면 = ① 쪽 = 자기음영 미반영.")
print("  단 정오 90°는 ②와도 일치 → 도8은 '내부 비일관'(엣지=무시, 정오=반영), [0190] 예시적 자인.")
