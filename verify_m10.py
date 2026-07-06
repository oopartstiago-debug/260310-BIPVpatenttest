# ==============================================================================
# 셀 규격 교체 이득 (2026-07-03) — 사용자: "여백을 M10 하프셀로 채우면 몇 %?"
#   현 M6 하프셀 83mm → M10 하프셀 91mm (182mm 웨이퍼 반절). 현 114mm 폭.
#   각 구성 = 각자 최적각(오라클)에서 연간 발전(per 파사드면적 = POA(유효음영) × 커버리지).
#   상단여백은 그림자 흡수용 → 91mm는 상단여백 최대 23mm(=현24와 유사)로 배치.
# 실행: .venv/bin/python verify_m10.py
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


def poa_cell(t, lo_frac, w_frac):
    tt = np.full(N, t, float); b = np.radians(tt)
    ov = P.strip_shade(panel_sf(tt, el, az, hd=C, p=P.PITCH, sa=SA), lo_frac, w_frac)
    aoi = pvlib.irradiance.aoi(tt, SA, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(tt.ravel(), a_r=P.A_R)
    ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    return np.maximum(bp * iam_b * (1 - ov)
                      + dhi * P.view_factor(tt, C, P.PITCH, "f_sky") * np.asarray(sky, float)
                      + ghi * P.ALBEDO * P.view_factor(tt, C, P.PITCH, "f_grd") * np.asarray(grd, float), 0).sum()


def best_facade(wmm, lomm):
    """셀폭 wmm·상단여백 lomm. 각자 최적각서 per 파사드면적 연간 발전 + 최적각."""
    wf, lf = wmm / C, lomm / C
    E = np.array([poa_cell(t, lf, wf) * wf for t in T])
    i = int(E.argmax()); return E[i], T[i]


# 현: M6 83mm, 상단24/하7
E_cur, t_cur = best_facade(83, 24)
# M10: 91mm, 상단여백 스윕(최대 23=114-91)에서 최적
cands = [(91, lo) for lo in [0, 7, 14, 23]]
print(f"주간 {N}행 · 현{C:.0f}/피치{P.PITCH:.0f} · 셀 길이방향은 발전밀도에 중립(면적비만)\n")
print(f"현재 (M6 하프 83mm, 상24/하7): 최적각 {t_cur:.0f}° · 커버 {83/C*100:.0f}% = 기준 100%\n")
print(f"{'구성':<28}{'상단여백':>7}{'하단':>6}{'커버%':>7}{'최적각':>7}{'vs현재':>9}")
print("─" * 66)
best = (None, -1)
for wmm, lomm in cands:
    if lomm + wmm > C: continue
    E, t = best_facade(wmm, lomm)
    rel = E / E_cur * 100
    print(f"{'M10 하프 91mm':<28}{lomm:>6}{C-lomm-wmm:>6.0f}{wmm/C*100:>6.0f}%{t:>6.0f}°{rel-100:>+8.1f}%")
    if E > best[1]: best = ((wmm, lomm, t), E)
(bw, blo, bt), bE = best
print("─" * 66)
print(f"★ M10 최선: 91mm·상단여백 {blo}mm·하단 {C-blo-bw:.0f}mm → 현재 대비 +{bE/E_cur*100-100:.1f}%")
print(f"\n[참고] 더 큰 규격도:")
for name, wmm in [("G12 하프 105mm", 105), ("M6 풀 166→미stack", 83)]:
    if wmm > C:
        print(f"  {name}: {wmm}mm > 현{C:.0f}mm → 단일 배치 불가(현방향 안 들어감)")
        continue
    E, t = best_facade(wmm, min(24, C - wmm))
    print(f"  {name}: 상단여백 {min(24,C-wmm):.0f}mm → +{E/E_cur*100-100:.1f}% (최적각 {t:.0f}°)")
print(f"\n※ 가정: 셀 효율(단위면적당 W)은 M6=M10 동일(같은 실리콘). 이득=순수 면적(커버리지)비.")
print(f"※ 배선·직렬수·조립 human-gate 별도. 길이방향 셀 개수 변화는 발전밀도 중립(면적만 반영).")
