# ==============================================================================
# 최대발전프로파일 분석 (2026-07-08) — 우리 오라클의 시간별 최적각 (HDC 도7A·7B 대응)
#   ① 4월·7월 시간별 최적각(월평균) = "최대발전프로파일" 실제 모양
#   ② "왜 각도 못 눕히나" = 특정 시각 발전 vs 각도 곡선 → 저각이 자기음영으로 죽는지 직접 표시
#   핵심 질문: 프로파일이 저각으로 내려가나? 안 내려가면 그 이유가 음영인가?
#   실행: .venv/bin/python verify_maxgen_profile.py
# ==============================================================================
import numpy as np, pandas as pd
import physics_v3 as P
from physics_v2 import panel_sf

df = pd.read_csv("bipv_ai_master_data_v17.csv")
df["timestamp"] = pd.to_datetime(df.timestamp)
df["month"] = df.timestamp.dt.month; df["hour"] = df.timestamp.dt.hour
dd = df[df.solar_elevation > 0].copy()
T = np.arange(0, 90.1, 1.0)


def hourly_profile(sub):
    """시간별 최대발전각 (그 월·시각 발전합 최대각) + 그 시각 평균 태양고도."""
    rows = []
    for h in sorted(sub.hour.unique()):
        s = sub[sub.hour == h]
        if len(s) < 5: continue
        el = s.solar_elevation.to_numpy(float); az = s.solar_azimuth.to_numpy(float)
        dni = s.dni.to_numpy(float); dhi = s.dhi.to_numpy(float)
        E = np.array([P.eff_poa(np.full(len(s), t), el, az, dni, dhi).sum() for t in T])
        rows.append((h, T[E.argmax()], el.mean()))
    return rows


for mn, mlabel in [(4, "4월 (HDC 도7A 대응)"), (7, "7월 (HDC 도7B 대응)")]:
    print(f"\n{'═'*56}\n최대발전프로파일 — {mlabel}")
    print(f"  {'시각':>4}{'최적각':>8}{'태양고도':>9}")
    for h, ang, el in hourly_profile(dd[dd.month == mn]):
        bar = "█" * int(ang / 90 * 20)
        print(f"  {h:>3}시{ang:>7.0f}°{el:>8.1f}°  {bar}")

# ── ② "왜 저각으로 못 내려가나" — 특정 시각 발전 vs 각도 (음영 대비) ──────────
print(f"\n{'═'*56}\n② 발전 vs 각도 곡선 — 저각이 왜 죽나 (7월, 태양 높은 정오 vs 낮은 아침)")
jul = dd[dd.month == 7]
for h, tag in [(12, "정오(태양 높음)"), (8, "아침(태양 낮음)")]:
    s = jul[jul.hour == h]
    el = s.solar_elevation.to_numpy(float); az = s.solar_azimuth.to_numpy(float)
    dni = s.dni.to_numpy(float); dhi = s.dhi.to_numpy(float)
    print(f"\n  [{h}시 {tag}, 평균 태양고도 {el.mean():.0f}°]")
    print(f"  {'각도':>5}{'발전(상대)':>10}{'자기음영률':>11}")
    E = {t: P.eff_poa(np.full(len(s), t), el, az, dni, dhi).sum() for t in [30, 40, 50, 60, 70, 80, 90]}
    Emax = max(E.values())
    for t in [30, 40, 50, 60, 70, 80, 90]:
        sf = panel_sf(np.full(len(s), float(t)), el, az, hd=P.CHORD, p=P.PITCH).mean()
        mark = " ← 최적" if E[t] == Emax else ""
        print(f"  {t:>4}°{E[t]/Emax*100:>9.1f}%{sf*100:>10.0f}%{mark}")

print(f"\n{'═'*56}")
print("핵심: 최적각 프로파일이 저각으로 안 내려가는 이유 = '발전 최대화'가 이미 음영을 회피.")
print("  저각일수록 자기음영률↑ → 발전↓ → 최대화가 알아서 고각을 고름. 별도 음영제한 불필요.")
print("  → HDC도 '최대발전' 목적이면 같은 물리서 같은 고각 프로파일이 나와야 정상.")
print("  → HDC 프로파일이 저각으로 내려간다면: 기하가 성겨서(겹침 적음) or 도면이 예시적([0190]).")
