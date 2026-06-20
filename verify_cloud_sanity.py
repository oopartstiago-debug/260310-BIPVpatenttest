# ==============================================================================
# AI Tilt — 구름감쇠 모델 sanity-check (2026-06-19)
#   질문: louver_cloud 모델이 흐린날 예측각 0°를 내는데 degenerate(망가짐)인가,
#         아니면 물리적으로 옳은 학습결과(확산광 지배 → 저각 유리)인가?
#   방법: 검증된 물리 oracle(physics_v3.eff_poa, C# 포팅 40/40 PASS의 원본)에
#         SolarDayEnv.AttenuateByCloud 와 동일한 구름감쇠를 적용해
#         oracle argmax 각도가 맑음→흐림에서 어떻게 움직이는지 계산.
#   판정: oracle 도 흐림에서 저각이면 → 모델의 0° 는 옳은 추종(degenerate 아님).
#         oracle 은 고각인데 모델만 0° 면 → degenerate.
# ==============================================================================
import json
import numpy as np
from physics_v3 import eff_poa, CHORD, PITCH, ALBEDO

TILTS = np.arange(0, 90.1, 1.0)  # OracleTilt 와 동일한 1° 그리드 argmax


def attenuate_by_cloud(dni_cs, dhi_cs, elev_deg, C):
    """SolarDayEnv.AttenuateByCloud 의 1:1 포팅 (코드값 사용: dni=(1-C)^1.5)."""
    sinE = max(0.0, np.sin(np.radians(elev_deg)))
    ghi_cs = dni_cs * sinE + dhi_cs
    ghi = ghi_cs * (1.0 - 0.75 * C ** 3.4)
    dni = dni_cs * (1.0 - C) ** 1.5
    dhi = max(0.0, ghi - dni * sinE)
    return dni, dhi


def oracle_tilt(elev, az, dni, dhi, c=CHORD, p=PITCH, albedo=ALBEDO):
    poa = eff_poa(TILTS, elev, az, dni, dhi, c=c, p=p, albedo=albedo)
    poa = np.asarray(poa).ravel()
    k = int(np.argmax(poa))
    return TILTS[k], poa[k]


# ---------------------------------------------------------------------------
# (a) 합성 스윕 — 한여름 정오 기하 고정, 구름 0→1
# ---------------------------------------------------------------------------
print("=" * 72)
print("(a) 합성 스윕: 서울 한여름 정오(elev=72°, az=180°), clear-sky dni=880/dhi=110")
print("    구름량 C 를 0(맑음)→1(완전흐림)로 올리며 oracle 최적각 추적")
print("=" * 72)
print(f"{'C':>5} {'DNI':>7} {'DHI':>7} {'beam비중':>8} {'oracle각':>9} {'oraclePOA':>10}")
elev, az, dni_cs, dhi_cs = 72.0, 180.0, 880.0, 110.0
for C in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]:
    dni, dhi = attenuate_by_cloud(dni_cs, dhi_cs, elev, C)
    sinE = np.sin(np.radians(elev))
    beam_frac = (dni * sinE) / max(1e-6, dni * sinE + dhi)
    ot, op = oracle_tilt(elev, az, dni, dhi)
    print(f"{C:>5.2f} {dni:>7.1f} {dhi:>7.1f} {beam_frac:>7.1%} {ot:>8.0f}° {op:>10.1f}")

# ---------------------------------------------------------------------------
# (b) 실제 하루 — sun_days.json 에서 맑은날 / 흐린날 뽑아 시간별 oracle 각도
# ---------------------------------------------------------------------------
LIB = "unity_viz/AITiltViz/AITiltViz/Assets/StreamingAssets/sun_days.json"
days = json.load(open(LIB))["days"]


def day_mean_cloud(d):
    fr = [f for f in d["f"] if f["elev"] > 5]
    return np.mean([f["cloud"] for f in fr]) / 10.0 if fr else 0.0


def day_peak_dni(d):
    return max(f["dni"] for f in d["f"])


# 여름(6~8월)만 추려 맑은날/흐린날 대표 1개씩
summer = [d for d in days if d["date"][5:7] in ("06", "07", "08")]
summer_sorted = sorted(summer, key=day_mean_cloud)
clear_day = summer_sorted[0]                 # 가장 맑은 여름날
cloudy_day = summer_sorted[-1]               # 가장 흐린 여름날


def print_day(d, label):
    print("\n" + "-" * 72)
    print(f"({label})  {d['date']}  평균구름={day_mean_cloud(d):.2f}  peakDNI(clear-sky)={day_peak_dni(d):.0f}")
    print("-" * 72)
    print(f"{'시':>4} {'elev':>6} {'cloud':>6} {'DNI_cs':>7} {'DNI':>7} {'DHI':>7} {'oracle각':>9}")
    for f in d["f"]:
        if f["elev"] < 5:
            continue
        C = f["cloud"] / 10.0
        dni, dhi = attenuate_by_cloud(f["dni"], f["dhi"], f["elev"], C)
        if dni + dhi < 1:
            continue
        ot, _ = oracle_tilt(f["elev"], f["az"], dni, dhi)
        print(f"{f['h']:>4d} {f['elev']:>6.1f} {C:>6.2f} {f['dni']:>7.0f} {dni:>7.0f} {dhi:>7.0f} {ot:>8.0f}°")


print_day(clear_day, "맑은 여름날")
print_day(cloudy_day, "흐린 여름날")

# ---------------------------------------------------------------------------
# (c) 판정 요약 — 흐림(C 큰) 정오 시간대 oracle 각 vs 맑음 정오 oracle 각
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("(c) 판정")
print("=" * 72)
dni0, dhi0 = attenuate_by_cloud(880, 110, 72, 0.0)
dni1, dhi1 = attenuate_by_cloud(880, 110, 72, 1.0)
ot0, _ = oracle_tilt(72, 180, dni0, dhi0)
ot1, _ = oracle_tilt(72, 180, dni1, dhi1)
print(f"  맑음 정오 oracle 최적각 = {ot0:.0f}°   (beam 지배)")
print(f"  완전흐림 정오 oracle 최적각 = {ot1:.0f}°   (확산 지배)")
print(f"  → 흐림에서 oracle 이 {'저각으로 내려감(모델 0° = 옳은 추종, degenerate 아님)' if ot1 < ot0 - 20 else '여전히 고각(모델 0° 면 degenerate 의심)'}")
