# ==============================================================================
# HDC α-envelope 비교계산 (2026-07-09) — f1·f2 몰라도 성립하는 상·하한 맞대결
#   근거(Fable 3방향 문헌추적 종결): KR102683082B1 [0077]·청구항4·도10로
#   α = [최대발전프로파일각 ↔ 완전개방 90°] 보간 가중치 구조 확정. 계수는 영업비밀.
#   → 어떤 f1·f2든 HDC 제어각은 θ(t)=θ_도8(t)+α·(90°−θ_도8(t)), α∈[0,1] 안.
#   → 발전량은 α=0(도8 그대로)~α=1(항상 90°) envelope로 엄밀히 바운드 가능.
#   도8(예시적[0190]): 8시30°·10시45°·12시90°·14시75°·16시45°·18시30° (0°=수평폐쇄).
#   가정: ①도8 시간보간=선형(민감도로 계단식도 병기) ②8시 이전/18시 이후 일조는
#         엣지값 30° 유지 ③α는 스냅샷 상수(실제는 시변이나 envelope엔 무관).
#   실행: .venv/bin/python verify_hdc_envelope.py
# ==============================================================================
import numpy as np, pandas as pd
import physics_v3 as P

df = pd.read_csv("bipv_ai_master_data_v17.csv")
df["timestamp"] = pd.to_datetime(df.timestamp)
df["month"] = df.timestamp.dt.month
df["hfrac"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60.0
dd = df[df.solar_elevation > 0].copy()
T = np.arange(0, 90.1, 1.0)

# ── HDC 도8 프로파일 (선형보간 / 계단식) ─────────────────────────────────────
H8 = np.array([8, 10, 12, 14, 16, 18], float)
A8 = np.array([30, 45, 90, 75, 45, 30], float)


def theta_hdc(hfrac, mode="linear"):
    if mode == "linear":
        return np.interp(hfrac, H8, A8)          # 범위 밖=엣지값(30°) 유지
    idx = np.clip(np.searchsorted(H8 + 1.0, hfrac), 0, len(A8) - 1)  # 계단: 각 시각±1h
    return A8[idx]


def season_energies(sub):
    """해당 부분집합에서 전 정책의 발전합."""
    el = sub.solar_elevation.to_numpy(float); az = sub.solar_azimuth.to_numpy(float)
    dni = sub.dni.to_numpy(float); dhi = sub.dhi.to_numpy(float)
    hf = sub.hfrac.to_numpy(float); N = len(sub)
    out = {}
    # 고정각 스윕 → 최고 고정각 + 순진 기준들
    Efix = {t: P.eff_poa(np.full(N, t), el, az, dni, dhi).sum() for t in T}
    bt = max(Efix, key=Efix.get)
    out["best_fixed_angle"] = bt
    out["best_fixed"] = Efix[bt]
    for t in (45, 38, 30, 15, 0):
        out[f"fixed_{t}"] = Efix[t]
    # 오라클(그 시각 발전최대각 = 이상적 최대발전프로파일)
    poa_all = np.stack([P.eff_poa(np.full(N, t), el, az, dni, dhi) for t in T])
    out["oracle"] = poa_all.max(axis=0).sum()
    # HDC envelope: α 스윕 (도8 선형보간)
    th0 = theta_hdc(hf, "linear")
    for a in np.arange(0.0, 1.001, 0.1):
        th = th0 + a * (90.0 - th0)
        out[f"hdc_a{a:.1f}"] = P.eff_poa(th, el, az, dni, dhi).sum()
    # 민감도: 계단식 도8, α=0
    out["hdc_a0_step"] = P.eff_poa(theta_hdc(hf, "step"), el, az, dni, dhi).sum()
    return out


def pct(x, base):
    return (x / base - 1) * 100


SEASONS = [("연간", dd), ("여름(6~8월)", dd[dd.month.isin([6, 7, 8])]),
           ("겨울(12~2월)", dd[dd.month.isin([12, 1, 2])])]

results = {}
for name, sub in SEASONS:
    results[name] = season_energies(sub)

# ── ① HDC envelope vs 정직 기준(최고 고정각) ────────────────────────────────
print("═" * 66)
print("① HDC 제어 envelope vs 최고 고정각 — 'f1·f2가 뭐든' 상·하한")
print(f"  {'구간':<12}{'최고고정':>7}{'도8(α=0)':>10}{'α최적':>12}{'완전개방(α=1)':>13}{'오라클':>9}")
for name, r in results.items():
    as_ = {a: r[f"hdc_a{a:.1f}"] for a in np.arange(0.0, 1.001, 0.1)}
    abest = max(as_, key=as_.get)
    b = r["best_fixed"]
    print(f"  {name:<12}{r['best_fixed_angle']:>5.0f}°"
          f"{pct(r['hdc_a0.0'], b):>+9.1f}%"
          f"{pct(as_[abest], b):>+8.1f}%(α={abest:.1f})"
          f"{pct(r['hdc_a1.0'], b):>+12.1f}%"
          f"{pct(r['oracle'], b):>+8.1f}%")
print("  (모든 %는 그 구간 최고 고정각 대비. HDC 실제값은 α=0~1 사이 어딘가)")

# ── ② "여름 +20% / 겨울 +12%"가 나오려면 어떤 기준이어야 하나 ────────────────
print("\n" + "═" * 66)
print("② HDC PR 수치 역추적 — envelope 최상단(α최적)조차 기준별로 몇 %인가")
print(f"  {'기준(고정식)':<22}{'여름 이득':>10}{'겨울 이득':>10}{'연간':>9}")
su, wi, yr = results["여름(6~8월)"], results["겨울(12~2월)"], results["연간"]
for lbl, key in [("최고 고정각(정직)", "best_fixed"), ("45° 관행", "fixed_45"),
                 ("38° 교과서", "fixed_38"), ("30° 순진", "fixed_30"),
                 ("15° 방치", "fixed_15"), ("0° 수평", "fixed_0")]:
    def envmax(r):
        return max(r[f"hdc_a{a:.1f}"] for a in np.arange(0.0, 1.001, 0.1))
    print(f"  {lbl:<22}{pct(envmax(su), su[key]):>+9.1f}%{pct(envmax(wi), wi[key]):>+9.1f}%"
          f"{pct(envmax(yr), yr[key]):>+8.1f}%")
print("  → '여름+20%·겨울+12%'와 동시에 맞는 행 = HDC가 암묵적으로 쓴 기준 후보")

# ── ③ 도8 프로파일의 물리 결함 재확인 + 냉각보정의 역설 ─────────────────────
print("\n" + "═" * 66)
print("③ 도8 그대로(α=0) vs 완전개방(α=1) — 아침·저녁 저각의 비용")
sub = dd[dd.month.isin([6, 7, 8])]
el = sub.solar_elevation.to_numpy(float); az = sub.solar_azimuth.to_numpy(float)
dni = sub.dni.to_numpy(float); dhi = sub.dhi.to_numpy(float)
hf = sub.hfrac.to_numpy(float)
th0 = theta_hdc(hf, "linear")
e0 = P.eff_poa(th0, el, az, dni, dhi)
e1 = P.eff_poa(np.full(len(sub), 90.0), el, az, dni, dhi)
for h0, h1, tag in [(5, 9, "아침(5~9시)"), (11, 13, "정오(11~13시)"), (17, 21, "저녁(17~21시)")]:
    m = (hf >= h0) & (hf < h1)
    print(f"  {tag:<14} 도8={e0[m].sum():>9.1f} vs 90°고정={e1[m].sum():>9.1f}"
          f"  → 도8이 {pct(e0[m].sum(), e1[m].sum()):+.1f}%")
print("  → 냉각보정(α↑)이 도8의 저각 결함을 '우연히' 메꾸는 구간 확인")

# ── ④ 민감도: 도8 보간 방식 ────────────────────────────────────────────────
print("\n" + "═" * 66)
print("④ 민감도 — 도8 선형보간 vs 계단식 (α=0, 최고고정 대비)")
for name, r in results.items():
    b = r["best_fixed"]
    print(f"  {name:<12} 선형 {pct(r['hdc_a0.0'], b):+.1f}%  계단 {pct(r['hdc_a0_step'], b):+.1f}%")

print("\n" + "═" * 66)
print("판정 가이드: envelope 전체(α=0~1)가 최고 고정각 아래면 → 'HDC 제어는 f1·f2와")
print("  무관하게 잘 고른 고정각을 못 이긴다'가 물리적 결론. PR 20%는 ②의 기준 선택 문제.")
