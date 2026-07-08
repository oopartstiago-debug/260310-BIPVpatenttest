# ==============================================================================
# "기준이 다르면 뭐가 맞나" 검증 (2026-07-08) — HDC 자동각도제어 vs 우리
#   확인: HDC "최대발전프로파일"(특허 [0158,0170]) = 우리 오라클(그 시각 발전최대각). 동일 물리.
#   그럼 "HDC 여름20% vs 우리 +2.6%"는 물리 이견인가, 비교 기준 차이인가?
#   → 같은 오라클을 여러 baseline에 대보면, 기준에 따라 숫자가 +2.6%~+수십% 다 나오는지 판정.
#   실행: .venv/bin/python verify_baseline.py
# ==============================================================================
import numpy as np, pandas as pd
import physics_v3 as P

df = pd.read_csv("bipv_ai_master_data_v17.csv")
df["month"] = pd.to_datetime(df.timestamp).dt.month
dd = df[df.solar_elevation > 0].copy()
T = np.arange(0, 90.1, 1.0)


def energies(sub):
    el = sub.solar_elevation.to_numpy(float); az = sub.solar_azimuth.to_numpy(float)
    dni = sub.dni.to_numpy(float); dhi = sub.dhi.to_numpy(float); N = len(sub)
    # 각 고정각 연간합
    Efix = {t: P.eff_poa(np.full(N, t), el, az, dni, dhi).sum() for t in T}
    best_t = max(Efix, key=Efix.get); Ebest = Efix[best_t]
    # 오라클 = 매 시각 최선각 (= HDC 최대발전프로파일 = 완전추적)
    poa_all = np.stack([P.eff_poa(np.full(N, t), el, az, dni, dhi) for t in T])  # [nT, N]
    Eoracle = poa_all.max(axis=0).sum()
    return Efix, best_t, Ebest, Eoracle


def report(name, sub):
    Efix, bt, Eb, Eo = energies(sub)
    print(f"\n{'═'*58}\n{name} ({len(sub)}시간)  · 오라클(=HDC 최대발전프로파일) 기준")
    print(f"  최적 고정각 = {bt:.0f}°")
    print(f"  {'비교 기준(baseline)':<26}{'오라클 이득':>14}")
    print(f"  {'─'*40}")
    rows = [("최고 고정각 "+f"{bt:.0f}°"+" (가장 정직)", Eb),
            ("회사관행 45°", Efix[45]), ("교과서 ≈38° (위도37.5, 1°격자)", Efix[38]),
            ("순진 30°", Efix[30]), ("방치 15°", Efix[15]), ("수평 0°", Efix[0])]
    for lbl, Eb0 in rows:
        print(f"  {lbl:<26}{(Eo/Eb0-1)*100:>+12.1f}%")


report("연간", dd)
report("여름 (6~8월)", dd[dd.month.isin([6, 7, 8])])
print(f"\n{'═'*58}")
print("판정: 오라클(=HDC와 동일 각도제어)의 이득은 baseline에 따라 +수%~+수십%.")
print("  우리 '+2.6%'=최고 고정각(가장 엄격) 기준. HDC PR '여름20%'=순진 고정각 기준일 때 재현되면")
print("  → 물리 이견 아님, 같은 제어법의 '기준 선택' 차이. 우리가 더 엄격/정직한 기준을 씀.")
