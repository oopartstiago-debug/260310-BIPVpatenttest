# ==============================================================================
# 허공-현실 재기준선 (2026-07-03) — 아파트 파사드·앞아래 허공 = 반사면 통제 불가.
#   지면반사는 "통제 못 하는 주변값"이므로, 알베도 0(순수허공)~0.20 스윕으로
#   ① 발전량 ② 최적 고정각 ③ 트래킹 이득 이 이 값에 얼마나 의존하는지 노출.
#   실기후 = PVGIS-SARAH2 서울 TMY. 밀폐 gcr1.169.
# ==============================================================================
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import pvlib
from pvlib.iotools import get_pvgis_tmy
import physics_v3 as P

CHORD, P0 = 114.0, 97.5
LAT, LON = 37.5665, 126.9780

data = get_pvgis_tmy(LAT, LON, map_variables=True)[0]
idx = data.index.tz_localize("UTC") if data.index.tz is None else data.index
sp = pvlib.solarposition.get_solarposition(idx, LAT, LON,
        pressure=data["pressure"].to_numpy(), temperature=data["temp_air"].to_numpy())
el = sp["apparent_elevation"].to_numpy(float); az = sp["azimuth"].to_numpy(float)
ghi = data["ghi"].to_numpy(float); dni = data["dni"].to_numpy(float); dhi = data["dhi"].to_numpy(float)
day = el > 3.0
e, a, dn, dh = el[day], az[day], dni[day], dhi[day]
N = day.sum()
TILTS = np.arange(0.0, 90.1, 1.0)

# 알베도별로 tilt별 연간 POA (eff_poa는 albedo 파라미터)
def annual_by_tilt(alb):
    return np.array([np.asarray(P.eff_poa(np.full(N, t), e, a, dn, dh,
                     c=CHORD, p=P0, albedo=alb)).ravel().sum() for t in TILTS])

# 기준 = 현 모델 albedo 0.15, 최고고정
ann015 = annual_by_tilt(0.15); base_gen = ann015.max(); base_t = TILTS[ann015.argmax()]
print(f"기준(현 모델): albedo 0.15 · 최고고정 tilt={base_t:.0f}° · 연간 발전(상대)={base_gen:.3e}\n")

print("지면 알베도 스윕 — 발전·최적각·트래킹 이득 민감도 (실기후 PVGIS):")
print(f"  {'알베도':>7}{'최적고정°':>9}{'연간발전 vs 0.15':>16}{'tilt 트래킹 이득':>16}")
print("  " + "-"*50)
for alb in (0.00, 0.05, 0.10, 0.15, 0.20):
    ann = annual_by_tilt(alb)
    best = ann.max(); bt = TILTS[ann.argmax()]
    # 트래킹 이득: 매 스텝 tilt 최적
    E = np.stack([np.asarray(P.eff_poa(np.full(N, t), e, a, dn, dh, c=CHORD, p=P0, albedo=alb)).ravel() for t in TILTS])
    trk = E.max(0).sum()
    print(f"  {alb:>7.2f}{bt:>8.0f}°{(best/base_gen-1)*100:>+14.2f}%{(trk/best-1)*100:>+15.2f}%")

print("\n지면반사 성분이 발전에서 차지하는 몫 (최적각 기준):")
for alb in (0.05, 0.10, 0.15):
    a0 = annual_by_tilt(0.0)   # 순수 허공(지면반사 0)
    aa = annual_by_tilt(alb)
    frac = (aa[int(base_t)] - a0[int(base_t)]) / aa[int(base_t)] * 100
    print(f"  albedo {alb:.2f}: 지면반사 몫 = {frac:.1f}%  (이만큼이 통제 불가 주변값)")

print("\n주해:")
print(" · 허공 위엔 반사면을 못 놓음 → 알베도를 '올릴' 수 없음(레버 A 사망).")
print(" · 실제 주변값=저 멀리 도시 지면·타 건물 → 대략 0.10~0.15 추정(통제 밖).")
print(" · 순수 허공(0.0) 가정은 하한. 이 표가 우리 발전 숫자의 낙관 여지를 보여줌.")
