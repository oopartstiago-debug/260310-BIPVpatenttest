# ==============================================================================
# 발전량 상승 레버 정량 (2026-07-03) — 트래킹(각도%)이 죽었으니 절대 발전 레버를 뒤진다.
#   실기후 = PVGIS-SARAH2 서울 TMY. 밀폐 gcr1.169, 최고고정 tilt.
#   레버 ① 양면 후면포집(infinite_sheds poa_back) vs 내부 알베도
#        ② 전면 알베도(루버 아래 반사면)
#        ③ 셀 커버리지(스트립 폭) — 블레이드당 셀 면적
# ==============================================================================
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import pvlib
from pvlib.iotools import get_pvgis_tmy
from pvlib.bifacial import infinite_sheds
import physics_v3 as P

CHORD, P0 = 114.0, 97.5
LAT, LON = 37.5665, 126.9780
BIFACIALITY = 0.70                    # 후면 감도 (일반 양면셀 0.65~0.75)

data = get_pvgis_tmy(LAT, LON, map_variables=True)[0]
idx = data.index.tz_localize("UTC") if data.index.tz is None else data.index
sp = pvlib.solarposition.get_solarposition(idx, LAT, LON,
        pressure=data["pressure"].to_numpy(), temperature=data["temp_air"].to_numpy())
el = sp["apparent_elevation"].to_numpy(float); az = sp["azimuth"].to_numpy(float)
zen = np.clip(sp["apparent_zenith"].to_numpy(float), 0, 89.9)
ghi = data["ghi"].to_numpy(float); dni = data["dni"].to_numpy(float); dhi = data["dhi"].to_numpy(float)
day = el > 3.0
e, a, z, dn, dh, gh = el[day], az[day], zen[day], dni[day], dhi[day], ghi[day]
N = day.sum()

# 최고고정 tilt (전면, 알베도 0.15 기준)
TILTS = np.arange(0.0, 90.1, 2.0)
front = np.stack([np.asarray(P.eff_poa(np.full(N, t), e, a, dn, dh, c=CHORD, p=P0, albedo=0.15)).ravel() for t in TILTS])
BT = TILTS[front.sum(1).argmax()]
base_front = front.sum(1).max()       # 기준 연간 발전(상대)
print(f"기준: 밀폐 gcr1.169 · 최고고정 tilt={BT:.0f}° · 전면 알베도 0.15 (실기후 PVGIS)")
print(f"      연간 전면 발전(상대단위) = {base_front:.3e}\n")

# --- ① 양면 후면포집 vs 내부 알베도 (infinite_sheds) ---
print("① 양면(bifacial) — 실외기실 내부 반사면 알베도별 후면 이득")
print(f"   {'내부 알베도':>10}{'후면/전면':>10}{'양면 이득(×0.70)':>16}")
g = min(CHORD / P0, 0.999)
for alb in (0.15, 0.30, 0.50, 0.70, 0.85):
    r = infinite_sheds.get_irradiance(surface_tilt=float(BT), surface_azimuth=180.0,
        solar_zenith=z, solar_azimuth=a, gcr=g, height=1.0, pitch=1.0,
        ghi=gh, dhi=dh, dni=dn, albedo=alb, model="isotropic", bifaciality=BIFACIALITY)
    pf = np.nan_to_num(np.asarray(r["poa_front"], float)).sum()
    pb = np.nan_to_num(np.asarray(r["poa_back"], float)).sum()
    print(f"   {alb:>10.2f}{pb/pf:>9.1%}{pb*BIFACIALITY/pf:>+15.2%}")
print("   주: 밀폐 gcr1.169서 후면은 뒤 블레이드에 크게 가림 → 후면 이득 억제 확인용.")

# --- ② 전면 알베도 (루버 아래 반사면) ---
print("\n② 전면 알베도 — 루버 하단 반사 sill/바닥 개선 (전면만)")
print(f"   {'전면 알베도':>10}{'전면 발전 이득':>14}")
for alb in (0.15, 0.30, 0.50, 0.70):
    F = np.stack([np.asarray(P.eff_poa(np.full(N, t), e, a, dn, dh, c=CHORD, p=P0, albedo=alb)).ravel() for t in (BT,)])
    print(f"   {alb:>10.2f}{(F.sum()/base_front-1):>+13.2%}")

# --- ③ 셀 커버리지(스트립 폭) ---
print("\n③ 셀 커버리지 — 블레이드당 PV 스트립 폭 (현 83mm=0.728). 발전 ∝ 폭×유효POA")
print(f"   {'스트립폭mm':>10}{'strip_frac':>11}{'블레이드당 발전 이득':>18}")
base_strip = 83.0
# 현 배치: 상단마진 24 / 셀83 / 하단7. 스트립 넓히면 하단쪽으로(하단마진 7→0) + 상단쪽으로.
for sw in (83.0, 90.0, 100.0, 107.0):
    lo = max(114.0 - sw - 7.0, 0.0) / 114.0      # 상단마진 축소(하단7 유지)
    frac = sw / 114.0
    Fw = np.asarray(P.eff_poa(np.full(N, BT), e, a, dn, dh, c=CHORD, p=P0, albedo=0.15,
                    strip_lo=lo, strip_frac=frac)).ravel()
    gen = frac * Fw.sum()
    base_gen = (base_strip/114.0) * base_front
    print(f"   {sw:>10.0f}{frac:>11.3f}{(gen/base_gen-1):>+17.2%}")
print("   주: 발전=스트립면적×유효POA. 넓히면 면적↑지만 음영대로 확장 → 순이득 정량.")

print("\n요약: 각 레버의 현실적 상한을 비교. 양면은 밀집 억제, 알베도는 시공 가능, 스트립은 셀 재배치.")
