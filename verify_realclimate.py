# ==============================================================================
# 트리 B — 실기후 재검증 (2026-07-03)
#   가설: 현 TMY(diffuse 0.44)는 낙관. 진짜 서울(더 흐림)이면 tilt 트래킹 이득이 커진다.
#   방법: PVGIS-SARAH2 위성관측 TMY(서울, 키 불필요) 시간별 GHI/DNI/DHI 로
#         ① 실 확산율·연간 GHI 확인 ② 밀폐(p0) tilt 트래킹 이득 재계산.
#   기준: 밀폐 피치 p0=97.5(고정) → 공간당 이득 = eff_poa 비 (gcr 상수라 상쇄).
# ==============================================================================
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import pvlib
from pvlib.iotools import get_pvgis_tmy
import physics_v3 as P

CHORD, P0, ALB = 114.0, 97.5, P.ALBEDO
LAT, LON = 37.5665, 126.9780        # 서울시청

# --- PVGIS 실기후 ---
data = get_pvgis_tmy(LAT, LON, map_variables=True)[0]
idx = data.index
if idx.tz is None:
    idx = idx.tz_localize("UTC")
sp = pvlib.solarposition.get_solarposition(idx, LAT, LON,
        pressure=data["pressure"].to_numpy(), temperature=data["temp_air"].to_numpy())
el = sp["apparent_elevation"].to_numpy(float)
az = sp["azimuth"].to_numpy(float)
ghi = data["ghi"].to_numpy(float); dni = data["dni"].to_numpy(float); dhi = data["dhi"].to_numpy(float)

day = el > 3.0                       # 주간 (지평 위)
print("===== PVGIS-SARAH2 서울 실기후 vs 현 TMY =====")
print(f"연간 GHI(수평) = {ghi.sum()/1000:.0f} kWh/m2/yr   (현 realistic TMY ≈ 1510)")
df_diff = dhi[day].sum() / ghi[day].sum()
print(f"연간 확산율 DHI/GHI(주간) = {df_diff:.3f}   (현 realistic TMY = 0.44)")
# 청명도 kt = ghi / ghi_clearsky(대략) 로 흐림 구간 분리
loc = pvlib.location.Location(LAT, LON)
cs = loc.get_clearsky(idx, model="ineichen")["ghi"].to_numpy(float)
kt = np.where(cs > 1, ghi / cs, 0.0)
cloudy = day & (kt < 0.45)           # 흐림 (청명도 낮음)
clear  = day & (kt > 0.7)            # 맑음
print(f"주간 시간 분포: 흐림(kt<0.45) {cloudy.sum()/day.sum()*100:.0f}% · 맑음(kt>0.7) {clear.sum()/day.sum()*100:.0f}%")

# --- 밀폐 tilt 트래킹 이득 (공간당=eff_poa 비, dense p0 고정) ---
TILTS = np.arange(0.0, 90.1, 2.0)
sel_all = day
def gain(mask, name):
    m = mask
    e, a, dn, dh = el[m], az[m], dni[m], dhi[m]
    E = np.stack([np.asarray(P.eff_poa(np.full(m.sum(), t), e, a, dn, dh,
                  c=CHORD, p=P0, albedo=ALB)).ravel() for t in TILTS])
    ann = E.sum(1); best = ann.max(); bt = TILTS[ann.argmax()]
    trk = E.max(0).sum()
    print(f"  {name:<22} 최고고정 tilt={bt:>4.0f}° · tilt 트래킹 이득 = {(trk/best-1)*100:+.2f}%")
    return (trk/best-1)*100

print("\n밀폐(p0=97.5, gcr1.169) tilt 트래킹 이득 — 실기후:")
gain(day,   "전체(주간)")
gain(cloudy,"흐림 kt<0.45")
gain(clear, "맑음 kt>0.7")

print("\n주해:")
print(" · 이 이득은 '밀폐 유지 + 각도만' 트래킹 (가변피치는 A에서 발전이득 0 확정).")
print(" · PVGIS-SARAH2 = 위성관측 다년 → 서울 '전형년'. 현 realistic TMY 와 직접 비교용.")
