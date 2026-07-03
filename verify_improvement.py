# ==============================================================================
# "밀집 루버는 발전 손해 아닌가? 개선 방향은?" (2026-07-01/02 사용자 질문)
#
# 두 기준으로 분해:
#   ① 블레이드 면적당(W/m²-of-blade): 자유 패널(자기음영X) 대비 손해율
#   ② 입면 공간당(발전밀도): 같은 개구부 면적에 블레이드를 얼마나 넣나 = gcr(현/피치)
#      공간밀도 = (면적당 효율) × gcr.  gcr↑ 하면 면적↑·효율↓ → 곱의 최적 gcr 존재?
#
# 개선 레버 = 현/피치(gcr). gcr↓(간격 벌림)=자기음영↓·효율↑·최적각↓(교과서로)·면적↓
# 실행: .venv/bin/python verify_improvement.py
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
C = P.CHORD


def poa_free(tilt):
    """자유 단일 패널: 자기음영X, 이웃차폐X (등방 하늘 (1±cosβ)/2)."""
    tilt = np.asarray(tilt, float); beta = np.radians(tilt)
    aoi = pvlib.irradiance.aoi(tilt, 180.0, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(np.atleast_1d(tilt).ravel(), a_r=P.A_R)
    iam_sky = np.asarray(sky, float).reshape(np.shape(tilt)); iam_grd = np.asarray(grd, float).reshape(np.shape(tilt))
    ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    return np.maximum(bp * iam_b + dhi * (1 + np.cos(beta)) / 2 * iam_sky
                      + ghi * P.ALBEDO * (1 - np.cos(beta)) / 2 * iam_grd, 0)


def per_area(tilt, gcr):
    """밀집 루버 면적당 (현=C 고정, 피치=C/gcr)."""
    p = C / gcr
    return P.eff_poa(np.asarray(tilt, float), el, az, dni, dhi, c=C, p=p)


def best_of(fn):
    E = np.array([fn(np.full(N, t)).sum() for t in TILTS])
    bi = int(E.argmax()); return TILTS[bi], E[bi]


# ── ① 면적당 손해 ────────────────────────────────────────────────────────────
tf, Ef = best_of(poa_free)
td, Ed = best_of(lambda t: per_area(t, 1.0))       # gcr=1.0 실기하
print(f"주간 {N}행 · 현={C}mm · 각도 0=수평 90=수직\n")
print("═" * 74)
print("① 블레이드 면적당 발전 (수광면 1m² 기준)")
print("─" * 74)
print(f"  자유 단일패널  최적 {tf:>3.0f}°  →  {Ef/1e6:.3f}  (기준 100%)")
print(f"  밀집 루버(현)  최적 {td:>3.0f}°  →  {Ed/1e6:.3f}  ({Ed/Ef*100:.1f}% = 면적당 {(1-Ed/Ef)*100:.0f}% 손해)")
print(f"  → 자기음영 밀집의 태생적 대가. 블레이드 한 장만 보면 손해 맞음.")

# ── ② 입면 공간당 발전밀도 = 면적당 × gcr ────────────────────────────────────
print()
print("═" * 74)
print("② 입면 공간당 발전밀도 (같은 개구부에 블레이드 밀도 gcr=현/피치 만큼)")
print("   공간밀도 = 면적당효율 × gcr   (자유패널 최적을 1.00 으로 정규화)")
print("─" * 74)
print("  gcr(현/피치)  간격     최적각   면적당(%자유)   공간밀도(×gcr, 자유=1.00)")
dens = []
for gcr in (0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00):
    t, E = best_of(lambda tt, g=gcr: per_area(tt, g))
    gap = C / gcr - C
    d = (E / Ef) * gcr           # 공간밀도 (자유 면적당 기준 정규화)
    dens.append((gcr, d))
    print(f"    {gcr:.2f}      {gap:>4.0f}mm    {t:>3.0f}°      {E/Ef*100:>5.1f}%          {d:.3f}")
best_gcr = max(dens, key=lambda x: x[1])
print(f"\n  → 공간당 발전밀도 최대 = gcr {best_gcr[0]:.2f} (밀도 {best_gcr[1]:.3f})")

# ── 개선 방향 정리 ───────────────────────────────────────────────────────────
print()
print("═" * 74)
print("개선 방향 (정량)")
print("─" * 74)
d_now = [d for g, d in dens if abs(g - 1.0) < 1e-9][0]
print(f"  · 현 설계(gcr 1.0=현≈피치) 공간밀도 = {d_now:.3f}")
print(f"  · 최적 gcr {best_gcr[0]:.2f} 로 간격 벌리면 공간밀도 {best_gcr[1]:.3f} "
      f"({(best_gcr[1]/d_now-1)*100:+.1f}%)")
print(f"  · 트레이드오프: 간격 벌리면 면적당↑·최적각↓(교과서 복귀)지만 블레이드 수↓")
print(f"  · BIPV 본질: 발전이 아니라 환기·차폐·외장이 본업 → 밀집(gcr高)은 그 기능 위한 것")
print(f"    발전 최대가 목표면 애초에 루버 아닌 자유 패널. 밀집은 '공간 통합'의 대가.")
