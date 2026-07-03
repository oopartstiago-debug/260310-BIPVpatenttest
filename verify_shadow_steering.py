# ==============================================================================
# "음영을 다른쪽(비발전 마진)으로 유도할 수 없나" (2026-07-02 사용자 아이디어)
#
# 자기음영은 광선추적상 항상 안쪽(피벗) 가장자리부터 [0, sf] 밴드로 든다(방향 일정).
# → PV 셀(83mm)을 바깥쪽으로 밀면 안쪽 마진이 그림자를 먼저 흡수 → 저각 PV 보호.
# gcr(현=피치=97.5) 고정 = 공간밀도 손해 없이 '배치'만 바꾸는 순수 개선.
#
# ① 현 블레이드(97.5mm)에서 PV 83mm 위치(strip_lo) 스윕: 안쪽→중앙→바깥
# ② 블레이드를 넓혀 안쪽 마진을 더 크게 주면(피치도 같이 커져 gcr↓) 회복 vs 공간대가
# 실행: .venv/bin/python verify_shadow_steering.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P

df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df.solar_elevation > 0].copy()
el = dd.solar_elevation.to_numpy(float); az = dd.solar_azimuth.to_numpy(float)
dni = dd.dni.to_numpy(float); dhi = dd.dhi.to_numpy(float)
N = len(dd)
TILTS = np.arange(0, 90.1, 1.0)
C = P.CHORD
FRAC = 83.0 / 97.5           # PV 폭/현 (현 블레이드)


def poa_free(tilt):
    tilt = np.asarray(tilt, float); beta = np.radians(tilt)
    aoi = pvlib.irradiance.aoi(tilt, 180.0, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(np.atleast_1d(tilt).ravel(), a_r=P.A_R)
    iam_sky = np.asarray(sky, float).reshape(np.shape(tilt)); iam_grd = np.asarray(grd, float).reshape(np.shape(tilt))
    ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    return np.maximum(bp * iam_b + dhi * (1 + np.cos(beta)) / 2 * iam_sky
                      + ghi * P.ALBEDO * (1 - np.cos(beta)) / 2 * iam_grd, 0)


def best(fn):
    E = np.array([fn(np.full(N, t)).sum() for t in TILTS])
    bi = int(E.argmax()); return TILTS[bi], E[bi]


tf, Ef = best(poa_free)
_, Emid = best(lambda t: P.eff_poa(t, el, az, dni, dhi, strip_lo=(1 - FRAC) / 2, strip_frac=FRAC))

print(f"주간 {N}행 · 현={C}mm · PV폭 83mm · 각도 0=수평 90=수직")
print(f"기준: 자유패널 {tf:.0f}°={Ef/1e6:.3f}(100%) · 현 중앙배치 루버={Emid/1e6:.3f}({Emid/Ef*100:.1f}%)\n")

print("═" * 78)
print("① PV 위치(strip_lo) 스윕 — 블레이드 97.5mm 고정, 셀만 이동 (공간대가 0)")
print("─" * 78)
print("  배치            안쪽마진  바깥마진  최적각   면적당(%자유)   중앙대비")
LOS = [(0.0, "안쪽 끝"), ((1 - FRAC) / 2, "중앙 ★현"), (1 - FRAC, "바깥 끝")]
# 더 촘촘히
grid = np.linspace(0, 1 - FRAC, 7)
for lo in grid:
    t, E = best(lambda tt, l=lo: P.eff_poa(tt, el, az, dni, dhi, strip_lo=l, strip_frac=FRAC))
    inner = lo * C; outer = (1 - FRAC - lo) * C
    tag = "★현" if abs(lo - (1 - FRAC) / 2) < 1e-6 else ""
    print(f"  lo={lo:.3f}         {inner:>4.1f}mm   {outer:>4.1f}mm    {t:>3.0f}°     "
          f"{E/Ef*100:>5.2f}%        {(E/Emid-1)*100:+.2f}% {tag}")
print("  → 바깥 끝 배치면 안쪽 14.5mm 마진이 그림자 흡수. 회복폭이 실측 개선여지.")

print()
print("═" * 78)
print("② 안쪽 마진 확대 (블레이드 넓혀 PV 83mm 바깥배치, 피치=블레이드폭=gcr1 유지)")
print("   블레이드 넓히면 피치도 커짐 → 공간밀도(×gcr, 자유=1) 트레이드 동시 표시")
print("─" * 78)
print("  블레이드폭  안쪽마진  최적각   면적당(%자유)  공간밀도(×gcr)")
for chord in (97.5, 110, 125, 140, 166):
    frac = 83.0 / chord
    lo = 1 - frac        # PV 바깥 끝, 안쪽 마진 = (chord-83)
    gcr = C / chord * 1.0  # 기준 대비? gcr=현83?... 여기선 현=chord=피치 → gcr=1 항상
    # 공간밀도: PV면적/입면 = 83 / chord (블레이드 넓을수록 PV 성김)
    dens_factor = 83.0 / chord
    t, E = best(lambda tt, c=chord, l=lo, fr=frac:
                P.eff_poa(tt, el, az, dni, dhi, c=c, p=c, strip_lo=l, strip_frac=fr))
    print(f"  {chord:>5.0f}mm   {chord-83:>4.0f}mm    {t:>3.0f}°     {E/Ef*100:>5.1f}%      "
          f"{E/Ef*dens_factor:.3f}")
print("  → 블레이드 넓혀 마진 주면 면적당↑·최적각↓ 하지만 PV가 성겨져 공간밀도↓.")
print("     (PV폭 83 고정이라 넓힌 블레이드는 비발전 마진만 늘어 공간당은 손해.)")
print()
print("결론: 공간대가 0 개선 = ①셀 바깥배치(그림자를 안쪽 마진으로 유도). "
      "②블레이드 넓히기는 공간당 손해라 순개선 아님.")
