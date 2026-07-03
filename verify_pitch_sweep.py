# ==============================================================================
# 피치(블레이드 간격) 스윕 (2026-06-23) — "얼마나 넓으면 트래킹 차이가 나나"
#   현(블레이드 폭)=97.5mm 고정, 피치를 넓혀가며 ratio=현/피치 ↓.
#   각 간격에서 이론 천장(완벽 트래커 오라클 vs 최고 고정각)을 두 독립모델로.
#   physics_v3 (자체 MC, ratio>=0.30 유효) + pvlib infinite_sheds (독립 표준).
# ==============================================================================
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from pvlib.bifacial import infinite_sheds
import physics_v3 as P

df = pd.read_csv("bipv_ai_master_data_v15.csv")
d = df[df.ghi_w_m2 >= 10].copy()
zen = np.clip(d.solar_zenith.to_numpy(float), 0, 89.9); el = d.solar_elevation.to_numpy(float)
saz = d.solar_azimuth.to_numpy(float); dni = d.dni.to_numpy(float); dhi = d.dhi.to_numpy(float)
ghi = dni * np.sin(np.radians(np.clip(el, 0, 90))) + dhi
A = np.arange(0, 90.1, 1.0); CHORD = 97.5; ALB = 0.15


def v3(pitch):
    E = np.stack([np.asarray(P.eff_poa(np.full(len(el), t), el, saz, dni, dhi,
                  c=CHORD, p=pitch, albedo=ALB)).ravel() for t in A])
    ann = E.sum(1); mx = ann.max(); best = A[ann.argmax()]
    w = A[ann >= 0.99 * mx]
    return best, w.max() - w.min(), (E.max(0).sum() / mx - 1) * 100


def isheds(ratio):
    E = []
    for t in A:
        r = infinite_sheds.get_irradiance(surface_tilt=float(t), surface_azimuth=180.0,
            solar_zenith=zen, solar_azimuth=saz, gcr=min(ratio, 0.999), height=1.0, pitch=1.0,
            ghi=ghi, dhi=dhi, dni=dni, albedo=ALB, model="isotropic")
        E.append(np.nan_to_num(np.asarray(r["poa_front"], float)))
    E = np.stack(E); ann = E.sum(1); mx = ann.max()
    return (E.max(0).sum() / mx - 1) * 100


print("현(블레이드 폭)=97.5mm 고정. 피치↑ = 간격↑ = ratio↓ = 덜 밀집.")
print(f"{'피치mm':>7}{'간격(gap)':>9}{'ratio':>7}{'│ physics_v3':>22}{'│ infinite_sheds':>18}")
print(f"{'':>7}{'':>9}{'':>7}{'최고°  고원폭  천장이득':>22}{'  천장이득':>18}")
print("-" * 74)
rows = []
for pitch in (97.5, 110, 122, 137, 163, 195, 244, 325, 488, 975):
    ratio = CHORD / pitch
    gap = pitch - CHORD
    if ratio >= 0.299:
        best, pw, g3 = v3(pitch); v3s = f"{best:>4.0f}°  {pw:>4.0f}°  {g3:>+6.2f}%"
    else:
        v3s = "  (ratio<0.30 표밖)"
    gis = isheds(ratio)
    print(f"{pitch:>7.0f}{gap:>8.0f}mm{ratio:>7.2f}│ {v3s:<20}│  {gis:>+6.2f}%")
    rows.append((pitch, ratio, gis))

print("\n임계 간격 (infinite_sheds 천장이득 기준):")
for thr in (1.0, 2.0, 5.0):
    hit = [r for r in rows if r[2] >= thr]
    if hit:
        p, rt, g = min(hit, key=lambda x: x[0])
        print(f"  트래킹 천장 ≥{thr:.0f}% 되려면: 피치 ≥ {p:.0f}mm (ratio ≤ {rt:.2f}, 간격 ≥ {p-CHORD:.0f}mm) → 그때 {g:+.1f}%")
    else:
        print(f"  트래킹 천장 ≥{thr:.0f}%: 스윕 범위 내 도달 안 함")
print("\n주: 실기하(도면 1043A)=피치 97.5mm(ratio 1.0). 이건 완벽 트래커 상한이고,")
print("    실제 RL/XGBoost 모델은 이보다 약간 낮음. ratio<0.30은 physics_v3 시야계수표 밖이라 infinite_sheds만.")
