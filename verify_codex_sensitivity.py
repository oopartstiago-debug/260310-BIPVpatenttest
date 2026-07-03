# ==============================================================================
# codex 적대 민감도 테스트 재현 (2026-06-23): 평탄 고원/낮은 천장이 특정 가정의
#   '인공물'인지 — 의심 가정을 하나씩 적대적으로 교체해 최고고정각·천장이 뒤집히나.
#   A 기준(현 physics_v3)  B 개방 등방성 시야계수  C 그림자 띠 바깥쪽
#   D IAM 제거  E 음영 제거  + f_grd 비율의존성 점검
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P

df = pd.read_csv("bipv_ai_master_data_v15.csv")
el = df.solar_elevation.to_numpy(float); az = df.solar_azimuth.to_numpy(float)
dni = df.dni.to_numpy(float); dhi = df.dhi.to_numpy(float)
day = df.ghi_w_m2.to_numpy(float) >= 10
A = np.arange(15, 91.0)
tilt = A[:, None]

aoi = pvlib.irradiance.aoi(tilt, 180, 90 - el[None, :], az[None, :])
bp = np.maximum(dni[None, :] * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=0.16), 0.0)
skyiam, grdiam = pvlib.iam.martin_ruiz_diffuse(A, a_r=0.16)
ghi = dni[None, :] * np.maximum(np.sin(np.radians(el[None, :])), 0) + dhi[None, :]
vfsky = P.view_factor(A, kind="f_sky")[:, None]
vfgrd = P.view_factor(A, kind="f_grd")[:, None]
sf = P.panel_sf(tilt, el[None, :], az[None, :], hd=97.5, p=97.5, sa=180)
ov = P.strip_shade(sf)
# 개방 등방성 시야계수 (이웃 블레이드/뒷벽 없음 = 단일 경사면)
vfsky_open = ((1 + np.cos(np.radians(A))) / 2)[:, None]
vfgrd_open = ((1 - np.cos(np.radians(A))) / 2)[:, None]
# 바깥쪽 그림자 띠 (피벗 안쪽 가정을 뒤집음)
lo, frac = P.STRIP_LO, P.STRIP_FRAC
ov_outer = np.clip(np.minimum(sf, 1 - lo) - (1 - lo - frac), 0, None) / frac


def report(name, E):
    ann = E[:, day].sum(1); mx = ann.max(); best = A[ann.argmax()]
    w = A[ann >= 0.99 * mx]; orac = E[:, day].max(0).sum()
    print(f"{name:<26} 최고={best:>4.0f}°  고원1%={w.min():.0f}-{w.max():.0f}°  "
          f"오라클vs최고={(orac/mx-1)*100:+.2f}%")


print("의심 가정을 적대적으로 교체 → 결론(최고81°·평탄고원·천장~0.5%)이 뒤집히나\n")
report("A 기준(physics_v3)", bp*iam_b*(1-ov) + dhi[None,:]*vfsky*skyiam[:,None] + ghi*0.15*vfgrd*grdiam[:,None])
report("B 개방등방성 시야계수", bp*iam_b*(1-ov) + dhi[None,:]*vfsky_open*skyiam[:,None] + ghi*0.15*vfgrd_open*grdiam[:,None])
report("C 그림자 띠 바깥쪽", bp*iam_b*(1-ov_outer) + dhi[None,:]*vfsky*skyiam[:,None] + ghi*0.15*vfgrd*grdiam[:,None])
report("D IAM 제거", bp*(1-ov) + dhi[None,:]*vfsky + ghi*0.15*vfgrd)
report("E 음영 제거", bp*iam_b + dhi[None,:]*vfsky*skyiam[:,None] + ghi*0.15*vfgrd*grdiam[:,None])
report("F 빔만(음영·IAM)", bp*iam_b*(1-ov))

print("\nf_grd / f_sky 의 현/피치(ratio) 의존성 (codex 플래그):")
import json
vf = json.load(open("view_factors_v17.json"))
rr = vf["ratios"]; tt = vf["tilts"]
for t in (30.0, 60.0, 81.0):
    k = tt.index(t)
    print(f"  tilt {t:>4.0f}°: " + "  ".join(
        f"r{rr[i]:.2f}>Fsky{vf['f_sky'][i][k]:.3f}/Fgrd{vf['f_grd'][i][k]:.3f}"
        for i in (0, len(rr)//2, len(rr)-1)))
