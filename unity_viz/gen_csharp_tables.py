#!/usr/bin/env python3
"""C# physics 포팅용 테이블/검증벡터 생성 (Python physics_v3 = 단일 진실).
  - iam_diffuse_v17.json: tilt -> (iam_sky, iam_grd)  [Martin-Ruiz diffuse, a_r=0.16]
  - physics_testvectors.json: 무작위 입력 -> eff_poa, panel_sf (C# 셀프테스트용)
  - view_factors_v17.json 은 StreamingAssets로 복사(별도)
"""
import sys, os, json
import numpy as np, pvlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import physics_v3 as P

HERE = os.path.dirname(os.path.abspath(__file__))

# 1) 확산 IAM 테이블 (tilt만 의존, a_r 고정)
tilts = np.arange(0, 90.0001, 1.0)
sky, grd = pvlib.iam.martin_ruiz_diffuse(tilts, a_r=P.A_R)
json.dump({"a_r": P.A_R, "tilts": [round(float(t),3) for t in tilts],
           "iam_sky": [round(float(x),6) for x in sky],
           "iam_grd": [round(float(x),6) for x in grd]},
          open(os.path.join(HERE, "iam_diffuse_v17.json"), "w"), ensure_ascii=False)

# 2) 검증벡터 (시드고정, 결정론)
rng = np.random.default_rng(42)
vecs = []
for _ in range(40):
    tilt = float(rng.uniform(0, 90)); elev = float(rng.uniform(1, 85))
    az = float(rng.uniform(60, 300)); dni = float(rng.uniform(0, 950)); dhi = float(rng.uniform(0, 300))
    poa = float(P.eff_poa(np.array([tilt]), np.array([elev]), np.array([az]), np.array([dni]), np.array([dhi]))[0])
    sf  = float(P.panel_sf(np.array([tilt]), np.array([elev]), np.array([az]), hd=P.CHORD, p=P.PITCH)[0])
    vecs.append({"tilt": round(tilt,4), "elev": round(elev,4), "az": round(az,4),
                 "dni": round(dni,3), "dhi": round(dhi,3),
                 "eff_poa": round(poa,5), "panel_sf": round(sf,6)})
json.dump({"chord": P.CHORD, "pitch": P.PITCH, "albedo": P.ALBEDO, "a_r": P.A_R,
           "strip_lo": P.STRIP_LO, "strip_frac": P.STRIP_FRAC, "tol": 0.02, "vectors": vecs},
          open(os.path.join(HERE, "physics_testvectors.json"), "w"), ensure_ascii=False, indent=1)

# 3) 시야계수 → C#용 평탄(flat) 배열 (JsonUtility는 중첩 2D 못 읽음)
_vf = json.load(open(os.path.join(HERE, "..", "view_factors_v17.json")))
_fs, _fg = np.array(_vf["f_sky"]), np.array(_vf["f_grd"])
json.dump({"ratios": [float(x) for x in _vf["ratios"]], "tilts": [float(x) for x in _vf["tilts"]],
           "n_ratio": int(_fs.shape[0]), "n_tilt": int(_fs.shape[1]),
           "f_sky_flat": [round(float(x), 6) for x in _fs.ravel()],
           "f_grd_flat": [round(float(x), 6) for x in _fg.ravel()]},
          open(os.path.join(HERE, "vf_flat_v17.json"), "w"), ensure_ascii=False)
print("generated iam_diffuse_v17.json, physics_testvectors.json (40 vecs), vf_flat_v17.json")
print("sample vec[0]:", json.dumps(vecs[0], ensure_ascii=False))
