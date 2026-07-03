# ==============================================================================
# "루버 뒤가 뚫려있다(실외기실)" — 후방 반공간 가정 재검증
#   (2026-07-01, 사용자: 후방벽 아니라 뚫림. 뒤가 하늘로 트이면 최적각 내려오나?)
#
# 현 모델 시야계수(view_factors_mc)의 핵심 = `fwd = dx>0`:
#   블레이드 앞면(외부 +x)으로 가는 광선만 하늘/지면으로 카운트.
#   뒤 반공간(x<0, 실외기실 안쪽)은 하늘 아님(=확산광 없음)으로 가정.
#
# 이 스크립트: fwd 조건을 3단계로 바꿔 F_sky/F_grd 재생성 → 최적 고정각 비교
#   (a) FRONT   : dx>0만 (현 모델 — 뒤는 하늘 아님/단면 셀)
#   (b) OPEN    : 뒤도 하늘 (실외기실 뒤가 하늘로 관통 or 양면 셀)
#   (c) HALF    : 뒤 하늘을 절반 가중 (뒤가 부분 트임)
#   이웃 블레이드 ±4 차폐는 3케이스 모두 유지(밀집은 실재).
# 실행: .venv/bin/python verify_backspace.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P
from physics_v2 import panel_sf

RATIOS = np.arange(0.30, 1.051, 0.05)
TILTS_VF = np.arange(0.0, 90.1, 2.5)


def vf_mc(tilt_deg, c_over_p, back="front", n_pts=64, n_dir=4000, n_nb=4, seed=42):
    """view_factors_mc 와 동일하되 뒤 반공간 처리만 파라미터화."""
    half, p = c_over_p / 2.0, 1.0
    rng = np.random.default_rng(seed)
    tr = np.radians(tilt_deg)
    n = np.array([np.sin(tr), np.cos(tr), 0.0])
    t1 = np.array([np.cos(tr), -np.sin(tr), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    u = (rng.random(n_pts) * 2 - 1) * half
    px, py = u * np.cos(tr), -u * np.sin(tr)
    ct_ = np.sqrt(rng.random((n_pts, n_dir))); st_ = np.sqrt(1 - ct_**2)
    ph = rng.random((n_pts, n_dir)) * 2 * np.pi
    d = (ct_[..., None] * n + (st_ * np.cos(ph))[..., None] * t1 + (st_ * np.sin(ph))[..., None] * t2)
    dx, dy = d[..., 0], d[..., 1]
    blocked = np.zeros((n_pts, n_dir), dtype=bool)
    for k in list(range(1, n_nb + 1)) + list(range(-n_nb, 0)):
        ax_, ay_ = -half * np.cos(tr), k * p + half * np.sin(tr)
        bx_, by_ = half * np.cos(tr), k * p - half * np.sin(tr)
        ex, ey = bx_ - ax_, by_ - ay_
        qx, qy = ax_ - px[:, None], ay_ - py[:, None]
        den = dx * ey - dy * ex
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        t_ray = (qx * ey - qy * ex) / den
        s_seg = (qx * dy - qy * dx) / den
        blocked |= (t_ray > 1e-9) & (s_seg >= 0.0) & (s_seg <= 1.0)
    # ★뒤 반공간 처리
    if back == "front":       # 현 모델: 앞쪽만 하늘
        w = (dx > 1e-9).astype(float)
    elif back == "open":      # 뒤도 하늘 (관통/양면)
        w = np.ones_like(dx)
    elif back == "half":      # 뒤 하늘 절반 가중
        w = np.where(dx > 1e-9, 1.0, 0.5)
    up = ((dy > 1e-9) & ~blocked).astype(float) * w
    dn = ((dy < -1e-9) & ~blocked).astype(float) * w
    return float(up.mean()), float(dn.mean())


def build_table(back):
    FS = np.zeros((len(RATIOS), len(TILTS_VF))); FG = np.zeros_like(FS)
    for i, r in enumerate(RATIOS):
        for j, t in enumerate(TILTS_VF):
            FS[i, j], FG[i, j] = vf_mc(t, r, back=back)
    return {"tilts": TILTS_VF, "ratios": RATIOS, "f_sky": FS, "f_grd": FG}


# 데이터
df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df.solar_elevation > 0].copy()
el = dd.solar_elevation.to_numpy(float); az = dd.solar_azimuth.to_numpy(float)
dni = dd.dni.to_numpy(float); dhi = dd.dhi.to_numpy(float)
N = len(dd)
TILTS = np.arange(0, 90.1, 1.0)
C = P.CHORD; PIT = P.PITCH
r_real = float(np.clip(C / PIT, RATIOS[0], RATIOS[-1]))


def vf_lookup(tbl, tilt, kind):
    t = np.asarray(tilt, float)
    ratios = tbl["ratios"]; i = int(np.clip(np.searchsorted(ratios, r_real) - 1, 0, len(ratios) - 2))
    w = (r_real - ratios[i]) / (ratios[i + 1] - ratios[i])
    arr = tbl["f_sky"] if kind == "sky" else tbl["f_grd"]
    row = (1 - w) * arr[i] + w * arr[i + 1]
    return np.interp(t.ravel(), tbl["tilts"], row).reshape(t.shape)


def poa(tilt, tbl):
    tilt = np.asarray(tilt, float)
    aoi = pvlib.irradiance.aoi(tilt, 180.0, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    ov = P.strip_shade(panel_sf(tilt, el, az, hd=C, p=PIT))
    sky, grd = pvlib.iam.martin_ruiz_diffuse(np.atleast_1d(tilt).ravel(), a_r=P.A_R)
    iam_sky = np.asarray(sky, float).reshape(np.shape(tilt)); iam_grd = np.asarray(grd, float).reshape(np.shape(tilt))
    ghi_est = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    return np.maximum(bp * iam_b * (1 - ov)
                      + dhi * vf_lookup(tbl, tilt, "sky") * iam_sky
                      + ghi_est * P.ALBEDO * vf_lookup(tbl, tilt, "grd") * iam_grd, 0)


print(f"주간 {N}행 · 현/피치={C}/{PIT} (ratio {r_real:.2f}) · 각도 0=수평 90=수직\n")
print("═" * 82)
print("루버 뒤 반공간 가정별 최고 고정각 + 트래킹 천장")
print("─" * 82)
for back, label in [("front", "(a) 앞쪽만 하늘  ← ★현 모델 (뒤=실외기실/단면셀)"),
                    ("half",  "(b) 뒤 절반 트임 (부분 관통)"),
                    ("open",  "(c) 뒤도 하늘   (완전 관통 or 양면셀)")]:
    tbl = build_table(back)
    E = np.array([poa(np.full(N, t), tbl).sum() for t in TILTS])
    bi = int(E.argmax()); mx = E[bi]
    w1 = TILTS[(mx - E) / mx <= 0.01]
    # 오라클 천장
    E15 = np.array([poa(np.full(N, t), tbl) for t in np.arange(15, 90.1, 1.0)])
    orac = E15.max(0).sum()
    fsky60 = float(vf_lookup(tbl, np.array([60.0]), "sky")[0])
    print(f"  {label:<46} 최고={bi and TILTS[bi]:>4.0f}°  고원1%={w1.min():.0f}-{w1.max():.0f}°  "
          f"천장={(orac/mx-1)*100:+.2f}%  (F_sky@60°={fsky60:.2f})")

print()
print("해석:")
print("  · (a)→(c) 로 갈수록 뒤 하늘이 저각 블레이드에 확산광을 보태 → 최적각 내려옴.")
print("  · 결정 변수 2개: ① 셀이 단면(뒷면 발전X)이냐 양면이냐  ② 실외기실 뒤가")
print("    하늘로 트였냐(최상층/관통) vs 어두운 내부 공동이냐. 단면+공동이면 (a) 유지.")
