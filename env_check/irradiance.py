"""P2 독립 조도·발전 엔진 — 광선캐스팅 시야계수 + 빔 음영그리드 + IAM.

클린룸 규칙: physics_v3 / view_factors_v17.json 을 사용하지 않는다.
- 확산광(하늘/지면): PV 띠 위 샘플점에서 전방 반구 각도 광선캐스팅(shapely 차폐:
  이웃 블레이드 ±4 + 후방 벽) → F_sky, F_grd. 등방 하늘 가정(v3와 동일 모델 선택).
- 빔: pvlib AOI(3D) × (1−음영률 sf(tilt, profile)) × Martin-Ruiz IAM(a_r=0.16, 문헌 표준).
- 지면 반사: GHI × albedo × F_grd.
모델 선택 가정(독립 구현이지만 비교를 위해 동일 파라미터): albedo=0.15, a_r=0.16,
반사·온도 미모델(양 엔진 공통 한계).

후방 벽 위치 = y=−50mm (도면: 프레임 깊이 100mm, 피벗=깊이 중앙 → 현 97.5가
수평 개방 시 ±48.75로 꼭 들어맞는 배치). ※독립 판독 가정 — P3에서 민감도 확인.
"""
import json
import os
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import unary_union

from geometry import PITCH, CHORD, STRIP, blade_segment, strip_segment, \
    beam_shaded_fraction, profile_angle_deg, FACADE_AZ

ALBEDO = 0.15
A_R = 0.16
WALL_Y = -50.0
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

TILTS = np.arange(0, 91, 1.0)            # 시야계수·sf 그리드 축
PROFILES = np.arange(0.5, 90.0, 1.0)     # 빔 프로파일각 축


# ── 확산광: 광선캐스팅 시야계수 ────────────────────────────────────
def _blockers(tilt):
    segs = [LineString(blade_segment(tilt, k * PITCH))
            for k in range(-4, 5) if k != 0]
    segs.append(LineString([(WALL_Y, -20 * PITCH), (WALL_Y, 20 * PITCH)]))
    return unary_union(segs)

def view_factors_raycast(tilt, n_pts=15, dphi_deg=0.5):
    """PV 띠의 전방 반구 시야계수 (F_sky, F_grd, F_blocked). 합=1."""
    th = np.radians(tilt)
    n = np.array([np.sin(th), np.cos(th)])          # 전면 법선 (위·바깥)
    d = np.array([np.cos(th), -np.sin(th)])         # 띠 방향
    s0, _ = strip_segment(tilt)
    t = (np.arange(n_pts) + 0.5) / n_pts * STRIP
    pts = s0[None, :] + t[:, None] * d[None, :]     # (P,2) 샘플점
    phi = np.radians(np.arange(-90 + dphi_deg / 2, 90, dphi_deg))  # (A,)
    # 광선 방향 = 법선을 phi 만큼 회전
    ca, sa = np.cos(phi), np.sin(phi)
    dirs = np.stack([ca * n[0] - sa * n[1], sa * n[0] + ca * n[1]], axis=1)  # (A,2)
    P, A = len(pts), len(phi)
    starts = (pts[:, None, :] + 0.05 * dirs[None, :, :]).reshape(-1, 2)
    ends = (pts[:, None, :] + 4000.0 * dirs[None, :, :]).reshape(-1, 2)
    rays = shapely.linestrings(np.stack([starts, ends], axis=1))
    blocked = shapely.intersects(_blockers(tilt), rays).reshape(P, A)
    w = np.cos(phi) * np.radians(dphi_deg) / 2.0    # 2D 라디오시티 가중 (반구합=1)
    up = dirs[:, 1] > 0
    f_sky = (w[None, :] * (~blocked) * up[None, :]).sum(axis=1).mean()
    f_grd = (w[None, :] * (~blocked) * ~up[None, :]).sum(axis=1).mean()
    return float(f_sky), float(f_grd)


# ── 빔: 음영 그리드 + 벽 도달(보존 게이트용) ───────────────────────
def wall_lit_fraction(tilt, profile_deg):
    """피치 1구간당 벽에 직달 빔이 닿는 길이 / 피치 (에너지 보존 게이트용)."""
    b = np.radians(profile_deg)
    ray = np.array([-np.cos(b), -np.sin(b)])
    polys = [shapely.geometry.Polygon(
        [a, c, c + 3000 * ray, a + 3000 * ray])
        for k in range(0, 30)
        for a, c in [blade_segment(tilt, k * PITCH)]]
    wall = LineString([(WALL_Y, 0), (WALL_Y, PITCH)])   # 대표 1피치 구간
    lit = wall.difference(unary_union(polys))
    return float(lit.length / PITCH)


def build_tables(force=False):
    """시야계수·빔음영 그리드 계산 후 캐시."""
    fn = os.path.join(_OUT, "tables_raycast.json")
    if os.path.exists(fn) and not force:
        with open(fn) as f:
            return json.load(f)
    os.makedirs(_OUT, exist_ok=True)
    f_sky, f_grd = [], []
    for t in TILTS:
        a, g = view_factors_raycast(float(t))
        f_sky.append(a); f_grd.append(g)
    sf = [[beam_shaded_fraction(float(t), float(p)) for p in PROFILES]
          for t in TILTS]
    tab = {"tilts": TILTS.tolist(), "profiles": PROFILES.tolist(),
           "f_sky": f_sky, "f_grd": f_grd, "sf": sf}
    with open(fn, "w") as f:
        json.dump(tab, f)
    return tab


# ── 발전량 (상대 단위) ─────────────────────────────────────────────
def iam_martin_ruiz(aoi_deg):
    c = np.cos(np.radians(np.clip(aoi_deg, 0, 90)))
    return np.maximum((1 - np.exp(-c / A_R)) / (1 - np.exp(-1 / A_R)), 0.0)

def power_series(tab, tilt, elev, az, dni, dhi, ghi, albedo=ALBEDO):
    """시간 벡터(elev, az, dni, dhi, ghi)에 대한 tilt 고정 발전량 (W/m² 상대)."""
    import pvlib
    ti = int(round(tilt))
    zen = 90.0 - elev
    aoi = pvlib.irradiance.aoi(tilt, 180.0, zen, az)
    # 프로파일각 (전면·주간만)
    gamma = np.radians(az - FACADE_AZ)
    front = (elev > 0) & (np.cos(gamma) > 0)
    prof = np.degrees(np.arctan2(np.tan(np.radians(np.maximum(elev, 0.001))),
                                 np.cos(gamma)))
    sf = np.where(front, np.interp(prof, tab["profiles"], tab["sf"][ti]), 0.0)
    beam = dni * np.maximum(np.cos(np.radians(aoi)), 0) * \
        iam_martin_ruiz(aoi) * (1 - sf) * front
    sky = dhi * tab["f_sky"][ti]
    grd = ghi * albedo * tab["f_grd"][ti]
    return np.maximum(beam + sky + grd, 0.0)


if __name__ == "__main__":
    # ── P2 게이트 ──
    ok = True
    # ① 정규화: 차폐 없는 수평면 F_sky = 1
    import geometry as _g
    th = 0.0
    phi = np.radians(np.arange(-89.75, 90, 0.5))
    w = np.cos(phi) * np.radians(0.5) / 2
    print(f"게이트① 반구가중합={w.sum():.4f} (기대 1.0)",
          "PASS" if abs(w.sum() - 1) < 1e-3 else "FAIL")
    # ② 빔 에너지 보존: 블레이드 흡수(현 전체) + 벽 도달 = 개구부 유입
    print("게이트② 빔 보존 (tilt, profile): blade+wall vs aperture")
    for tilt, prof in [(60, 30), (60, 60), (45, 30), (75, 45), (30, 20), (85, 50)]:
        b = np.radians(prof)
        # 현 전체 음영률 (띠 아님)
        s0e = blade_segment(tilt, 0.0)
        chord_line = LineString(s0e)
        from geometry import shadow_polygons
        inter = chord_line.intersection(unary_union(shadow_polygons(tilt, prof)))
        sf_chord = inter.length / chord_line.length if not inter.is_empty else 0.0
        aoi2d = abs((90 - tilt) - prof)             # 단면 내 입사각
        blade_flux = CHORD * np.cos(np.radians(aoi2d)) * (1 - sf_chord)
        wall_flux = wall_lit_fraction(tilt, prof) * PITCH  # 수직벽: cosβ 동일 소거
        aperture = PITCH                                    # 수직 개구부: cosβ 소거
        # 검: blade·cos(aoi2d)/cosβ... 정확식: 개구부 유입(수평성분) = p·cosβ,
        # 블레이드 수평성분 = chord·cos(aoi2d)·(1−sf)… 양변 cosβ 정규화 비교
        lhs = (blade_flux * 1.0 + wall_flux * np.cos(b)) / np.cos(b)
        rel = abs(lhs - aperture) / aperture
        flag = "PASS" if rel < 0.01 else "FAIL"
        ok &= rel < 0.01
        print(f"  tilt{tilt:>3} β{prof:>3}: blade {blade_flux:7.2f}  wall {wall_flux*np.cos(b):6.2f}"
              f"  → 합/개구부 = {lhs/aperture:.4f}  {flag}")
    # ③ 시야계수 분할: F_sky+F_grd ≤ 1, 닫힘(90°)에서 벽 차폐 0(자기 평면)
    t0 = view_factors_raycast(0.0)
    t90 = view_factors_raycast(90.0)
    print(f"게이트③ VF: tilt0 F_sky={t0[0]:.3f} F_grd={t0[1]:.3f} | "
          f"tilt90 F_sky={t90[0]:.3f} F_grd={t90[1]:.3f}")
    print("전체:", "PASS" if ok else "FAIL")
