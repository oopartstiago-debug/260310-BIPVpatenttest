"""클린룸 기하 엔진 — shapely 다각형 투영 기반 자기음영 계산.

클린룸 규칙: physics_v3.py / train_v17.py / app.py 를 임포트·참조하지 않는다.
기하 상수는 도면 blade_drawing_1043A.png 판독값만 사용한다.

좌표계 (2D 단면, 파사드 법선 + 연직면):
  y = 파사드 바깥(남쪽) 방향 +  /  z = 상방 +
  블레이드 피벗 = (0, k*PITCH), 피벗은 현(chord)의 중앙.
  tilt 0° = 수평 개방, 90° = 수직 완전닫힘.

빔 음영 방법 (기존 해석적 선분교차와 다른 알고리즘):
  이웃 블레이드 선분을 태양 광선 진행방향으로 길게 스윕한 그림자
  평행사변형(shapely Polygon)들의 합집합과 대상 블레이드 PV 띠
  LineString의 교차 길이 / 띠 길이 = 음영률.
"""
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

# ── 도면 판독값 (blade_drawing_1043A.png) ──────────────────────────
PITCH = 97.5          # mm, 날개 간격 (도면 명기 "날개 간격: 97.5mm")
CHORD = 97.5          # mm, A-A 단면 닫힘 연속면 → 현 = 피치
STRIP = 83.0          # mm, PV 셀 띠 폭 (M6 half-cut 83mm, 현 중앙 배치)
MARGIN = (CHORD - STRIP) / 2.0   # 7.25mm 양끝 마진 (Rotation Pin / Support)
FRAME_DEPTH = 100.0   # mm, 제품 두께 (후방 벽 시각화·확산 차폐용)
FACADE_AZ = 180.0     # 남향 가정 (동~서향은 P2 파라미터)

_SWEEP = 20.0 * PITCH  # 그림자 스윕 길이 (충분히 김)


def blade_segment(tilt_deg: float, z_pivot: float = 0.0):
    """블레이드 현 선분의 (inner, outer) 끝점. inner=건물쪽(위), outer=바깥쪽(아래)."""
    th = np.radians(tilt_deg)
    d = np.array([np.cos(th), -np.sin(th)])   # 피벗→바깥 방향 (tilt↑ → 바깥끝이 내려감)
    pivot = np.array([0.0, z_pivot])
    return pivot - (CHORD / 2) * d, pivot + (CHORD / 2) * d


def strip_segment(tilt_deg: float, z_pivot: float = 0.0):
    """PV 띠(83mm, 중앙 배치) 선분의 끝점."""
    th = np.radians(tilt_deg)
    d = np.array([np.cos(th), -np.sin(th)])
    pivot = np.array([0.0, z_pivot])
    return pivot - (STRIP / 2) * d, pivot + (STRIP / 2) * d


def profile_angle_deg(elevation_deg: float, azimuth_deg: float) -> float | None:
    """태양 프로파일각(단면 투영 고도). 후면/지평선 아래면 None."""
    if elevation_deg <= 0:
        return None
    gamma = np.radians(azimuth_deg - FACADE_AZ)
    if np.cos(gamma) <= 0:          # 태양이 파사드 뒤
        return None
    return float(np.degrees(np.arctan(np.tan(np.radians(elevation_deg)) / np.cos(gamma))))


def shadow_polygons(tilt_deg: float, profile_deg: float, n_neighbors: int = 4):
    """위쪽 이웃 블레이드들이 광선 진행방향으로 드리우는 그림자 평행사변형 목록."""
    b = np.radians(profile_deg)
    ray = np.array([-np.cos(b), -np.sin(b)])  # 광선 진행방향 (태양→장면)
    polys = []
    for k in range(1, n_neighbors + 1):
        a, c = blade_segment(tilt_deg, z_pivot=k * PITCH)
        polys.append(Polygon([a, c, c + _SWEEP * ray, a + _SWEEP * ray]))
    return polys


def beam_shaded_fraction(tilt_deg: float, profile_deg: float | None,
                         n_neighbors: int = 4) -> float:
    """대상 블레이드(피벗 z=0) PV 띠의 빔 음영률 [0,1]."""
    if profile_deg is None:
        return 0.0
    s0, s1 = strip_segment(tilt_deg)
    strip = LineString([s0, s1])
    shadow = unary_union(shadow_polygons(tilt_deg, profile_deg, n_neighbors))
    inter = strip.intersection(shadow)
    return float(inter.length / strip.length) if not inter.is_empty else 0.0


def shaded_interval_local(tilt_deg: float, profile_deg: float | None):
    """음영 구간을 띠 로컬좌표(0=안쪽끝, 1=바깥끝)로 반환 — '어느 가장자리부터' 검증용."""
    if profile_deg is None:
        return []
    s0, s1 = strip_segment(tilt_deg)
    strip = LineString([s0, s1])
    inter = strip.intersection(unary_union(shadow_polygons(tilt_deg, profile_deg)))
    if inter.is_empty:
        return []
    geoms = getattr(inter, "geoms", [inter])
    out = []
    for g in geoms:
        ts = [strip.project(__import__("shapely").geometry.Point(p)) / strip.length
              for p in g.coords]
        out.append((round(min(ts), 4), round(max(ts), 4)))
    return sorted(out)


if __name__ == "__main__":
    # 기하 불변식 빠른 점검 (육안 게이트와 별개의 수치 sanity)
    checks = []
    # ① 90° 완전닫힘 = 동일 평면 → 빔 자기음영 0
    checks.append(("90° 닫힘 빔음영=0",
                   all(beam_shaded_fraction(90.0, b) < 1e-9 for b in (10, 30, 60, 85))))
    # ② 손계산 앵커 1: tilt 0°·β=75° → 광선이 z=97.5 통과점 y=t+97.5/tan75,
    #    차단조건 t∈[−74.9, 22.62] ∩ 띠[−41.5,41.5] = 64.12mm → sf=0.7726
    checks.append(("앵커 sf(0°,75°)=0.773±0.01",
                   abs(beam_shaded_fraction(0.0, 75.0) - 0.7726) < 0.01))
    # ③ 손계산 앵커 2: tilt 60°·β=60° → 위 블레이드 바깥끝 그림자 t=−7.54,
    #    음영 [−41.5, −7.54] = 33.96mm → sf=0.4092
    checks.append(("앵커 sf(60°,60°)=0.409±0.01",
                   abs(beam_shaded_fraction(60.0, 60.0) - 0.4092) < 0.01))
    # ③b 낮은 태양은 틈을 통과 → 음영 작음 (수평 블레이드 직관 반대 주의)
    checks.append(("sf(60°,10°)<0.05", beam_shaded_fraction(60.0, 10.0) < 0.05))
    # ④ 음영은 띠 안쪽(피벗쪽 위) 가장자리부터 시작
    iv = shaded_interval_local(60.0, 30.0)
    checks.append(("음영구간이 안쪽 끝(0)에서 시작", bool(iv) and iv[0][0] < 1e-6))
    for name, ok in checks:
        print(("PASS" if ok else "FAIL"), name)
    print("60° 음영구간(로컬):", iv, " sf:", round(beam_shaded_fraction(60.0, 30.0), 4))
