"""P1 정지 렌더 — 단면 장면 9컷 (계절·시각 3 × tilt 3).

각 패널: 블레이드 단면(회색) + PV 띠(파랑) + 음영 구간(빨강) +
태양 광선(주황 화살표·평행선) + 그림자 영역(연회색) + 후방 벽.
태양 위치 = pvlib (서울 37.5665N, 126.9780E, Asia/Seoul).
"""
import numpy as np
import pandas as pd
import pvlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from geometry import (PITCH, CHORD, STRIP, FRAME_DEPTH, blade_segment,
                      strip_segment, profile_angle_deg, shadow_polygons,
                      beam_shaded_fraction)

# 한글 폰트
for f in ("AppleGothic", "Apple SD Gothic Neo"):
    if any(font.name == f for font in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

LOC = pvlib.location.Location(37.5665, 126.9780, tz="Asia/Seoul")

SCENES = [
    ("하지 정오", "2024-06-21 12:30"),
    ("하지 아침", "2024-06-21 09:00"),
    ("동지 정오", "2024-12-21 12:30"),
]
TILTS = [60.0, 83.0, 90.0]
N_BLADES = 9          # 그림에 그릴 블레이드 수 (대상 = 중앙)
TARGET = N_BLADES // 2


def solar(ts: str):
    sp = LOC.get_solarposition(pd.DatetimeIndex([ts], tz="Asia/Seoul"))
    return float(sp["apparent_elevation"].iloc[0]), float(sp["azimuth"].iloc[0])


def draw_panel(ax, tilt, elev, az):
    prof = profile_angle_deg(elev, az)
    # 그림자 영역: 모든 블레이드가 드리우는 그림자 (시각용)
    if prof is not None:
        b = np.radians(prof)
        ray = np.array([-np.cos(b), -np.sin(b)])
        polys = []
        for k in range(N_BLADES):
            z = (k - TARGET) * PITCH
            a, c = blade_segment(tilt, z)
            polys.append(Polygon([a, c, c + 2000 * ray, a + 2000 * ray]))
        for g in getattr(unary_union(polys), "geoms", [unary_union(polys)]):
            ax.fill(*g.exterior.xy, color="0.82", alpha=0.55, zorder=1)
        # 태양 광선 평행선 + 화살표
        for zline in np.arange(-2.2, 2.3, 0.55) * PITCH:
            p0 = np.array([260.0, zline + 260.0 * np.tan(b)])
            ax.plot([p0[0], p0[0] - 520], [p0[1], p0[1] - 520 * np.tan(b)],
                    color="orange", lw=0.7, alpha=0.5, zorder=2)
        ax.annotate("", xy=(120 - 90 * np.cos(b), 200 - 90 * np.sin(b)),
                    xytext=(120, 200),
                    arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2.2))
    # 후방 벽 (프레임 깊이 100mm → 피벗 뒤 약 50mm)
    ax.fill_betweenx([-2.6 * PITCH, 2.6 * PITCH], -50 - 14, -50,
                     color="0.45", zorder=3)
    # 블레이드 + PV 띠
    for k in range(N_BLADES):
        z = (k - TARGET) * PITCH
        a, c = blade_segment(tilt, z)
        ax.plot([a[0], c[0]], [a[1], c[1]], color="0.35", lw=4.5,
                solid_capstyle="butt", zorder=4)
        s0, s1 = strip_segment(tilt, z)
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color="#1f6fd0", lw=2.6,
                solid_capstyle="butt", zorder=5)
    # 대상 블레이드(중앙)의 음영 구간 = 빨강
    sf = beam_shaded_fraction(tilt, prof)
    if prof is not None and sf > 0:
        s0, s1 = strip_segment(tilt, 0.0)
        strip = LineString([s0, s1])
        inter = strip.intersection(unary_union(shadow_polygons(tilt, prof)))
        for g in getattr(inter, "geoms", [inter]):
            xs, zs = zip(*g.coords)
            ax.plot(xs, zs, color="crimson", lw=3.4, solid_capstyle="butt", zorder=6)
    # 대상 표시
    ax.annotate("대상", xy=(-CHORD / 2 - 8, 0), ha="right", va="center",
                fontsize=8, color="0.3")
    ax.set_xlim(-150, 260)
    ax.set_ylim(-2.6 * PITCH, 2.6 * PITCH)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return prof, sf


def main():
    fig, axes = plt.subplots(len(SCENES), len(TILTS), figsize=(13.5, 13.5))
    for i, (label, ts) in enumerate(SCENES):
        elev, az = solar(ts)
        for j, tilt in enumerate(TILTS):
            prof, sf = draw_panel(axes[i][j], tilt, elev, az)
            ptxt = f"{prof:.1f}°" if prof is not None else "후면/일몰"
            axes[i][j].set_title(
                f"{label} {ts[11:16]} | tilt {tilt:.0f}°\n"
                f"고도 {elev:.1f}° 방위 {az:.1f}° 프로파일 {ptxt} | 띠 음영 {sf*100:.1f}%",
                fontsize=9.5)
    fig.suptitle(
        "env_check P1 — 클린룸 기하(shapely) 자기음영 검증 장면\n"
        "회색=블레이드 현 97.5  파랑=PV 띠 83(중앙)  빨강=음영 구간  "
        "주황=태양광선(프로파일각)  연회색=그림자 영역", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "env_check/out/p1_scenes.png"
    import os; os.makedirs("env_check/out", exist_ok=True)
    fig.savefig(out, dpi=140)
    print("saved:", out)


if __name__ == "__main__":
    main()
