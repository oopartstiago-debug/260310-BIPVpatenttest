"""P4 pygame 실시간 시뮬레이터 — 클린룸 엔진 단독 구동 (physics_v3 미사용).

태양이 하루를 가로지르며 이동, 블레이드 단면·그림자·PV 띠 음영이 실시간 갱신.
조작:
  SPACE 재생/정지   ←/→ 10분 스텝   ↑/↓ 수동 tilt ±1°
  A     AI(오라클) 모드 토글 — 매 시각 클린룸 물리 argmax 각도로 자동 회전
  1/2/3 날짜 (하지/춘분/동지, 맑은날 clear-sky)   ESC 종료
표시: 발전량 막대(현재 tilt vs 오라클 최대), 음영률, 태양 고도/방위/프로파일각.
셀프테스트:  python sim_pygame.py --selftest  → out/p4_frames/*.png 저장 후 종료.
"""
import os
import sys

import numpy as np
import pandas as pd
import pvlib
import pygame
from shapely.geometry import LineString
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (PITCH, CHORD, blade_segment, strip_segment,
                      profile_angle_deg, shadow_polygons, beam_shaded_fraction)
from irradiance import build_tables, power_series, WALL_Y

LOC = pvlib.location.Location(37.5665, 126.9780, tz="Asia/Seoul")
DAYS = {"1": ("하지", "2024-06-21"), "2": ("춘분", "2024-03-21"),
        "3": ("동지", "2024-12-21")}
W, H = 1080, 720
SCALE = 1.05                      # px / mm
OX, OY = 430, H // 2              # 피벗(대상 블레이드) 화면 위치
N_BLADES = 9
TARGET = N_BLADES // 2
TILT_GRID = np.arange(0, 91, 1.0)


def day_series(date):
    """5분 간격 태양위치 + clear-sky 일사 (서울)."""
    times = pd.date_range(f"{date} 05:00", f"{date} 20:00",
                          freq="5min", tz="Asia/Seoul")
    sp = LOC.get_solarposition(times)
    cs = LOC.get_clearsky(times)            # Ineichen: ghi/dni/dhi
    return {"t": times, "elev": sp["apparent_elevation"].to_numpy(),
            "az": sp["azimuth"].to_numpy(), "dni": cs["dni"].to_numpy(),
            "dhi": cs["dhi"].to_numpy(), "ghi": cs["ghi"].to_numpy()}


def w2s(p):
    return (int(OX + p[0] * SCALE), int(OY - p[1] * SCALE))


class Sim:
    def __init__(self):
        self.tab = build_tables()
        self.day_key = "1"
        self.data = day_series(DAYS[self.day_key][1])
        self.i = len(self.data["t"]) // 2
        self.tilt = 83.0
        self.ai = True
        self.playing = True

    def set_day(self, key):
        self.day_key = key
        self.data = day_series(DAYS[key][1])
        self.i = min(self.i, len(self.data["t"]) - 1)

    def instant(self):
        d, i = self.data, self.i
        return (d["elev"][i], d["az"][i], d["dni"][i], d["dhi"][i], d["ghi"][i])

    def power(self, tilt):
        e, a, dni, dhi, ghi = self.instant()
        return float(power_series(self.tab, tilt, np.array([e]), np.array([a]),
                                  np.array([dni]), np.array([dhi]),
                                  np.array([ghi]))[0])

    def oracle(self):
        p = [self.power(float(t)) for t in TILT_GRID]
        j = int(np.argmax(p))
        return float(TILT_GRID[j]), p[j]

    def step(self, di=1):
        self.i = (self.i + di) % len(self.data["t"])


def draw(screen, font, sim):
    screen.fill((248, 247, 243))
    e, a, dni, dhi, ghi = sim.instant()
    prof = profile_angle_deg(e, a)
    opt_tilt, opt_p = sim.oracle()
    if sim.ai:
        sim.tilt = opt_tilt
    cur_p = sim.power(sim.tilt)
    sf = beam_shaded_fraction(sim.tilt, prof)
    # 그림자 영역
    if prof is not None:
        b = np.radians(prof)
        ray = np.array([-np.cos(b), -np.sin(b)])
        polys = []
        for k in range(N_BLADES):
            z = (k - TARGET) * PITCH
            s0, s1 = blade_segment(sim.tilt, z)
            polys.append(shapely_poly([s0, s1, s1 + 1500 * ray, s0 + 1500 * ray]))
        for g in getattr(unary_union(polys), "geoms",
                         [unary_union(polys)]):
            pygame.draw.polygon(screen, (224, 222, 215),
                                [w2s(p) for p in g.exterior.coords])
        # 광선
        for zl in np.arange(-2.4, 2.5, 0.5) * PITCH:
            p0 = np.array([320.0, zl + 320.0 * np.tan(b)])
            p1 = p0 - 900 * np.array([np.cos(b), np.sin(b)])
            pygame.draw.line(screen, (250, 190, 80), w2s(p0), w2s(p1), 1)
    # 벽
    pygame.draw.rect(screen, (120, 118, 112),
                     (*w2s((WALL_Y - 12, 2.9 * PITCH)),
                      int(12 * SCALE), int(5.8 * PITCH * SCALE)))
    # 블레이드 + 띠
    for k in range(N_BLADES):
        z = (k - TARGET) * PITCH
        s0, s1 = blade_segment(sim.tilt, z)
        pygame.draw.line(screen, (70, 70, 70), w2s(s0), w2s(s1), 6)
        t0, t1 = strip_segment(sim.tilt, z)
        pygame.draw.line(screen, (35, 110, 205), w2s(t0), w2s(t1), 3)
    # 대상 블레이드 음영 구간 = 빨강
    if prof is not None and sf > 0:
        t0, t1 = strip_segment(sim.tilt, 0.0)
        strip = LineString([t0, t1])
        inter = strip.intersection(unary_union(shadow_polygons(sim.tilt, prof)))
        for g in getattr(inter, "geoms", [inter]):
            cs = list(g.coords)
            pygame.draw.line(screen, (205, 35, 60), w2s(cs[0]), w2s(cs[-1]), 5)
    # HUD
    name, date = DAYS[sim.day_key]
    ts = sim.data["t"][sim.i].strftime("%H:%M")
    ptxt = f"{prof:.1f}" if prof is not None else "-"
    lines = [
        f"{name} {date}  {ts}   [{'AI' if sim.ai else '수동'}]",
        f"tilt {sim.tilt:.0f}°  (오라클 {opt_tilt:.0f}°)   띠 음영 {sf*100:.0f}%",
        f"고도 {e:.1f}°  방위 {a:.1f}°  프로파일 {ptxt}°",
        f"DNI {dni:.0f}  DHI {dhi:.0f}  GHI {ghi:.0f} W/m²",
        "SPACE 재생  ←→ 시간  ↑↓ 각도  A 모드  1/2/3 날짜",
    ]
    for li, txt in enumerate(lines):
        screen.blit(font.render(txt, True, (40, 40, 40)), (18, 14 + 24 * li))
    # 발전량 막대: 현재 vs 오라클
    bx, by, bw = 18, H - 70, 330
    ref = max(opt_p, 1e-9)
    pygame.draw.rect(screen, (210, 208, 200), (bx, by, bw, 18))
    pygame.draw.rect(screen, (35, 110, 205),
                     (bx, by, int(bw * min(cur_p / ref, 1.0)), 18))
    pygame.draw.rect(screen, (210, 208, 200), (bx, by + 24, bw, 18))
    pygame.draw.rect(screen, (90, 170, 90), (bx, by + 24, bw, 18))
    screen.blit(font.render(
        f"현재 {cur_p:.0f}  /  오라클 최대 {opt_p:.0f} W/m²  "
        f"({(cur_p/ref*100 if ref > 1e-6 else 0):.1f}%)", True, (40, 40, 40)),
        (bx, by - 26))


def shapely_poly(pts):
    from shapely.geometry import Polygon
    return Polygon([tuple(p) for p in pts])


def pick_font():
    for name in ("applesdgothicneo", "appligothic", "applegothic"):
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, 17)
    return pygame.font.Font(None, 20)   # 한글 폰트 없으면 기본(영문)


def main():
    selftest = "--selftest" in sys.argv
    if selftest:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("env_check P4 — AI Tilt 클린룸 시뮬레이터")
    font = pick_font()
    clock = pygame.time.Clock()
    sim = Sim()
    if selftest:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "out", "p4_frames")
        os.makedirs(outdir, exist_ok=True)
        shots = [("1", 60, False), ("1", 84, True), ("3", 84, True),
                 ("3", 60, False)]
        for n, (dk, idx_frac, ai) in enumerate(shots):
            sim.set_day(dk)
            sim.i = int(len(sim.data["t"]) * idx_frac / 100)
            sim.ai = ai
            if not ai:
                sim.tilt = 60.0
            draw(screen, font, sim)
            pygame.display.flip()
            pygame.image.save(screen, os.path.join(outdir, f"frame{n}.png"))
        print("selftest frames saved:", outdir)
        pygame.quit()
        return
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                if ev.key == pygame.K_SPACE:
                    sim.playing = not sim.playing
                if ev.key == pygame.K_LEFT:
                    sim.step(-2)
                if ev.key == pygame.K_RIGHT:
                    sim.step(2)
                if ev.key == pygame.K_a:
                    sim.ai = not sim.ai
                if ev.key == pygame.K_UP and not sim.ai:
                    sim.tilt = min(90.0, sim.tilt + 1)
                if ev.key == pygame.K_DOWN and not sim.ai:
                    sim.tilt = max(0.0, sim.tilt - 1)
                if ev.unicode in DAYS:
                    sim.set_day(ev.unicode)
        if sim.playing:
            sim.step(1)
        draw(screen, font, sim)
        pygame.display.flip()
        clock.tick(12)


if __name__ == "__main__":
    main()
