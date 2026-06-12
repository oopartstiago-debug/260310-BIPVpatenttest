"""P4 pygame 실시간 시뮬레이터 — 클린룸 엔진 단독 구동 (physics_v3 미사용).

태양이 하루를 가로지르며 이동, 블레이드 단면·그림자·PV 띠 음영이 실시간 갱신.
모든 수치 = env_check 클린룸 엔진 (시늉/가짜 데이터 없음).
조작:
  SPACE 재생/정지   ←/→ 10분 스텝   ↑/↓ 수동 tilt ±1°
  A     AI(오라클) 모드 토글 — 매 시각 클린룸 물리 argmax 각도로 자동 회전
  1/2/3 날짜 (하지/춘분/동지, 맑은날 clear-sky)   ESC 종료
표시: 시간대별 하늘·태양 디스크, 하루 발전 곡선(오라클 vs 현재각), 발전 게이지,
      음영률·고도/방위/프로파일각.
셀프테스트:  python sim_pygame.py --selftest  → out/p4_frames/*.png 저장 후 종료.
"""
import os
import sys

import numpy as np
import pandas as pd
import pvlib
import pygame
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (PITCH, blade_segment, strip_segment,
                      profile_angle_deg, shadow_polygons, beam_shaded_fraction)
from irradiance import build_tables, power_series, WALL_Y

LOC = pvlib.location.Location(37.5665, 126.9780, tz="Asia/Seoul")
DAYS = {"1": ("하지", "2024-06-21"), "2": ("춘분", "2024-03-21"),
        "3": ("동지", "2024-12-21")}
W, H = 1180, 860
SCENE_H = 660                     # 단면 뷰 높이 (아래는 발전 곡선 패널)
SCALE = 0.95                      # px / mm
OX, OY = 520, SCENE_H // 2        # 피벗(대상 블레이드) 화면 위치
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


def lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Sim:
    def __init__(self):
        self.tab = build_tables()
        self.tilt = 83.0
        self.ai = True
        self.playing = True
        self.set_day("1")
        self.i = len(self.data["t"]) // 2

    def set_day(self, key):
        self.day_key = key
        d = day_series(DAYS[key][1])
        self.data = d
        # 하루 전체 발전 행렬 (91 tilt × T) — 곡선·오라클 즉답용
        self.P_day = np.stack([
            power_series(self.tab, float(t), d["elev"], d["az"],
                         d["dni"], d["dhi"], d["ghi"]) for t in TILT_GRID])
        self.curve_oracle = self.P_day.max(axis=0)
        self.i = min(getattr(self, "i", 0), len(d["t"]) - 1)

    def instant(self):
        d, i = self.data, self.i
        return (d["elev"][i], d["az"][i], d["dni"][i], d["dhi"][i], d["ghi"][i])

    def power(self, tilt):
        return float(self.P_day[int(round(tilt)), self.i])

    def oracle(self):
        j = int(self.P_day[:, self.i].argmax())
        return float(TILT_GRID[j]), float(self.P_day[j, self.i])

    def step(self, di=1):
        self.i = (self.i + di) % len(self.data["t"])


def draw_sky(screen, elev, az):
    """시간대별 하늘 그라디언트 + 태양 디스크."""
    night = ((24, 30, 54), (40, 46, 72))
    dawn = ((255, 168, 110), (250, 214, 160))
    day = ((132, 178, 235), (208, 228, 248))
    if elev <= -2:
        top, bot = night
    elif elev < 12:                      # 박명~저녁 보간
        t = (elev + 2) / 14
        top = lerp(lerp(night[0], dawn[0], min(t * 2, 1)), day[0], max(t * 2 - 1, 0))
        bot = lerp(lerp(night[1], dawn[1], min(t * 2, 1)), day[1], max(t * 2 - 1, 0))
    else:
        top, bot = day
    for y in range(0, SCENE_H, 4):
        pygame.draw.rect(screen, lerp(top, bot, y / SCENE_H), (0, y, W, 4))
    # 태양 디스크: x=방위(동90→서270), y=고도
    if elev > 0:
        sx = int(np.interp(az, [75, 285], [60, W - 60]))
        sy = int(np.interp(elev, [0, 80], [SCENE_H - 60, 40]))
        for r, a in ((34, 36), (24, 70), (14, 255)):
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 218, 120, a), (r, r), r)
            screen.blit(s, (sx - r, sy - r))


def draw_scene(screen, sim, prof):
    """블레이드 단면 + 그림자 + PV 띠 + 광선."""
    e, a, dni, dhi, ghi = sim.instant()
    sf = beam_shaded_fraction(sim.tilt, prof)
    if prof is not None:
        b = np.radians(prof)
        ray = np.array([-np.cos(b), -np.sin(b)])
        polys = []
        for k in range(N_BLADES):
            z = (k - TARGET) * PITCH
            s0, s1 = blade_segment(sim.tilt, z)
            polys.append(Polygon([s0, s1, s1 + 1500 * ray, s0 + 1500 * ray]))
        merged = unary_union(polys)
        for g in getattr(merged, "geoms", [merged]):
            sh = pygame.Surface((W, SCENE_H), pygame.SRCALPHA)
            pygame.draw.polygon(sh, (30, 36, 60, 70),
                                [w2s(p) for p in g.exterior.coords])
            screen.blit(sh, (0, 0))
        for zl in np.arange(-2.6, 2.7, 0.45) * PITCH:
            p0 = np.array([340.0, zl + 340.0 * np.tan(b)])
            p1 = p0 - 1000 * np.array([np.cos(b), np.sin(b)])
            pygame.draw.line(screen, (255, 226, 150), w2s(p0), w2s(p1), 1)
    # 후방 벽 + 실내 측 어둡게
    wx = w2s((WALL_Y, 0))[0]
    pygame.draw.rect(screen, (52, 54, 60), (0, 0, wx, SCENE_H))
    pygame.draw.rect(screen, (96, 94, 90), (wx - 10, 0, 12, SCENE_H))
    # 블레이드 + PV 띠
    for k in range(N_BLADES):
        z = (k - TARGET) * PITCH
        s0, s1 = blade_segment(sim.tilt, z)
        pygame.draw.line(screen, (228, 228, 232), w2s(s0), w2s(s1), 8)
        pygame.draw.line(screen, (140, 140, 150), w2s(s0), w2s(s1), 2)
        t0, t1 = strip_segment(sim.tilt, z)
        pygame.draw.line(screen, (28, 96, 190), w2s(t0), w2s(t1), 4)
    # 대상 블레이드 음영 구간 = 빨강
    if prof is not None and sf > 0:
        t0, t1 = strip_segment(sim.tilt, 0.0)
        strip = LineString([t0, t1])
        inter = strip.intersection(unary_union(shadow_polygons(sim.tilt, prof)))
        for g in getattr(inter, "geoms", [inter]):
            cs = list(g.coords)
            pygame.draw.line(screen, (230, 50, 70), w2s(cs[0]), w2s(cs[-1]), 6)
    return sf


def draw_curve_panel(screen, font, small, sim):
    """하루 발전 곡선: 오라클(녹색) vs 현재 tilt 고정(파랑) + 현재 시각 커서."""
    y0 = SCENE_H
    pygame.draw.rect(screen, (28, 30, 38), (0, y0, W, H - y0))
    pad, pw, ph = 70, W - 140, H - y0 - 58
    top = y0 + 34
    ref = max(float(sim.curve_oracle.max()), 1e-9)
    n = len(sim.curve_oracle)
    xs = np.linspace(pad, pad + pw, n)
    cur = sim.P_day[int(round(sim.tilt))]

    def pts(curve):
        return [(int(x), int(top + ph * (1 - v / ref)))
                for x, v in zip(xs, curve)]
    pygame.draw.lines(screen, (90, 200, 120), False, pts(sim.curve_oracle), 3)
    pygame.draw.lines(screen, (90, 150, 240), False, pts(cur), 2)
    # 시간 눈금
    t = sim.data["t"]
    for hh in range(6, 21, 2):
        idx = np.searchsorted(t.hour * 60 + t.minute, hh * 60)
        if idx < n:
            x = int(xs[idx])
            pygame.draw.line(screen, (70, 72, 84), (x, top), (x, top + ph), 1)
            screen.blit(small.render(f"{hh:02d}", True, (150, 152, 165)),
                        (x - 9, top + ph + 4))
    # 현재 시각 커서
    cx = int(xs[sim.i])
    pygame.draw.line(screen, (250, 210, 110), (cx, top - 6), (cx, top + ph + 6), 2)
    # 범례 + 하루 적산 비교
    e_or, e_cur = float(sim.curve_oracle.sum()), float(cur.sum())
    pct = e_cur / e_or * 100 if e_or > 0 else 0
    screen.blit(font.render(
        f"하루 발전 곡선  ─ 오라클(AI)  ─ 현재각 {sim.tilt:.0f}° 고정  "
        f"|  하루 적산: 현재각 = 오라클의 {pct:.1f}%", True, (225, 226, 232)),
        (pad, y0 + 8))


def draw_hud(screen, font, big, sim, prof, sf):
    e, a, dni, dhi, ghi = sim.instant()
    opt_tilt, opt_p = sim.oracle()
    cur_p = sim.power(sim.tilt)
    name, date = DAYS[sim.day_key]
    ts = sim.data["t"][sim.i].strftime("%H:%M")
    panel = pygame.Surface((332, 252), pygame.SRCALPHA)
    panel.fill((16, 18, 26, 185))
    screen.blit(panel, (14, 14))
    screen.blit(big.render(f"{name} {ts}", True, (255, 255, 255)), (30, 26))
    mode = "AI 오라클" if sim.ai else "수동"
    col = (120, 230, 150) if sim.ai else (250, 200, 110)
    screen.blit(font.render(f"모드 {mode}", True, col), (30, 62))
    ptxt = f"{prof:.1f}°" if prof is not None else "후면/일몰"
    lines = [
        f"tilt {sim.tilt:.0f}°   (오라클 {opt_tilt:.0f}°)",
        f"PV 띠 음영 {sf*100:.0f}%",
        f"고도 {e:.1f}°  방위 {a:.1f}°",
        f"프로파일각 {ptxt}",
        f"DNI {dni:.0f}  DHI {dhi:.0f}  GHI {ghi:.0f}",
    ]
    for li, txt in enumerate(lines):
        screen.blit(font.render(txt, True, (228, 229, 235)), (30, 92 + 26 * li))
    # 발전 게이지 (현재 vs 오라클)
    gx, gy, gw = 30, 92 + 26 * 5 + 8, 286
    ref = max(opt_p, 1e-9)
    pygame.draw.rect(screen, (60, 62, 75), (gx, gy, gw, 16), border_radius=4)
    fill = (90, 200, 120) if cur_p / ref > 0.98 else (90, 150, 240)
    pygame.draw.rect(screen, fill,
                     (gx, gy, int(gw * min(cur_p / ref, 1.0)), 16),
                     border_radius=4)
    screen.blit(font.render(
        f"{cur_p:.0f} / {opt_p:.0f} W/m²  ({cur_p/ref*100:.1f}%)",
        True, (228, 229, 235)), (gx, gy + 20))
    # 조작 도움말 (우상단)
    help_txt = "SPACE 재생 · ←→ 시간 · ↑↓ 각도 · A 모드 · 1/2/3 날짜 · ESC"
    s = font.render(help_txt, True, (245, 246, 250))
    bg = pygame.Surface((s.get_width() + 20, s.get_height() + 10), pygame.SRCALPHA)
    bg.fill((16, 18, 26, 150))
    screen.blit(bg, (W - s.get_width() - 34, 14))
    screen.blit(s, (W - s.get_width() - 24, 19))


def draw(screen, fonts, sim):
    font, big, small = fonts
    e, a, *_ = sim.instant()
    prof = profile_angle_deg(e, a)
    if sim.ai:
        sim.tilt = sim.oracle()[0]
    draw_sky(screen, e, a)
    sf = draw_scene(screen, sim, prof)
    draw_curve_panel(screen, font, small, sim)
    draw_hud(screen, font, big, sim, prof, sf)


def pick_fonts():
    for name in ("applesdgothicneo", "applegothic"):
        path = pygame.font.match_font(name)
        if path:
            return (pygame.font.Font(path, 17), pygame.font.Font(path, 26),
                    pygame.font.Font(path, 13))
    f = pygame.font.Font(None, 20)
    return (f, pygame.font.Font(None, 30), pygame.font.Font(None, 15))


def main():
    selftest = "--selftest" in sys.argv
    if selftest:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("env_check P4 — AI Tilt 클린룸 시뮬레이터")
    fonts = pick_fonts()
    clock = pygame.time.Clock()
    sim = Sim()
    if selftest:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "out", "p4_frames")
        os.makedirs(outdir, exist_ok=True)
        shots = [("1", 50, True), ("1", 60, False), ("3", 60, True),
                 ("3", 35, False), ("1", 95, True)]
        for n, (dk, idx_frac, ai) in enumerate(shots):
            sim.set_day(dk)
            sim.i = int(len(sim.data["t"]) * idx_frac / 100)
            sim.ai = ai
            if not ai:
                sim.tilt = 60.0
            draw(screen, fonts, sim)
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
        draw(screen, fonts, sim)
        pygame.display.flip()
        clock.tick(12)


if __name__ == "__main__":
    main()
