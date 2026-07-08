# ==============================================================================
# 개별 블레이드 각도제어 이득 검증 (2026-07-08) — 사용자 질문
#   지금: 전 블레이드 동일 각도(균일). 블레이드마다 다른 각도로 주면 발전↑?
#   물리 긴장: 한 장을 눕히면 자기 발전↑ 但 이웃에 그림자↑ or 이웃 그림자에 들어감.
#   주기 N=2 배열(θ0,θ1) 광선추적 자기음영 → 비균일 최적 vs 균일 최적 비교.
#   조건=연간 대표 표본(균등 간격 600시간). 발전=per 파사드(블레이드당 평균).
#   ★한계: 확산광 시야계수는 각 블레이드 자기각의 uniform-neighbor VF로 근사(빔은 엄밀 광선추적).
#   실행: .venv/bin/python verify_individual_blade.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P

C, PIT, SA = P.CHORD, P.PITCH, 180.0
LO, FR = P.STRIP_LO, P.STRIP_FRAC     # 셀 스트립 (상단여백/폭)

df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df.solar_elevation > 3].reset_index(drop=True)
idx = np.linspace(0, len(dd) - 1, 600).astype(int)     # 연간 균등 대표 표본
s = dd.iloc[idx]
el = s.solar_elevation.to_numpy(float); az = s.solar_azimuth.to_numpy(float)
dni = s.dni.to_numpy(float); dhi = s.dhi.to_numpy(float)
ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
M = len(s)
# 태양 2D 프로파일 방향 (sf_group·panel_sf 검증된 관례)
er = np.radians(np.clip(el, 0.1, 89.9))
D = np.stack([np.cos(er) * np.cos(np.radians(az - SA)), np.sin(er)], axis=1)   # [M,2] 점→태양(위로)
# IAM 사전
def iam_terms(t):
    aoi = pvlib.irradiance.aoi(np.full(M, t), SA, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    ib = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(np.array([float(t)]), a_r=P.A_R)
    return bp, ib, float(sky[0]), float(grd[0])
IAM = {t: iam_terms(t) for t in np.arange(0, 90.1, 1.0)}
VF = {t: (float(P.view_factor(t, C, PIT, "f_sky")), float(P.view_factor(t, C, PIT, "f_grd"))) for t in np.arange(0, 90.1, 1.0)}


def endpoints(k, ang):
    cy = k * PIT
    dirv = np.array([np.cos(np.radians(ang)), -np.sin(np.radians(ang))])
    top = np.array([0.0, cy]) + (C / 2) * dirv       # 현 상단(+C/2)
    bot = np.array([0.0, cy]) - (C / 2) * dirv       # 현 하단
    return top, bot


def cell_shadow(angles, jt, nS=16, G=2):
    """주기 angles(길이 N)에서 target 블레이드 jt의 셀 스트립 음영률 [M] (빔에너지 무관, 순수 기하)."""
    N = len(angles)
    segs = [endpoints(g * N + j, angles[j]) for g in range(-G, G + 1) for j in range(N)]
    self_k = (0) * N + jt   # target = 중앙 주기 g=0
    top_t, bot_t = endpoints(jt, angles[jt])
    ss = np.linspace(LO, LO + FR, nS)                # 셀 스트립만 샘플
    shaded = np.zeros(M)
    for u in ss:
        p = top_t + u * (bot_t - top_t)              # 현 상단부터 u 비율
        hit = np.zeros(M, bool)
        for (A, B) in segs:
            if np.allclose(A, top_t) and np.allclose(B, bot_t):
                continue
            e = B - A; q = A - p
            den = e[0] * D[:, 1] - D[:, 0] * e[1]
            ok = np.abs(den) > 1e-12
            t = np.where(ok, (e[0] * q[1] - e[1] * q[0]) / np.where(ok, den, 1), -1)
            spar = np.where(ok, (D[:, 0] * q[1] - D[:, 1] * q[0]) / np.where(ok, den, 1), -1)
            hit |= ok & (t > 1e-6) & (spar >= 0) & (spar <= 1)
        shaded += hit
    return shaded / nS


def gen_perfacade(angles):
    """per 파사드(블레이드당 평균) 연간대표 발전."""
    N = len(angles); tot = 0.0
    for j in range(N):
        cs = cell_shadow(angles, j)
        aj = round(angles[j])
        bp, ib, sky, grd = IAM[aj]; fsky, fgrd = VF[aj]
        poa = np.maximum(bp * ib * (1 - cs) + dhi * fsky * sky + ghi * P.ALBEDO * fgrd * grd, 0)
        tot += poa.sum()
    return tot / N


# ── 균일 최적 (같은 모델) ──
GA = np.arange(60, 90.1, 2.0)          # 균일 스윕 (관심 고각대 + 여유)
uni = [(a, gen_perfacade([a, a])) for a in GA]
ubest_a, uE = max(uni, key=lambda x: x[1])

# ── 비균일 N=2 최적 (그리드) ──
grid = np.arange(30, 90.1, 4.0)
best = (None, -1)
for a0 in grid:
    for a1 in grid:
        if a1 < a0:      # 대칭 중복 skip
            continue
        E = gen_perfacade([a0, a1])
        if E > best[1]:
            best = ((a0, a1), E)
(b0, b1), bE = best

print(f"주간 대표표본 {M} · 현{C:.0f}/피치{PIT:.0f} gcr{C/PIT:.3f} · 셀 상{LO*C:.0f}/폭{FR*C:.0f}")
print(f"\n균일 최적 (전 블레이드 동일):  {ubest_a:.0f}°  = {uE:.4g}  (기준 100%)")
print(f"비균일 N=2 최적 (블레이드별):  ({b0:.0f}°, {b1:.0f}°)  = {bE:.4g}  ({bE/uE*100:.2f}%)")
print(f"\n→ 개별 제어 이득 = {(bE/uE-1)*100:+.2f}%")
if abs(b0 - b1) < 1e-6 or (bE / uE - 1) < 0.003:
    print("  판정: 비균일 이득 ≈ 0 → 균일이 최적(또는 동률). 개별 각도제어 무의미.")
else:
    print(f"  판정: 비균일이 균일을 {(bE/uE-1)*100:.2f}% 이김 → 개별 제어 실익 있음(검토).")

# 참고: 상위 몇 개 비균일 후보
print("\n[참고] 비균일 상위 5 (a0,a1,상대%):")
allc = sorted([((a0, a1), gen_perfacade([a0, a1])) for a0 in grid for a1 in grid if a1 >= a0],
              key=lambda x: -x[1])[:5]
for (a0, a1), E in allc:
    print(f"   ({a0:.0f}°,{a1:.0f}°)  {E/uE*100:.2f}%")
