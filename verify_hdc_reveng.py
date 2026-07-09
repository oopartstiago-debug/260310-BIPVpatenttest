# ==============================================================================
# HDC 계산모델 역공학 v2 (2026-07-09) — KR102460661B1 성능표 16조건(웹조사 확보)
#   가설 M1(패널 무음영률)은 실패(저고도서 100% vs 그들 28.9%). → 지표 재추정:
#   M2 = "창 개구부에 입사한 직달광선 중, 첫 교차가 블레이드 PV면(태양쪽 면)인 비율"
#        (원문 "패널에 닿는 선 vs 닿지 않는 선"과 정합 — 닿지 않는 선 = 틈으로 통과 or 뒷면)
#   자유 파라미터: gcr(현/피치) × 기울임 방향(바깥/안쪽) → 16 baseline 피팅
#   검증: 피팅 기하로 편심(1번축+24%/2번 항상수직/3번축-11%) 예측 vs 그들 편심 16개
#   실행: .venv/bin/python verify_hdc_reveng.py
# ==============================================================================
import numpy as np

C = 1.0

# 특허 표: {태양고도: {회전각ρ(0=수직닫힘): (종래%, 편심%)}}
TBL = {
    52: {20: (88.57, 88.86), 40: (86.91, 88.86), 60: (84.68, 87.74), 80: (79.94, 84.40)},
    76: {20: (75.66, 78.95), 40: (70.40, 82.24), 60: (69.08, 81.58), 80: (63.82, 78.29)},
    10: {20: (82.62, 82.62), 40: (72.34, 75.71), 60: (54.08, 62.94), 80: (28.90, 45.21)},
    20: {20: (88.10, 86.62), 40: (82.29, 83.27), 60: (67.84, 72.49), 80: (44.05, 56.32)},
}
CONDS = [(e, r) for e in (52, 76, 10, 20) for r in (20, 40, 60, 80)]


NB = 21   # 블레이드 수 (21=사실상 무한 / 소수=유한 모듈, 특허 도면은 ~6장)


def blades_geom(rho, gcr, mode, lean, K=None):
    """블레이드 세그먼트 목록 [(top, bot, pv_normal)]. lean=+1: 상단이 바깥(태양쪽), -1: 안쪽.
    mode='uni'(전부 중앙축 ρ) | 'ecc'(1번 축+0.24C·3번 축-0.11C 공간고정, 2번 항상 수직).
    유한 모듈: k=0..NB-1 (NB 전역)."""
    p = C / gcr
    out = []
    for k in range(0, NB):
        if mode == "uni":
            o, r = 0.0, rho
        else:
            j = k % 3
            o = (0.24, 0.0, -0.11)[j]
            r = 0.0 if j == 1 else rho
        beta = np.radians(90.0 - r)                      # tilt from horizontal
        dirv = np.array([lean * np.cos(beta), -np.sin(beta)])   # top→bot 방향
        axis = np.array([0.0, k * p])
        top = axis + (0.5 * C - o) * dirv
        bot = axis - (0.5 * C + o) * dirv
        # PV면 법선: 태양쪽(+x)/위쪽을 향하는 수직벡터
        n = np.array([-dirv[1], dirv[0]])
        if n[1] < 0 or (abs(n[1]) < 1e-9 and n[0] < 0):
            n = -n
        if abs(n[1]) < 1e-9:                             # 수직 블레이드: 태양쪽(+x) 면
            n = n if n[0] > 0 else -n
        out.append((top, bot, n))
    return out, p


def pv_hit_fraction(rho, elev, gcr, mode, lean, N=800, denom="aperture"):
    """평행광선 추적. 반환 = PV면 첫도달 광선 비율(%).
    denom='aperture': 분모=개구부 전체 광선(통과광 포함) / 'blocked': 분모=블레이드에 걸린 광선만."""
    segs, p = blades_geom(rho, gcr, mode, lean)
    er = np.radians(elev)
    u = np.array([-np.cos(er), -np.sin(er)])             # 광선 진행방향(태양→안쪽·아래)
    y_lo, y_hi = -0.5 * p, (NB - 0.5) * p                # 프레임 개구부(수직) 범위
    X0 = 5.0
    hits = 0; blocked = 0; tot = 0
    for y0 in np.linspace(y_lo, y_hi, N, endpoint=False):
        start = np.array([X0, y0 + X0 * np.tan(er)])     # x=0에서 y0을 지나는 광선
        best_t = None; best_pv = False
        for (A, B, n) in segs:
            e_ = B - A; q = A - start
            den = e_[0] * u[1] - u[0] * e_[1]
            if abs(den) < 1e-12: continue
            t = (e_[0] * q[1] - e_[1] * q[0]) / den
            s = (u[0] * q[1] - u[1] * q[0]) / den
            if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9:
                if best_t is None or t < best_t:
                    best_t = t
                    best_pv = float(np.dot(u, n)) < 0     # PV면 쪽에서 입사
        tot += 1
        if best_t is not None:
            blocked += 1
            if best_pv: hits += 1
    d = tot if denom == "aperture" else max(blocked, 1)
    return hits / d * 100


def sse_baseline(gcr, lean, N=300, denom="aperture"):
    return sum((pv_hit_fraction(r, e, gcr, "uni", lean, N, denom) - TBL[e][r][0]) ** 2 for e, r in CONDS)


print("① 가설 전수탐색 — 분모(개구부/걸린광선) × 기울임 × 블레이드수 × gcr")
best = (None, None, None, None, 1e18)
for denom in ("aperture", "blocked"):
    for lean in (+1, -1):
        for nb in (5, 6, 8, 21):
            NB = nb
            for gcr in np.arange(0.50, 1.31, 0.05):
                s = sse_baseline(gcr, lean, N=250, denom=denom)
                if s < best[4]: best = (gcr, nb, lean, denom, s)
g0, nb0, l0, d0, _ = best
NB = nb0
for gcr in np.arange(max(0.3, g0 - 0.05), g0 + 0.051, 0.01):
    s = sse_baseline(gcr, l0, N=600, denom=d0)
    if s < best[4]: best = (gcr, nb0, l0, d0, s)
GCR, NB, LEAN, DEN, SSE = best
print(f"  최적: gcr={GCR:.2f} · {NB}장 · 기울임={'바깥아래' if LEAN>0 else '안쪽아래'} · 분모={DEN} · RMS {np.sqrt(SSE/16):.2f}%p")

print(f"\n② 복원모델 vs 특허 표 — 종래 / 편심(1번+24%·2번수직고정·3번-11%)")
print(f"  {'고도':>4}{'ρ':>5} | {'종래:특허':>9}{'재현':>7}{'오차':>6} | {'편심:특허':>9}{'재현':>7}{'오차':>6}")
eu = []; ec = []
for e, r in CONDS:
    pu, pc = TBL[e][r]
    su = pv_hit_fraction(r, e, GCR, "uni", LEAN, denom=DEN)
    sc = pv_hit_fraction(r, e, GCR, "ecc", LEAN, denom=DEN)
    eu.append(abs(su - pu)); ec.append(abs(sc - pc))
    print(f"  {e:>3}°{r:>4}° | {pu:>8.2f}{su:>7.1f}{su-pu:>+6.1f} | {pc:>8.2f}{sc:>7.1f}{sc-pc:>+6.1f}")
print(f"  RMS오차: 종래 {np.sqrt(np.mean(np.square(eu))):.1f}%p · 편심 {np.sqrt(np.mean(np.square(ec))):.1f}%p · 최대 {max(eu):.1f}/{max(ec):.1f}%p")
print("\n판정: RMS 수%p 이내면 그들 계산기 복원 성공 → 지표·기하·기울임 확정.")
