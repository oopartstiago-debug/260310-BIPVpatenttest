# ==============================================================================
# HDC 계산모델 재분석 (2026-07-13) — "높은 해서 왜 100% 포화?" 기전 분해
#   광선을 PV정면 / 뒷면 / 틈통과 로 나눠 세어, 특허표(<100%)와의 갭 원인 특정.
#   그다음 후보지표(입사각가중·PV스트립부분) 테스트.
# ==============================================================================
import numpy as np

C = 1.0
TBL = {  # {고도: {ρ(0=수직닫힘): 종래%}}
    52: {20: 88.57, 40: 86.91, 60: 84.68, 80: 79.94},
    76: {20: 75.66, 40: 70.40, 60: 69.08, 80: 63.82},
    10: {20: 82.62, 40: 72.34, 60: 54.08, 80: 28.90},
    20: {20: 88.10, 40: 82.29, 60: 67.84, 80: 44.05},
}
CONDS = [(e, r) for e in (52, 76, 10, 20) for r in (20, 40, 60, 80)]


def blades(rho, gcr, lean, NB, strip_lo=0.0, strip_hi=1.0):
    """블레이드 세그먼트 + PV구간(s∈[strip_lo,strip_hi]만 발전). top→bot 파라미터 s=0..1."""
    p = C / gcr
    out = []
    for k in range(NB):
        beta = np.radians(90.0 - rho)
        dirv = np.array([lean * np.cos(beta), -np.sin(beta)])
        axis = np.array([0.0, k * p])
        top = axis + 0.5 * C * dirv
        bot = axis - 0.5 * C * dirv
        n = np.array([-dirv[1], dirv[0]])
        if n[1] < 0 or (abs(n[1]) < 1e-9 and n[0] < 0):
            n = -n
        if abs(n[1]) < 1e-9:
            n = n if n[0] > 0 else -n
        out.append((top, bot, n))
    return out, p


def trace(rho, elev, gcr, lean, NB=21, N=1200, strip_lo=0.0, strip_hi=1.0, iam=False):
    """반환: dict(pv_front%, back%, gap%, pv_cosw) — pv_cosw=입사각가중 PV정면(정규화)."""
    segs, p = blades(rho, gcr, lean, NB)
    er = np.radians(elev)
    u = np.array([-np.cos(er), -np.sin(er)])
    y_lo, y_hi = -0.5 * p, (NB - 0.5) * p
    X0 = 5.0
    n_pv = n_back = n_gap = 0
    cosw = 0.0
    tot = 0
    for y0 in np.linspace(y_lo, y_hi, N, endpoint=False):
        start = np.array([X0, y0 + X0 * np.tan(er)])
        best_t = None; best = None
        for (A, B, nrm) in segs:
            e_ = B - A; q = A - start
            den = e_[0] * u[1] - u[0] * e_[1]
            if abs(den) < 1e-12: continue
            t = (e_[0] * q[1] - e_[1] * q[0]) / den
            s = (u[0] * q[1] - u[1] * q[0]) / den
            if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9:
                if best_t is None or t < best_t:
                    best_t = t; best = (nrm, s)
        tot += 1
        if best is None:
            n_gap += 1
        else:
            nrm, s = best
            cos_i = -float(np.dot(u, nrm))          # 광선-법선 정면도 (>0 = 정면 입사)
            on_strip = strip_lo <= s <= strip_hi
            if cos_i > 0 and on_strip:
                n_pv += 1
                cosw += cos_i
            elif cos_i > 0:                          # 정면이지만 PV 밖(맨 여백)
                n_back += 0                          # 여백 = 미이용(뒤로 분류 안 하고 gap과 구분위해 pv에서 뺌)
                n_gap += 0
                n_back += 1
            else:
                n_back += 1
    f = 100.0 / tot
    return dict(pv=n_pv * f, back=n_back * f, gap=n_gap * f, cosw=cosw * f)


print("=== 기전 분해: 광선이 어디로 가나 (PV정면 / 뒷면·여백 / 틈통과) ===")
for gcr in (0.82, 1.169):
    print(f"\n--- gcr={gcr} (전면 PV, strip 전체) ---")
    print(f"{'고도':>4}{'ρ':>4} | {'특허':>6} | {'PV정면':>7}{'뒷면':>6}{'틈':>6} | {'입사각가중PV':>11}")
    for e, r in CONDS:
        d = trace(r, e, gcr, +1, strip_lo=0, strip_hi=1)
        print(f"{e:>3}°{r:>3}° | {TBL[e][r]:>6.1f} | {d['pv']:>7.1f}{d['back']:>6.1f}{d['gap']:>6.1f} | {d['cosw']:>11.1f}")

print("\n※ 특허값이 PV정면(현모델)보다 낮으면 → 그 차이만큼 '어디로' 새는지가 관건.")
print("  입사각가중PV 컬럼이 특허와 가까우면 → 그들 지표=입사각가중(정면도) 가설 지지.")
