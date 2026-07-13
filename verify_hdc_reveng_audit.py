# ==============================================================================
# HDC 역공학 자가감사 (2026-07-13) — "0.82가 정말 단단한 값인가?"
#   사용자 지적: 실물 루버는 밀폐라 gcr≥1이어야 하는데 우리 피팅이 0.82(비밀폐)를 뱉음.
#   → 0.82가 argmin일 뿐인지(=1.0 이상도 비슷하게 맞음=취약), 정말 압도적인지 정량.
#   verify_hdc_reveng.py의 기하·지표를 그대로 재사용(복붙, 원본 불변).
# ==============================================================================
import numpy as np

C = 1.0
TBL = {
    52: {20: (88.57, 88.86), 40: (86.91, 88.86), 60: (84.68, 87.74), 80: (79.94, 84.40)},
    76: {20: (75.66, 78.95), 40: (70.40, 82.24), 60: (69.08, 81.58), 80: (63.82, 78.29)},
    10: {20: (82.62, 82.62), 40: (72.34, 75.71), 60: (54.08, 62.94), 80: (28.90, 45.21)},
    20: {20: (88.10, 86.62), 40: (82.29, 83.27), 60: (67.84, 72.49), 80: (44.05, 56.32)},
}
CONDS = [(e, r) for e in (52, 76, 10, 20) for r in (20, 40, 60, 80)]


def blades_geom(rho, gcr, mode, lean, NB):
    p = C / gcr
    out = []
    for k in range(0, NB):
        if mode == "uni":
            o, r = 0.0, rho
        else:
            j = k % 3
            o = (0.24, 0.0, -0.11)[j]
            r = 0.0 if j == 1 else rho
        beta = np.radians(90.0 - r)
        dirv = np.array([lean * np.cos(beta), -np.sin(beta)])
        axis = np.array([0.0, k * p])
        top = axis + (0.5 * C - o) * dirv
        bot = axis - (0.5 * C + o) * dirv
        n = np.array([-dirv[1], dirv[0]])
        if n[1] < 0 or (abs(n[1]) < 1e-9 and n[0] < 0):
            n = -n
        if abs(n[1]) < 1e-9:
            n = n if n[0] > 0 else -n
        out.append((top, bot, n))
    return out, p


def pv_hit_fraction(rho, elev, gcr, mode, lean, NB, N=600, denom="aperture"):
    segs, p = blades_geom(rho, gcr, mode, lean, NB)
    er = np.radians(elev)
    u = np.array([-np.cos(er), -np.sin(er)])
    y_lo, y_hi = -0.5 * p, (NB - 0.5) * p
    X0 = 5.0
    hits = 0; blocked = 0; tot = 0
    for y0 in np.linspace(y_lo, y_hi, N, endpoint=False):
        start = np.array([X0, y0 + X0 * np.tan(er)])
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
                    best_pv = float(np.dot(u, n)) < 0
        tot += 1
        if best_t is not None:
            blocked += 1
            if best_pv: hits += 1
    d = tot if denom == "aperture" else max(blocked, 1)
    return hits / d * 100


def rms_conv(gcr, lean, NB, denom):
    e2 = [(pv_hit_fraction(r, e, gcr, "uni", lean, NB, denom=denom) - TBL[e][r][0]) ** 2 for e, r in CONDS]
    return np.sqrt(np.mean(e2))


# 원본이 고른 최적 설정 재현: lean/denom/NB는 원본 스윕 결과를 그대로 씀
# 원본은 (lean,denom,NB) 조합도 스윕 → 여기선 대표조합들에서 gcr곡선을 그려 비교
print("=== gcr 스윕: 종래 컬럼 RMS(%p) — 낮을수록 잘 맞음 ===\n")
for denom in ("aperture", "blocked"):
    for lean in (+1, -1):
        for NB in (6, 21):
            row = []
            for gcr in (0.70, 0.82, 0.90, 1.00, 1.10, 1.169, 1.25):
                row.append(rms_conv(gcr, lean, NB, denom))
            best_gcr_grid = np.arange(0.5, 1.31, 0.02)
            best_rms = min((rms_conv(g, lean, NB, denom), g) for g in best_gcr_grid)
            tag = f"denom={denom:8s} lean={'바깥' if lean>0 else '안쪽'} NB={NB:2d}"
            print(tag)
            print("   gcr :  0.70   0.82   0.90   1.00   1.10  1.169  1.25")
            print("   RMS : " + " ".join(f"{v:5.1f}" for v in row))
            print(f"   → 최적 gcr={best_rms[1]:.2f} (RMS {best_rms[0]:.1f})   |  0.82 대비 1.169 손해 = {row[5]-row[1]:+.1f}%p\n")

print("=== 판정 기준 ===")
print("• 0.82와 1.169의 RMS 차가 작으면(수%p 이내) → 0.82는 argmin일 뿐, 밀폐 기하도 표 설명 가능 = 0.82 취약")
print("• 어느 조합서도 gcr≥1의 RMS가 크게 나쁘면 → 우리 지표(M2) 가정이 실물 밀폐와 안 맞는다는 신호")
