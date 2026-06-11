# ==============================================================================
# v17 불변식 게이트 배터리 — 적대검토(2026-06-11)가 잡은 결함의 영구 회귀 테스트
# 실행: .venv/bin/python invariants_v17.py   → exit 0 = 전 게이트 PASS
# 원칙: by-construction 금지 — 각 게이트는 독립 구현(광선추적) 또는 해석해 앵커에 대조
# G1 독립 광선추적 ↔ panel_sf (순환 차단)     G2 개구부 항등식 (capture=1)
# G3 MC 현/피치 의미론 (★구 2배 버그 재발 차단)  G4 PV 띠 음영 성질
# G5 IAM 앵커                                  G6 라벨 ↔ 물리 정합 + 모델 regret
# G7 결정 스냅샷 (최적각 사다리 드리프트 감지, WARN 전용)
# ==============================================================================
import json
import sys
import numpy as np
import pandas as pd
import pvlib

import physics_v3 as P3
from physics_v2 import panel_sf
from train_v17 import view_factors_mc, ANGLES

C, P = P3.CHORD, P3.PITCH
RES = {}
FAIL = []


def gate(name, ok, detail, warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    RES[name] = {"status": tag, "detail": detail}
    if not ok and not warn_only:
        FAIL.append(name)
    print(f" [{tag}] {name}: {detail}")


# ── G1+G2: 독립 전방 광선추적 (panel_sf 미사용 구현 — verify_adversarial 계보) ─
def trace_beam(tilt, elev, az, c=C, p=P, sa=180.0, n_rays=40000):
    tr = np.radians(tilt); er = np.radians(np.clip(elev, 0.1, 89.9))
    adr = np.radians(az - sa)
    rdx, rdy = -np.cos(er) * np.cos(adr), -np.sin(er)
    if rdx >= -1e-9:
        return 1.0, 0.0
    nbx, nby = np.sin(tr), np.cos(tr)
    cos_aoi = -(rdx * nbx + rdy * nby)
    L = c * 2 + 10.0
    y_at_ap = (np.arange(n_rays) + 0.5) / n_rays * p
    x0 = np.full(n_rays, L)
    y0 = y_at_ap - rdy / rdx * (0.0 - L)
    y_min = min(y_at_ap.min(), y0.min()) - c; y_max = max(y_at_ap.max(), y0.max()) + c
    k_lo = int(np.floor((y_min - c) / p)) - 1; k_hi = int(np.ceil((y_max + c) / p)) + 1
    best_t = np.full(n_rays, np.inf); best_front = np.zeros(n_rays, dtype=bool)
    ex, ey = c * np.cos(tr), -c * np.sin(tr)
    for k in range(k_lo, k_hi + 1):
        qx, qy = 0.0 - x0, k * p - y0
        den = rdx * ey - rdy * ex
        if abs(den) < 1e-12:
            continue
        t_ray = (qx * ey - qy * ex) / den
        s_seg = (qx * rdy - qy * rdx) / den
        hit = (t_ray > 1e-9) & (s_seg >= 0.0) & (s_seg <= 1.0) & (t_ray < best_t)
        best_t = np.where(hit, t_ray, best_t)
        best_front = np.where(hit, cos_aoi > 0, best_front)
    P_ap = abs(rdx) * p
    P_front = (np.isfinite(best_t) & best_front).sum() / n_rays * P_ap
    P_noshade = max(cos_aoi, 0) * c
    sf_true = 1.0 - P_front / P_noshade if P_noshade > 1e-12 else 1.0
    return float(np.clip(sf_true, 0, 1)), float(P_front / P_ap)


CASES = [(30.0, 180.0), (74.0, 180.0), (52.0, 180.0), (20.0, 220.0), (45.0, 110.0)]
errs, caps = [], []
for e_, a_ in CASES:
    for t in [0.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
        sf_m = float(panel_sf(t, e_, a_, hd=C, p=P))
        sf_t, cap = trace_beam(t, e_, a_)
        errs.append(abs(sf_m - sf_t))
        if t >= 45:
            caps.append(cap)
gate("G1_tracer_vs_panel_sf", max(errs) < 0.02, f"30케이스 최대오차 {max(errs):.4f} (<0.02)")
gate("G2_aperture_identity", min(caps) > 0.999, f"tilt≥45° capture 최소 {min(caps):.4f} (=1.0)")

# ── G3: MC 현/피치 의미론 (구 2배 버그 재발 차단) ────────────────────────────
fs90, fg90 = view_factors_mc(90.0, 1.0)
fs0_10, _ = view_factors_mc(0.0, 1.0)
fs0_05, _ = view_factors_mc(0.0, 0.5)
# 앵커1: 90°(동일평면) = 수직벽 해석값 0.5/0.5 (현 무관)
ok_a = abs(fs90 - 0.5) < 0.01 and abs(fg90 - 0.5) < 0.01
# 앵커2: tilt0 에서 현/피치 1.0 → F_sky≈0.30 (구버그의 2.0 기하면 0.197 로 나옴 = 재발 감지)
ok_b = abs(fs0_10 - 0.300) < 0.02 and fs0_10 > 0.25
# 앵커3: 단조성 — 현이 좁을수록(0.5) tilt0 하늘이 더 보임
ok_c = fs0_05 > fs0_10 + 0.05
# 앵커4: MC 시드 안정성 (수렴 검사는 고표본 — 기본 n_dir=4000 의 노이즈 ±0.02 는 정상)
fs60_s2, _ = view_factors_mc(60.0, 1.0, n_pts=128, n_dir=12000, seed=7)
fs60_s1, _ = view_factors_mc(60.0, 1.0, n_pts=128, n_dir=12000)
ok_d = abs(fs60_s1 - fs60_s2) < 0.015
gate("G3_vf_chord_semantics", ok_a and ok_b and ok_c and ok_d,
     f"90°=({fs90:.3f},{fg90:.3f})≈0.5 | tilt0 c/p1.0={fs0_10:.3f}(버그면 0.197) "
     f"c/p0.5={fs0_05:.3f} | 시드차 {abs(fs60_s1 - fs60_s2):.4f}")

# ── G4: PV 띠 음영 성질 ──────────────────────────────────────────────────────
lo = P3.STRIP_LO
sf_test = np.array([0.0, lo, lo + 0.01, 0.5, 1.0])
ov = P3.strip_shade(sf_test)
ok = (ov[0] == 0 and ov[1] == 0 and 0 < ov[2] < 0.02 and abs(ov[4] - 1.0) < 1e-9
      and np.all(np.diff(ov) >= 0))
gate("G4_strip_shade", bool(ok), f"sf={sf_test.tolist()} → ov={np.round(ov, 3).tolist()} (마진흡수·단조·sf1→1)")

# ── G5: IAM 앵커 ─────────────────────────────────────────────────────────────
aoi_g = np.array([0.0, 30.0, 60.0, 74.0, 85.0])
iam = np.asarray(pvlib.iam.martin_ruiz(aoi_g, a_r=P3.A_R), dtype=float)
ok = abs(iam[0] - 1.0) < 1e-6 and np.all(np.diff(iam) < 0) and abs(iam[2] - 0.958) < 0.01 and abs(iam[3] - 0.823) < 0.01
gate("G5_iam_anchors", bool(ok), f"IAM(0/30/60/74/85)={np.round(iam, 3).tolist()} (1.0·단조↓·.958·.823)")

# ── G6: 라벨 ↔ 물리 정합 + 모델 에너지 regret ────────────────────────────────
df = pd.read_csv("bipv_ai_master_data_v17.csv")
day = df["ghi_w_m2"].to_numpy(float) >= 10
rng = np.random.default_rng(0)
idx_s = rng.choice(np.where(day)[0], 2000, replace=False)
el = df["solar_elevation"].to_numpy(float)[idx_s]; az = df["solar_azimuth"].to_numpy(float)[idx_s]
dni = df["dni"].to_numpy(float)[idx_s]; dhi = df["dhi"].to_numpy(float)[idx_s]
tgt = df["target_angle_v17"].to_numpy(float)[idx_s]
E = P3.eff_poa(ANGLES[:, None], el[None, :], az[None, :], dni[None, :], dhi[None, :])
opt = ANGLES[np.argmax(E, axis=0)]
e_lbl = E[np.searchsorted(ANGLES, tgt), np.arange(len(tgt))]
e_opt = E.max(axis=0)
mask = e_opt > 0
eloss = float(np.mean(1 - e_lbl[mask] / e_opt[mask])) * 100
gate("G6a_labels_match_physics", eloss < 0.1,
     f"라벨 vs argmax 에너지 손실 {eloss:.4f}% (<0.1%, 각도일치 {np.mean(np.abs(opt - tgt) <= 1) * 100:.1f}%)")
with open("model_metrics_v17.json", encoding="utf-8") as f:
    m = json.load(f)
gate("G6b_model_regret", m["energy_regret_pct"] < 0.5, f"홀드아웃 에너지 regret {m['energy_regret_pct']}% (<0.5%)")

# ── G7: 결정 스냅샷 (드리프트 감지, WARN 전용) ───────────────────────────────
el_a = df["solar_elevation"].to_numpy(float)[day]; az_a = df["solar_azimuth"].to_numpy(float)[day]
dni_a = df["dni"].to_numpy(float)[day]; dhi_a = df["dhi"].to_numpy(float)[day]
tgt_a = df["target_angle_v17"].to_numpy(float)[day]
e_ai = P3.eff_poa(tgt_a, el_a, az_a, dni_a, dhi_a).sum()
lad = {}
for fa in [45.0, 60.0, 75.0, 90.0]:
    lad[int(fa)] = round(float(e_ai / P3.eff_poa(np.full_like(el_a, fa), el_a, az_a, dni_a, dhi_a).sum() - 1) * 100, 2)
GOLDEN = {45: 8.0, 60: 4.2, 75: 1.0, 90: 6.5}   # 2026-06-11 v17 초기값 (±1.5%p 드리프트 감지)
drift = {k: round(lad[k] - GOLDEN[k], 2) for k in lad}
gate("G7_decision_snapshot", all(abs(d) <= 1.5 for d in drift.values()),
     f"AI vs 고정 사다리 {lad} (golden 대비 드리프트 {drift})", warn_only=True)

with open("invariants_v17_results.json", "w") as f:
    json.dump(RES, f, ensure_ascii=False, indent=1)
n_pass = sum(1 for v in RES.values() if v["status"] == "PASS")
print(f"\n{n_pass}/{len(RES)} PASS | FAIL: {FAIL or '없음'} → invariants_v17_results.json")
sys.exit(1 if FAIL else 0)
