"""P3 전수 교차대조 — 클린룸 엔진(env_check) vs physics_v3.

이 파일만 physics_v3 임포트 허용 (클린룸 규칙의 예외 = 대조 하네스).
레벨별 대조:
  L1 빔 음영 그리드  : shapely 투영  vs  panel_sf+strip_shade (해석적 선분교차)
  L2 시야계수 테이블 : 광선캐스팅    vs  MC view_factors_v17.json
  L3 시간별 결정     : 양 엔진 최적각·교차 regret (상대 발전량 전 시간)
  L4 연간 사다리     : AI vs 고정 60/81/83/90 — POSITIONING.md 수치와 대조
알려진 모델 선택 차이(버그 아님, 분리 보고):
  v3 = 확산 IAM 적용 + ghi_est=dni·sinα+dhi  /  클린룸 = 확산 IAM 미적용 + CSV ghi.
실행: 프로젝트 루트에서  .venv/bin/python env_check/crosscheck.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from geometry import beam_shaded_fraction                      # 클린룸
from irradiance import build_tables, power_series              # 클린룸
import physics_v3 as v3                                        # 대조 대상

TILT_GRID = np.arange(0, 91, 1.0)


def l1_sf_grid():
    rows = []
    for tilt in range(0, 91, 5):
        for prof in range(5, 90, 5):
            mine = beam_shaded_fraction(float(tilt), float(prof))
            ref = float(v3.strip_shade(v3.panel_sf(
                np.array([float(tilt)]), np.array([float(prof)]),
                np.array([180.0]), hd=97.5, p=97.5))[0])
            rows.append((tilt, prof, mine, ref, abs(mine - ref)))
    diffs = np.array([r[4] for r in rows])
    worst = sorted(rows, key=lambda r: -r[4])[:5]
    return {"max_abs": float(diffs.max()), "mean_abs": float(diffs.mean()),
            "worst5": [(int(t), int(p), round(m, 4), round(r, 4))
                       for t, p, m, r, _ in worst]}


def l2_vf():
    tab = build_tables()
    out = {}
    for kind in ("f_sky", "f_grd"):
        mine = np.array(tab[kind])
        ref = v3.view_factor(TILT_GRID, 97.5, 97.5, kind)
        d = np.abs(mine - ref)
        out[kind] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                     "argmax_tilt": int(TILT_GRID[d.argmax()]),
                     "at_0": [round(float(mine[0]), 4), round(float(ref[0]), 4)],
                     "at_83": [round(float(mine[83]), 4), round(float(ref[83]), 4)],
                     "at_90": [round(float(mine[90]), 4), round(float(ref[90]), 4)]}
    return out


def l3_l4(df):
    tab = build_tables()
    elev = df["solar_elevation"].to_numpy()
    az = df["solar_azimuth"].to_numpy()
    dni = df["dni"].to_numpy()
    dhi = df["dhi"].to_numpy()
    ghi = df["ghi_w_m2"].to_numpy()
    n = len(df)
    P_mine = np.empty((len(TILT_GRID), n))
    P_v3 = np.empty((len(TILT_GRID), n))
    for i, t in enumerate(TILT_GRID):
        P_mine[i] = power_series(tab, float(t), elev, az, dni, dhi, ghi)
        P_v3[i] = v3.eff_poa(np.full(n, t), elev, az, dni, dhi)
    opt_mine = TILT_GRID[P_mine.argmax(axis=0)]
    opt_v3 = TILT_GRID[P_v3.argmax(axis=0)]
    lit = P_mine.max(axis=0) > 1e-6                  # 발전 가능 시간만
    # 각도 차 (발전 가능 시간)
    dang = np.abs(opt_mine - opt_v3)[lit]
    # ★교차 regret: v3의 결정을 클린룸 물리로 평가했을 때 잃는 에너지
    e_opt_mine = P_mine.max(axis=0).sum()
    e_v3_in_mine = P_mine[np.searchsorted(TILT_GRID, opt_v3), np.arange(n)].sum()
    cross_regret = (e_opt_mine - e_v3_in_mine) / e_opt_mine
    # 반대 방향도 (대칭 확인)
    e_opt_v3 = P_v3.max(axis=0).sum()
    e_mine_in_v3 = P_v3[np.searchsorted(TILT_GRID, opt_mine), np.arange(n)].sum()
    cross_regret_rev = (e_opt_v3 - e_mine_in_v3) / e_opt_v3
    # 사다리 (클린룸 엔진 기준)
    def ladder(P):
        e_ai = P.max(axis=0).sum()
        return {f"vs{int(t)}": round(float((e_ai / P[int(t)].sum() - 1) * 100), 2)
                for t in (60, 81, 83, 90)}
    # CSV v17 라벨도 클린룸 물리로 평가
    lab = df["target_angle_v17"].to_numpy()
    e_lab_in_mine = P_mine[np.searchsorted(TILT_GRID, np.clip(lab, 0, 90)),
                           np.arange(n)].sum()
    label_regret = (e_opt_mine - e_lab_in_mine) / e_opt_mine
    best_fixed = int(TILT_GRID[P_mine.sum(axis=1).argmax()])
    return {
        "hours": int(n), "lit_hours": int(lit.sum()),
        "angle_diff_mean_deg": round(float(dang.mean()), 2),
        "angle_diff_p95_deg": round(float(np.percentile(dang, 95)), 1),
        "cross_regret_v3_decisions_in_cleanroom_%": round(float(cross_regret * 100), 4),
        "cross_regret_cleanroom_decisions_in_v3_%": round(float(cross_regret_rev * 100), 4),
        "v17_label_regret_in_cleanroom_%": round(float(label_regret * 100), 4),
        "ladder_cleanroom": ladder(P_mine),
        "ladder_v3": ladder(P_v3),
        "best_fixed_cleanroom": best_fixed,
        "best_fixed_v3": int(TILT_GRID[P_v3.sum(axis=1).argmax()]),
        "opt_median_cleanroom": float(np.median(opt_mine[lit])),
        "opt_median_v3": float(np.median(opt_v3[lit])),
    }


def main():
    df = pd.read_csv(os.path.join(_ROOT, "bipv_ai_master_data_v17.csv"))
    df = df[df["solar_elevation"] > 0].reset_index(drop=True)
    res = {"L1_sf_grid": l1_sf_grid(), "L2_view_factors": l2_vf(),
           "L3_L4_decisions": l3_l4(df)}
    out = os.path.join(_HERE, "out", "crosscheck_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
