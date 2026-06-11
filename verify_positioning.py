# ==============================================================================
# AI Tilt 포지셔닝 수치 재현 스크립트 (POSITIONING.md 의 모든 수치 단일 소스)
# 실행: .venv/bin/python verify_positioning.py  →  verify_positioning_results.json
# 물리: physics_v3.eff_poa (ADVERSARIAL_REVIEW_RESULTS.md, v17 실기하) · 데이터: bipv_ai_master_data_v17.csv
# ==============================================================================
import json
import numpy as np
import pandas as pd
import physics_v3 as P

EL, AZ, DNI, DHI = "solar_elevation", "solar_azimuth", "dni", "dhi"

df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df[EL] > 0].copy()
dd["month"] = pd.to_datetime(dd.timestamp).dt.month
dd["hour"] = pd.to_datetime(dd.timestamp).dt.hour
TILTS = np.arange(0, 91, 1.0)
R = {"rows_daylight": int(len(dd))}


def energy(tilt_arr, sub=dd):
    return float(P.eff_poa(np.asarray(tilt_arr, dtype=float), sub[EL].values,
                           sub[AZ].values, sub[DNI].values, sub[DHI].values).sum())


e_ai = energy(dd.target_angle_v17.values)

# [1] AI vs 고정각 사다리 (+ 브루트포스 최고 고정각)
fixed = {int(t): energy(np.full(len(dd), t)) for t in TILTS}
best_fix = max(fixed, key=fixed.get)
R["ai_vs_fixed_pct"] = {t: round((e_ai / fixed[t] - 1) * 100, 2)
                        for t in (15, 30, 37, 45, best_fix, 60, 75, 83, 90)}
R["best_fixed_deg"] = best_fix

# [2] 월별 최적 고정 스케줄 (오라클 산출) + AI 우위
sched, e_monthly = {}, 0.0
for m, g in dd.groupby("month"):
    es = [energy(np.full(len(g), t), g) for t in TILTS]
    sched[int(m)] = int(TILTS[int(np.argmax(es))])
    e_monthly += max(es)
R["monthly_schedule"] = sched
R["ai_vs_monthly_pct"] = round((e_ai / e_monthly - 1) * 100, 2)

# [3] 연 2회 조정 (여름 5~9월 / 겨울 나머지, 각 기간 최적 고정)
e2 = 0.0
for g in (dd[dd.month.isin([5, 6, 7, 8, 9])], dd[~dd.month.isin([5, 6, 7, 8, 9])]):
    e2 += max(energy(np.full(len(g), t), g) for t in TILTS)
R["ai_vs_twice_yearly_pct"] = round((e_ai / e2 - 1) * 100, 2)

# [4] 교과서 룰 (서울 위도 37.5°): 고정=위도각 / 계절조정=여름 lat-15·겨울 lat+15·봄가을 lat
R["ai_vs_textbook_fixed_pct"] = round((e_ai / energy(np.full(len(dd), 37.5)) - 1) * 100, 2)
textbook = np.where(dd.month.isin([5, 6, 7, 8]), 22.5,
                    np.where(dd.month.isin([11, 12, 1, 2]), 52.5, 37.5))
R["ai_vs_textbook_seasonal_pct"] = round((e_ai / energy(textbook) - 1) * 100, 2)

# [5] 흐림/맑음 분리 (vs 최고 고정각)
for key, g in (("cloudy_cc_ge_0.7", dd[dd.cloud_cover >= 0.7]),
               ("clear_cc_le_0.3", dd[dd.cloud_cover <= 0.3])):
    R[f"ai_vs_bestfix_{key}_pct"] = round(
        (energy(g.target_angle_v17.values, g) / energy(np.full(len(g), float(best_fix)), g) - 1) * 100, 2)

# [6] 환기 시나리오 (가정: 여름 6~8월 13~17시 블레이드 강제 ≤30° 개방)
vent = (dd.month.isin([6, 7, 8]) & dd.hour.between(13, 17)).values
ai_vent = dd.target_angle_v17.values.copy(); ai_vent[vent] = 30.0
neglect = dd.target_angle_v17.values.copy(); neglect[dd.month.isin([6, 7, 8]).values] = 30.0
R["vent_ai_return_vs_neglect_pct"] = round((energy(ai_vent) / energy(neglect) - 1) * 100, 2)
R["vent_constraint_cost_pct"] = round((1 - energy(ai_vent) / e_ai) * 100, 2)

print(json.dumps(R, indent=1, ensure_ascii=False))
with open("verify_positioning_results.json", "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
