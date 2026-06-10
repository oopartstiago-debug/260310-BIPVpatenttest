# ==============================================================================
# v16 파이프라인: 시야계수 테이블 생성 → 라벨 재생성 → XGBoost 재학습 → 평가
# 실행: .venv/bin/python train_v16.py
# 산출: view_factors_v16.json / bipv_ai_master_data_v16.csv /
#       bipv_xgboost_model_v16.pkl / model_metrics_v16.json
# 근거: FABLE_REVIEW.md §C·§D
# ==============================================================================
import json
import numpy as np
import pandas as pd

AMIN, AMAX, ANIGHT = 15, 90, 90
ANGLES = np.arange(AMIN, AMAX + 1, 1.0)

# ── 1. 시야계수 테이블 (MC, verify_fable2.py 와 동일 기하·벽 포함) ────────────
def view_factors_mc(tilt_deg, ratio, n_pts=64, n_dir=4000, n_nb=4, seed=42):
    hd, p = ratio, 1.0  # 비율만 의미 있음
    rng = np.random.default_rng(seed)
    tr = np.radians(tilt_deg)
    n = np.array([np.sin(tr), np.cos(tr), 0.0])
    t1 = np.array([np.cos(tr), -np.sin(tr), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    u = (rng.random(n_pts) * 2 - 1) * hd
    px, py = u * np.cos(tr), -u * np.sin(tr)
    ct_ = np.sqrt(rng.random((n_pts, n_dir))); st_ = np.sqrt(1 - ct_**2)
    ph = rng.random((n_pts, n_dir)) * 2 * np.pi
    d = (ct_[..., None] * n + (st_ * np.cos(ph))[..., None] * t1 + (st_ * np.sin(ph))[..., None] * t2)
    dx, dy = d[..., 0], d[..., 1]
    blocked = np.zeros((n_pts, n_dir), dtype=bool)
    for k in list(range(1, n_nb + 1)) + list(range(-n_nb, 0)):
        ax_, ay_ = -hd * np.cos(tr), k * p + hd * np.sin(tr)
        bx_, by_ = hd * np.cos(tr), k * p - hd * np.sin(tr)
        ex, ey = bx_ - ax_, by_ - ay_
        qx, qy = ax_ - px[:, None], ay_ - py[:, None]
        den = dx * ey - dy * ex
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        t_ray = (qx * ey - qy * ex) / den
        s_seg = (qx * dy - qy * dx) / den
        blocked |= (t_ray > 1e-9) & (s_seg >= 0.0) & (s_seg <= 1.0)
    fwd = dx > 1e-9  # 후방 = 실내(벽)
    return (float(((dy > 1e-9) & fwd & ~blocked).mean()),
            float(((dy < -1e-9) & fwd & ~blocked).mean()))

TILTS = np.arange(0.0, 90.1, 2.5)
RATIOS = np.arange(0.30, 0.751, 0.05)
print(f"[1] 시야계수 테이블 생성: {len(RATIOS)} ratios × {len(TILTS)} tilts MC ...")
FS = np.zeros((len(RATIOS), len(TILTS))); FG = np.zeros_like(FS)
for i, r in enumerate(RATIOS):
    for j, t in enumerate(TILTS):
        FS[i, j], FG[i, j] = view_factors_mc(t, r)
with open("view_factors_v16.json", "w") as f:
    json.dump({"tilts": TILTS.tolist(), "ratios": [round(float(r), 2) for r in RATIOS],
               "f_sky": np.round(FS, 4).tolist(), "f_grd": np.round(FG, 4).tolist(),
               "gen": "train_v16.py MC n_pts=64 n_dir=4000 n_nb=4 seed=42 wall=True"}, f)
print(f"    ratio 0.50 spot: F_sky@60°={FS[4, 24]:.3f} F_grd@60°={FG[4, 24]:.3f}")

# ── 2. 라벨 재생성 (physics_v2.eff_poa 그대로 사용 = 앱과 동일 코드 경로) ─────
import physics_v2 as P2
df = pd.read_csv("bipv_ai_master_data_v15.csv")
el = df["solar_elevation"].to_numpy(float); az = df["solar_azimuth"].to_numpy(float)
dni = df["dni"].to_numpy(float); dhi = df["dhi"].to_numpy(float)
ghi = df["ghi_w_m2"].to_numpy(float)
day = ghi >= 10
print(f"[2] 라벨 재생성: {len(df)}행 (주간 {day.sum()})")
E = P2.eff_poa(ANGLES[:, None], el[None, :], az[None, :], dni[None, :], dhi[None, :])
idx = np.argmax(E, axis=0)
tgt16 = ANGLES[idx].copy()
tgt16[~day] = float(ANIGHT)
df["target_angle_v16"] = tgt16
df.to_csv("bipv_ai_master_data_v16.csv", index=False)
d15 = df["target_angle_v15"].to_numpy(float)
print(f"    주간 중앙값 v15={np.median(d15[day]):.0f}° → v16={np.median(tgt16[day]):.0f}° | 평균이동 {+(tgt16[day]-d15[day]).mean():.1f}°")

# ── 3. 재학습 (v15 와 동일 하이퍼파라미터) ───────────────────────────────────
from xgboost import XGBRegressor
FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "ghi_w_m2", "cloud_cover", "temp_actual"]
X = df[FEATS].to_numpy(float)
y = tgt16
n = len(df); n_tr = int(n * 0.8)  # 시간순 홀드아웃 20%
model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.7,
                     colsample_bytree=0.5, min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
                     random_state=42, n_jobs=-1)
print("[3] 재학습 (홀드아웃=시간순 마지막 20%) ...")
model.fit(X[:n_tr], y[:n_tr])

# ── 4. 평가: R²/MAE + 에너지 regret ──────────────────────────────────────────
def r2(a, b): return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
pred_ho = np.clip(model.predict(X[n_tr:]), AMIN, AMAX)
y_ho = y[n_tr:]; day_ho = day[n_tr:]
m_ho = {"r2_all": round(r2(y_ho, pred_ho), 4), "mae_all": round(float(np.abs(y_ho - pred_ho).mean()), 2),
        "r2_day": round(r2(y_ho[day_ho], pred_ho[day_ho]), 4),
        "mae_day": round(float(np.abs(y_ho[day_ho] - pred_ho[day_ho]).mean()), 2)}
sl = slice(n_tr, None)
e_pred = P2.eff_poa(pred_ho[day_ho], el[sl][day_ho], az[sl][day_ho], dni[sl][day_ho], dhi[sl][day_ho]).sum()
e_orac = P2.eff_poa(y_ho[day_ho], el[sl][day_ho], az[sl][day_ho], dni[sl][day_ho], dhi[sl][day_ho]).sum()
regret = round(float(1 - e_pred / e_orac) * 100, 3)
# 전체 데이터 재학습본 저장 (배포용) — 평가는 위 홀드아웃 수치로 고정
model_full = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.7,
                          colsample_bytree=0.5, min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=-1)
model_full.fit(X, y)
import joblib
joblib.dump(model_full, "bipv_xgboost_model_v16.pkl")
yrs = sorted(pd.to_datetime(df["timestamp"]).dt.year.unique())
metrics = {**m_ho, "energy_regret_pct": regret,
           "training_rows": int(n), "year_range": f"{yrs[0]}–{yrs[-1]}",
           "label": "target_angle_v16 (physics_v2: 선형 retain + MC 시야계수 + albedo 0.2)",
           "holdout": "chronological last 20%",
           "importance": {f: round(float(v), 3) for f, v in zip(FEATS, model_full.feature_importances_)}}
with open("model_metrics_v16.json", "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=1)
print("[4] 홀드아웃:", m_ho, f"| 에너지 regret {regret}%")
print("    importance:", metrics["importance"])
print("saved → bipv_xgboost_model_v16.pkl / model_metrics_v16.json / bipv_ai_master_data_v16.csv / view_factors_v16.json")
