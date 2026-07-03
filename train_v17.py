# ==============================================================================
# v17 파이프라인: 시야계수 테이블(현/피치 의미론 교정) → 라벨 → XGBoost → 평가
# 실행: .venv/bin/python train_v17.py   (★__main__ 가드 있음 — import 안전)
# 산출: view_factors_v17.json / bipv_ai_master_data_v17.csv /
#       bipv_xgboost_model_v17.pkl / model_metrics_v17.json
# 근거: ADVERSARIAL_REVIEW_RESULTS.md (v17 게이트 5항목)
# ==============================================================================
import json
import numpy as np
import pandas as pd

AMIN, AMAX, ANIGHT = 15, 90, 90
ANGLES = np.arange(AMIN, AMAX + 1, 1.0)
TILTS = np.arange(0.0, 90.1, 2.5)
RATIOS = np.arange(0.30, 1.201, 0.05)   # 현/피치 (실기하 1.169=현114/피치97.5 겹침밀폐 포함, ≤1.20)


def view_factors_mc(tilt_deg, c_over_p, n_pts=64, n_dir=4000, n_nb=4, seed=42):
    """★v16 버그 수정판: c_over_p = 현/피치 그대로 (구판은 현=2×ratio 였음 — 검토 §A).
    독립 재구현(verify_adversarial.vf_mc_clean)과 동일 — 게이트 G3 이 교차검증."""
    half, p = c_over_p / 2.0, 1.0
    rng = np.random.default_rng(seed)
    tr = np.radians(tilt_deg)
    n = np.array([np.sin(tr), np.cos(tr), 0.0])
    t1 = np.array([np.cos(tr), -np.sin(tr), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    u = (rng.random(n_pts) * 2 - 1) * half
    px, py = u * np.cos(tr), -u * np.sin(tr)
    ct_ = np.sqrt(rng.random((n_pts, n_dir))); st_ = np.sqrt(1 - ct_**2)
    ph = rng.random((n_pts, n_dir)) * 2 * np.pi
    d = (ct_[..., None] * n + (st_ * np.cos(ph))[..., None] * t1 + (st_ * np.sin(ph))[..., None] * t2)
    dx, dy = d[..., 0], d[..., 1]
    blocked = np.zeros((n_pts, n_dir), dtype=bool)
    for k in list(range(1, n_nb + 1)) + list(range(-n_nb, 0)):
        ax_, ay_ = -half * np.cos(tr), k * p + half * np.sin(tr)
        bx_, by_ = half * np.cos(tr), k * p - half * np.sin(tr)
        ex, ey = bx_ - ax_, by_ - ay_
        qx, qy = ax_ - px[:, None], ay_ - py[:, None]
        den = dx * ey - dy * ex
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        t_ray = (qx * ey - qy * ex) / den
        s_seg = (qx * dy - qy * dx) / den
        blocked |= (t_ray > 1e-9) & (s_seg >= 0.0) & (s_seg <= 1.0)
    fwd = dx > 1e-9
    return (float(((dy > 1e-9) & fwd & ~blocked).mean()),
            float(((dy < -1e-9) & fwd & ~blocked).mean()))


def main():
    # ── 1. 시야계수 테이블 ────────────────────────────────────────────────────
    print(f"[1] 시야계수 테이블: {len(RATIOS)} ratios(현/피치) × {len(TILTS)} tilts MC ...")
    FS = np.zeros((len(RATIOS), len(TILTS))); FG = np.zeros_like(FS)
    for i, r in enumerate(RATIOS):
        for j, t in enumerate(TILTS):
            FS[i, j], FG[i, j] = view_factors_mc(t, r)
    with open("view_factors_v17.json", "w") as f:
        json.dump({"tilts": TILTS.tolist(), "ratios": [round(float(r), 2) for r in RATIOS],
                   "f_sky": np.round(FS, 4).tolist(), "f_grd": np.round(FG, 4).tolist(),
                   "gen": "train_v17.py MC n_pts=64 n_dir=4000 n_nb=4 seed=42 wall=True "
                          "ratio=chord/pitch (v16 반의미 교정)"}, f)
    i10 = int(np.argmin(np.abs(RATIOS - 1.0)))
    print(f"    ratio 1.00 spot: F_sky@60°={FS[i10, 24]:.3f} F_grd@60°={FG[i10, 24]:.3f}")

    # ── 2. 라벨 재생성 (physics_v3 = 앱과 동일 코드 경로) ────────────────────
    import physics_v3 as P3
    df = pd.read_csv("bipv_ai_master_data_v15_realistic.csv")  # TMY 현실화(cloud 감쇠+Erbs, 2026-07-02)
    el = df["solar_elevation"].to_numpy(float); az = df["solar_azimuth"].to_numpy(float)
    dni = df["dni"].to_numpy(float); dhi = df["dhi"].to_numpy(float)
    ghi = df["ghi_w_m2"].to_numpy(float)
    day = ghi >= 10
    print(f"[2] 라벨 재생성: {len(df)}행 (주간 {day.sum()})")
    E = P3.eff_poa(ANGLES[:, None], el[None, :], az[None, :], dni[None, :], dhi[None, :])
    idx = np.argmax(E, axis=0)
    tgt = ANGLES[idx].copy()
    tgt[~day] = float(ANIGHT)
    df["target_angle_v17"] = tgt
    df.to_csv("bipv_ai_master_data_v17.csv", index=False)
    print(f"    주간 중앙값 v17={np.median(tgt[day]):.0f}° (p10–90 "
          f"{np.percentile(tgt[day], 10):.0f}–{np.percentile(tgt[day], 90):.0f}°)")

    # ── 3. 재학습 (v15/16 과 동일 하이퍼파라미터) ────────────────────────────
    from xgboost import XGBRegressor
    FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "ghi_w_m2", "cloud_cover", "temp_actual"]
    X = df[FEATS].to_numpy(float); y = tgt
    n = len(df); n_tr = int(n * 0.8)
    hp = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.7,
              colsample_bytree=0.5, min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
              random_state=42, n_jobs=-1)
    print("[3] 재학습 (홀드아웃=시간순 마지막 20%) ...")
    model = XGBRegressor(**hp); model.fit(X[:n_tr], y[:n_tr])

    # ── 4. 평가: R²/MAE + 에너지 regret ──────────────────────────────────────
    def r2(a, b): return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
    pred_ho = np.clip(model.predict(X[n_tr:]), AMIN, AMAX)
    y_ho = y[n_tr:]; day_ho = day[n_tr:]
    m_ho = {"r2_all": round(r2(y_ho, pred_ho), 4), "mae_all": round(float(np.abs(y_ho - pred_ho).mean()), 2),
            "r2_day": round(r2(y_ho[day_ho], pred_ho[day_ho]), 4),
            "mae_day": round(float(np.abs(y_ho[day_ho] - pred_ho[day_ho]).mean()), 2)}
    sl = slice(n_tr, None)
    e_pred = P3.eff_poa(pred_ho[day_ho], el[sl][day_ho], az[sl][day_ho], dni[sl][day_ho], dhi[sl][day_ho]).sum()
    e_orac = P3.eff_poa(y_ho[day_ho], el[sl][day_ho], az[sl][day_ho], dni[sl][day_ho], dhi[sl][day_ho]).sum()
    regret = round(float(1 - e_pred / e_orac) * 100, 3)
    model_full = XGBRegressor(**hp); model_full.fit(X, y)
    import joblib
    joblib.dump(model_full, "bipv_xgboost_model_v17.pkl")
    yrs = sorted(pd.to_datetime(df["timestamp"]).dt.year.unique())
    metrics = {**m_ho, "energy_regret_pct": regret,
               "training_rows": int(n), "year_range": f"{yrs[0]}–{yrs[-1]}",
               "label": "target_angle_v17 (physics_v3: 실기하 현114/피치97.5 gcr1.169 + PV띠 상24/하7 + IAM + alb 0.15)",
               "holdout": "chronological last 20%",
               "importance": {f: round(float(v), 3) for f, v in zip(FEATS, model_full.feature_importances_)}}
    with open("model_metrics_v17.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("[4] 홀드아웃:", m_ho, f"| 에너지 regret {regret}%")
    print("    importance:", metrics["importance"])
    print("saved → bipv_xgboost_model_v17.pkl / model_metrics_v17.json / "
          "bipv_ai_master_data_v17.csv / view_factors_v17.json")


if __name__ == "__main__":
    main()
