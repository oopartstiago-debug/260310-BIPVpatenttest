# ==============================================================================
# 사용자 재검증 (2026-06-23): "AI tilt vs 진짜 고정식 이득이 이렇게 낮은 게 맞나"
#   독립 재계산 — verify_positioning.py 의 e_ai(오라클 라벨)를 재사용하지 않고
#   ① 고정각 에너지 지형 전체 스캔(0~90°,1°) → 최고고정각·고원폭 확인
#   ② 오라클(per-row argmax) 이론 천장 vs 최고고정
#   ③ ★실제 XGBoost v17 모델 예측 각도의 에너지 → 모델 vs 최고고정/60/90
#   ④ 핵심 레버(현/피치) 민감도: ratio 바뀌면 최고고정각·천장이 어떻게 변하나
# 실행: .venv/bin/python verify_user_recheck.py
# ==============================================================================
import json
import numpy as np
import pandas as pd
import physics_v3 as P

EL, AZ, DNI, DHI = "solar_elevation", "solar_azimuth", "dni", "dhi"
df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df[EL] > 0].copy()
el = dd[EL].values; az = dd[AZ].values; dni = dd[DNI].values; dhi = dd[DHI].values
N = len(dd)
print(f"주간 행수 = {N}  (연수 {pd.to_datetime(dd.timestamp).dt.year.nunique()})\n")


def E(tilt_arr, c=P.CHORD, p=P.PITCH):
    return float(P.eff_poa(np.asarray(tilt_arr, float), el, az, dni, dhi, c=c, p=p).sum())


# ── ① 고정각 에너지 지형 (0~90°, 1°) ─────────────────────────────────────────
TILTS = np.arange(0, 90.1, 1.0)
fixedE = np.array([E(np.full(N, t)) for t in TILTS])
best_i = int(fixedE.argmax()); best_t = TILTS[best_i]; best_E = fixedE[best_i]
within1 = TILTS[(best_E - fixedE) / best_E <= 0.01]      # 최고 대비 1% 이내 = 고원
within2 = TILTS[(best_E - fixedE) / best_E <= 0.02]
print("① 고정각 지형")
print(f"   최고 고정각 = {best_t:.0f}°  (연 에너지 {best_E/1000:.1f} kWh/m²급 상대단위)")
print(f"   1% 이내 고원 = {within1.min():.0f}~{within1.max():.0f}°  (폭 {within1.max()-within1.min():.0f}°)")
print(f"   2% 이내 고원 = {within2.min():.0f}~{within2.max():.0f}°")
for t in (15, 30, 45, 60, 75, 81, 83, 90):
    e = E(np.full(N, float(t)))
    print(f"   고정 {t:>2.0f}°: 최고 대비 {(e/best_E-1)*100:+6.2f}%")

# ── ② 오라클(이론 천장) ─────────────────────────────────────────────────────
# per-row 최적각: 0~90° 전구간 argmax (라벨은 15~90° 제약이었음 — 둘 다 산출)
Eall = P.eff_poa(TILTS[:, None], el[None, :], az[None, :], dni[None, :], dhi[None, :])
orac_full = TILTS[np.argmax(Eall, axis=0)]
e_orac_full = float(P.eff_poa(orac_full, el, az, dni, dhi).sum())
T15 = np.arange(15, 90.1, 1.0)
E15 = P.eff_poa(T15[:, None], el[None, :], az[None, :], dni[None, :], dhi[None, :])
orac_15 = T15[np.argmax(E15, axis=0)]
e_orac_15 = float(P.eff_poa(orac_15, el, az, dni, dhi).sum())
print("\n② 오라클(완벽한 시간별 트래킹) = 이론 천장")
print(f"   오라클(0~90°)  vs 최고고정 = {(e_orac_full/best_E-1)*100:+.2f}%")
print(f"   오라클(15~90°) vs 최고고정 = {(e_orac_15/best_E-1)*100:+.2f}%  (←라벨 정의, POSITIONING +0.53% 와 대조)")
print(f"   오라클 각도 분포: 중앙 {np.median(orac_full):.0f}° / p10–90 {np.percentile(orac_full,10):.0f}–{np.percentile(orac_full,90):.0f}°")

# ── ③ 실제 XGBoost v17 모델 예측 각도의 에너지 ──────────────────────────────
print("\n③ 실제 XGBoost v17 모델 (대시보드가 쓰는 그 모델)")
try:
    import joblib
    model = joblib.load("bipv_xgboost_model_v17.pkl")
    FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "ghi_w_m2", "cloud_cover", "temp_actual"]
    pred = np.clip(model.predict(dd[FEATS].to_numpy(float)), 15, 90)
    e_model = float(P.eff_poa(pred.astype(float), el, az, dni, dhi).sum())
    print(f"   모델 예측각: 중앙 {np.median(pred):.0f}° / p10–90 {np.percentile(pred,10):.0f}–{np.percentile(pred,90):.0f}°")
    print(f"   모델 vs 최고고정({best_t:.0f}°) = {(e_model/best_E-1)*100:+.2f}%   ← ★진짜 'AI vs 최적 고정식' 이득")
    print(f"   모델 vs 고정 60°          = {(e_model/E(np.full(N,60.0))-1)*100:+.2f}%   (대시보드 헤드라인 +4.4%)")
    print(f"   모델 vs 고정 90°          = {(e_model/E(np.full(N,90.0))-1)*100:+.2f}%")
    print(f"   모델 vs 오라클(천장)      = {(e_model/e_orac_15-1)*100:+.2f}%  (regret = {(1-e_model/e_orac_15)*100:.3f}%)")
except Exception as ex:
    print("   (모델 로드 실패:", ex, ")")

# ── ④ 핵심 레버: 현/피치(ratio) 민감도 ──────────────────────────────────────
print("\n④ 현/피치(밀집도) 민감도 — 평탄 고원이 기하 가정의 산물인지")
print("   ratio   최고고정각   오라클(15~90)vs최고고정")
for ratio in (0.5, 0.7, 0.85, 1.0):
    p = P.CHORD / ratio    # chord 고정, pitch 키워 ratio 낮춤
    fe = np.array([E(np.full(N, t), c=P.CHORD, p=p) for t in TILTS])
    bi = int(fe.argmax()); bt = TILTS[bi]
    Er = P.eff_poa(T15[:, None], el[None, :], az[None, :], dni[None, :], dhi[None, :])  # POA는 ratio가 c,p로 들어가야함
    # ratio 반영해 다시
    orow = T15[np.argmax(
        np.stack([P.eff_poa(np.full(N, t), el, az, dni, dhi, c=P.CHORD, p=p) for t in T15]), axis=0)]
    eo = float(P.eff_poa(orow, el, az, dni, dhi, c=P.CHORD, p=p).sum())
    print(f"   {ratio:.2f}    {bt:>6.0f}°      {(eo/fe[bi]-1)*100:+.2f}%")
print("\n해석: ratio가 1.0(실기하)일수록 밀집→고원 평탄→천장 작음. 이게 실측 도면값이면 낮은 이득은 물리 사실.")
