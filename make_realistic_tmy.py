# ==============================================================================
# TMY 현실화 (2026-07-02, Fable 지적: clear-sky GHI 2061·diffuse 0.22 비현실)
#   방법: 태양위치·시간축 유지, 일사만 보정
#     ① cloud_cover(실측 octas 0~10)로 clear-sky GHI 감쇠 (Kasten-Czeplak)
#     ② pvlib Erbs 로 dni/dhi 재분해 → diffuse fraction 이 kt(청명지수) 따라 현실화
#   목표: 연 GHI ~1400 kWh/m² · 주간 diffuse fraction ~0.5 (서울 KMA 실측)
# 산출: bipv_ai_master_data_v15_realistic.csv (dni/dhi/ghi_w_m2 교체, 나머지 유지)
# 실행: .venv/bin/python make_realistic_tmy.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib

src = "bipv_ai_master_data_v15.csv"
df = pd.read_csv(src)
el = df["solar_elevation"].to_numpy(float)
zen = np.clip(90 - el, 0, 89.9)
C = np.clip(df["cloud_cover"].to_numpy(float) / 10.0, 0, 1)   # octas → 0~1
ghi_cs = df["ghi_w_m2"].to_numpy(float)
doy = pd.to_datetime(df["timestamp"]).dt.dayofyear.to_numpy()

# ① 구름 감쇠 (Kasten-Czeplak, octas 표준식)
ghi = ghi_cs * (1 - 0.75 * C ** 3.4)
ghi = np.clip(ghi, 0, None)

# ② Erbs 분해 (kt 기반 diffuse fraction)
out = pvlib.irradiance.erbs(ghi, zen, doy)
dni = np.nan_to_num(out["dni"].to_numpy(float) if hasattr(out["dni"], "to_numpy") else np.asarray(out["dni"]))
dhi = np.nan_to_num(out["dhi"].to_numpy(float) if hasattr(out["dhi"], "to_numpy") else np.asarray(out["dhi"]))

# 야간 0
night = el <= 0
ghi[night] = 0; dni[night] = 0; dhi[night] = 0

df["ghi_w_m2"] = ghi; df["dni"] = dni; df["dhi"] = dhi
out_fn = "bipv_ai_master_data_v15_realistic.csv"
df.to_csv(out_fn, index=False)

# 검증
yrs = pd.to_datetime(df["timestamp"]).dt.year.nunique()
d = df[df["solar_elevation"] > 0]
ghi_yr = d["ghi_w_m2"].sum() / yrs / 1000
dfrac = d["dhi"].sum() / ((d["dni"] * np.sin(np.radians(d["solar_elevation"])) + d["dhi"]).sum())
print(f"보정 전:  연GHI 2062 kWh/m² · diffuse 0.223")
print(f"보정 후:  연GHI {ghi_yr:.0f} kWh/m² · diffuse {dfrac:.3f}")
print(f"서울 실측: 연GHI ~1300-1450  · diffuse ~0.5")
print(f"→ {out_fn} 저장 ({len(df)}행)")
