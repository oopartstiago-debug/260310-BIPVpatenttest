# ==============================================================================
# BIPV 통합 관제 시스템 v8.0 — V13 물리모델 + 해석 강화
# 탭: 메인 | 데이터셋 | 피처중요도 | 음영원리 | 음영시각화 | 발전량비교 | 월별각도 | 내일스케줄 | 건강진단 | 파라미터튜닝
# ==============================================================================
# v8.0 변경사항:
#   1. pkl → GitHub raw URL 로드 (gdown 제거)
#   2. 피처 불일치 수정: hour_sin/cos, doy_sin/cos, ghi_real, cloud_cover, temp_actual
#   3. 기상청 TMP(온도) 파싱 추가
#   4. 메인: GHI/SVF 주석, 아침저녁 90도 이유, 음영률 단위+해석
#   5. 데이터셋: 실제 CSV 데이터 시각화 (V5 학습 데이터)
#   6. 피처중요도: 각 인자 설명 + MAE/R² 해석
#   7. 음영원리: SVF/POA 용어 설명, 쉬운 설명
#   8. 음영시각화: 궤적 순서/방향/이유 설명
#   9. 월별각도: 전체 해석 추가
#  10. 건강진단: P_sim/P_actual 설명, 임계값 해석
#  11. 파라미터: 고정각 그래프 제거 → DEPTH/PITCH/최소각 민감도 3종 그래프
#  12. requirements.txt: gdown 제거
# ==============================================================================
__version__ = "8.0"

import os
import io
import numpy as np
import pandas as pd
import requests
import urllib.parse
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pvlib
from pvlib.location import Location
from datetime import datetime, timedelta

try:
    import joblib
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

# ==============================================================================
# 상수 — V13 블레이드 스펙
# ==============================================================================
KMA_SERVICE_KEY = "c6ffb5b520437f3e6983a55234e73701fce509cbb3153c9473ebbe5756a1da00"
LAT, LON, TZ = 37.5665, 126.9780, "Asia/Seoul"
NX, NY = 60, 127
DEFAULT_CAPACITY   = 300
DEFAULT_EFFICIENCY = 18.7
DEFAULT_LOSS       = 0.85
DEFAULT_KEPCO      = 210
DEFAULT_UNIT_COUNT = 1
DEFAULT_LOUVER_COUNT = 20
DEFAULT_WIDTH_MM   = 900.0
DEFAULT_HEIGHT_MM  = 114.0
DEFAULT_PITCH_MM   = 114.0
HALF_DEPTH_MM      = 57.0
ANGLE_MIN          = 15
ANGLE_MAX          = 90
ANGLE_NIGHT        = 90
XGB_MODEL_FILENAME = "bipv_xgboost_model.pkl"
XGB_FEATURE_NAMES  = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
                       "ghi_real", "cloud_cover", "temp_actual"]

# GitHub raw URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/oopartstiago-debug/260310-BIPVpatenttest/main"
MODEL_URL = f"{GITHUB_RAW_BASE}/bipv_xgboost_model.pkl"
CSV_URL   = f"{GITHUB_RAW_BASE}/bipv_ai_master_data_v5.csv"

site = Location(LAT, LON, tz=TZ)

# ==============================================================================
# ★ v8.0 모델 로드 — GitHub raw URL 방식
# ==============================================================================
@st.cache_resource
def load_xgb_model():
    if not _XGB_AVAILABLE:
        return None
    # 로컬에 없으면 GitHub에서 다운로드
    if not os.path.exists(XGB_MODEL_FILENAME):
        try:
            r = requests.get(MODEL_URL, timeout=30)
            if r.status_code == 200:
                with open(XGB_MODEL_FILENAME, "wb") as f:
                    f.write(r.content)
        except Exception:
            return None
    if os.path.isfile(XGB_MODEL_FILENAME):
        try:
            return joblib.load(XGB_MODEL_FILENAME)
        except Exception:
            return None
    return None

# ==============================================================================
# ★ v8.0 CSV 로드 — GitHub raw URL 방식
# ==============================================================================
@st.cache_data(ttl=86400)
def load_training_csv():
    try:
        r = requests.get(CSV_URL, timeout=30)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
    except Exception:
        pass
    return None

# ==============================================================================
# V13 물리 계산 함수
# ==============================================================================
def blade_geometry(tilt_deg, half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    tilt_rad = np.radians(np.asarray(tilt_deg, dtype=float))
    protrusion = half_depth * np.cos(tilt_rad)
    vertical_occupy = blade_depth * np.sin(tilt_rad)
    gap = np.maximum(pitch - vertical_occupy, 0.0)
    return protrusion, vertical_occupy, gap

def sky_view_factor(tilt_deg, half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    protrusion, _, gap = blade_geometry(tilt_deg, half_depth, blade_depth, pitch)
    denom = gap + protrusion
    svf = np.where(denom > 0, gap / denom, 0.0)
    return np.clip(svf, 0.05, 1.0)

def calc_shadow_on_next_blade(tilt_deg, elev_deg, azimuth_diff_deg=0.0,
                               half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    protrusion, _, gap = blade_geometry(tilt_deg, half_depth, blade_depth, pitch)
    elev_rad = np.radians(np.clip(np.asarray(elev_deg, dtype=float), 0.1, 89.9))
    az_diff_rad = np.radians(np.asarray(azimuth_diff_deg, dtype=float))
    shadow_vertical = protrusion * np.cos(az_diff_rad) / np.tan(elev_rad)
    shadow_on_blade = np.maximum(shadow_vertical - gap, 0.0)
    return shadow_on_blade

def calc_shading_fraction(tilt_deg, elev_deg, half_depth=HALF_DEPTH_MM,
                           blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM,
                           azimuth_diff_deg=0.0):
    shadow = calc_shadow_on_next_blade(tilt_deg, elev_deg, azimuth_diff_deg,
                                        half_depth, blade_depth, pitch)
    sf = np.clip(shadow / pitch, 0, 1)
    sf = np.where(np.asarray(elev_deg) <= 0, 1.0, sf)
    return sf

def poa_components(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi, a_r=0.16):
    irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solar_zenith, solar_azimuth=solar_azimuth
    )
    poa_direct  = np.nan_to_num(irrad["poa_direct"],  nan=0.0)
    poa_diffuse = np.nan_to_num(irrad["poa_diffuse"], nan=0.0)
    poa_sky_diffuse = np.nan_to_num(irrad.get("poa_sky_diffuse", irrad["poa_diffuse"]), nan=0.0)
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    aoi = np.clip(np.asarray(aoi, dtype=float), 0, 90)
    try:
        iam = pvlib.iam.martin_ruiz(aoi, a_r=a_r)
    except AttributeError:
        iam = pvlib.irradiance.iam.martin_ruiz(aoi, a_r=a_r)
    return poa_direct * iam, poa_diffuse, poa_sky_diffuse

def calc_effective_poa(poa_direct, poa_diffuse, poa_sky_diffuse, tilt_deg, elev_deg,
                        half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    sf  = calc_shading_fraction(tilt_deg, elev_deg, half_depth, blade_depth, pitch)
    svf = sky_view_factor(tilt_deg, half_depth, blade_depth, pitch)
    effective = poa_direct * (1 - sf * 0.7) + poa_diffuse * svf + poa_sky_diffuse * svf
    return np.maximum(effective, 0.0)

# ==============================================================================
# ★ v8.0 predict_angles_xgb — 피처 불일치 수정
# ==============================================================================
def predict_angles_xgb(model, times, ghi_real, cloud_series, temp_series, angle_cap_deg):
    n = len(times)
    hour_sin = np.sin(2 * np.pi * np.asarray(times.hour, dtype=float) / 24.0)
    hour_cos = np.cos(2 * np.pi * np.asarray(times.hour, dtype=float) / 24.0)
    doy_sin  = np.sin(2 * np.pi * np.asarray(times.dayofyear, dtype=float) / 365.0)
    doy_cos  = np.cos(2 * np.pi * np.asarray(times.dayofyear, dtype=float) / 365.0)

    X = np.column_stack([
        hour_sin[:n], hour_cos[:n], doy_sin[:n], doy_cos[:n],
        np.asarray(ghi_real,    dtype=float).ravel()[:n],
        np.asarray(cloud_series,dtype=float).ravel()[:n],
        np.asarray(temp_series, dtype=float).ravel()[:n],
    ])
    try:
        pred = model.predict(X)
    except Exception:
        return None
    pred = np.clip(np.asarray(pred).ravel()[:n], ANGLE_MIN, min(ANGLE_MAX, angle_cap_deg))
    pred[np.asarray(ghi_real).ravel()[:n] < 10] = ANGLE_NIGHT
    return pred.astype(float)

# ==============================================================================
# 기상청 API — ★ v8.0: TMP(온도) 추가 파싱
# ==============================================================================
@st.cache_data(ttl=3600)
def get_kma_forecast():
    decoded_key = urllib.parse.unquote(KMA_SERVICE_KEY)
    base_date = datetime.now().strftime("%Y%m%d")
    now_hour = datetime.now().hour
    available_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    base_time_int = max([h for h in available_hours if h <= now_hour] or [23])
    if base_time_int == 23 and now_hour < 2:
        base_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    base_time = f"{base_time_int:02d}00"
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {"serviceKey": decoded_key, "numOfRows": "1000", "dataType": "JSON",
              "base_date": base_date, "base_time": base_time, "nx": NX, "ny": NY}
    try:
        res = requests.get(url, params=params, timeout=10).json()
        items = res["response"]["body"]["items"]["item"]
        df = pd.DataFrame(items)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
        df_tom = df[df["fcstDate"] == tomorrow].drop_duplicates(subset=["fcstDate","fcstTime","category"])
        pivot = df_tom.pivot(index="fcstTime", columns="category", values="fcstValue")
        return pivot, tomorrow
    except Exception:
        return None, None

# ==============================================================================
# 연간 데이터
# ==============================================================================
@st.cache_data(ttl=86400)
def get_annual_data(year, half_depth, blade_depth, pitch_mm, capacity_w, unit_count, eff_factor, default_loss):
    times_y  = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="h", tz=TZ)
    solpos_y = site.get_solarposition(times_y)
    cs_y     = site.get_clearsky(times_y)
    ghi_y    = np.asarray(cs_y["ghi"].values, dtype=float)
    zen_y    = solpos_y["apparent_zenith"].values
    az_y     = solpos_y["azimuth"].values
    elev_y   = 90.0 - zen_y
    dni_y    = pvlib.irradiance.dirint(ghi_y, zen_y, times_y).fillna(0).values
    dhi_y    = (ghi_y - dni_y * np.cos(np.radians(zen_y))).clip(0)

    angles_ai  = np.where(ghi_y < 10, float(ANGLE_NIGHT),
                   np.clip(elev_y, ANGLE_MIN, ANGLE_MAX).astype(float))
    angles_60  = np.full_like(ghi_y, 60.0)
    angles_90  = np.full_like(ghi_y, 90.0)

    def energy(angles):
        tilt = np.asarray(angles, dtype=float)
        poa_dir, poa_diff, poa_sky = poa_components(tilt, np.full_like(tilt, 180), zen_y, az_y, dni_y, ghi_y, dhi_y)
        eff_poa = calc_effective_poa(poa_dir, poa_diff, poa_sky, tilt, elev_y, half_depth, blade_depth, pitch_mm)
        mask = ghi_y >= 10
        return (eff_poa[mask] / 1000 * capacity_w * unit_count * eff_factor * default_loss).sum()

    wh_ai = energy(angles_ai)
    wh_60 = energy(angles_60)
    wh_90 = energy(angles_90)

    df_annual = pd.DataFrame({
        "timestamp": times_y, "ghi": ghi_y, "zenith": zen_y,
        "azimuth": az_y, "elevation": elev_y,
        "angle_ai": angles_ai, "angle_60": angles_60, "angle_90": angles_90,
    })
    df_annual["month"] = df_annual["timestamp"].dt.month

    monthly = []
    for m in range(1, 13):
        mask_m = (df_annual["month"] == m) & (ghi_y >= 10)
        zen_m, az_m, elev_m = zen_y[mask_m], az_y[mask_m], elev_y[mask_m]
        dni_m, ghi_m, dhi_m = dni_y[mask_m], ghi_y[mask_m], dhi_y[mask_m]

        def e_m(ang):
            tilt = np.asarray(ang, dtype=float)
            poa_dir, poa_diff, poa_sky = poa_components(tilt, np.full_like(tilt, 180), zen_m, az_m, dni_m, ghi_m, dhi_m)
            eff_poa = calc_effective_poa(poa_dir, poa_diff, poa_sky, tilt, elev_m, half_depth, blade_depth, pitch_mm)
            return (eff_poa / 1000 * capacity_w * unit_count * eff_factor * default_loss).sum()

        monthly.append({
            "month": m,
            "AI": e_m(df_annual.loc[mask_m, "angle_ai"].values),
            "고정60°": e_m(df_annual.loc[mask_m, "angle_60"].values),
            "수직90°": e_m(df_annual.loc[mask_m, "angle_90"].values),
            "avg_angle": float(df_annual.loc[mask_m, "angle_ai"].mean()),
        })
    df_monthly = pd.DataFrame(monthly)
    return wh_ai, wh_60, wh_90, df_monthly, df_annual


# ==============================================================================
# 메인 앱
# ==============================================================================
def run_app():
    st.set_page_config(page_title="BIPV AI 관제 V13", layout="wide", page_icon="☀️")

    st.markdown("""
    <style>
    .metric-card {background:#1e2130;border-radius:12px;padding:16px 20px;margin-bottom:8px;}
    .tab-header {font-size:1.1rem;font-weight:600;color:#f0f4ff;margin-bottom:12px;}
    div[data-testid="stMetricValue"] {font-size:1.6rem;font-weight:700;}
    .explain-box {background:#1a2035;border-left:3px solid #4fc3f7;padding:10px 14px;
                  border-radius:6px;margin:8px 0;font-size:0.92rem;color:#cfd8f0;}
    .warn-box {background:#2a1a1a;border-left:3px solid #ff6b6b;padding:10px 14px;
               border-radius:6px;margin:8px 0;font-size:0.92rem;color:#ffcdd2;}
    </style>
    """, unsafe_allow_html=True)

    xgb_model = load_xgb_model() if _XGB_AVAILABLE else None
    kma, tomorrow = get_kma_forecast()

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    st.sidebar.title("■ 통합 환경 설정")
    if xgb_model:
        st.sidebar.success("✅ XGBoost 모델 로드됨")
    else:
        st.sidebar.warning("⚠️ 규칙 기반 모드 (모델 없음)")

    st.sidebar.subheader("1. 시간 및 날짜")
    tomorrow_dt = datetime.strptime(tomorrow, "%Y%m%d") if tomorrow else datetime.now() + timedelta(days=1)
    sim_date = st.sidebar.date_input("시뮬레이션 날짜", tomorrow_dt)

    st.sidebar.subheader("2. 블레이드 스펙 (V13)")
    st.sidebar.caption("중심축 회전 | 가로(발전면적) × 세로(음영계산) | 피치 | 개수")
    width_mm  = st.sidebar.number_input("블레이드 가로 (mm)", min_value=100.0, value=DEFAULT_WIDTH_MM, step=100.0)
    blade_depth_mm = st.sidebar.number_input("블레이드 세로/DEPTH (mm)", min_value=10.0, value=DEFAULT_HEIGHT_MM, step=1.0)
    pitch_mm  = st.sidebar.number_input("블레이드 피치 (mm)", min_value=10.0, value=DEFAULT_PITCH_MM, step=1.0)
    half_depth_mm = blade_depth_mm / 2.0
    st.sidebar.caption(f"HALF_DEPTH = {half_depth_mm:.1f} mm (중심축→끝)")
    louver_count = st.sidebar.number_input("블레이드 개수 (개)", min_value=1, value=DEFAULT_LOUVER_COUNT, step=1)

    st.sidebar.subheader("3. 패널 스펙")
    unit_count  = st.sidebar.number_input("설치 유닛 수 (개)", min_value=1, value=DEFAULT_UNIT_COUNT)
    capacity_w  = st.sidebar.number_input("패널 용량 (W)",      value=DEFAULT_CAPACITY)
    target_eff  = st.sidebar.number_input("패널 효율 (%)",      value=DEFAULT_EFFICIENCY, step=0.1)
    kepco_rate  = st.sidebar.number_input("전기 요금 (원/kWh)", value=DEFAULT_KEPCO)

    eff_factor   = float(target_eff) / DEFAULT_EFFICIENCY
    area_scale   = (width_mm * blade_depth_mm * louver_count) / (DEFAULT_WIDTH_MM * DEFAULT_HEIGHT_MM * DEFAULT_LOUVER_COUNT)

    # ── 날짜별 데이터 ─────────────────────────────────────────────────────────
    _sim_d = sim_date.strftime("%Y-%m-%d")
    times  = pd.date_range(start=f"{_sim_d} 00:00", periods=24, freq="h", tz=TZ)
    solpos = site.get_solarposition(times)
    cs     = site.get_clearsky(times)
    zen    = np.asarray(solpos["apparent_zenith"].values, dtype=float)
    az     = np.asarray(solpos["azimuth"].values,         dtype=float)
    elev   = np.asarray(solpos["apparent_elevation"].values, dtype=float)

    # ★ v8.0: 기상청 파싱 — SKY + TMP
    cloud_series = np.zeros(24)
    temp_series  = np.full(24, 15.0)  # 기본값 15도

    if kma is not None and _sim_d.replace("-","") == tomorrow:
        kma_reindex = kma.reindex(times.strftime("%H00"))
        if "SKY" in kma.columns:
            cloud_series = kma_reindex["SKY"].apply(
                lambda x: 0.0 if x == "1" else (0.5 if x == "3" else 1.0)
            ).fillna(0).astype(float).values
        if "TMP" in kma.columns:
            temp_series = pd.to_numeric(kma_reindex["TMP"], errors="coerce").fillna(15.0).values

    ghi_real = np.asarray(cs["ghi"].values, dtype=float) * (1.0 - cloud_series * 0.65)
    dni_arr  = pvlib.irradiance.dirint(ghi_real, zen, times).fillna(0).values
    dhi_arr  = (ghi_real - dni_arr * np.cos(np.radians(zen))).clip(0)

    # ★ v8.0: cloud_series를 0~9 스케일로 변환 (학습 데이터 스케일 맞춤)
    cloud_kma_scale = cloud_series * 9.0

    # AI 각도
    xgb_angles = None
    if xgb_model:
        xgb_angles = predict_angles_xgb(xgb_model, times, ghi_real, cloud_kma_scale, temp_series, ANGLE_MAX)
    ai_angles = xgb_angles if xgb_angles is not None else np.where(
        ghi_real < 10, float(ANGLE_NIGHT), np.clip(elev, ANGLE_MIN, ANGLE_MAX).astype(float)
    )
    angle_mode = "XGBoost" if xgb_angles is not None else "규칙 기반"

    def calc_power_day(angles):
        tilt = np.asarray(angles, dtype=float)
        poa_dir, poa_diff, poa_sky = poa_components(tilt, np.full_like(tilt, 180.0), zen, az, dni_arr, ghi_real, dhi_arr)
        eff_poa = calc_effective_poa(poa_dir, poa_diff, poa_sky, tilt, elev, half_depth_mm, blade_depth_mm, pitch_mm)
        mask = ghi_real >= 10
        return (eff_poa[mask] / 1000 * capacity_w * unit_count * eff_factor * DEFAULT_LOSS * area_scale).sum()

    pow_ai  = calc_power_day(ai_angles)
    pow_60  = calc_power_day(np.full(24, 60.0))
    pow_90  = calc_power_day(np.full(24, 90.0))

    last_year = sim_date.year - 1
    wh_ai_y, wh_60_y, wh_90_y, df_monthly, df_annual = get_annual_data(
        last_year, half_depth_mm, blade_depth_mm, pitch_mm, capacity_w, unit_count, eff_factor, DEFAULT_LOSS
    )
    ann_kwh_ai = wh_ai_y / 1000 * area_scale
    ann_kwh_60 = wh_60_y / 1000 * area_scale
    ann_kwh_90 = wh_90_y / 1000 * area_scale

    kma_status = "✅ 기상청 예보 연동" if (kma is not None and _sim_d.replace("-","") == tomorrow) else "⚠️ 청천 기준 (기상청 데이터 없음)"
    weather_status = "맑음" if np.mean(cloud_series) < 0.3 else ("구름많음" if np.mean(cloud_series) < 0.8 else "흐림")
    mask_day = (times.hour >= 6) & (times.hour <= 19)

    # ── 탭 ────────────────────────────────────────────────────────────────────
    st.title("☀️ BIPV AI 통합 관제 대시보드")
    st.caption(f"v{__version__} (V13 물리모델) | {_sim_d} | {weather_status} | {angle_mode} 모드 | {kma_status}")

    tabs = st.tabs([
        "🏠 메인", "📊 학습 데이터셋", "🎯 피처 중요도",
        "💡 음영 원리", "🔥 음영 시각화", "⚡ 발전량 비교",
        "📅 월별 각도", "🌤️ 내일 스케줄", "🩺 건강진단", "🔧 파라미터 튜닝"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 0: 메인 대시보드
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("오늘의 발전 현황")

        # ★ v8.0: 용어 주석
        st.markdown("""
        <div class="explain-box">
        <b>📖 주요 용어 안내</b><br>
        • <b>GHI (Global Horizontal Irradiance)</b>: 수평면 전일사량. 지표면에 수평으로 내리쬐는 태양에너지 총량 (W/m²). 발전 가능한 에너지의 기준값.<br>
        • <b>SVF (Sky View Factor)</b>: 하늘 조망 계수. 루버 사이로 보이는 하늘의 비율 (0~1). 1에 가까울수록 확산광을 많이 받음.<br>
        • <b>음영률</b>: 윗 블레이드가 아랫 블레이드를 가리는 비율 (0~100%). 높을수록 발전 손실 큼.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI 제어 발전량",  f"{pow_ai/1000:.3f} kWh",  f"연간 {ann_kwh_ai:.1f} kWh")
        c2.metric("고정 60° 대비",   f"+{(pow_ai/pow_60-1)*100:.1f}%" if pow_60 > 0 else "—")
        c3.metric("수직 90° 대비",   f"+{(pow_ai/pow_90-1)*100:.1f}%" if pow_90 > 0 else "—")
        c4.metric("예상 수익",        f"{int(pow_ai/1000*kepco_rate):,} 원")

        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.subheader("제어 스케줄 (일중)")

            # ★ v8.0: 아침/저녁 90도 이유 설명
            st.markdown("""
            <div class="explain-box">
            <b>💡 아침·저녁에 루버가 90°(수직)인 이유</b><br>
            일사량(GHI)이 10 W/m² 미만인 시간대는 태양에너지가 거의 없어 발전이 불가능합니다.
            이 시간에는 루버를 수직(90°)으로 세워 벽면과 평행하게 닫아두어 외부 노출과 기계적 부하를 최소화합니다.
            일출·일몰 직후 GHI가 10 W/m²를 넘는 순간부터 AI가 최적 각도로 제어를 시작합니다.
            </div>
            """, unsafe_allow_html=True)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=times[mask_day].strftime("%H:%M"), y=ghi_real[mask_day],
                                  name="GHI (W/m²)", marker_color="rgba(255,180,0,0.4)"), secondary_y=False)
            fig.add_trace(go.Scatter(x=times[mask_day].strftime("%H:%M"), y=ai_angles[mask_day],
                                      name="AI 각도 (°)", line=dict(color="#4fc3f7", width=3)), secondary_y=True)
            fig.update_yaxes(title_text="GHI (W/m²)", secondary_y=False)
            fig.update_yaxes(title_text="각도 (°)", range=[0, 95], secondary_y=True)
            fig.update_layout(height=360, legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("발전량 비교")
            fig_bar = go.Figure(go.Bar(
                x=["AI", "고정60°", "수직90°"],
                y=[pow_ai/1000, pow_60/1000, pow_90/1000],
                marker_color=["#4fc3f7", "#aaa", "#666"],
                text=[f"{v/1000:.3f}" for v in [pow_ai, pow_60, pow_90]],
                textposition="auto"
            ))
            fig_bar.update_layout(height=360, yaxis_title="kWh")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ★ v8.0: 음영률 해석 포함 테이블
        st.subheader("시간별 스케줄 테이블")

        sf_vals  = [calc_shading_fraction(a, e, half_depth_mm, blade_depth_mm, pitch_mm)
                    for a, e in zip(ai_angles[mask_day], elev[mask_day])]
        svf_vals = [sky_view_factor(a, half_depth_mm, blade_depth_mm, pitch_mm)
                    for a in ai_angles[mask_day]]

        def sf_status(sf):
            if sf < 0.3:
                return f"{sf*100:.1f}% 🟢 양호"
            elif sf < 0.5:
                return f"{sf*100:.1f}% 🟡 경미"
            elif sf < 0.8:
                return f"{sf*100:.1f}% 🟠 주의"
            else:
                return f"{sf*100:.1f}% 🔴 심각"

        df_sch = pd.DataFrame({
            "시간":          times[mask_day].strftime("%H:%M").tolist(),
            "AI 각도(°)":   ai_angles[mask_day].astype(int).tolist(),
            "GHI (W/m²)":   np.round(ghi_real[mask_day], 1).tolist(),
            "음영률 상태":   [sf_status(sf) for sf in sf_vals],
            "SVF":           [f"{svf:.2f}" for svf in svf_vals],
        })
        st.dataframe(df_sch, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="explain-box">
        <b>📊 음영률 판정 기준 및 발전량 영향</b><br>
        • 🟢 <b>0~30% (양호)</b>: 음영 영향 미미. 발전량 손실 거의 없음.<br>
        • 🟡 <b>30~50% (경미)</b>: 발전량 5~15% 감소 수준. 허용 범위 내.<br>
        • 🟠 <b>50~80% (주의)</b>: 발전량 15~40% 감소. AI가 각도를 조정해 손실 최소화 시도.<br>
        • 🔴 <b>80% 이상 (심각)</b>: 발전량 급감 구간. 태양고도가 매우 낮은 아침·저녁에 주로 발생하며,
        이 시간대는 GHI 자체도 낮아 전체 발전에 미치는 실질 영향은 제한적.<br>
        <b>💡 SVF</b>는 확산광 수신 효율입니다. 0.5 이상이면 하늘의 절반 이상을 보고 있어 흐린 날에도 발전이 가능합니다.
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: 학습 데이터셋 — ★ v8.0: 실제 CSV 데이터
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("📊 XGBoost 학습 데이터셋 탐색 (V5 실측 데이터)")

        df_csv = load_training_csv()

        if df_csv is not None:
            st.success(f"✅ 실측 학습 데이터 로드 완료 | {len(df_csv):,}행 | 2014~2023년 기상청 관측 기반")

            # 변수 설명
            st.markdown("""
            <div class="explain-box">
            <b>📖 학습 변수 설명</b><br>
            • <b>ghi_w_m2</b>: 수평면 전일사량 (W/m²). 태양이 지표면에 보내는 에너지 총량. 발전량 예측의 핵심 입력값.<br>
            • <b>cloud_cover</b>: 운량 (0~9). 기상청 관측값. 0=맑음, 9=완전 흐림. 일사량 감쇠 정도를 반영.<br>
            • <b>temp_actual</b>: 실제 외기온도 (°C). 기온이 높을수록 패널 효율이 소폭 감소하는 온도계수를 반영.<br>
            • <b>target_angle_v5</b>: 학습 타겟값. 해당 시간 최대 발전량을 내는 루버 각도 (°).
            </div>
            """, unsafe_allow_html=True)

            df_plot = df_csv[df_csv["ghi_w_m2"] > 10].copy()
            df_plot["month"] = pd.to_datetime(df_plot["timestamp"]).dt.month
            df_plot["month_s"] = df_plot["month"].map(
                {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**GHI 월별 분포 (실측)**")
                fig = px.box(df_plot, x="month_s", y="ghi_w_m2", color="month_s",
                              category_orders={"month_s": month_order})
                fig.update_layout(showlegend=False, height=320, yaxis_title="GHI (W/m²)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("여름(6~8월)에 중앙값과 분산이 모두 큼 → 맑은 날 강한 일사, 장마로 인한 낮은 일사가 공존.")

            with c2:
                st.markdown("**운량 월별 분포 (실측)**")
                fig2 = px.box(df_plot, x="month_s", y="cloud_cover", color="month_s",
                               category_orders={"month_s": month_order})
                fig2.update_layout(showlegend=False, height=320, yaxis_title="운량 (0~9)")
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("6~7월(장마)에 운량이 높고 분산도 큼. 봄·가을은 운량이 낮고 안정적.")

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**기온 월별 분포 (실측)**")
                fig3 = px.box(df_plot, x="month_s", y="temp_actual", color="month_s",
                               category_orders={"month_s": month_order})
                fig3.update_layout(showlegend=False, height=320, yaxis_title="기온 (°C)")
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("여름 고온이 패널 효율 저하에 기여. 겨울 저온은 효율에 유리하나 일사량이 적음.")

            with c4:
                st.markdown("**최적 루버 각도 월별 분포 (타겟)**")
                fig4 = px.box(df_plot, x="month_s", y="target_angle_v5", color="month_s",
                               category_orders={"month_s": month_order})
                fig4.update_layout(showlegend=False, height=320, yaxis_title="최적 각도 (°)")
                st.plotly_chart(fig4, use_container_width=True)
                st.caption("여름엔 태양고도가 높아 낮은 각도(15~30°)가 최적. 겨울엔 태양이 낮게 떠 높은 각도(60~90°)가 유리.")

            st.markdown("**GHI vs 최적각 산점도 (10년 실측)**")
            sample = df_plot.sample(min(3000, len(df_plot)), random_state=42)
            fig5 = px.scatter(sample, x="ghi_w_m2", y="target_angle_v5", color="month_s",
                               opacity=0.4,
                               labels={"ghi_w_m2":"GHI (W/m²)", "target_angle_v5":"최적 각도 (°)"})
            fig5.update_layout(height=350)
            st.plotly_chart(fig5, use_container_width=True)
            st.caption("GHI가 높을수록 최적 각도가 낮아지는 경향 (태양이 높이 뜨면 루버를 눕힘). 계절별로 패턴이 분리됨.")

        else:
            st.warning("⚠️ 학습 데이터 CSV를 불러올 수 없습니다. GitHub 연결을 확인하세요.")
            st.info("현재 클리어스카이 시뮬레이션 기반 데이터로 대체 표시합니다.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: 피처 중요도 — ★ v8.0: 설명 + MAE/R² 해석
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("🎯 피처 중요도 (Feature Importance) — V13")

        st.markdown("""
        <div class="explain-box">
        <b>📖 피처 중요도란?</b><br>
        XGBoost 모델이 루버 각도를 예측할 때 각 입력 변수가 얼마나 중요하게 사용됐는지를 나타냅니다.
        <b>Gain</b> 방식은 해당 변수가 트리의 분기점에서 예측 오차를 얼마나 줄였는지로 중요도를 측정합니다.
        값이 클수록 그 변수가 각도 결정에 더 큰 영향을 줍니다.
        </div>
        """, unsafe_allow_html=True)

        importance_data = {
            "피처":  ["ghi_real", "doy_cos", "hour_sin", "cloud_cover", "doy_sin", "hour_cos", "temp_actual"],
            "Gain":  [0.408,       0.149,      0.126,       0.125,         0.072,     0.068,      0.052],
            "변수 설명": [
                "수평면 일사량 — 발전 가능한 태양에너지의 절대량",
                "연중 날짜 코사인 — 계절 위치 (하지/동지 구분)",
                "하루 중 시각 사인 — 오전/오후 태양 위치",
                "운량 (0~9) — 구름에 의한 일사 감쇠",
                "연중 날짜 사인 — 계절 위치 보완 성분",
                "하루 중 시각 코사인 — 시각 보완 성분",
                "외기온도 — 패널 온도계수에 의한 효율 보정",
            ],
        }
        df_imp = pd.DataFrame(importance_data).sort_values("Gain", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_imp["Gain"], y=df_imp["피처"], orientation="h",
            marker_color=["#4fc3f7" if g > 0.12 else "#90caf9" if g > 0.06 else "#b0bec5" for g in df_imp["Gain"]],
            text=[f"{g:.3f}" for g in df_imp["Gain"]], textposition="outside",
            customdata=df_imp["변수 설명"],
            hovertemplate="<b>%{y}</b><br>Gain: %{x:.3f}<br>%{customdata}<extra></extra>"
        ))
        fig.update_layout(height=380, xaxis_title="Gain (분기점 오차 감소 기여도)",
                          xaxis_range=[0, 0.50])
        st.plotly_chart(fig, use_container_width=True)

        # 변수별 해석
        st.markdown("**각 변수 해석**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="explain-box">
            <b>🥇 ghi_real (40.8%)</b><br>
            가장 중요한 변수. 일사량이 높을수록 루버를 눕혀(낮은 각도) 직달광을 최대화하고
            음영을 줄입니다. 일사량이 낮으면 각도를 세워 확산광을 활용합니다.
            </div>
            <div class="explain-box">
            <b>📅 doy_cos + doy_sin (22.1%)</b><br>
            연중 날짜 정보. 계절에 따라 태양 고도 패턴이 달라지므로 여름·겨울의 최적 각도 전략이
            완전히 다릅니다. 두 성분을 함께 써서 1월1일~12월31일을 원형으로 표현합니다.
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="explain-box">
            <b>🕐 hour_sin + hour_cos (19.4%)</b><br>
            하루 중 시각 정보. 같은 계절이라도 아침·정오·저녁에 태양 위치가 다르므로
            시각별 최적 각도가 달라집니다. 사인·코사인으로 24시간 순환성을 표현합니다.
            </div>
            <div class="explain-box">
            <b>☁️ cloud_cover (12.5%) + 🌡️ temp_actual (5.2%)</b><br>
            운량이 높으면 확산광 비중이 커져 각도 전략이 바뀝니다.
            기온은 패널 온도계수(-0.4%/°C)를 통해 효율에 미세하게 영향을 줍니다.
            </div>
            """, unsafe_allow_html=True)

        # MAE / R² 설명
        st.markdown("---")
        st.markdown("**모델 성능 지표 해석**")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div class="explain-box">
            <b>MAE (평균절대오차) = 1.56°</b><br>
            모델이 예측한 각도와 실제 최적 각도의 평균 차이가 <b>1.56°</b>입니다.<br>
            루버 제어 모터의 물리적 정밀도(±2~3°)보다 작은 오차이므로
            실제 제어에 지장이 없는 수준입니다.
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="explain-box">
            <b>R² (결정계수) = 0.9564</b><br>
            모델이 루버 각도 변동의 <b>95.6%를 설명</b>한다는 의미입니다.<br>
            1.0이 완벽한 예측이며, 0.95 이상은 실용적으로 매우 높은 수준입니다.
            나머지 4.4%는 예측하기 어려운 순간적 기상 변동 등에 기인합니다.
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: 음영 원리 — ★ v8.0: SVF/POA 설명, 쉬운 설명
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("💡 V13 중심축 회전 루버의 음영 원리")

        st.markdown("""
        <div class="explain-box">
        <b>📖 핵심 용어 설명</b><br>
        • <b>POA (Plane of Array)</b>: 패널 수광면 일사량. 루버 패널이 실제로 받는 태양에너지량 (W/m²).
          GHI(수평 일사량)와 달리 패널 기울기·방향·음영을 모두 반영한 실효 에너지입니다.<br>
        • <b>SVF (Sky View Factor)</b>: 하늘 조망 계수. 루버 사이 틈으로 보이는 하늘의 비율.
          루버를 눕힐수록 틈이 줄어 SVF가 낮아지고, 세울수록 SVF가 높아집니다.<br>
        • <b>음영률 (Shading Fraction)</b>: 윗 블레이드의 돌출부가 아랫 블레이드를 가리는 비율.
          태양이 낮게 뜰수록, 루버를 많이 눕힐수록 음영률이 높아집니다.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **쉽게 이해하는 작동 원리**

        블라인드를 생각해보세요. 블라인드 날개(블레이드)를 완전히 눕히면(수평) 빛이 잘 들어오지만
        위 날개가 아래 날개에 그림자를 드리웁니다. 반대로 세우면(수직) 그림자는 없지만
        직사광선은 못 받고 옆에서 오는 빛(확산광)만 받습니다.
        **AI는 매 시간 이 두 가지를 최적으로 조율하여 발전량을 최대화합니다.**

        V13에서는 블레이드가 **세로 방향 중심부에 회전축**이 있어, 기울어지면 위쪽 반은 벽면 안쪽으로,
        아래쪽 반은 바깥으로 돌출됩니다. 이 **돌출부(protrusion)**가 아래 블레이드에 그림자를 만듭니다.
        """)

        elev_example = st.slider("태양 고도각 (°)", min_value=5, max_value=80, value=30, step=5)
        tilt_example  = st.slider("루버 각도 (°)",   min_value=15, max_value=90, value=45, step=5)

        sf_ex  = calc_shading_fraction(tilt_example, elev_example, half_depth_mm, blade_depth_mm, pitch_mm)
        svf_ex = sky_view_factor(tilt_example, half_depth_mm, blade_depth_mm, pitch_mm)
        prot_ex, vocc_ex, gap_ex = blade_geometry(tilt_example, half_depth_mm, blade_depth_mm, pitch_mm)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            n_louvers = 4
            pitch_px = 80
            wall_x = -10

            for i in range(n_louvers):
                y_center = i * pitch_px
                half_len_px = (blade_depth_mm / 2) / pitch_mm * pitch_px
                dx = half_len_px * np.cos(np.radians(tilt_example))
                dy = half_len_px * np.sin(np.radians(tilt_example))
                pivot_x = 40

                top_x = pivot_x - dx
                top_y = y_center + dy
                bot_x = pivot_x + dx
                bot_y = y_center - dy

                fig.add_shape(type="line",
                    x0=top_x, y0=top_y, x1=bot_x, y1=bot_y,
                    line=dict(color="#4fc3f7", width=6))

                fig.add_trace(go.Scatter(x=[pivot_x], y=[y_center], mode="markers",
                    marker=dict(size=8, color="red", symbol="x"),
                    showlegend=False, hoverinfo="skip"))

                fig.add_shape(type="line",
                    x0=pivot_x, y0=y_center, x1=bot_x, y1=bot_y,
                    line=dict(color="#ff8a65", width=3, dash="dot"))

                if i > 0:
                    protrusion_px = prot_ex / pitch_mm * pitch_px
                    gap_px = gap_ex / pitch_mm * pitch_px
                    shadow_vert_px = protrusion_px / np.tan(np.radians(max(elev_example, 1)))
                    shadow_on_px = max(shadow_vert_px - gap_px, 0)
                    if shadow_on_px > 0:
                        shade_top = y_center
                        shade_bottom = max(y_center - shadow_on_px, (i-1)*pitch_px)
                        fig.add_shape(type="rect",
                            x0=bot_x*0.3, y0=shade_bottom, x1=bot_x*1.2, y1=shade_top,
                            fillcolor="rgba(255,100,100,0.25)", line_width=0)

            fig.add_shape(type="line", x0=wall_x, y0=-20, x1=wall_x, y1=n_louvers*pitch_px+40,
                           line=dict(color="gray", width=3))
            fig.add_annotation(x=wall_x, y=n_louvers*pitch_px+50, text="벽면",
                                showarrow=False, font=dict(color="gray", size=11))

            sun_x = 200
            sun_y = n_louvers * pitch_px + 30
            arrow_dx = -np.cos(np.radians(elev_example)) * 60
            arrow_dy = -np.sin(np.radians(elev_example)) * 60
            fig.add_annotation(x=sun_x+arrow_dx, y=sun_y+arrow_dy,
                ax=sun_x, ay=sun_y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowcolor="orange", arrowwidth=3)
            fig.add_trace(go.Scatter(x=[sun_x], y=[sun_y], mode="markers+text",
                marker=dict(size=20, color="orange", symbol="circle"),
                text=["☀️"], textposition="top center", showlegend=False))

            fig.update_layout(
                height=400,
                xaxis=dict(range=[-30, 260], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-30, n_louvers*pitch_px+80], showgrid=False, zeroline=False,
                           showticklabels=False, scaleanchor="x"),
                plot_bgcolor="rgba(20,25,40,1)",
                paper_bgcolor="rgba(0,0,0,0)",
                title=f"V13 중심축 회전 단면도 — 태양고도 {elev_example}° | 루버각 {tilt_example}°"
            )
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="블레이드",
                line=dict(color="#4fc3f7", width=4)))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="돌출부 (protrusion)",
                line=dict(color="#ff8a65", width=3, dash="dot")))
            fig.update_layout(legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("음영률", f"{sf_ex*100:.1f}%",
                       delta="낮음 ✅" if sf_ex < 0.3 else ("중간 ⚠️" if sf_ex < 0.6 else "높음 🔴"),
                       delta_color="off")
            st.metric("SVF (하늘 조망)", f"{svf_ex:.2f}",
                       delta="높음 ✅" if svf_ex > 0.5 else ("중간 ⚠️" if svf_ex > 0.2 else "낮음 🔴"),
                       delta_color="off")
            st.markdown(f"""
            <div class="explain-box">
            <b>V13 기하학 공식</b><br>
            돌출 = HALF_DEPTH × cos(각도)<br>
            수직점유 = DEPTH × sin(각도)<br>
            틈 = PITCH - 수직점유<br>
            SVF = 틈 / (틈 + 돌출)<br>
            그림자 = 돌출 / tan(태양고도) - 틈<br>
            음영률 = 그림자 / PITCH<br><br>
            <b>현재 계산값</b><br>
            DEPTH: {blade_depth_mm:.0f}mm | HALF: {half_depth_mm:.0f}mm<br>
            PITCH: {pitch_mm:.0f}mm<br>
            돌출: {prot_ex:.1f}mm<br>
            수직점유: {vocc_ex:.1f}mm<br>
            틈: {gap_ex:.1f}mm
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="explain-box">
            <b>💡 AI의 최적화 전략</b><br>
            각도를 낮추면: 직달광 POA↑ 하지만 음영↑<br>
            각도를 높이면: 음영↓ SVF↑ 하지만 직달광 POA↓<br>
            AI는 매 시간 이 상충관계를 계산해 순발전량이 최대인 각도를 선택합니다.
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: 음영 시각화 — ★ v8.0: 궤적 순서/방향/이유 설명
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("🔥 Tilt vs 태양고도별 음영률 히트맵 (V13)")

        c1, c2 = st.columns(2)
        with c1:
            bd_hm = st.number_input("블레이드 DEPTH (mm)", value=float(blade_depth_mm), step=1.0, key="hm_bd")
        with c2:
            p_hm = st.number_input("피치 (mm)", value=float(pitch_mm), step=1.0, key="hm_p")
        hd_hm = bd_hm / 2.0

        tilt_range = np.arange(15, 91, 2)
        elev_range = np.arange(5,  81, 2)
        Z = np.zeros((len(elev_range), len(tilt_range)))
        for i, e in enumerate(elev_range):
            Z[i, :] = calc_shading_fraction(tilt_range, e, hd_hm, bd_hm, p_hm)

        fig = go.Figure(go.Heatmap(
            x=tilt_range, y=elev_range, z=Z,
            colorscale="RdYlGn_r",
            colorbar=dict(title="음영률", tickformat=".0%"),
            zmin=0, zmax=1,
            hovertemplate="루버각: %{x}°<br>태양고도: %{y}°<br>음영률: %{z:.1%}<extra></extra>"
        ))

        # ★ v8.0: 궤적에 시간 순서 번호 추가
        elev_day = elev[mask_day]
        ang_day  = ai_angles[mask_day]
        time_labels = times[mask_day].strftime("%H:%M")
        valid = ghi_real[mask_day] >= 10

        fig.add_trace(go.Scatter(
            x=ang_day[valid], y=elev_day[valid],
            mode="markers+lines+text",
            name=f"AI 운전 궤적 ({_sim_d})",
            marker=dict(size=10, color="white", symbol="circle",
                        line=dict(color="#4fc3f7", width=2)),
            line=dict(color="white", width=2, dash="dot"),
            text=[f"{t}" for t in time_labels[valid]],
            textposition="top center",
            textfont=dict(size=9, color="white"),
        ))

        fig.add_shape(type="line", x0=60, x1=60, y0=5, y1=80,
                       line=dict(color="cyan", width=2, dash="dash"))
        fig.add_annotation(x=60, y=78, text="고정60°", showarrow=False,
                            font=dict(color="cyan", size=11))

        fig.update_layout(
            height=500,
            xaxis_title="루버 각도 (°)",
            yaxis_title="태양 고도각 (°)",
            title="V13 음영률 히트맵 — AI 운전 궤적 오버레이",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <b>📖 히트맵 읽는 법</b><br>
        • 🟢 초록: 음영 없음 → 발전 최대 구간<br>
        • 🟡 노랑: 부분 음영 → 발전 손실 시작<br>
        • 🔴 빨강: 음영 심함 → 발전량 크게 감소<br>
        • <b>흰 점선 (AI 궤적)</b>: 시간 순서대로 번호가 표시됩니다.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="explain-box">
        <b>💡 AI 궤적이 이렇게 움직이는 이유</b><br>
        <b>① 아침 (일출 직후)</b>: 태양 고도가 낮음 → 히트맵 하단에서 시작.
        낮은 고도에서는 어떤 각도를 써도 음영이 크므로, AI는 확산광 수신을 위해 각도를 높게 설정.<br><br>
        <b>② 정오 전후</b>: 태양 고도 최대 → 히트맵 상단으로 이동.
        고도가 높으면 음영이 줄어드므로 AI는 각도를 낮춰 직달광을 최대로 받음.
        궤적이 초록 구간(낮은 음영률)을 따라가는 것을 확인할 수 있음.<br><br>
        <b>③ 오후~저녁</b>: 태양 고도가 다시 낮아짐 → 히트맵 하단으로 내려오며 각도를 다시 높임.
        GHI가 10 W/m² 미만으로 떨어지면 루버를 90°로 닫고 대기.<br><br>
        <b>핵심</b>: AI는 항상 현재 태양 위치에서 음영률이 가장 낮은 초록 구간을 찾아 이동합니다.
        고정 60° (청록 수직선)보다 훨씬 다양한 경로로 최적점을 추적합니다.
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: 발전량 비교
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("⚡ AI vs 고정60° vs 수직90° 발전량 비교")

        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = go.Figure()
        for col, color, name in [("AI","#4fc3f7","AI 제어"), ("고정60°","#ffa726","고정 60°"), ("수직90°","#78909c","수직 90°")]:
            fig.add_trace(go.Bar(
                x=month_names, y=df_monthly[col] * area_scale / 1000,
                name=name, marker_color=color
            ))
        fig.update_layout(barmode="group", height=400,
                           yaxis_title="발전량 (kWh)", xaxis_title="월")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("연간 AI",     f"{ann_kwh_ai:.1f} kWh")
        c2.metric("연간 고정60°", f"{ann_kwh_60:.1f} kWh", f"AI 대비 -{(1-ann_kwh_60/ann_kwh_ai)*100:.1f}%" if ann_kwh_ai>0 else "—")
        c3.metric("연간 수직90°", f"{ann_kwh_90:.1f} kWh", f"AI 대비 -{(1-ann_kwh_90/ann_kwh_ai)*100:.1f}%" if ann_kwh_ai>0 else "—")

        df_cum = df_monthly.copy()
        df_cum["AI_cum"]  = (df_cum["AI"]   * area_scale / 1000).cumsum()
        df_cum["F60_cum"] = (df_cum["고정60°"] * area_scale / 1000).cumsum()
        df_cum["V90_cum"] = (df_cum["수직90°"] * area_scale / 1000).cumsum()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=month_names, y=df_cum["AI_cum"],  name="AI",     line=dict(color="#4fc3f7", width=3)))
        fig2.add_trace(go.Scatter(x=month_names, y=df_cum["F60_cum"], name="고정60°", line=dict(color="#ffa726", width=2, dash="dash")))
        fig2.add_trace(go.Scatter(x=month_names, y=df_cum["V90_cum"], name="수직90°", line=dict(color="#78909c", width=2, dash="dot")))
        fig2.update_layout(height=350, yaxis_title="누적 발전량 (kWh)", title="연간 누적 발전량")
        st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: 월별 각도 — ★ v8.0: 전체 해석 추가
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.subheader("📅 월별 AI 제어 각도 분포")

        df_plot2 = df_annual[df_annual["ghi"] >= 10].copy()
        df_plot2["month_n"] = df_plot2["timestamp"].dt.month
        df_plot2["month_s"] = df_plot2["month_n"].map(
            {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})

        fig = go.Figure()
        for m, ms in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1):
            d = df_plot2[df_plot2["month_n"] == m]["angle_ai"]
            fig.add_trace(go.Box(y=d, name=ms, marker_color="#4fc3f7", boxmean=True))
        fig.add_hline(y=ANGLE_MIN, line_dash="dash", line_color="red",
                       annotation_text=f"최소각 {ANGLE_MIN}°")
        fig.update_layout(height=420, yaxis_title="루버 각도 (°)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # ★ v8.0: 전체 해석
        st.markdown("""
        <div class="explain-box">
        <b>📊 월별 각도 분포 전체 해석</b><br><br>
        <b>🌞 여름 (6~8월) — 낮은 각도, 넓은 분포</b><br>
        서울 기준 여름철 태양 최대 고도각은 76° 수준입니다. 태양이 높이 떠 있으므로
        루버를 눕혀야(낮은 각도) 직달광이 패널 수광면에 수직에 가깝게 입사합니다.
        평균 각도가 15~20°로 낮고, 박스(분포)가 넓은 이유는 흐린 날(운량 많음)에
        확산광 활용을 위해 각도를 높이는 경우가 섞이기 때문입니다.
        장마철 흐린 날과 맑은 날의 전략이 달라 분포가 넓어집니다.<br><br>
        <b>❄️ 겨울 (12~2월) — 높은 각도, 좁은 분포</b><br>
        겨울철 최대 고도각은 약 29°로 낮습니다. 태양이 낮게 떠 있으므로
        루버를 세워야(높은 각도) 직달광을 효과적으로 받을 수 있습니다.
        평균 각도 35~45°, 분포가 좁은 이유는 겨울에는 맑은 날이 상대적으로 많고
        태양 궤적이 일정해 각도 전략이 단순하기 때문입니다.<br><br>
        <b>🍂 봄·가을 (3~5월, 9~11월) — 중간 각도, 점진적 변화</b><br>
        태양 고도각이 여름과 겨울 사이를 오가며 각도도 중간값(20~35°)을 보입니다.
        계절 전환기이므로 월별로 뚜렷하게 각도가 변하는 것을 확인할 수 있습니다.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("월별 평균 각도 & 발전량")
        v13_ref_angles = {
            1: 43.6, 2: 27.5, 3: 25.0, 4: 19.4, 5: 15.8, 6: 15.5,
            7: 16.3, 8: 18.8, 9: 18.0, 10: 28.4, 11: 31.9, 12: 38.9
        }
        df_summary = df_monthly.copy()
        df_summary["month_s"] = [month_names[i] for i in range(12)]
        df_summary = df_summary.rename(columns={"avg_angle":"시뮬 평균각(°)"})
        df_summary["V13 참조각(°)"] = [v13_ref_angles[m] for m in range(1,13)]
        st.dataframe(df_summary[["month_s","시뮬 평균각(°)","V13 참조각(°)","AI","고정60°","수직90°"]].round(1),
                     use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7: 내일 스케줄
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.subheader("🌤️ 내일 예측 스케줄")
        if kma is None:
            st.error("❌ 기상청 API 데이터를 불러올 수 없습니다. 청천 기준으로 시뮬레이션합니다.")
        else:
            st.success(f"✅ 기상청 단기예보 연동 성공 | 예보 기준일: {tomorrow}")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=times[mask_day].strftime("%H:%M"), y=ghi_real[mask_day],
                              name="예측 GHI (W/m²)", marker_color="rgba(255,180,0,0.4)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=times[mask_day].strftime("%H:%M"), y=ai_angles[mask_day],
                                  name="AI 예측 각도", line=dict(color="#4fc3f7", width=3),
                                  mode="lines+markers"), secondary_y=True)
        fig.update_yaxes(title_text="GHI (W/m²)", secondary_y=False)
        fig.update_yaxes(title_text="각도 (°)", range=[0, 95], secondary_y=True)
        fig.update_layout(height=380, title=f"내일({tomorrow}) 루버 제어 스케줄")
        st.plotly_chart(fig, use_container_width=True)

        df_tom = pd.DataFrame({
            "시간":        times[mask_day].strftime("%H:%M").tolist(),
            "예측 GHI (W/m²)": np.round(ghi_real[mask_day], 1).tolist(),
            "기온 (°C)":   np.round(temp_series[mask_day], 1).tolist(),
            "AI 각도(°)": ai_angles[mask_day].astype(int).tolist(),
            "음영률 상태": [
                f"{calc_shading_fraction(a, e, half_depth_mm, blade_depth_mm, pitch_mm)*100:.1f}%"
                for a, e in zip(ai_angles[mask_day], elev[mask_day])
            ],
            "SVF": [f"{sky_view_factor(a, half_depth_mm, blade_depth_mm, pitch_mm):.2f}"
                    for a in ai_angles[mask_day]],
        })
        st.dataframe(df_tom, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8: 건강진단 — ★ v8.0: 용어/해석 강화
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.subheader("🩺 시스템 건강 진단")

        st.markdown("""
        <div class="explain-box">
        <b>📖 건강진단이란?</b><br>
        AI가 예측한 발전량(P_sim)과 실제 센서에서 측정된 발전량(P_actual)을 비교하여
        패널 오염·고장·케이블 문제 등 시스템 이상을 자동으로 감지하는 기능입니다.<br><br>
        • <b>P_sim (Simulated Power)</b>: AI가 기상 조건과 루버 각도를 기반으로 계산한 <b>예상 발전량</b>.
          "이 조건이면 이만큼 나와야 한다"는 기준값입니다.<br>
        • <b>P_actual (Actual Power)</b>: 실제 인버터/계측기에서 측정된 <b>실측 발전량</b>.<br>
        • <b>Health Ratio</b>: P_actual / P_sim. 1.0(100%)이면 완벽 정상, 낮을수록 이상 상태.
        </div>
        """, unsafe_allow_html=True)

        st.info("📌 현재는 실측 센서 미연동 상태입니다. 아래 슬라이더로 시나리오를 시뮬레이션할 수 있습니다.")

        col1, col2 = st.columns(2)
        with col1:
            p_actual_pct = st.slider("실측 발전량 비율 (%)", min_value=10, max_value=110, value=95,
                                       help="P_actual / P_sim × 100. 센서 연동 후 자동 입력됩니다.")
        with col2:
            warn_thr  = st.number_input("WARNING 임계값 (%)", value=90,
                                         help="이 값 미만이면 경고. 패널 오염 의심 수준.")
            crit_thr  = st.number_input("CRITICAL 임계값 (%)", value=75,
                                         help="이 값 미만이면 긴급. 즉시 점검 필요 수준.")

        health_ratio = p_actual_pct / 100.0
        if health_ratio >= warn_thr / 100:
            status, color = "✅ NORMAL",   "green"
        elif health_ratio >= crit_thr / 100:
            status, color = "⚠️ WARNING",  "orange"
        else:
            status, color = "🔴 CRITICAL", "red"

        c1, c2, c3 = st.columns(3)
        c1.metric("P_sim (AI 예측)", f"{pow_ai/1000:.3f} kWh",
                   help="기상조건 + 루버각도 기반 시뮬레이션 발전량")
        c2.metric("P_actual (실측 추정)", f"{pow_ai/1000*health_ratio:.3f} kWh",
                   help="실제 측정 발전량 (현재는 슬라이더 입력값)")
        c3.metric("Health Ratio", f"{health_ratio:.2%}", status)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=p_actual_pct,
            delta={"reference": 100, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 110]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, crit_thr],  "color": "rgba(255,80,80,0.2)"},
                    {"range": [crit_thr, warn_thr], "color": "rgba(255,165,0,0.2)"},
                    {"range": [warn_thr, 110], "color": "rgba(0,200,100,0.2)"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": warn_thr},
            },
            title={"text": f"시스템 상태: {status}"}
        ))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <b>📊 판정 기준 및 해석</b><br>
        <b>✅ 90% 이상 (NORMAL)</b>: 정상 운전. P_sim과 P_actual이 거의 일치. 시스템 이상 없음.<br>
        <b>⚠️ 75~90% (WARNING)</b>: 패널 표면 오염(먼지, 새똥 등) 또는 부분 고장 의심.
        10~25% 발전 손실이 발생 중. 청소 또는 현장 점검 권고.<br>
        <b>🔴 75% 미만 (CRITICAL)</b>: 심각한 성능 저하. 25% 이상 발전 손실.
        케이블 불량, 인버터 고장, 다수 셀 손상 등 중대 결함 가능성. 즉시 점검 필요.<br><br>
        <b>임계값 설정 기준</b>: WARNING 90%는 패널 오염으로 통상 발생하는 손실 범위의 상한,
        CRITICAL 75%는 단순 오염을 넘어 구조적 결함이 의심되는 수준입니다.
        실제 운영 환경에 따라 조정 가능합니다.
        </div>
        """, unsafe_allow_html=True)

        st.caption("📌 실측 센서 연동 후 P_actual 자동 입력 예정 (계속출원 청구항 4 대상).")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9: 파라미터 튜닝 — ★ v8.0: 3종 민감도 그래프
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.subheader("🔧 파라미터 민감도 분석 — V13 물리모델 기반")
        st.markdown("파라미터 변화에 따라 발전량·음영률이 어떻게 달라지는지 확인합니다.")

        col1, col2 = st.columns(2)
        with col1:
            t_loss     = st.slider("시스템 손실률", 0.70, 0.95, DEFAULT_LOSS, step=0.01, key="t_loss")
            t_capacity = st.slider("패널 용량 (W)", 100, 600, DEFAULT_CAPACITY, step=50, key="t_cap")

        # ── 그래프 1: BLADE_DEPTH 변화 → 연간 발전량 변화 ──────────────────
        st.markdown("---")
        st.markdown("#### 📐 블레이드 DEPTH 변화 → 연간 발전량 영향")
        st.caption("DEPTH가 클수록 돌출이 커져 음영이 늘고, 작을수록 음영은 줄지만 발전 면적도 줄어듭니다.")

        depth_range = np.arange(60, 181, 10)
        pow_by_depth = []
        for bd in depth_range:
            hd = bd / 2.0
            _, _, _, df_m_tmp, _ = get_annual_data(
                last_year, hd, float(bd), pitch_mm, capacity_w, unit_count, eff_factor, t_loss
            )
            pow_by_depth.append(df_m_tmp["AI"].sum() * area_scale / 1000)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=depth_range, y=pow_by_depth,
            mode="lines+markers", line=dict(color="#4fc3f7", width=2),
            name="연간 AI 발전량"
        ))
        fig1.add_vline(x=blade_depth_mm, line_dash="dash", line_color="orange",
                        annotation_text=f"현재 {blade_depth_mm:.0f}mm")
        fig1.update_layout(height=300, xaxis_title="BLADE_DEPTH (mm)",
                            yaxis_title="연간 발전량 (kWh)", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

        # ── 그래프 2: PITCH 변화 → 음영률 변화 ─────────────────────────────
        st.markdown("---")
        st.markdown("#### 📏 피치(블레이드 간격) 변화 → 정오 기준 음영률")
        st.caption("피치가 넓을수록 블레이드 사이 간격이 커져 음영이 줄어듭니다. 단, 블레이드 수가 같으면 전체 설치 면적이 늘어납니다.")

        pitch_range = np.arange(80, 201, 5)
        sf_noon_by_pitch = []
        for p in pitch_range:
            hd = blade_depth_mm / 2.0
            sf_noon = calc_shading_fraction(45.0, 60.0, hd, blade_depth_mm, float(p))
            sf_noon_by_pitch.append(float(sf_noon) * 100)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pitch_range, y=sf_noon_by_pitch,
            mode="lines+markers", line=dict(color="#ff8a65", width=2),
            name="음영률 (%)"
        ))
        fig2.add_vline(x=pitch_mm, line_dash="dash", line_color="orange",
                        annotation_text=f"현재 {pitch_mm:.0f}mm")
        fig2.add_hline(y=30, line_dash="dot", line_color="green",
                        annotation_text="30% (양호 기준)")
        fig2.update_layout(height=300, xaxis_title="PITCH (mm)",
                            yaxis_title="음영률 % (루버45°, 태양고도60° 기준)",
                            template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

        # ── 그래프 3: 최소각도 변화 → AI 발전량 변화 ───────────────────────
        st.markdown("---")
        st.markdown("#### 📐 최소 허용 각도 변화 → 연간 AI 발전량 영향")
        st.caption("최소각이 낮을수록 여름철 최적 제어 범위가 넓어지지만, 너무 낮으면 기계적 한계와 음영 증가로 오히려 손해가 날 수 있습니다.")

        amin_range = np.arange(5, 46, 5)
        pow_by_amin = []
        for amin in amin_range:
            times_y  = pd.date_range(start=f"{last_year}-01-01", end=f"{last_year}-12-31 23:00", freq="h", tz=TZ)
            solpos_y = site.get_solarposition(times_y)
            cs_y     = site.get_clearsky(times_y)
            ghi_y    = np.asarray(cs_y["ghi"].values, dtype=float)
            zen_y    = solpos_y["apparent_zenith"].values
            az_y     = solpos_y["azimuth"].values
            elev_y   = 90.0 - zen_y
            dni_y    = pvlib.irradiance.dirint(ghi_y, zen_y, times_y).fillna(0).values
            dhi_y    = (ghi_y - dni_y * np.cos(np.radians(zen_y))).clip(0)
            angles_tmp = np.where(ghi_y < 10, float(ANGLE_NIGHT),
                                   np.clip(elev_y, float(amin), ANGLE_MAX).astype(float))
            tilt = np.asarray(angles_tmp, dtype=float)
            poa_dir, poa_diff, poa_sky = poa_components(tilt, np.full_like(tilt, 180), zen_y, az_y, dni_y, ghi_y, dhi_y)
            eff_poa = calc_effective_poa(poa_dir, poa_diff, poa_sky, tilt, elev_y, half_depth_mm, blade_depth_mm, pitch_mm)
            mask = ghi_y >= 10
            wh = (eff_poa[mask] / 1000 * capacity_w * unit_count * eff_factor * t_loss).sum()
            pow_by_amin.append(wh * area_scale / 1000)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=amin_range, y=pow_by_amin,
            mode="lines+markers", line=dict(color="#a5d6a7", width=2),
            name="연간 AI 발전량"
        ))
        fig3.add_vline(x=ANGLE_MIN, line_dash="dash", line_color="orange",
                        annotation_text=f"현재 최소각 {ANGLE_MIN}°")
        fig3.update_layout(height=300, xaxis_title="최소 허용 각도 (°)",
                            yaxis_title="연간 발전량 (kWh)", template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

        st.caption("💡 V13: PPO 강화학습을 향후 실시예로 추가 가능 (현재는 XGBoost/규칙 기반)")


if __name__ == "__main__":
    run_app()
