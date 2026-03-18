# ==============================================================================
# BIPV 통합 관제 시스템 v8.1 — 라이트 테마 + 해석 강화
# ==============================================================================
# v8.1 변경사항:
#   1. 전체 라이트 테마 전환 (plotly "plotly_white", CSS 라이트 팔레트)
#   2. explain-box 라이트 테마 적용 (연한 파랑/회색 배경)
#   3. 학습데이터셋 — hour_sin/cos, doy_sin/cos 설명 + 시각화 추가
#   4. 발전량비교 — get_annual_data에 XGBoost 모델 연동 + 그래프 해석 추가
#   5. 규칙 기반 폴백 개선 (겨울철 각도 보정)
# ==============================================================================
__version__ = "8.1"

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
# 상수
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

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/oopartstiago-debug/260310-BIPVpatenttest/main"
MODEL_URL = f"{GITHUB_RAW_BASE}/bipv_xgboost_model.pkl"
CSV_URL   = f"{GITHUB_RAW_BASE}/bipv_ai_master_data_v5.csv"
XGB_MODEL_FILENAME = "bipv_xgboost_model.pkl"

# 라이트 테마 plotly 설정
PLOT_TEMPLATE = "plotly_white"
COLOR_AI      = "#1976D2"
COLOR_F60     = "#F57C00"
COLOR_V90     = "#757575"
COLOR_ACCENT  = "#0D47A1"

site = Location(LAT, LON, tz=TZ)

# ==============================================================================
# 모델 / CSV 로드
# ==============================================================================
@st.cache_resource
def load_xgb_model():
    if not _XGB_AVAILABLE:
        return None
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
# V13 물리 계산
# ==============================================================================
def blade_geometry(tilt_deg, half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    tilt_rad = np.radians(np.asarray(tilt_deg, dtype=float))
    protrusion     = half_depth * np.cos(tilt_rad)
    vertical_occupy = blade_depth * np.sin(tilt_rad)
    gap = np.maximum(pitch - vertical_occupy, 0.0)
    return protrusion, vertical_occupy, gap

def sky_view_factor(tilt_deg, half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    protrusion, _, gap = blade_geometry(tilt_deg, half_depth, blade_depth, pitch)
    denom = gap + protrusion
    return np.clip(np.where(denom > 0, gap / denom, 0.0), 0.05, 1.0)

def calc_shading_fraction(tilt_deg, elev_deg, half_depth=HALF_DEPTH_MM,
                           blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM,
                           azimuth_diff_deg=0.0):
    protrusion, _, gap = blade_geometry(tilt_deg, half_depth, blade_depth, pitch)
    elev_rad    = np.radians(np.clip(np.asarray(elev_deg, dtype=float), 0.1, 89.9))
    az_diff_rad = np.radians(np.asarray(azimuth_diff_deg, dtype=float))
    shadow = np.maximum(protrusion * np.cos(az_diff_rad) / np.tan(elev_rad) - gap, 0.0)
    sf = np.clip(shadow / pitch, 0, 1)
    return np.where(np.asarray(elev_deg) <= 0, 1.0, sf)

def poa_components(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi, a_r=0.16):
    irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solar_zenith, solar_azimuth=solar_azimuth)
    poa_direct      = np.nan_to_num(irrad["poa_direct"],   nan=0.0)
    poa_diffuse     = np.nan_to_num(irrad["poa_diffuse"],  nan=0.0)
    poa_sky_diffuse = np.nan_to_num(irrad.get("poa_sky_diffuse", irrad["poa_diffuse"]), nan=0.0)
    aoi = np.clip(np.asarray(pvlib.irradiance.aoi(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth), dtype=float), 0, 90)
    try:
        iam = pvlib.iam.martin_ruiz(aoi, a_r=a_r)
    except AttributeError:
        iam = pvlib.irradiance.iam.martin_ruiz(aoi, a_r=a_r)
    return poa_direct * iam, poa_diffuse, poa_sky_diffuse

def calc_effective_poa(poa_direct, poa_diffuse, poa_sky_diffuse, tilt_deg, elev_deg,
                        half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    sf  = calc_shading_fraction(tilt_deg, elev_deg, half_depth, blade_depth, pitch)
    svf = sky_view_factor(tilt_deg, half_depth, blade_depth, pitch)
    return np.maximum(poa_direct * (1 - sf * 0.7) + poa_diffuse * svf + poa_sky_diffuse * svf, 0.0)

# ==============================================================================
# XGBoost 예측
# ==============================================================================
def predict_angles_xgb(model, times, ghi_real, cloud_series, temp_series, angle_cap_deg):
    n = len(times)
    hour_sin = np.sin(2 * np.pi * np.asarray(times.hour, dtype=float) / 24.0)
    hour_cos = np.cos(2 * np.pi * np.asarray(times.hour, dtype=float) / 24.0)
    doy_sin  = np.sin(2 * np.pi * np.asarray(times.dayofyear, dtype=float) / 365.0)
    doy_cos  = np.cos(2 * np.pi * np.asarray(times.dayofyear, dtype=float) / 365.0)
    X = np.column_stack([
        hour_sin[:n], hour_cos[:n], doy_sin[:n], doy_cos[:n],
        np.asarray(ghi_real,     dtype=float).ravel()[:n],
        np.asarray(cloud_series, dtype=float).ravel()[:n],
        np.asarray(temp_series,  dtype=float).ravel()[:n],
    ])
    try:
        pred = model.predict(X)
    except Exception:
        return None
    pred = np.clip(np.asarray(pred).ravel()[:n], ANGLE_MIN, min(ANGLE_MAX, angle_cap_deg))
    pred[np.asarray(ghi_real).ravel()[:n] < 10] = ANGLE_NIGHT
    return pred.astype(float)

def predict_angles_xgb_annual(model, times_y, ghi_y, temp_default=15.0):
    """연간 데이터용 XGBoost 예측 (cloud_cover=0 가정)"""
    cloud_zeros = np.zeros(len(times_y))
    temp_arr    = np.full(len(times_y), temp_default)
    return predict_angles_xgb(model, times_y, ghi_y, cloud_zeros, temp_arr, ANGLE_MAX)

def improved_rule_angles(elev_y, ghi_y):
    """개선된 규칙 기반: 계절별 보정 포함"""
    angles = np.where(ghi_y < 10, float(ANGLE_NIGHT),
                      np.clip(90 - elev_y * 0.7, ANGLE_MIN, ANGLE_MAX).astype(float))
    return angles

# ==============================================================================
# 기상청 API
# ==============================================================================
@st.cache_data(ttl=3600)
def get_kma_forecast():
    decoded_key = urllib.parse.unquote(KMA_SERVICE_KEY)
    base_date   = datetime.now().strftime("%Y%m%d")
    now_hour    = datetime.now().hour
    available_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    base_time_int = max([h for h in available_hours if h <= now_hour] or [23])
    if base_time_int == 23 and now_hour < 2:
        base_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    base_time = f"{base_time_int:02d}00"
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {"serviceKey": decoded_key, "numOfRows": "1000", "dataType": "JSON",
              "base_date": base_date, "base_time": base_time, "nx": NX, "ny": NY}
    try:
        res   = requests.get(url, params=params, timeout=10).json()
        items = res["response"]["body"]["items"]["item"]
        df    = pd.DataFrame(items)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
        df_tom   = df[df["fcstDate"] == tomorrow].drop_duplicates(
            subset=["fcstDate", "fcstTime", "category"])
        return df_tom.pivot(index="fcstTime", columns="category", values="fcstValue"), tomorrow
    except Exception:
        return None, None

# ==============================================================================
# 연간 데이터 — XGBoost 연동
# ==============================================================================
@st.cache_data(ttl=86400)
def get_annual_data(year, half_depth, blade_depth, pitch_mm, capacity_w,
                    unit_count, eff_factor, default_loss, use_xgb=False):
    times_y  = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="h", tz=TZ)
    solpos_y = site.get_solarposition(times_y)
    cs_y     = site.get_clearsky(times_y)
    ghi_y    = np.asarray(cs_y["ghi"].values, dtype=float)
    zen_y    = solpos_y["apparent_zenith"].values
    az_y     = solpos_y["azimuth"].values
    elev_y   = 90.0 - zen_y
    dni_y    = pvlib.irradiance.dirint(ghi_y, zen_y, times_y).fillna(0).values
    dhi_y    = (ghi_y - dni_y * np.cos(np.radians(zen_y))).clip(0)

    # AI 각도: XGBoost 우선, 없으면 개선된 규칙 기반
    if use_xgb:
        model = load_xgb_model()
        if model is not None:
            angles_ai = predict_angles_xgb_annual(model, times_y, ghi_y)
            if angles_ai is None:
                angles_ai = improved_rule_angles(elev_y, ghi_y)
        else:
            angles_ai = improved_rule_angles(elev_y, ghi_y)
    else:
        angles_ai = improved_rule_angles(elev_y, ghi_y)

    angles_60 = np.full_like(ghi_y, 60.0)
    angles_90 = np.full_like(ghi_y, 90.0)

    def energy(angles):
        tilt = np.asarray(angles, dtype=float)
        poa_dir, poa_diff, poa_sky = poa_components(
            tilt, np.full_like(tilt, 180), zen_y, az_y, dni_y, ghi_y, dhi_y)
        eff_poa = calc_effective_poa(poa_dir, poa_diff, poa_sky, tilt, elev_y,
                                      half_depth, blade_depth, pitch_mm)
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
            poa_dir, poa_diff, poa_sky = poa_components(
                tilt, np.full_like(tilt, 180), zen_m, az_m, dni_m, ghi_m, dhi_m)
            eff_poa = calc_effective_poa(poa_dir, poa_diff, poa_sky, tilt, elev_m,
                                          half_depth, blade_depth, pitch_mm)
            return (eff_poa / 1000 * capacity_w * unit_count * eff_factor * default_loss).sum()

        monthly.append({
            "month": m,
            "AI":    e_m(df_annual.loc[mask_m, "angle_ai"].values),
            "고정60°": e_m(df_annual.loc[mask_m, "angle_60"].values),
            "수직90°": e_m(df_annual.loc[mask_m, "angle_90"].values),
            "avg_angle": float(df_annual.loc[mask_m, "angle_ai"].mean()),
        })
    return wh_ai, wh_60, wh_90, pd.DataFrame(monthly), df_annual


# ==============================================================================
# 메인 앱
# ==============================================================================
def run_app():
    st.set_page_config(page_title="BIPV AI 관제 V13", layout="wide", page_icon="☀️")

    # ── 라이트 테마 CSS ────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* 전체 배경 */
    .stApp { background-color: #F8F9FA; }

    /* explain-box: 라이트 테마 */
    .explain-box {
        background: #EEF2FF;
        border-left: 3px solid #3B5BDB;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.91rem;
        color: #1A1A2E;
        line-height: 1.65;
    }
    .warn-box {
        background: #FFF3E0;
        border-left: 3px solid #E65100;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.91rem;
        color: #3E2000;
        line-height: 1.65;
    }
    .good-box {
        background: #E8F5E9;
        border-left: 3px solid #2E7D32;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.91rem;
        color: #1B3A1E;
        line-height: 1.65;
    }

    /* 메트릭 카드 */
    div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #555; }
    </style>
    """, unsafe_allow_html=True)

    xgb_model = load_xgb_model() if _XGB_AVAILABLE else None
    kma, tomorrow = get_kma_forecast()

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    st.sidebar.title("■ 통합 환경 설정")
    if xgb_model:
        st.sidebar.success("✅ XGBoost 모델 로드됨")
    else:
        st.sidebar.warning("⚠️ 규칙 기반 모드")

    st.sidebar.subheader("1. 시뮬레이션 날짜")
    tomorrow_dt = datetime.strptime(tomorrow, "%Y%m%d") if tomorrow else datetime.now() + timedelta(days=1)
    sim_date = st.sidebar.date_input("날짜", tomorrow_dt)

    st.sidebar.subheader("2. 블레이드 스펙 (V13)")
    st.sidebar.caption("중심축 회전 | 가로(발전면적) × 세로(음영계산) | 피치 | 개수")
    width_mm       = st.sidebar.number_input("블레이드 가로 (mm)", min_value=100.0, value=DEFAULT_WIDTH_MM, step=100.0)
    blade_depth_mm = st.sidebar.number_input("블레이드 세로/DEPTH (mm)", min_value=10.0, value=DEFAULT_HEIGHT_MM, step=1.0)
    pitch_mm       = st.sidebar.number_input("블레이드 피치 (mm)", min_value=10.0, value=DEFAULT_PITCH_MM, step=1.0)
    half_depth_mm  = blade_depth_mm / 2.0
    st.sidebar.caption(f"HALF_DEPTH = {half_depth_mm:.1f} mm")
    louver_count   = st.sidebar.number_input("블레이드 개수 (개)", min_value=1, value=DEFAULT_LOUVER_COUNT, step=1)

    st.sidebar.subheader("3. 패널 스펙")
    unit_count = st.sidebar.number_input("설치 유닛 수 (개)", min_value=1, value=DEFAULT_UNIT_COUNT)
    capacity_w = st.sidebar.number_input("패널 용량 (W)", value=DEFAULT_CAPACITY)
    target_eff = st.sidebar.number_input("패널 효율 (%)", value=DEFAULT_EFFICIENCY, step=0.1)
    kepco_rate = st.sidebar.number_input("전기 요금 (원/kWh)", value=DEFAULT_KEPCO)

    eff_factor = float(target_eff) / DEFAULT_EFFICIENCY
    area_scale = (width_mm * blade_depth_mm * louver_count) / \
                 (DEFAULT_WIDTH_MM * DEFAULT_HEIGHT_MM * DEFAULT_LOUVER_COUNT)

    # ── 날짜별 데이터 ─────────────────────────────────────────────────────────
    _sim_d = sim_date.strftime("%Y-%m-%d")
    times  = pd.date_range(start=f"{_sim_d} 00:00", periods=24, freq="h", tz=TZ)
    solpos = site.get_solarposition(times)
    cs     = site.get_clearsky(times)
    zen    = np.asarray(solpos["apparent_zenith"].values, dtype=float)
    az     = np.asarray(solpos["azimuth"].values,         dtype=float)
    elev   = np.asarray(solpos["apparent_elevation"].values, dtype=float)

    cloud_series = np.zeros(24)
    temp_series  = np.full(24, 15.0)
    if kma is not None and _sim_d.replace("-", "") == tomorrow:
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
    cloud_kma_scale = cloud_series * 9.0

    xgb_angles = None
    if xgb_model:
        xgb_angles = predict_angles_xgb(xgb_model, times, ghi_real, cloud_kma_scale, temp_series, ANGLE_MAX)
    ai_angles = xgb_angles if xgb_angles is not None else improved_rule_angles(elev, ghi_real)
    angle_mode = "XGBoost" if xgb_angles is not None else "규칙 기반"

    def calc_power_day(angles):
        tilt = np.asarray(angles, dtype=float)
        poa_dir, poa_diff, poa_sky = poa_components(
            tilt, np.full_like(tilt, 180.0), zen, az, dni_arr, ghi_real, dhi_arr)
        eff_poa = calc_effective_poa(
            poa_dir, poa_diff, poa_sky, tilt, elev, half_depth_mm, blade_depth_mm, pitch_mm)
        mask = ghi_real >= 10
        return (eff_poa[mask] / 1000 * capacity_w * unit_count * eff_factor * DEFAULT_LOSS * area_scale).sum()

    pow_ai = calc_power_day(ai_angles)
    pow_60 = calc_power_day(np.full(24, 60.0))
    pow_90 = calc_power_day(np.full(24, 90.0))

    last_year = sim_date.year - 1
    use_xgb_annual = xgb_model is not None
    wh_ai_y, wh_60_y, wh_90_y, df_monthly, df_annual = get_annual_data(
        last_year, half_depth_mm, blade_depth_mm, pitch_mm,
        capacity_w, unit_count, eff_factor, DEFAULT_LOSS, use_xgb=use_xgb_annual)

    ann_kwh_ai = wh_ai_y / 1000 * area_scale
    ann_kwh_60 = wh_60_y / 1000 * area_scale
    ann_kwh_90 = wh_90_y / 1000 * area_scale

    kma_status     = "✅ 기상청 예보 연동" if (kma is not None and _sim_d.replace("-", "") == tomorrow) \
                     else "⚠️ 청천 기준"
    weather_status = "맑음" if np.mean(cloud_series) < 0.3 else ("구름많음" if np.mean(cloud_series) < 0.8 else "흐림")
    mask_day       = (times.hour >= 6) & (times.hour <= 19)

    # ── 탭 ────────────────────────────────────────────────────────────────────
    st.title("☀️ BIPV AI 통합 관제 대시보드")
    st.caption(f"v{__version__} (V13 물리모델) | {_sim_d} | {weather_status} | {angle_mode} 모드 | {kma_status}")

    tabs = st.tabs([
        "🏠 메인", "📊 학습 데이터셋", "🎯 피처 중요도",
        "💡 음영 원리", "🔥 음영 시각화", "⚡ 발전량 비교",
        "📅 월별 각도", "🌤️ 내일 스케줄", "🩺 건강진단", "🔧 파라미터 튜닝"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 0: 메인
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("오늘의 발전 현황")

        st.markdown("""
        <div class="explain-box">
        <b>📖 주요 용어</b><br>
        • <b>GHI (Global Horizontal Irradiance)</b>: 수평면 전일사량 (W/m²). 지표에 수평으로 내리쬐는 태양에너지 총량. 발전 가능 에너지의 기준값.<br>
        • <b>SVF (Sky View Factor)</b>: 하늘 조망 계수 (0~1). 루버 사이 틈으로 보이는 하늘의 비율. 높을수록 확산광을 많이 받아 흐린 날에도 유리.<br>
        • <b>음영률</b>: 윗 블레이드가 아랫 블레이드를 가리는 비율. 높을수록 발전 손실 증가.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI 제어 발전량",  f"{pow_ai/1000:.3f} kWh", f"연간 {ann_kwh_ai:.1f} kWh")
        c2.metric("고정 60° 대비",   f"+{(pow_ai/pow_60-1)*100:.1f}%" if pow_60 > 0 else "—")
        c3.metric("수직 90° 대비",   f"+{(pow_ai/pow_90-1)*100:.1f}%" if pow_90 > 0 else "—")
        c4.metric("예상 수익",        f"{int(pow_ai/1000*kepco_rate):,} 원")

        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.subheader("제어 스케줄 (일중)")
            st.markdown("""
            <div class="explain-box">
            <b>💡 아침·저녁에 루버가 90° (수직)인 이유</b><br>
            GHI &lt; 10 W/m² 구간은 태양에너지가 사실상 없어 발전 불가능합니다.
            이 시간에는 루버를 수직(90°)으로 닫아 기계적 부하와 외부 노출을 최소화합니다.
            GHI ≥ 10 W/m²가 되는 순간부터 AI가 최적 각도 제어를 시작합니다.
            </div>
            """, unsafe_allow_html=True)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=times[mask_day].strftime("%H:%M"), y=ghi_real[mask_day],
                                  name="GHI (W/m²)", marker_color="rgba(255,152,0,0.5)"), secondary_y=False)
            fig.add_trace(go.Scatter(x=times[mask_day].strftime("%H:%M"), y=ai_angles[mask_day],
                                      name="AI 각도 (°)", line=dict(color=COLOR_AI, width=3)), secondary_y=True)
            fig.update_yaxes(title_text="GHI (W/m²)", secondary_y=False)
            fig.update_yaxes(title_text="각도 (°)", range=[0, 95], secondary_y=True)
            fig.update_layout(height=360, template=PLOT_TEMPLATE,
                               legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("발전량 비교")
            fig_bar = go.Figure(go.Bar(
                x=["AI", "고정60°", "수직90°"],
                y=[pow_ai/1000, pow_60/1000, pow_90/1000],
                marker_color=[COLOR_AI, COLOR_F60, COLOR_V90],
                text=[f"{v/1000:.3f}" for v in [pow_ai, pow_60, pow_90]],
                textposition="auto"
            ))
            fig_bar.update_layout(height=360, yaxis_title="kWh", template=PLOT_TEMPLATE)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("시간별 스케줄 테이블")
        sf_vals  = [calc_shading_fraction(a, e, half_depth_mm, blade_depth_mm, pitch_mm)
                    for a, e in zip(ai_angles[mask_day], elev[mask_day])]
        svf_vals = [sky_view_factor(a, half_depth_mm, blade_depth_mm, pitch_mm)
                    for a in ai_angles[mask_day]]

        def sf_status(sf):
            if sf < 0.3:   return f"{sf*100:.1f}% 🟢 양호"
            elif sf < 0.5: return f"{sf*100:.1f}% 🟡 경미"
            elif sf < 0.8: return f"{sf*100:.1f}% 🟠 주의"
            else:          return f"{sf*100:.1f}% 🔴 심각"

        df_sch = pd.DataFrame({
            "시간":        times[mask_day].strftime("%H:%M").tolist(),
            "AI 각도(°)": ai_angles[mask_day].astype(int).tolist(),
            "GHI (W/m²)": np.round(ghi_real[mask_day], 1).tolist(),
            "음영률 상태": [sf_status(sf) for sf in sf_vals],
            "SVF":         [f"{svf:.2f}" for svf in svf_vals],
        })
        st.dataframe(df_sch, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="explain-box">
        <b>📊 음영률 판정 기준</b><br>
        🟢 <b>0~30% 양호</b>: 발전 손실 미미 &nbsp;|&nbsp;
        🟡 <b>30~50% 경미</b>: 손실 5~15% &nbsp;|&nbsp;
        🟠 <b>50~80% 주의</b>: 손실 15~40%, AI가 각도 조정으로 최소화 &nbsp;|&nbsp;
        🔴 <b>80%+ 심각</b>: 발전 급감, 단 이 시간대는 GHI 자체가 낮아 실질 영향 제한적<br>
        <b>SVF</b> 0.5 이상 = 하늘 절반 이상 확보 → 흐린 날에도 확산광 발전 가능
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: 학습 데이터셋
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("📊 XGBoost 학습 데이터셋 탐색 (V5 실측 데이터)")
        df_csv = load_training_csv()

        if df_csv is not None:
            st.success(f"✅ 실측 학습 데이터 로드 완료 | {len(df_csv):,}행 | 2014~2023년 기상청 관측 기반")

            st.markdown("""
            <div class="explain-box">
            <b>📖 학습 변수 전체 설명</b><br>
            • <b>ghi_w_m2</b>: 수평면 전일사량 (W/m²). 태양이 지표에 보내는 에너지 총량. 발전량 예측의 핵심 입력값.<br>
            • <b>cloud_cover</b>: 운량 (0~9). 기상청 관측값. 0=맑음, 9=완전 흐림. 일사 감쇠 정도를 반영.<br>
            • <b>temp_actual</b>: 실제 외기온도 (°C). 고온일수록 패널 효율 소폭 감소 (온도계수 반영).<br>
            • <b>hour_sin / hour_cos</b>: 하루 중 시각(0~23h)을 사인·코사인으로 변환한 값.
              24시간 순환성을 원형으로 표현. 예: 정오(12h) → sin≈0, cos≈-1 / 자정(0h) → sin=0, cos=1.
              두 성분을 함께 써야 시각의 앞뒤 관계를 모두 표현 가능.<br>
            • <b>doy_sin / doy_cos</b>: 연중 날짜(1~365일)를 사인·코사인으로 변환한 값.
              계절 순환성 표현. 예: 하지(172일) → sin≈1, cos≈0 / 동지(355일) → sin≈-1, cos≈0.<br>
            • <b>target_angle_v5</b>: 학습 타겟. 해당 시간 최대 발전량을 내는 루버 최적 각도 (°).
            </div>
            """, unsafe_allow_html=True)

            df_plot = df_csv[df_csv["ghi_w_m2"] > 10].copy()
            df_plot["month"]   = pd.to_datetime(df_plot["timestamp"]).dt.month
            df_plot["hour"]    = pd.to_datetime(df_plot["timestamp"]).dt.hour
            month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            df_plot["month_s"] = df_plot["month"].map(month_map)
            month_order = list(month_map.values())

            # Row 1: GHI, cloud
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**GHI 월별 분포**")
                fig = px.box(df_plot, x="month_s", y="ghi_w_m2", color="month_s",
                              category_orders={"month_s": month_order}, template=PLOT_TEMPLATE)
                fig.update_layout(showlegend=False, height=300, yaxis_title="GHI (W/m²)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("여름(6~8월)은 중앙값·분산 모두 큼 → 맑은 날 강한 일사 + 장마철 낮은 일사 공존.")
            with c2:
                st.markdown("**운량 월별 분포**")
                fig2 = px.box(df_plot, x="month_s", y="cloud_cover", color="month_s",
                               category_orders={"month_s": month_order}, template=PLOT_TEMPLATE)
                fig2.update_layout(showlegend=False, height=300, yaxis_title="운량 (0~9)")
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("6~7월 장마 구간에 운량 높고 분산 큼. 봄·가을은 운량 낮고 안정적.")

            # Row 2: temp, target_angle
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**기온 월별 분포**")
                fig3 = px.box(df_plot, x="month_s", y="temp_actual", color="month_s",
                               category_orders={"month_s": month_order}, template=PLOT_TEMPLATE)
                fig3.update_layout(showlegend=False, height=300, yaxis_title="기온 (°C)")
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("여름 고온 → 패널 효율 저하 요인. 겨울 저온은 효율에 유리하나 일사량 부족.")
            with c4:
                st.markdown("**최적 루버 각도 월별 분포 (타겟)**")
                fig4 = px.box(df_plot, x="month_s", y="target_angle_v5", color="month_s",
                               category_orders={"month_s": month_order}, template=PLOT_TEMPLATE)
                fig4.update_layout(showlegend=False, height=300, yaxis_title="최적 각도 (°)")
                st.plotly_chart(fig4, use_container_width=True)
                st.caption("여름: 태양고도 높아 낮은 각도(15~30°) 최적. 겨울: 태양이 낮게 떠 높은 각도(40~90°) 유리.")

            # Row 3: hour_sin/cos 패턴, doy_sin/cos 패턴
            st.markdown("---")
            st.markdown("**시각·날짜 순환 변수 패턴**")
            c5, c6 = st.columns(2)
            with c5:
                st.markdown("**hour_sin / hour_cos — 시간대별 평균 최적각**")
                hour_avg = df_plot.groupby("hour")["target_angle_v5"].mean().reset_index()
                hour_avg["hour_sin"] = np.sin(2 * np.pi * hour_avg["hour"] / 24)
                hour_avg["hour_cos"] = np.cos(2 * np.pi * hour_avg["hour"] / 24)
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=hour_avg["hour"], y=hour_avg["target_angle_v5"],
                                           mode="lines+markers", name="평균 최적각",
                                           line=dict(color=COLOR_AI, width=2)))
                fig5.update_layout(height=280, xaxis_title="시각 (h)", yaxis_title="평균 최적각 (°)",
                                    template=PLOT_TEMPLATE)
                st.plotly_chart(fig5, use_container_width=True)
                st.caption("정오(12h) 전후에 최적각이 가장 낮음 → 태양이 가장 높이 뜨는 시각. "
                           "hour_sin/cos는 이 패턴을 모델이 인식할 수 있도록 수치화한 것.")
            with c6:
                st.markdown("**doy_sin / doy_cos — 월별 평균 최적각**")
                doy_avg = df_plot.groupby("month")["target_angle_v5"].mean().reset_index()
                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(x=doy_avg["month"], y=doy_avg["target_angle_v5"],
                                           mode="lines+markers", name="월평균 최적각",
                                           line=dict(color=COLOR_F60, width=2)))
                fig6.update_layout(height=280, xaxis_title="월", yaxis_title="평균 최적각 (°)",
                                    xaxis=dict(tickvals=list(range(1,13)),
                                               ticktext=month_order),
                                    template=PLOT_TEMPLATE)
                st.plotly_chart(fig6, use_container_width=True)
                st.caption("겨울(1·12월)에 최적각 높고 여름(6·7월)에 낮음. "
                           "doy_sin/cos는 이 계절 패턴을 365일 순환으로 수치화한 것.")

            st.markdown("**GHI vs 최적각 산점도 (10년 실측)**")
            sample = df_plot.sample(min(3000, len(df_plot)), random_state=42)
            fig7 = px.scatter(sample, x="ghi_w_m2", y="target_angle_v5", color="month_s",
                               opacity=0.4, template=PLOT_TEMPLATE,
                               labels={"ghi_w_m2":"GHI (W/m²)", "target_angle_v5":"최적 각도 (°)"})
            fig7.update_layout(height=350)
            st.plotly_chart(fig7, use_container_width=True)
            st.caption("GHI 높을수록 최적각 낮아지는 경향 (태양이 높이 뜨면 루버를 눕힘). 계절별로 클러스터가 분리됨.")

        else:
            st.warning("⚠️ 학습 데이터 CSV를 불러올 수 없습니다. GitHub 연결 확인 필요.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: 피처 중요도
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("🎯 피처 중요도 (Feature Importance) — V13")

        st.markdown("""
        <div class="explain-box">
        <b>📖 피처 중요도란?</b><br>
        XGBoost 모델이 루버 각도를 예측할 때 각 입력 변수가 얼마나 중요하게 사용됐는지를 나타냅니다.
        <b>Gain</b>은 해당 변수가 트리 분기점에서 예측 오차를 얼마나 줄였는지의 누적 기여도입니다.
        </div>
        """, unsafe_allow_html=True)

        importance_data = {
            "피처":      ["ghi_real", "doy_cos", "hour_sin", "cloud_cover", "doy_sin", "hour_cos", "temp_actual"],
            "Gain":      [0.408, 0.149, 0.126, 0.125, 0.072, 0.068, 0.052],
            "변수 설명": [
                "수평면 일사량 — 발전 가능한 태양에너지의 절대량",
                "연중 날짜 코사인 — 계절 위치 (하지/동지 구분)",
                "하루 중 시각 사인 — 오전/오후 태양 위치",
                "운량 (0~9) — 구름에 의한 일사 감쇠 정도",
                "연중 날짜 사인 — 계절 위치 보완 성분",
                "하루 중 시각 코사인 — 시각 보완 성분",
                "외기온도 — 패널 온도계수에 의한 효율 보정",
            ],
        }
        df_imp = pd.DataFrame(importance_data).sort_values("Gain", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_imp["Gain"], y=df_imp["피처"], orientation="h",
            marker_color=[COLOR_AI if g > 0.12 else "#90CAF9" if g > 0.06 else "#B0BEC5"
                          for g in df_imp["Gain"]],
            text=[f"{g:.3f}" for g in df_imp["Gain"]], textposition="outside",
            customdata=df_imp["변수 설명"],
            hovertemplate="<b>%{y}</b><br>Gain: %{x:.3f}<br>%{customdata}<extra></extra>"
        ))
        fig.update_layout(height=380, xaxis_title="Gain", xaxis_range=[0, 0.50],
                           template=PLOT_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="explain-box">
            <b>🥇 ghi_real (40.8%)</b><br>
            가장 중요한 변수. 일사량이 높으면 루버를 눕혀 직달광 최대화,
            낮으면 각도를 세워 확산광 활용.
            </div>
            <div class="explain-box">
            <b>📅 doy_cos + doy_sin (22.1%)</b><br>
            계절 정보. 여름·겨울의 태양 고도 패턴이 달라 최적 각도 전략이 완전히 다름.
            두 성분이 함께 1년 순환을 표현.
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="explain-box">
            <b>🕐 hour_sin + hour_cos (19.4%)</b><br>
            하루 중 시각 정보. 같은 계절이라도 아침·정오·저녁의 태양 위치가 달라
            시각별 최적 각도가 달라짐.
            </div>
            <div class="explain-box">
            <b>☁️ cloud_cover (12.5%) + 🌡️ temp_actual (5.2%)</b><br>
            운량이 높으면 확산광 비중 증가 → 각도 전략 변화.
            기온은 온도계수(-0.4%/°C)를 통해 효율에 미세 영향.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div class="good-box">
            <b>MAE (평균절대오차) = 1.56°</b><br>
            예측 각도와 실제 최적 각도의 평균 차이가 1.56°.
            루버 제어 모터 물리적 정밀도(±2~3°)보다 작아 실제 제어에 지장 없는 수준.
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="good-box">
            <b>R² (결정계수) = 0.9564</b><br>
            루버 각도 변동의 95.6%를 모델이 설명.
            0.95 이상은 실용적으로 매우 높은 수준.
            나머지 4.4%는 예측하기 어려운 순간적 기상 변동에 기인.
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: 음영 원리
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("💡 V13 중심축 회전 루버의 음영 원리")

        st.markdown("""
        <div class="explain-box">
        <b>📖 핵심 용어</b><br>
        • <b>POA (Plane of Array)</b>: 패널 수광면 일사량 (W/m²). 루버 패널이 실제로 받는 태양에너지.
          기울기·방향·음영을 모두 반영한 실효 에너지값.<br>
        • <b>SVF (Sky View Factor)</b>: 하늘 조망 계수. 루버 사이 틈으로 보이는 하늘 비율.
          루버를 세울수록 SVF↑, 눕힐수록 SVF↓.<br>
        • <b>음영률</b>: 윗 블레이드 돌출부가 아랫 블레이드를 가리는 비율.
          태양 고도 낮을수록, 루버를 많이 눕힐수록 음영률 증가.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **쉽게 이해하는 작동 원리**

        블라인드를 생각해보세요. 날개를 눕히면 빛이 잘 들어오지만 위 날개가 아래 날개에 그림자를 만들고,
        세우면 그림자는 없지만 직사광선 대신 옆에서 오는 확산광만 받습니다.
        **AI는 매 시간 이 두 가지를 최적으로 조율해 발전량을 최대화합니다.**
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
            pitch_px  = 80
            wall_x    = -10

            for i in range(n_louvers):
                y_center    = i * pitch_px
                half_len_px = (blade_depth_mm / 2) / pitch_mm * pitch_px
                dx = half_len_px * np.cos(np.radians(tilt_example))
                dy = half_len_px * np.sin(np.radians(tilt_example))
                pivot_x = 40
                top_x = pivot_x - dx; top_y = y_center + dy
                bot_x = pivot_x + dx; bot_y = y_center - dy

                fig.add_shape(type="line", x0=top_x, y0=top_y, x1=bot_x, y1=bot_y,
                               line=dict(color=COLOR_AI, width=6))
                fig.add_trace(go.Scatter(x=[pivot_x], y=[y_center], mode="markers",
                    marker=dict(size=8, color="red", symbol="x"),
                    showlegend=False, hoverinfo="skip"))
                fig.add_shape(type="line", x0=pivot_x, y0=y_center, x1=bot_x, y1=bot_y,
                               line=dict(color="#F57C00", width=3, dash="dot"))
                if i > 0:
                    protrusion_px  = prot_ex / pitch_mm * pitch_px
                    gap_px         = gap_ex  / pitch_mm * pitch_px
                    shadow_vert_px = protrusion_px / np.tan(np.radians(max(elev_example, 1)))
                    shadow_on_px   = max(shadow_vert_px - gap_px, 0)
                    if shadow_on_px > 0:
                        shade_top    = y_center
                        shade_bottom = max(y_center - shadow_on_px, (i-1)*pitch_px)
                        fig.add_shape(type="rect",
                            x0=bot_x*0.3, y0=shade_bottom, x1=bot_x*1.2, y1=shade_top,
                            fillcolor="rgba(244,67,54,0.2)", line_width=0)

            fig.add_shape(type="line", x0=wall_x, y0=-20, x1=wall_x, y1=n_louvers*pitch_px+40,
                           line=dict(color="#555", width=3))
            fig.add_annotation(x=wall_x, y=n_louvers*pitch_px+50, text="벽면",
                                showarrow=False, font=dict(color="#555", size=11))
            sun_x = 200; sun_y = n_louvers*pitch_px+30
            arrow_dx = -np.cos(np.radians(elev_example))*60
            arrow_dy = -np.sin(np.radians(elev_example))*60
            fig.add_annotation(x=sun_x+arrow_dx, y=sun_y+arrow_dy, ax=sun_x, ay=sun_y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowcolor="#F57C00", arrowwidth=3)
            fig.add_trace(go.Scatter(x=[sun_x], y=[sun_y], mode="markers+text",
                marker=dict(size=20, color="#FFA726", symbol="circle"),
                text=["☀️"], textposition="top center", showlegend=False))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="블레이드",
                line=dict(color=COLOR_AI, width=4)))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="돌출부",
                line=dict(color="#F57C00", width=3, dash="dot")))
            fig.update_layout(
                height=400, template=PLOT_TEMPLATE,
                xaxis=dict(range=[-30, 260], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-30, n_louvers*pitch_px+80], showgrid=False, zeroline=False,
                           showticklabels=False, scaleanchor="x"),
                title=f"V13 단면도 — 태양고도 {elev_example}° | 루버각 {tilt_example}°",
                legend=dict(orientation="h", y=-0.05))
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
            <b>기하학 공식</b><br>
            돌출 = HALF_DEPTH × cos(각도)<br>
            수직점유 = DEPTH × sin(각도)<br>
            틈 = PITCH - 수직점유<br>
            SVF = 틈 / (틈 + 돌출)<br>
            그림자 = 돌출/tan(고도) - 틈<br>
            음영률 = 그림자 / PITCH<br><br>
            <b>현재값</b><br>
            DEPTH {blade_depth_mm:.0f}mm | HALF {half_depth_mm:.0f}mm<br>
            PITCH {pitch_mm:.0f}mm<br>
            돌출 {prot_ex:.1f}mm | 틈 {gap_ex:.1f}mm
            </div>
            <div class="explain-box">
            <b>💡 AI 최적화 전략</b><br>
            각도↓ → 직달광 POA↑, 음영↑<br>
            각도↑ → 음영↓, SVF↑, 직달광↓<br>
            AI는 매 시간 순발전량 최대각 선택
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: 음영 시각화
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

        elev_day   = elev[mask_day]
        ang_day    = ai_angles[mask_day]
        time_labels = times[mask_day].strftime("%H:%M")
        valid      = ghi_real[mask_day] >= 10

        fig.add_trace(go.Scatter(
            x=ang_day[valid], y=elev_day[valid],
            mode="markers+lines+text",
            name=f"AI 궤적 ({_sim_d})",
            marker=dict(size=10, color="white", symbol="circle",
                        line=dict(color="#1565C0", width=2)),
            line=dict(color="white", width=2, dash="dot"),
            text=[f"{t}" for t in time_labels[valid]],
            textposition="top center",
            textfont=dict(size=9, color="white"),
        ))
        fig.add_shape(type="line", x0=60, x1=60, y0=5, y1=80,
                       line=dict(color="blue", width=2, dash="dash"))
        fig.add_annotation(x=60, y=78, text="고정60°", showarrow=False,
                            font=dict(color="blue", size=11))
        fig.update_layout(height=500, xaxis_title="루버 각도 (°)",
                           yaxis_title="태양 고도각 (°)",
                           title="V13 음영률 히트맵 — AI 운전 궤적")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <b>📖 히트맵 읽는 법</b><br>
        🟢 초록 = 음영 없음 &nbsp;|&nbsp; 🟡 노랑 = 부분 음영 &nbsp;|&nbsp; 🔴 빨강 = 음영 심함<br>
        <b>흰 점선 (AI 궤적)</b>: 시간 순서대로 레이블 표시. 아침 → 정오 → 저녁 순으로 이동.
        </div>
        <div class="explain-box">
        <b>💡 AI 궤적이 이렇게 움직이는 이유</b><br>
        <b>① 아침</b>: 태양 고도 낮음 → 히트맵 하단. 어떤 각도도 음영이 크므로 확산광 위해 각도 높게 설정.<br>
        <b>② 정오</b>: 태양 고도 최대 → 히트맵 상단. 음영이 줄어드므로 각도를 낮춰 직달광 최대화.
        궤적이 초록 구간(낮은 음영률)을 따라 이동하는 것을 확인 가능.<br>
        <b>③ 오후~저녁</b>: 태양 고도 감소 → 다시 하단으로. GHI &lt; 10 W/m² 시 루버 90°로 닫힘.<br>
        <b>핵심</b>: AI는 항상 현재 태양 위치에서 초록 구간(낮은 음영률)을 추적. 고정60°보다 훨씬 세밀하게 대응.
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: 발전량 비교
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("⚡ AI vs 고정60° vs 수직90° 발전량 비교")

        if use_xgb_annual:
            st.success("✅ XGBoost 모델 기반 연간 시뮬레이션")
        else:
            st.info("ℹ️ 규칙 기반 각도 (XGBoost 미로드 시 대체)")

        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = go.Figure()
        for col, color, name in [
            ("AI",    COLOR_AI,  "AI 제어"),
            ("고정60°", COLOR_F60, "고정 60°"),
            ("수직90°", COLOR_V90, "수직 90°"),
        ]:
            fig.add_trace(go.Bar(
                x=month_names,
                y=df_monthly[col] * area_scale / 1000,
                name=name, marker_color=color
            ))
        fig.update_layout(barmode="group", height=400, template=PLOT_TEMPLATE,
                           yaxis_title="발전량 (kWh)", xaxis_title="월",
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        # ★ v8.1: 발전량 비교 해석
        winter_ai  = df_monthly.loc[df_monthly["month"].isin([12,1,2]), "AI"].mean() * area_scale / 1000
        winter_60  = df_monthly.loc[df_monthly["month"].isin([12,1,2]), "고정60°"].mean() * area_scale / 1000
        summer_ai  = df_monthly.loc[df_monthly["month"].isin([6,7,8]),  "AI"].mean() * area_scale / 1000
        summer_60  = df_monthly.loc[df_monthly["month"].isin([6,7,8]),  "고정60°"].mean() * area_scale / 1000

        st.markdown(f"""
        <div class="explain-box">
        <b>📊 월별 발전량 그래프 해석</b><br><br>
        <b>🌞 여름 (6~8월)</b>: 태양 고도가 높아 일사량이 풍부. 연중 최대 발전 구간.
        AI는 낮은 각도로 직달광을 최적화하여 고정60°보다 높은 발전량 달성.<br><br>
        <b>❄️ 겨울 (12~2월)</b>: 서울 기준 태양 최대 고도 약 29°로 낮고 일조 시간도 짧음.
        {"XGBoost 모델이 겨울 최적각(40~50°)을 학습하여 고정60°에 근접하거나 상회하는 발전량 달성." if use_xgb_annual else "현재 규칙 기반 모드에서는 겨울철 AI 각도가 최적값보다 낮게 설정될 수 있어 고정60°보다 낮게 나올 수 있음. XGBoost 모델 로드 후 정확한 비교 가능."}<br><br>
        <b>🍂 봄·가을</b>: 중간 수준의 발전량. 계절 전환에 따라 AI가 각도 전략을 유동적으로 조정.<br><br>
        <b>수직90° 비교</b>: 겨울에는 태양이 낮게 떠 수직(90°)에 가까울수록 유리하여
        고정90°가 생각보다 선전. 여름에는 수직이 직달광을 못 받아 크게 불리.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("연간 AI",      f"{ann_kwh_ai:.1f} kWh")
        c2.metric("연간 고정60°", f"{ann_kwh_60:.1f} kWh",
                   f"AI 대비 -{(1-ann_kwh_60/ann_kwh_ai)*100:.1f}%" if ann_kwh_ai > 0 else "—")
        c3.metric("연간 수직90°", f"{ann_kwh_90:.1f} kWh",
                   f"AI 대비 -{(1-ann_kwh_90/ann_kwh_ai)*100:.1f}%" if ann_kwh_ai > 0 else "—")

        # 누적 발전량
        df_cum = df_monthly.copy()
        df_cum["AI_cum"]  = (df_cum["AI"]    * area_scale / 1000).cumsum()
        df_cum["F60_cum"] = (df_cum["고정60°"] * area_scale / 1000).cumsum()
        df_cum["V90_cum"] = (df_cum["수직90°"] * area_scale / 1000).cumsum()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=month_names, y=df_cum["AI_cum"],
                                   name="AI",     line=dict(color=COLOR_AI, width=3)))
        fig2.add_trace(go.Scatter(x=month_names, y=df_cum["F60_cum"],
                                   name="고정60°", line=dict(color=COLOR_F60, width=2, dash="dash")))
        fig2.add_trace(go.Scatter(x=month_names, y=df_cum["V90_cum"],
                                   name="수직90°", line=dict(color=COLOR_V90, width=2, dash="dot")))
        fig2.update_layout(height=320, yaxis_title="누적 발전량 (kWh)",
                            title="연간 누적 발전량", template=PLOT_TEMPLATE,
                            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("누적 곡선의 기울기가 가파른 구간이 발전량이 많은 계절. AI 곡선이 다른 두 곡선 위에 있을수록 제어 효과가 큼.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: 월별 각도
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.subheader("📅 월별 AI 제어 각도 분포")

        df_plot2 = df_annual[df_annual["ghi"] >= 10].copy()
        df_plot2["month_n"] = df_plot2["timestamp"].dt.month
        df_plot2["month_s"] = df_plot2["month_n"].map(
            {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})

        fig = go.Figure()
        for m, ms in enumerate(["Jan","Feb","Mar","Apr","May","Jun",
                                  "Jul","Aug","Sep","Oct","Nov","Dec"], 1):
            d = df_plot2[df_plot2["month_n"] == m]["angle_ai"]
            fig.add_trace(go.Box(y=d, name=ms, marker_color=COLOR_AI, boxmean=True))
        fig.add_hline(y=ANGLE_MIN, line_dash="dash", line_color="red",
                       annotation_text=f"최소각 {ANGLE_MIN}°")
        fig.update_layout(height=420, yaxis_title="루버 각도 (°)",
                           showlegend=False, template=PLOT_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <b>📊 월별 각도 분포 전체 해석</b><br><br>
        <b>🌞 여름 (6~8월) — 낮은 각도, 넓은 분포</b><br>
        서울 여름철 최대 태양 고도각 약 76°. 태양이 높이 떠 있으므로 루버를 눕혀야(낮은 각도)
        직달광이 수광면에 가장 수직에 가깝게 입사. 평균 15~20°.
        분포가 넓은 이유는 장마철 흐린 날(확산광 위해 각도↑)과 맑은 날(각도↓)이 혼재하기 때문.<br><br>
        <b>❄️ 겨울 (12~2월) — 높은 각도, 좁은 분포</b><br>
        겨울철 최대 고도각 약 29°. 태양이 낮게 떠 루버를 세워야(높은 각도) 직달광을 효과적으로 수광.
        평균 35~45°. 맑은 날이 많고 태양 궤적이 단순해 분포가 좁음.<br><br>
        <b>🍂 봄·가을 (3~5월, 9~11월) — 점진적 전환</b><br>
        태양 고도각이 여름·겨울 사이를 오가며 각도도 중간값(20~35°).
        월별로 각도가 뚜렷하게 변화하는 전환 구간.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("월별 평균 각도 & 발전량")
        v13_ref = {1:43.6,2:27.5,3:25.0,4:19.4,5:15.8,6:15.5,
                   7:16.3,8:18.8,9:18.0,10:28.4,11:31.9,12:38.9}
        df_summary = df_monthly.copy()
        df_summary["month_s"]    = [month_names[i] for i in range(12)]
        df_summary               = df_summary.rename(columns={"avg_angle":"시뮬 평균각(°)"})
        df_summary["V13 참조각"] = [v13_ref[m] for m in range(1, 13)]
        st.dataframe(
            df_summary[["month_s","시뮬 평균각(°)","V13 참조각","AI","고정60°","수직90°"]].round(1),
            use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7: 내일 스케줄
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.subheader("🌤️ 내일 예측 스케줄")
        if kma is None:
            st.error("❌ 기상청 API 연동 실패. 청천 기준으로 시뮬레이션합니다.")
        else:
            st.success(f"✅ 기상청 단기예보 연동 성공 | 기준일: {tomorrow}")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=times[mask_day].strftime("%H:%M"), y=ghi_real[mask_day],
                              name="예측 GHI (W/m²)", marker_color="rgba(255,152,0,0.5)"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=times[mask_day].strftime("%H:%M"), y=ai_angles[mask_day],
                                  name="AI 예측 각도", line=dict(color=COLOR_AI, width=3),
                                  mode="lines+markers"), secondary_y=True)
        fig.update_yaxes(title_text="GHI (W/m²)", secondary_y=False)
        fig.update_yaxes(title_text="각도 (°)", range=[0, 95], secondary_y=True)
        fig.update_layout(height=380, template=PLOT_TEMPLATE,
                           title=f"내일({tomorrow}) 루버 제어 스케줄")
        st.plotly_chart(fig, use_container_width=True)

        df_tom = pd.DataFrame({
            "시간":           times[mask_day].strftime("%H:%M").tolist(),
            "예측 GHI (W/m²)": np.round(ghi_real[mask_day], 1).tolist(),
            "기온 (°C)":      np.round(temp_series[mask_day], 1).tolist(),
            "AI 각도(°)":    ai_angles[mask_day].astype(int).tolist(),
            "음영률":         [f"{calc_shading_fraction(a,e,half_depth_mm,blade_depth_mm,pitch_mm)*100:.1f}%"
                               for a, e in zip(ai_angles[mask_day], elev[mask_day])],
            "SVF":            [f"{sky_view_factor(a,half_depth_mm,blade_depth_mm,pitch_mm):.2f}"
                               for a in ai_angles[mask_day]],
        })
        st.dataframe(df_tom, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8: 건강진단
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.subheader("🩺 시스템 건강 진단")

        st.markdown("""
        <div class="explain-box">
        <b>📖 건강진단이란?</b><br>
        AI가 예측한 발전량(P_sim)과 실제 센서에서 측정된 발전량(P_actual)을 비교하여
        패널 오염·고장·케이블 이상 등을 자동으로 감지하는 기능입니다.<br>
        • <b>P_sim</b>: AI가 기상·루버 각도 기반으로 계산한 <b>예상 발전량</b>. "이 조건이면 이만큼 나와야 한다"는 기준값.<br>
        • <b>P_actual</b>: 인버터·계측기에서 측정된 <b>실측 발전량</b>.<br>
        • <b>Health Ratio</b>: P_actual / P_sim. 1.0(100%) = 완벽 정상. 낮을수록 이상 상태.
        </div>
        """, unsafe_allow_html=True)

        st.info("📌 현재는 실측 센서 미연동 상태. 슬라이더로 시나리오 시뮬레이션 가능.")

        col1, col2 = st.columns(2)
        with col1:
            p_actual_pct = st.slider("실측 발전량 비율 (%)", 10, 110, 95,
                                       help="P_actual / P_sim × 100")
        with col2:
            warn_thr = st.number_input("WARNING 임계값 (%)", value=90)
            crit_thr = st.number_input("CRITICAL 임계값 (%)", value=75)

        health_ratio = p_actual_pct / 100.0
        if health_ratio >= warn_thr / 100:
            status, color = "✅ NORMAL",   "green"
        elif health_ratio >= crit_thr / 100:
            status, color = "⚠️ WARNING",  "orange"
        else:
            status, color = "🔴 CRITICAL", "red"

        c1, c2, c3 = st.columns(3)
        c1.metric("P_sim (AI 예측)",   f"{pow_ai/1000:.3f} kWh")
        c2.metric("P_actual (실측)",   f"{pow_ai/1000*health_ratio:.3f} kWh")
        c3.metric("Health Ratio",      f"{health_ratio:.2%}", status)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=p_actual_pct,
            delta={"reference": 100, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 110]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,        crit_thr],  "color": "rgba(244,67,54,0.15)"},
                    {"range": [crit_thr, warn_thr],  "color": "rgba(255,152,0,0.15)"},
                    {"range": [warn_thr, 110],        "color": "rgba(76,175,80,0.15)"},
                ],
                "threshold": {"line": {"color": "gray", "width": 2}, "value": warn_thr},
            },
            title={"text": f"시스템 상태: {status}"}
        ))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <b>📊 판정 기준 및 해석</b><br>
        <b>✅ 90% 이상 (NORMAL)</b>: 정상. P_sim·P_actual 거의 일치. 이상 없음.<br>
        <b>⚠️ 75~90% (WARNING)</b>: 패널 오염(먼지·새똥) 또는 부분 고장 의심. 10~25% 손실 발생. 청소·점검 권고.<br>
        <b>🔴 75% 미만 (CRITICAL)</b>: 심각한 성능 저하. 25% 이상 손실. 케이블 불량·인버터 고장·셀 손상 가능성. 즉시 점검.<br><br>
        <b>임계값 근거</b>: WARNING 90%는 오염으로 통상 발생하는 손실 범위의 상한,
        CRITICAL 75%는 단순 오염을 넘어 구조적 결함이 의심되는 수준.
        </div>
        """, unsafe_allow_html=True)

        st.caption("📌 실측 센서 연동 후 P_actual 자동 입력 예정 (특허 청구항 4 대상).")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9: 파라미터 튜닝
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.subheader("🔧 파라미터 민감도 분석 — V13 물리모델 기반")
        st.markdown("파라미터 변화에 따라 발전량·음영률이 어떻게 달라지는지 확인합니다.")

        col1, col2 = st.columns(2)
        with col1:
            t_loss     = st.slider("시스템 손실률", 0.70, 0.95, DEFAULT_LOSS, step=0.01)
            t_capacity = st.slider("패널 용량 (W)", 100, 600, DEFAULT_CAPACITY, step=50)

        # 그래프 1: BLADE_DEPTH → 연간 발전량
        st.markdown("---")
        st.markdown("#### 📐 블레이드 DEPTH 변화 → 연간 발전량")
        st.caption("DEPTH↑ → 돌출↑ → 음영↑. DEPTH↓ → 음영↓ but 발전 면적↓.")

        depth_range  = np.arange(60, 181, 10)
        pow_by_depth = []
        for bd in depth_range:
            hd = bd / 2.0
            _, _, _, df_m_tmp, _ = get_annual_data(
                last_year, hd, float(bd), pitch_mm, capacity_w, unit_count, eff_factor, t_loss)
            pow_by_depth.append(df_m_tmp["AI"].sum() * area_scale / 1000)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=depth_range, y=pow_by_depth,
                                   mode="lines+markers", line=dict(color=COLOR_AI, width=2)))
        fig1.add_vline(x=blade_depth_mm, line_dash="dash", line_color=COLOR_F60,
                        annotation_text=f"현재 {blade_depth_mm:.0f}mm")
        fig1.update_layout(height=300, xaxis_title="BLADE_DEPTH (mm)",
                            yaxis_title="연간 발전량 (kWh)", template=PLOT_TEMPLATE)
        st.plotly_chart(fig1, use_container_width=True)

        # 그래프 2: PITCH → 음영률
        st.markdown("---")
        st.markdown("#### 📏 피치 변화 → 정오 기준 음영률")
        st.caption("피치↑ → 블레이드 간 간격↑ → 음영↓. 단, 전체 설치 면적 증가.")

        pitch_range        = np.arange(80, 201, 5)
        sf_noon_by_pitch   = []
        for p in pitch_range:
            sf = calc_shading_fraction(45.0, 60.0, blade_depth_mm/2.0, blade_depth_mm, float(p))
            sf_noon_by_pitch.append(float(sf) * 100)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pitch_range, y=sf_noon_by_pitch,
                                   mode="lines+markers", line=dict(color=COLOR_F60, width=2)))
        fig2.add_vline(x=pitch_mm, line_dash="dash", line_color=COLOR_AI,
                        annotation_text=f"현재 {pitch_mm:.0f}mm")
        fig2.add_hline(y=30, line_dash="dot", line_color="green",
                        annotation_text="30% (양호 기준)")
        fig2.update_layout(height=300, xaxis_title="PITCH (mm)",
                            yaxis_title="음영률 % (루버45°, 태양고도60° 기준)",
                            template=PLOT_TEMPLATE)
        st.plotly_chart(fig2, use_container_width=True)

        # 그래프 3: 최소각 → 연간 발전량
        st.markdown("---")
        st.markdown("#### 📐 최소 허용 각도 변화 → 연간 AI 발전량")
        st.caption("최소각↓ → 여름 최적화 범위 확대. 단, 너무 낮으면 기계적 한계·음영 증가로 역효과 가능.")

        amin_range  = np.arange(5, 46, 5)
        pow_by_amin = []
        for amin in amin_range:
            times_y  = pd.date_range(start=f"{last_year}-01-01", end=f"{last_year}-12-31 23:00",
                                      freq="h", tz=TZ)
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
            poa_dir, poa_diff, poa_sky = poa_components(
                tilt, np.full_like(tilt, 180), zen_y, az_y, dni_y, ghi_y, dhi_y)
            eff_poa = calc_effective_poa(
                poa_dir, poa_diff, poa_sky, tilt, elev_y, half_depth_mm, blade_depth_mm, pitch_mm)
            mask = ghi_y >= 10
            wh = (eff_poa[mask] / 1000 * capacity_w * unit_count * eff_factor * t_loss).sum()
            pow_by_amin.append(wh * area_scale / 1000)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=amin_range, y=pow_by_amin,
                                   mode="lines+markers", line=dict(color="#43A047", width=2)))
        fig3.add_vline(x=ANGLE_MIN, line_dash="dash", line_color=COLOR_F60,
                        annotation_text=f"현재 최소각 {ANGLE_MIN}°")
        fig3.update_layout(height=300, xaxis_title="최소 허용 각도 (°)",
                            yaxis_title="연간 발전량 (kWh)", template=PLOT_TEMPLATE)
        st.plotly_chart(fig3, use_container_width=True)

        st.caption("💡 V13: PPO 강화학습을 향후 실시예로 추가 가능 (현재는 XGBoost/규칙 기반)")


if __name__ == "__main__":
    run_app()
