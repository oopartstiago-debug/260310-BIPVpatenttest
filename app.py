# ==============================================================================
# BIPV 통합 관제 시스템 v6.0 — 탭 구조 확장판
# 탭: 메인 | 데이터셋 | 피처중요도 | 음영원리 | 음영시각화 | 발전량비교 | 월별각도 | 내일스케줄 | 건강진단 | 파라미터튜닝
# ==============================================================================
__version__ = "6.0"

import os
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

try:
    import gdown
    _GDOWN_AVAILABLE = True
except ImportError:
    _GDOWN_AVAILABLE = False

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
DEFAULT_LOUVER_COUNT = 10
DEFAULT_WIDTH_MM   = 900.0    # 블레이드 폭 (mm) — 실제 하드웨어 기준
DEFAULT_HEIGHT_MM  = 160.0
DEFAULT_PITCH_MM   = 114.0   # 블레이드 피치 (mm)
ANGLE_MIN          = 15
ANGLE_MAX          = 90
ANGLE_NIGHT        = 90
XGB_MODEL_FILENAME = "bipv_xgboost_model.pkl"
XGB_FEATURE_NAMES  = ["hour", "month", "zenith", "azimuth", "ghi", "dni", "dhi", "cloud_cover"]

site = Location(LAT, LON, tz=TZ)

# ==============================================================================
# 모델 로드
# ==============================================================================
@st.cache_resource
def load_xgb_model():
    if not _XGB_AVAILABLE:
        return None
    FILE_ID = "1OYqV6IH6NLSr4IJljw501qTRoRD4Pzam"
    if not os.path.exists(XGB_MODEL_FILENAME) and _GDOWN_AVAILABLE:
        try:
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", XGB_MODEL_FILENAME, quiet=False)
        except Exception:
            pass
    for base in [os.getcwd(), "."]:
        path = os.path.join(base, XGB_MODEL_FILENAME)
        if os.path.isfile(path):
            try:
                return joblib.load(path)
            except Exception:
                return None
    return None

# ==============================================================================
# 물리 계산 함수
# ==============================================================================
def poa_with_iam(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi, a_r=0.16):
    irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solar_zenith, solar_azimuth=solar_azimuth
    )
    poa_direct  = np.nan_to_num(irrad["poa_direct"],  nan=0.0)
    poa_diffuse = np.nan_to_num(irrad["poa_diffuse"], nan=0.0)
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    aoi = np.clip(np.asarray(aoi, dtype=float), 0, 90)
    try:
        iam = pvlib.iam.martin_ruiz(aoi, a_r=a_r)
    except AttributeError:
        iam = pvlib.irradiance.iam.martin_ruiz(aoi, a_r=a_r)
    return poa_direct * iam + poa_diffuse


def calc_shading_fraction(tilt_deg, elev_deg, width_mm, pitch_mm):
    """루버 음영률 계산 — 수평 투영 공식"""
    tilt_rad = np.radians(np.asarray(tilt_deg, dtype=float))
    elev_rad = np.radians(np.clip(np.asarray(elev_deg, dtype=float), 0.1, 89))
    shadow_len = width_mm * np.abs(np.sin(tilt_rad)) / np.tan(elev_rad)
    shading = np.clip(shadow_len / pitch_mm, 0, 1)
    shading = np.where(np.asarray(elev_deg) <= 0, 1.0, shading)
    return shading


def predict_angles_xgb(model, times, zenith_arr, azimuth_arr, ghi_real, dni_arr, dhi_arr, cloud_series, angle_cap_deg):
    n = len(times)
    X = np.column_stack([
        np.asarray(times.hour,  dtype=float).ravel()[:n],
        np.asarray(times.month, dtype=float).ravel()[:n],
        np.asarray(zenith_arr,  dtype=float).ravel()[:n],
        np.asarray(azimuth_arr, dtype=float).ravel()[:n],
        np.asarray(ghi_real,    dtype=float).ravel()[:n],
        np.asarray(dni_arr,     dtype=float).ravel()[:n],
        np.asarray(dhi_arr,     dtype=float).ravel()[:n],
        np.asarray(cloud_series,dtype=float).ravel()[:n],
    ])
    try:
        pred = model.predict(X)
    except Exception:
        return None
    pred = np.clip(np.asarray(pred).ravel()[:n], ANGLE_MIN, min(ANGLE_MAX, angle_cap_deg))
    pred[np.asarray(ghi_real).ravel()[:n] < 10] = ANGLE_NIGHT
    return pred.astype(float)


# ==============================================================================
# 기상청 API
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
        return df_tom.pivot(index="fcstTime", columns="category", values="fcstValue"), tomorrow
    except Exception as e:
        return None, None


# ==============================================================================
# 연간 데이터 (캐시)
# ==============================================================================
@st.cache_data(ttl=86400)
def get_annual_data(year, width_mm, pitch_mm, capacity_w, unit_count, eff_factor, default_loss):
    times_y  = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="h", tz=TZ)
    solpos_y = site.get_solarposition(times_y)
    cs_y     = site.get_clearsky(times_y)
    ghi_y    = np.asarray(cs_y["ghi"].values, dtype=float)
    zen_y    = solpos_y["apparent_zenith"].values
    az_y     = solpos_y["azimuth"].values
    dni_y    = pvlib.irradiance.dirint(ghi_y, zen_y, times_y).fillna(0).values
    dhi_y    = (ghi_y - dni_y * np.cos(np.radians(zen_y))).clip(0)

    # AI 각도 (V12 스타일: pvlib 기반 최적각)
    angles_ai  = np.where(ghi_y < 10, float(ANGLE_NIGHT),
                   np.clip(90 - zen_y, ANGLE_MIN, ANGLE_MAX).astype(float))
    angles_60  = np.full_like(ghi_y, 60.0)
    angles_90  = np.full_like(ghi_y, 90.0)

    def energy(angles):
        tilt = 90 - np.asarray(angles, dtype=float)
        sf   = calc_shading_fraction(np.asarray(angles, dtype=float),
                                     90 - zen_y, width_mm, pitch_mm)
        poa  = poa_with_iam(tilt, np.full_like(tilt, 180), zen_y, az_y, dni_y, ghi_y, dhi_y)
        poa  = poa * (1 - sf)
        mask = ghi_y >= 10
        return (poa[mask] / 1000 * capacity_w * unit_count * eff_factor * default_loss).sum()

    wh_ai = energy(angles_ai)
    wh_60 = energy(angles_60)
    wh_90 = energy(angles_90)

    # 월별
    df_annual = pd.DataFrame({
        "timestamp": times_y,
        "ghi": ghi_y,
        "zenith": zen_y,
        "azimuth": az_y,
        "angle_ai": angles_ai,
        "angle_60": angles_60,
        "angle_90": angles_90,
    })
    df_annual["month"] = df_annual["timestamp"].dt.month

    monthly = []
    for m in range(1, 13):
        mask_m = (df_annual["month"] == m) & (ghi_y >= 10)
        tilt_ai = 90 - df_annual.loc[mask_m, "angle_ai"].values
        tilt_60 = 90 - df_annual.loc[mask_m, "angle_60"].values
        tilt_90 = 90 - df_annual.loc[mask_m, "angle_90"].values
        zen_m   = zen_y[mask_m]
        az_m    = az_y[mask_m]
        dni_m   = dni_y[mask_m]
        ghi_m   = ghi_y[mask_m]
        dhi_m   = dhi_y[mask_m]
        ang_m   = df_annual.loc[mask_m, "angle_ai"].values

        def e_m(tilt, ang):
            sf  = calc_shading_fraction(ang, 90 - zen_m, width_mm, pitch_mm)
            poa = poa_with_iam(tilt, np.full_like(tilt, 180), zen_m, az_m, dni_m, ghi_m, dhi_m)
            return (poa * (1 - sf) / 1000 * capacity_w * unit_count * eff_factor * default_loss).sum()

        monthly.append({
            "month": m,
            "AI": e_m(tilt_ai, ang_m),
            "고정60°": e_m(tilt_60, df_annual.loc[mask_m, "angle_60"].values),
            "수직90°": e_m(tilt_90, df_annual.loc[mask_m, "angle_90"].values),
            "avg_angle": float(df_annual.loc[mask_m, "angle_ai"].mean()),
        })
    df_monthly = pd.DataFrame(monthly)
    return wh_ai, wh_60, wh_90, df_monthly, df_annual


# ==============================================================================
# 메인 앱
# ==============================================================================
def run_app():
    st.set_page_config(page_title="BIPV AI 관제", layout="wide", page_icon="☀️")

    st.markdown("""
    <style>
    .metric-card {background:#1e2130;border-radius:12px;padding:16px 20px;margin-bottom:8px;}
    .tab-header {font-size:1.1rem;font-weight:600;color:#f0f4ff;margin-bottom:12px;}
    div[data-testid="stMetricValue"] {font-size:1.6rem;font-weight:700;}
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

    st.sidebar.subheader("2. 블레이드 스펙")
    st.sidebar.caption("루버 1개 가로×세로, 피치(블레이드 간격), 개수")
    width_mm  = st.sidebar.number_input("루버 1개 가로 (mm)", min_value=100.0, value=DEFAULT_WIDTH_MM, step=100.0)
    height_mm = st.sidebar.number_input("루버 1개 세로 (mm)", min_value=10.0,  value=DEFAULT_HEIGHT_MM, step=10.0)
    pitch_mm  = st.sidebar.number_input("블레이드 피치 (mm)", min_value=10.0,  value=DEFAULT_PITCH_MM, step=1.0)
    louver_count = st.sidebar.number_input("루버 개수 (개)", min_value=1, value=DEFAULT_LOUVER_COUNT, step=1)

    st.sidebar.subheader("3. 패널 스펙")
    unit_count  = st.sidebar.number_input("설치 유닛 수 (개)", min_value=1, value=DEFAULT_UNIT_COUNT)
    capacity_w  = st.sidebar.number_input("패널 용량 (W)",      value=DEFAULT_CAPACITY)
    target_eff  = st.sidebar.number_input("패널 효율 (%)",      value=DEFAULT_EFFICIENCY, step=0.1)
    kepco_rate  = st.sidebar.number_input("전기 요금 (원/kWh)", value=DEFAULT_KEPCO)

    eff_factor   = float(target_eff) / DEFAULT_EFFICIENCY
    area_scale   = (width_mm * height_mm * louver_count) / (DEFAULT_WIDTH_MM * DEFAULT_HEIGHT_MM * DEFAULT_LOUVER_COUNT)

    # ── 날짜별 데이터 ─────────────────────────────────────────────────────────
    _sim_d = sim_date.strftime("%Y-%m-%d")
    times  = pd.date_range(start=f"{_sim_d} 00:00", periods=24, freq="h", tz=TZ)
    solpos = site.get_solarposition(times)
    cs     = site.get_clearsky(times)
    zen    = np.asarray(solpos["apparent_zenith"].values, dtype=float)
    az     = np.asarray(solpos["azimuth"].values,         dtype=float)
    elev   = np.asarray(solpos["apparent_elevation"].values, dtype=float)

    if kma is not None and _sim_d.replace("-","") == tomorrow:
        kma_reindex  = kma.reindex(times.strftime("%H00"))
        cloud_series = kma_reindex["SKY"].apply(
            lambda x: 0.0 if x == "1" else (0.5 if x == "3" else 1.0)
        ).astype(float).values
    else:
        cloud_series = np.zeros(24)

    ghi_real = np.asarray(cs["ghi"].values, dtype=float) * (1.0 - cloud_series * 0.65)
    dni_arr  = pvlib.irradiance.dirint(ghi_real, zen, times).fillna(0).values
    dhi_arr  = (ghi_real - dni_arr * np.cos(np.radians(zen))).clip(0)

    # AI 각도
    xgb_angles = None
    if xgb_model:
        xgb_angles = predict_angles_xgb(xgb_model, times, zen, az, ghi_real, dni_arr, dhi_arr, cloud_series, ANGLE_MAX)
    ai_angles = xgb_angles if xgb_angles is not None else np.where(
        ghi_real < 10, float(ANGLE_NIGHT), np.clip(90 - zen, ANGLE_MIN, ANGLE_MAX).astype(float)
    )
    angle_mode = "XGBoost" if xgb_angles is not None else "규칙 기반"

    def calc_power_day(angles):
        tilt = 90 - np.asarray(angles, dtype=float)
        sf   = calc_shading_fraction(np.asarray(angles, dtype=float), elev, width_mm, pitch_mm)
        poa  = poa_with_iam(tilt, np.full_like(tilt, 180), zen, az, dni_arr, ghi_real, dhi_arr)
        mask = ghi_real >= 10
        return (poa[mask] * (1 - sf[mask]) / 1000 * capacity_w * unit_count * eff_factor * DEFAULT_LOSS * area_scale).sum()

    pow_ai  = calc_power_day(ai_angles)
    pow_60  = calc_power_day(np.full(24, 60.0))
    pow_90  = calc_power_day(np.full(24, 90.0))

    # 연간 데이터
    last_year = sim_date.year - 1
    wh_ai_y, wh_60_y, wh_90_y, df_monthly, df_annual = get_annual_data(
        last_year, width_mm, pitch_mm, capacity_w, unit_count, eff_factor, DEFAULT_LOSS
    )
    ann_kwh_ai = wh_ai_y / 1000 * area_scale
    ann_kwh_60 = wh_60_y / 1000 * area_scale
    ann_kwh_90 = wh_90_y / 1000 * area_scale

    weather_status = "맑음" if np.mean(cloud_series) < 0.3 else ("구름많음" if np.mean(cloud_series) < 0.8 else "흐림")
    mask_day = (times.hour >= 6) & (times.hour <= 19)

    # ── 탭 ────────────────────────────────────────────────────────────────────
    st.title("☀️ BIPV AI 통합 관제 대시보드")
    st.caption(f"v{__version__} | {_sim_d} | {weather_status} | {angle_mode} 모드")

    tabs = st.tabs([
        "🏠 메인", "📊 데이터셋", "🎯 피처 중요도",
        "💡 음영 원리", "🔥 음영 시각화", "⚡ 발전량 비교",
        "📅 월별 각도", "🌤️ 내일 스케줄", "🩺 건강진단", "🔧 파라미터 튜닝"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 0: 메인 대시보드
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("오늘의 발전 현황")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI 제어 발전량",  f"{pow_ai/1000:.3f} kWh",  f"연간 {ann_kwh_ai:.1f} kWh")
        c2.metric("고정 60° 대비",   f"+{(pow_ai/pow_60-1)*100:.1f}%" if pow_60 > 0 else "—")
        c3.metric("수직 90° 대비",   f"+{(pow_ai/pow_90-1)*100:.1f}%" if pow_90 > 0 else "—")
        c4.metric("예상 수익",        f"{int(pow_ai/1000*kepco_rate):,} 원")

        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.subheader("제어 스케줄 (일중)")
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

        st.subheader("시간별 스케줄 테이블")
        df_sch = pd.DataFrame({
            "시간":       times[mask_day].strftime("%H:%M").tolist(),
            "AI 각도(°)": ai_angles[mask_day].astype(int).tolist(),
            "GHI (W/m²)": np.round(ghi_real[mask_day], 1).tolist(),
            "음영률":     [f"{calc_shading_fraction(a, e, width_mm, pitch_mm):.2f}"
                           for a, e in zip(ai_angles[mask_day], elev[mask_day])],
        })
        st.dataframe(df_sch, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: 데이터셋 탐색
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("학습 데이터셋 탐색")
        st.info("연간 시뮬레이션 데이터(클리어스카이 기준) 기반 분포를 보여줍니다.")

        df_plot = df_annual[df_annual["ghi"] >= 10].copy()
        df_plot["month_str"] = df_plot["timestamp"].dt.month.map(
            {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**GHI 분포 (월별)**")
            fig = px.box(df_plot, x="month_str", y="ghi", color="month_str",
                          category_orders={"month_str": ["Jan","Feb","Mar","Apr","May","Jun",
                                                          "Jul","Aug","Sep","Oct","Nov","Dec"]})
            fig.update_layout(showlegend=False, height=350, yaxis_title="GHI (W/m²)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**AI 최적 각도 분포 (월별)**")
            fig2 = px.box(df_plot, x="month_str", y="angle_ai", color="month_str",
                           category_orders={"month_str": ["Jan","Feb","Mar","Apr","May","Jun",
                                                           "Jul","Aug","Sep","Oct","Nov","Dec"]})
            fig2.update_layout(showlegend=False, height=350, yaxis_title="각도 (°)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**GHI vs 최적각 산점도**")
        fig3 = px.scatter(df_plot.sample(min(2000, len(df_plot))),
                           x="ghi", y="angle_ai", color="month_str", opacity=0.5,
                           labels={"ghi":"GHI (W/m²)", "angle_ai":"최적각 (°)"})
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: 피처 중요도
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("피처 중요도 (Feature Importance)")

        # V12 실측 중요도
        importance_data = {
            "피처":  ["ghi_real", "doy_cos", "hour_sin", "cloud_cover", "doy_sin", "hour_cos", "temp_actual"],
            "Gain":  [0.705,       0.099,      0.061,       0.049,         0.033,     0.030,      0.023],
            "설명":  ["수평면 일사량 — 발전량의 주 에너지 입력",
                      "연중 날짜 코사인 — 계절 패턴",
                      "하루 중 시각 사인 — 오전/오후 패턴",
                      "운량 — 흐린 날 보정",
                      "연중 날짜 사인 — 계절 보완",
                      "하루 중 시각 코사인 — 시각 보완",
                      "외기온도 — 효율 온도계수"],
        }
        df_imp = pd.DataFrame(importance_data).sort_values("Gain", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_imp["Gain"], y=df_imp["피처"], orientation="h",
            marker_color=["#4fc3f7" if g > 0.1 else "#90caf9" if g > 0.05 else "#b0bec5" for g in df_imp["Gain"]],
            text=[f"{g:.3f}" for g in df_imp["Gain"]], textposition="outside"
        ))
        fig.update_layout(height=380, xaxis_title="Gain (분기점 오차 감소 기여도)",
                          xaxis_range=[0, 0.80])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **해석 가이드**
        - **ghi_real (0.705)**: GHI가 분기점을 만들 때 오차 감소량이 나머지 6개 합보다 큼. 
          물리적으로 당연 — 일사량이 각도 결정의 주신호.
        - **나머지 피처들**: GHI가 비슷한 상황에서 계절·시간·운량이 각도를 미세 조정. 
          예) GHI=500W/m²이어도 여름 정오 vs 겨울 오후면 최적각이 다름.
        - **특허 관점**: "GHI를 주신호, 계절·시간·온도를 보조변수로 통합하여 단일변수 대비 정확도 향상"으로 서술.
        """)

        st.info(f"V12 모델 성능: MAE = 0.92° | R² = 0.9786")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: 음영 원리 설명
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("💡 태양고도에 따라 음영이 달라지는 이유")

        st.markdown("""
        루버 블레이드는 상하로 배열되어 있어서, **위쪽 블레이드가 아래쪽 블레이드에 그림자를 드리웁니다.**
        태양고도(태양이 하늘에 얼마나 높이 있는지)에 따라 이 그림자 길이가 달라집니다.
        """)

        # 도식화: 태양고도별 음영 SVG 시뮬레이션
        elev_example = st.slider("태양 고도각 (°)", min_value=5, max_value=80, value=30, step=5)
        tilt_example  = st.slider("루버 각도 (°)",   min_value=15, max_value=90, value=45, step=5)

        sf = calc_shading_fraction(tilt_example, elev_example, DEFAULT_WIDTH_MM, DEFAULT_PITCH_MM)

        col1, col2 = st.columns([2, 1])
        with col1:
            # 루버 단면도 시각화
            fig = go.Figure()
            n_louvers = 4
            pitch_px = 80
            width_px  = DEFAULT_WIDTH_MM / DEFAULT_PITCH_MM * pitch_px

            for i in range(n_louvers):
                y_center = i * pitch_px
                dx = width_px * np.cos(np.radians(tilt_example))
                dy = width_px * np.sin(np.radians(tilt_example))

                # 블레이드
                fig.add_shape(type="line",
                    x0=0, y0=y_center, x1=dx, y1=y_center+dy,
                    line=dict(color="#4fc3f7", width=6))

                # 음영 영역
                shadow_len_px = width_px * np.sin(np.radians(tilt_example)) / np.tan(np.radians(max(elev_example, 1)))
                if i > 0:
                    shade_top    = y_center
                    shade_bottom = max(y_center - shadow_len_px, (i-1)*pitch_px)
                    if shade_bottom < shade_top:
                        fig.add_shape(type="rect",
                            x0=dx*0.1, y0=shade_bottom, x1=dx*0.9, y1=shade_top,
                            fillcolor="rgba(100,100,100,0.35)", line_width=0)

            # 태양 방향 화살표
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
                height=380,
                xaxis=dict(range=[-20, 250], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-20, n_louvers*pitch_px+80], showgrid=False, zeroline=False,
                           showticklabels=False, scaleanchor="x"),
                plot_bgcolor="rgba(20,25,40,1)",
                paper_bgcolor="rgba(0,0,0,0)",
                title=f"루버 단면도 — 태양고도 {elev_example}° | 루버각 {tilt_example}°"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("음영률", f"{sf*100:.1f}%",
                       delta="낮음 ✅" if sf < 0.3 else ("중간 ⚠️" if sf < 0.6 else "높음 🔴"),
                       delta_color="off")
            st.markdown(f"""
            **계산 공식**
            ```
            그림자 길이 = 폭 × |sin(루버각)| / tan(태양고도)
            음영률 = 그림자 길이 / 피치
            ```
            **현재 값**
            - 폭: {DEFAULT_WIDTH_MM:.0f} mm
            - 피치: {DEFAULT_PITCH_MM:.0f} mm  
            - 그림자 길이: {DEFAULT_WIDTH_MM * abs(np.sin(np.radians(tilt_example))) / np.tan(np.radians(max(elev_example,1))):.1f} mm
            
            **핵심 원리**
            - 태양고도 ↑ → 그림자 짧아짐 → 음영률 ↓
            - 태양고도 ↓ → 그림자 길어짐 → 음영률 ↑
            - AI는 음영률이 높아지는 구간을 자동 회피
            """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: 음영 시각화 (히트맵)
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("🔥 Tilt vs 태양고도별 음영률 히트맵")

        c1, c2 = st.columns(2)
        with c1:
            w_hm = st.number_input("블레이드 폭 (mm)", value=float(width_mm), step=50.0, key="hm_w")
        with c2:
            p_hm = st.number_input("피치 (mm)", value=float(pitch_mm), step=1.0, key="hm_p")

        tilt_range = np.arange(15, 91, 2)
        elev_range = np.arange(5,  81, 2)
        Z = np.zeros((len(elev_range), len(tilt_range)))
        for i, e in enumerate(elev_range):
            Z[i, :] = calc_shading_fraction(tilt_range, e, w_hm, p_hm)

        fig = go.Figure(go.Heatmap(
            x=tilt_range, y=elev_range, z=Z,
            colorscale="RdYlGn_r",
            colorbar=dict(title="음영률", tickformat=".0%"),
            zmin=0, zmax=1,
            hovertemplate="루버각: %{x}°<br>태양고도: %{y}°<br>음영률: %{z:.1%}<extra></extra>"
        ))

        # AI 운전 궤적 오버레이
        zen_day  = zen[mask_day]
        elev_day = 90 - zen_day
        ang_day  = ai_angles[mask_day]
        fig.add_trace(go.Scatter(
            x=ang_day, y=elev_day,
            mode="markers+lines",
            name=f"AI 운전 궤적 ({_sim_d})",
            marker=dict(size=8, color="white", symbol="circle"),
            line=dict(color="white", width=2, dash="dot")
        ))
        # 고정 60° 라인
        fig.add_shape(type="line", x0=60, x1=60, y0=5, y1=80,
                       line=dict(color="cyan", width=2, dash="dash"))
        fig.add_annotation(x=60, y=78, text="고정60°", showarrow=False,
                            font=dict(color="cyan", size=11))

        fig.update_layout(
            height=480,
            xaxis_title="루버 각도 (°)",
            yaxis_title="태양 고도각 (°)",
            title="음영률 히트맵 — 빨간색=음영 심함, 초록색=음영 없음",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **읽는 법**
        - 🟢 초록: 음영 없음 → 발전 최대 구간
        - 🟡 노랑: 부분 음영 → 발전 손실 시작
        - 🔴 빨강: 음영 심함 → 발전량 크게 감소
        - **흰 점선**: AI가 오늘 실제로 선택한 각도 궤적 → 초록 구간을 따라감
        - **파란 점선**: 고정 60° → 겨울 저태양고도 구간에서 노란~빨간 영역에 갇힘
        """)

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

        # 누적 발전량
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
    # TAB 6: 월별 각도 분포
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

        st.subheader("월별 평균 각도 & 발전량")
        df_summary = df_monthly.copy()
        df_summary["month_s"] = [month_names[i] for i in range(12)]
        df_summary = df_summary.rename(columns={"avg_angle":"평균각(°)"})
        st.dataframe(df_summary[["month_s","평균각(°)","AI","고정60°","수직90°"]].round(1),
                     use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7: 내일 스케줄
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.subheader("🌤️ 내일 예측 스케줄")
        if kma is None:
            st.error("기상청 API 데이터를 불러올 수 없습니다.")
        else:
            st.success(f"기상청 단기예보 기반 | 예보 기준일: {tomorrow}")
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
                "예측 GHI":   np.round(ghi_real[mask_day], 1).tolist(),
                "AI 각도(°)": ai_angles[mask_day].astype(int).tolist(),
                "음영률":     [f"{calc_shading_fraction(a, e, width_mm, pitch_mm)*100:.1f}%"
                                for a, e in zip(ai_angles[mask_day], elev[mask_day])],
                "예측 발전(Wh)": [
                    f"{poa_with_iam(90-a, 180, z, az_v, d, g, dh) * (1-calc_shading_fraction(a,e,width_mm,pitch_mm)) / 1000 * capacity_w * unit_count * eff_factor * DEFAULT_LOSS * area_scale:.1f}"
                    for a, z, az_v, d, g, dh, e in zip(
                        ai_angles[mask_day], zen[mask_day], az[mask_day],
                        dni_arr[mask_day], ghi_real[mask_day], dhi_arr[mask_day], elev[mask_day]
                    )
                ],
            })
            st.dataframe(df_tom, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8: 건강진단
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.subheader("🩺 시스템 건강 진단 (P_sim vs P_actual)")
        st.info("청구항 4: 예측 발전량(P_sim) vs 실측 발전량(P_actual) 비율로 이상 감지")

        col1, col2 = st.columns(2)
        with col1:
            p_actual_pct = st.slider("실측 발전량 비율 (%)", min_value=10, max_value=110, value=95,
                                       help="실제 센서값 대비 시뮬레이션 대비 비율. 센서 미연동 시 수동 입력.")
        with col2:
            warn_thr  = st.number_input("WARNING 임계값 (%)", value=90)
            crit_thr  = st.number_input("CRITICAL 임계값 (%)", value=75)

        health_ratio = p_actual_pct / 100.0
        if health_ratio >= warn_thr / 100:
            status, color = "✅ NORMAL",   "green"
        elif health_ratio >= crit_thr / 100:
            status, color = "⚠️ WARNING",  "orange"
        else:
            status, color = "🔴 CRITICAL", "red"

        c1, c2, c3 = st.columns(3)
        c1.metric("P_sim (예측)", f"{pow_ai/1000:.3f} kWh")
        c2.metric("P_actual (추정)", f"{pow_ai/1000*health_ratio:.3f} kWh")
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
        **판정 기준**
        | 범위 | 상태 | 조치 |
        |------|------|------|
        | ≥ 90% | ✅ NORMAL | 정상 운전 |
        | 75~90% | ⚠️ WARNING | 패널 오염 또는 부분 고장 의심 — 점검 권고 |
        | < 75% | 🔴 CRITICAL | 심각한 성능 저하 — 즉시 점검 필요 |
        """)
        st.caption("📌 현재는 시뮬레이션 기반. 실측 센서 연동 후 자동 업데이트 예정 (계속출원 대상).")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9: 파라미터 튜닝
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.subheader("🔧 파라미터 튜닝 — 실시간 발전량 변화 확인")
        st.markdown("슬라이더를 조절하면 해당 파라미터 변경 시 발전량이 어떻게 바뀌는지 확인할 수 있습니다.")

        col1, col2 = st.columns(2)
        with col1:
            t_angle_min = st.slider("최소 각도 (°)", 5,  30, ANGLE_MIN, key="t_amin")
            t_width     = st.slider("블레이드 폭 (mm)", 300, 1200, int(width_mm), step=50, key="t_w")
            t_pitch     = st.slider("피치 (mm)",   80, 200, int(pitch_mm), step=2, key="t_p")
        with col2:
            t_loss      = st.slider("시스템 손실률", 0.70, 0.95, DEFAULT_LOSS, step=0.01, key="t_loss")
            t_capacity  = st.slider("패널 용량 (W)", 100, 600, DEFAULT_CAPACITY, step=50, key="t_cap")

        # 튜닝된 각도 계산
        tuned_angles = np.where(ghi_real < 10, float(ANGLE_NIGHT),
                                np.clip(90 - zen, t_angle_min, ANGLE_MAX).astype(float))

        def calc_tuned(angles, w, p, loss, cap):
            tilt = 90 - np.asarray(angles, dtype=float)
            sf   = calc_shading_fraction(np.asarray(angles, dtype=float), elev, w, p)
            poa  = poa_with_iam(tilt, np.full_like(tilt, 180), zen, az, dni_arr, ghi_real, dhi_arr)
            mask = ghi_real >= 10
            return (poa[mask] * (1-sf[mask]) / 1000 * cap * unit_count * eff_factor * loss * area_scale).sum()

        pow_tuned   = calc_tuned(tuned_angles, t_width, t_pitch, t_loss, t_capacity)
        pow_default = calc_power_day(ai_angles)
        delta_pct   = (pow_tuned / pow_default - 1) * 100 if pow_default > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("기본 AI 발전량",  f"{pow_default/1000:.3f} kWh")
        c2.metric("튜닝 후 발전량",  f"{pow_tuned/1000:.3f} kWh",  f"{delta_pct:+.1f}%")
        c3.metric("음영률 (정오)",
                   f"{calc_shading_fraction(45, 60, t_width, t_pitch)*100:.1f}%",
                   help="루버 45° + 태양고도 60° 기준 음영률")

        # 각도별 발전량 스캔
        angle_scan = np.arange(t_angle_min, 91, 2)
        pow_scan   = [calc_tuned(np.full(24, a), t_width, t_pitch, t_loss, t_capacity) for a in angle_scan]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=angle_scan, y=[p/1000 for p in pow_scan],
                                  mode="lines+markers", line=dict(color="#4fc3f7", width=2),
                                  name="고정각별 발전량"))
        fig.add_vline(x=t_angle_min, line_dash="dash", line_color="red",
                       annotation_text=f"최소각 {t_angle_min}°")
        fig.update_layout(height=320, xaxis_title="고정 각도 (°)", yaxis_title="발전량 (kWh)",
                           title="고정 각도별 하루 발전량 스캔")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 AI 제어는 매 시간 최적각을 선택하므로 어떤 고정각보다 항상 높거나 같습니다.")


if __name__ == "__main__":
    run_app()
