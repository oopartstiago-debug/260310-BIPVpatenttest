# ==============================================================================
# BIPV 통합 관제 시스템 v9.0 — V15 선분교차 물리모델
# ==============================================================================
# v9.0 변경사항:
#   1. 음영 계산: 선분교차(Ray-Blade Intersection) 방식으로 전면 교체
#      - calc_panel_shading_vec: 벡터화 정밀 음영률
#      - 발전면(Front Face = 피봇~바깥끝) 타겟, 90° → SF=0% ✅
#   2. calc_effective_poa_v15: Colab V15와 동일 공식
#   3. 피처중요도/MAE/R²/참조각 V15 수치 반영
#   4. 음영 시각화 탭: shadow_comparison 스타일 단면도
#   5. secrets 관리 유지 (KMA_SERVICE_KEY, GH_TOKEN)
# ==============================================================================
__version__ = "9.0"

import os, io
import numpy as np
import pandas as pd
import requests, urllib.parse
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pvlib
from pvlib.location import Location
from datetime import datetime, timedelta

try:
    import joblib, xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

# ==============================================================================
# 상수
# ==============================================================================
KMA_SERVICE_KEY = st.secrets.get("KMA_SERVICE_KEY", "")
GH_TOKEN        = st.secrets.get("GH_TOKEN", "")
GH_HEADERS      = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}

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
ANGLE_MIN, ANGLE_MAX, ANGLE_NIGHT = 15, 90, 90

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/oopartstiago-debug/260310-BIPVpatenttest/main"
MODEL_URL = f"{GITHUB_RAW_BASE}/bipv_xgboost_model_v15.pkl"
CSV_URL   = f"{GITHUB_RAW_BASE}/bipv_ai_master_data_v15.csv"
XGB_MODEL_FILENAME = "bipv_xgboost_model_v15.pkl"

PLOT_TEMPLATE = "plotly_white"
COLOR_AI, COLOR_F60, COLOR_V90 = "#1976D2", "#F57C00", "#757575"
site = Location(LAT, LON, tz=TZ)

# ==============================================================================
# 모델 / CSV 로드
# ==============================================================================
@st.cache_resource
def load_xgb_model():
    if not _XGB_AVAILABLE: return None
    if not os.path.exists(XGB_MODEL_FILENAME):
        try:
            r = requests.get(MODEL_URL, headers=GH_HEADERS, timeout=30)
            if r.status_code == 200:
                with open(XGB_MODEL_FILENAME, "wb") as f: f.write(r.content)
        except Exception: return None
    if os.path.isfile(XGB_MODEL_FILENAME):
        try: return joblib.load(XGB_MODEL_FILENAME)
        except Exception: return None
    return None

@st.cache_data(ttl=86400)
def load_training_csv():
    try:
        r = requests.get(CSV_URL, headers=GH_HEADERS, timeout=30)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Seoul")
            return df
    except Exception: pass
    return None

# ==============================================================================
# V15 물리 계산 — 선분교차 기반
# ==============================================================================
def blade_geometry(tilt_deg, half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    tilt_rad = np.radians(np.asarray(tilt_deg, dtype=float))
    protrusion = half_depth * np.cos(tilt_rad)
    vertical_occupy = blade_depth * np.sin(tilt_rad)
    gap = np.maximum(pitch - vertical_occupy, 0.0)
    return protrusion, vertical_occupy, gap

def calc_panel_shading_vec(tilt_deg, solar_elevation, solar_azimuth,
                            half_depth=HALF_DEPTH_MM, pitch=DEFAULT_PITCH_MM, surface_azimuth=180.0):
    """V15 벡터화 선분교차 음영률 — Colab과 동일 로직"""
    tilt = np.asarray(tilt_deg, dtype=float)
    elev = np.asarray(solar_elevation, dtype=float)
    az = np.asarray(solar_azimuth, dtype=float)
    t_rad = np.radians(tilt)
    e_rad = np.radians(np.clip(elev, 0.1, 89.9))
    cos_t, sin_t = np.cos(t_rad), np.sin(t_rad)

    r_x = half_depth * cos_t
    r_y = pitch - half_depth * sin_t
    f_ex = half_depth * cos_t
    f_ey = -half_depth * sin_t

    az_diff = np.radians(az - surface_azimuth)
    ray_dx = -np.cos(e_rad) * np.cos(az_diff)
    ray_dy = -np.sin(e_rad)

    dx_f, dy_f = f_ex, f_ey
    denom = dx_f * ray_dy - dy_f * ray_dx
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    drx, dry = r_x, r_y
    t_b = (drx * ray_dy - dry * ray_dx) / denom_safe
    t_a = (drx * dy_f - dry * dx_f) / denom_safe

    sf = np.where((t_a > 0) & (t_b > 1.0), 1.0,
         np.where((t_a > 0) & (t_b > 0) & (t_b <= 1.0), t_b, 0.0))
    sf = np.where(elev <= 0, 0.0, sf)
    return np.clip(sf, 0.0, 1.0)

def sky_view_factor(tilt_deg, half_depth=HALF_DEPTH_MM, pitch=DEFAULT_PITCH_MM):
    protrusion = half_depth * np.cos(np.radians(np.asarray(tilt_deg, dtype=float)))
    return np.clip(1.0 - protrusion / pitch, 0.05, 1.0)

def calc_effective_poa_v15(tilt_deg, solar_elevation, solar_azimuth, dni, dhi,
                            half_depth=HALF_DEPTH_MM, pitch=DEFAULT_PITCH_MM):
    """V15 유효 POA = 직달×(1-SF×0.7) + 산란×SVF"""
    tilt = np.asarray(tilt_deg, dtype=float)
    elev = np.asarray(solar_elevation, dtype=float)
    az = np.asarray(solar_azimuth, dtype=float)
    sf = calc_panel_shading_vec(tilt, elev, az, half_depth, pitch)
    svf = sky_view_factor(tilt, half_depth, pitch)
    poa_direct = np.maximum(pvlib.irradiance.beam_component(
        surface_tilt=tilt, surface_azimuth=180.0,
        solar_zenith=90.0 - elev, solar_azimuth=az, dni=dni), 0.0)
    poa_diffuse = dhi * (1 + np.cos(np.radians(tilt))) / 2
    return np.maximum(poa_direct * (1 - sf * 0.7) + poa_diffuse * svf, 0.0)

# ==============================================================================
# XGBoost 예측
# ==============================================================================
def predict_angles_xgb(model, times, ghi_real, cloud_series, temp_series, angle_cap_deg):
    n = len(times)
    h = np.asarray(times.hour, dtype=float)
    d = np.asarray(times.dayofyear, dtype=float)
    X = np.column_stack([
        np.sin(2*np.pi*h/24)[:n], np.cos(2*np.pi*h/24)[:n],
        np.sin(2*np.pi*d/365)[:n], np.cos(2*np.pi*d/365)[:n],
        np.asarray(ghi_real, dtype=float).ravel()[:n],
        np.asarray(cloud_series, dtype=float).ravel()[:n],
        np.asarray(temp_series, dtype=float).ravel()[:n],
    ])
    try: pred = model.predict(X)
    except Exception: return None
    pred = np.clip(np.asarray(pred).ravel()[:n], ANGLE_MIN, min(ANGLE_MAX, angle_cap_deg))
    pred[np.asarray(ghi_real).ravel()[:n] < 10] = ANGLE_NIGHT
    return pred.astype(float)

def predict_angles_xgb_annual(model, times_y, ghi_y, temp_default=15.0):
    return predict_angles_xgb(model, times_y, ghi_y, np.zeros(len(times_y)),
                               np.full(len(times_y), temp_default), ANGLE_MAX)

def improved_rule_angles(elev_y, ghi_y):
    return np.where(ghi_y < 10, float(ANGLE_NIGHT),
                    np.clip(90 - elev_y * 0.7, ANGLE_MIN, ANGLE_MAX).astype(float))

# ==============================================================================
# 기상청 API
# ==============================================================================
@st.cache_data(ttl=3600)
def get_kma_forecast():
    decoded_key = urllib.parse.unquote(KMA_SERVICE_KEY)
    base_date = datetime.now().strftime("%Y%m%d")
    now_hour = datetime.now().hour
    avail = [2,5,8,11,14,17,20,23]
    bti = max([h for h in avail if h <= now_hour] or [23])
    if bti == 23 and now_hour < 2:
        base_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    bt = f"{bti:02d}00"
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {"serviceKey":decoded_key,"numOfRows":"1000","dataType":"JSON",
              "base_date":base_date,"base_time":bt,"nx":NX,"ny":NY}
    try:
        res = requests.get(url, params=params, timeout=10).json()
        items = res["response"]["body"]["items"]["item"]
        df = pd.DataFrame(items)
        tom = (datetime.now()+timedelta(days=1)).strftime("%Y%m%d")
        df_t = df[df["fcstDate"]==tom].drop_duplicates(subset=["fcstDate","fcstTime","category"])
        return df_t.pivot(index="fcstTime",columns="category",values="fcstValue"), tom
    except Exception: return None, None

# ==============================================================================
# 연간 데이터
# ==============================================================================
@st.cache_data(ttl=86400)
def get_annual_data(year, half_depth, pitch_mm, capacity_w,
                    unit_count, eff_factor, default_loss, use_xgb=False):
    times_y = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="h", tz=TZ)
    solpos_y = site.get_solarposition(times_y)
    cs_y = site.get_clearsky(times_y)
    ghi_y = np.asarray(cs_y["ghi"].values, dtype=float)
    zen_y, az_y = solpos_y["apparent_zenith"].values, solpos_y["azimuth"].values
    elev_y = 90.0 - zen_y
    dni_y = pvlib.irradiance.dirint(ghi_y, zen_y, times_y).fillna(0).values
    dhi_y = (ghi_y - dni_y * np.cos(np.radians(zen_y))).clip(0)

    if use_xgb:
        model = load_xgb_model()
        if model is not None:
            angles_ai = predict_angles_xgb_annual(model, times_y, ghi_y)
            if angles_ai is None: angles_ai = improved_rule_angles(elev_y, ghi_y)
        else: angles_ai = improved_rule_angles(elev_y, ghi_y)
    else: angles_ai = improved_rule_angles(elev_y, ghi_y)

    def energy(angles):
        eff_poa = calc_effective_poa_v15(np.asarray(angles,dtype=float), elev_y, az_y,
                                          dni_y, dhi_y, half_depth, pitch_mm)
        mask = ghi_y >= 10
        return (eff_poa[mask]/1000*capacity_w*unit_count*eff_factor*default_loss).sum()

    wh_ai = energy(angles_ai)
    wh_60 = energy(np.full_like(ghi_y, 60.0))
    wh_90 = energy(np.full_like(ghi_y, 90.0))

    df_ann = pd.DataFrame({"timestamp":times_y,"ghi":ghi_y,"zenith":zen_y,
                            "azimuth":az_y,"elevation":elev_y,"angle_ai":angles_ai})
    df_ann["month"] = df_ann["timestamp"].dt.month

    monthly = []
    for m in range(1,13):
        mm = (df_ann["month"]==m)&(ghi_y>=10)
        def em(ang):
            eff = calc_effective_poa_v15(np.asarray(ang,dtype=float),elev_y[mm],az_y[mm],
                                          dni_y[mm],dhi_y[mm],half_depth,pitch_mm)
            return (eff/1000*capacity_w*unit_count*eff_factor*default_loss).sum()
        monthly.append({"month":m,"AI":em(df_ann.loc[mm,"angle_ai"].values),
                        "고정60°":em(np.full(mm.sum(),60.0)),
                        "수직90°":em(np.full(mm.sum(),90.0)),
                        "avg_angle":float(df_ann.loc[mm,"angle_ai"].mean())})
    return wh_ai, wh_60, wh_90, pd.DataFrame(monthly), df_ann

# ==============================================================================
# 메인 앱
# ==============================================================================
def run_app():
    st.set_page_config(page_title="BIPV AI V15", layout="wide", page_icon="☀️")
    st.markdown("""<style>
    .stApp{background-color:#F8F9FA}
    .explain-box{background:#EEF2FF;border-left:3px solid #3B5BDB;padding:12px 16px;
        border-radius:6px;margin:10px 0;font-size:.91rem;color:#1A1A2E;line-height:1.65}
    .warn-box{background:#FFF3E0;border-left:3px solid #E65100;padding:12px 16px;
        border-radius:6px;margin:10px 0;font-size:.91rem;color:#3E2000;line-height:1.65}
    .good-box{background:#E8F5E9;border-left:3px solid #2E7D32;padding:12px 16px;
        border-radius:6px;margin:10px 0;font-size:.91rem;color:#1B3A1E;line-height:1.65}
    div[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:700}
    div[data-testid="stMetricLabel"]{font-size:.85rem;color:#555}
    </style>""", unsafe_allow_html=True)

    xgb_model = load_xgb_model() if _XGB_AVAILABLE else None
    kma, tomorrow = get_kma_forecast()

    # ── 사이드바 ──
    st.sidebar.title("■ 통합 환경 설정")
    st.sidebar.success("✅ XGBoost V15 로드") if xgb_model else st.sidebar.warning("⚠️ 규칙 기반 모드")

    st.sidebar.subheader("1. 시뮬레이션 날짜")
    tomorrow_dt = datetime.strptime(tomorrow, "%Y%m%d") if tomorrow else datetime.now()+timedelta(days=1)
    sim_date = st.sidebar.date_input("날짜", tomorrow_dt)

    st.sidebar.subheader("2. 블레이드 스펙 (V15)")
    width_mm = st.sidebar.number_input("블레이드 가로(mm)",min_value=100.0,value=DEFAULT_WIDTH_MM,step=100.0)
    blade_depth_mm = st.sidebar.number_input("블레이드 세로/DEPTH(mm)",min_value=10.0,value=DEFAULT_HEIGHT_MM,step=1.0)
    pitch_mm = st.sidebar.number_input("블레이드 피치(mm)",min_value=10.0,value=DEFAULT_PITCH_MM,step=1.0)
    half_depth_mm = blade_depth_mm / 2.0
    st.sidebar.caption(f"HALF_DEPTH={half_depth_mm:.1f}mm | 비율={pitch_mm/blade_depth_mm:.2f}")
    louver_count = st.sidebar.number_input("블레이드 개수",min_value=1,value=DEFAULT_LOUVER_COUNT,step=1)

    st.sidebar.subheader("3. 패널 스펙")
    unit_count = st.sidebar.number_input("설치 유닛 수",min_value=1,value=DEFAULT_UNIT_COUNT)
    capacity_w = st.sidebar.number_input("패널 용량(W)",value=DEFAULT_CAPACITY)
    target_eff = st.sidebar.number_input("패널 효율(%)",value=DEFAULT_EFFICIENCY,step=0.1)
    kepco_rate = st.sidebar.number_input("전기 요금(원/kWh)",value=DEFAULT_KEPCO)

    eff_factor = float(target_eff)/DEFAULT_EFFICIENCY
    area_scale = (width_mm*blade_depth_mm*louver_count)/(DEFAULT_WIDTH_MM*DEFAULT_HEIGHT_MM*DEFAULT_LOUVER_COUNT)

    # ── 날짜별 데이터 ──
    _sim_d = sim_date.strftime("%Y-%m-%d")
    times = pd.date_range(start=f"{_sim_d} 00:00",periods=24,freq="h",tz=TZ)
    solpos = site.get_solarposition(times)
    cs = site.get_clearsky(times)
    zen = np.asarray(solpos["apparent_zenith"].values,dtype=float)
    az = np.asarray(solpos["azimuth"].values,dtype=float)
    elev = np.asarray(solpos["apparent_elevation"].values,dtype=float)

    cloud_series = np.zeros(24); temp_series = np.full(24,15.0)
    if kma is not None and _sim_d.replace("-","")==tomorrow:
        kma_r = kma.reindex(times.strftime("%H00"))
        if "SKY" in kma.columns:
            cloud_series = kma_r["SKY"].apply(lambda x:0.0 if x=="1" else(0.5 if x=="3" else 1.0)).fillna(0).astype(float).values
        if "TMP" in kma.columns:
            temp_series = pd.to_numeric(kma_r["TMP"],errors="coerce").fillna(15.0).values

    ghi_real = np.asarray(cs["ghi"].values,dtype=float)*(1.0-cloud_series*0.65)
    dni_arr = pvlib.irradiance.dirint(ghi_real,zen,times).fillna(0).values
    dhi_arr = (ghi_real-dni_arr*np.cos(np.radians(zen))).clip(0)
    cloud_kma = cloud_series*9.0

    xgb_angles = None
    if xgb_model:
        xgb_angles = predict_angles_xgb(xgb_model,times,ghi_real,cloud_kma,temp_series,ANGLE_MAX)
    ai_angles = xgb_angles if xgb_angles is not None else improved_rule_angles(elev,ghi_real)
    angle_mode = "XGBoost V15" if xgb_angles is not None else "규칙 기반"

    def calc_power_day(angles):
        eff = calc_effective_poa_v15(np.asarray(angles,dtype=float),elev,az,dni_arr,dhi_arr,half_depth_mm,pitch_mm)
        mask = ghi_real >= 10
        return (eff[mask]/1000*capacity_w*unit_count*eff_factor*DEFAULT_LOSS*area_scale).sum()

    pow_ai = calc_power_day(ai_angles)
    pow_60 = calc_power_day(np.full(24,60.0))
    pow_90 = calc_power_day(np.full(24,90.0))

    last_year = sim_date.year - 1
    use_xgb_ann = xgb_model is not None
    wh_ai_y,wh_60_y,wh_90_y,df_monthly,df_annual = get_annual_data(
        last_year,half_depth_mm,pitch_mm,capacity_w,unit_count,eff_factor,DEFAULT_LOSS,use_xgb_ann)
    ann_kwh_ai = wh_ai_y/1000*area_scale
    ann_kwh_60 = wh_60_y/1000*area_scale
    ann_kwh_90 = wh_90_y/1000*area_scale

    kma_st = "✅ 기상청 예보 연동" if (kma is not None and _sim_d.replace("-","")==tomorrow) else "⚠️ 청천 기준"
    weather_st = "맑음" if np.mean(cloud_series)<0.3 else("구름많음" if np.mean(cloud_series)<0.8 else "흐림")
    mask_day = (times.hour>=6)&(times.hour<=19)

    # ── 탭 ──
    st.title("☀️ BIPV AI 통합 관제 대시보드")
    st.caption(f"v{__version__} (V15 선분교차 물리모델) | {_sim_d} | {weather_st} | {angle_mode} | {kma_st}")

    tabs = st.tabs(["🏠 메인","📊 학습데이터","🎯 피처중요도","💡 음영원리",
                     "🔥 음영시각화","⚡ 발전량비교","📅 월별각도",
                     "🌤️ 내일스케줄","🩺 건강진단","🔧 파라미터튜닝"])

    # ═══ TAB 0: 메인 ═══
    with tabs[0]:
        st.subheader("오늘의 발전 현황")
        st.markdown("""<div class="explain-box"><b>📖 V15 핵심 개선</b><br>
        음영 계산을 <b>선분교차(Ray-Blade Intersection)</b> 방식으로 전면 교체.
        기울어진 발전면 위의 그림자를 정밀하게 계산하여 여름 고각도 자기 음영 문제를 정확히 반영합니다.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("AI 제어 발전량",f"{pow_ai/1000:.3f} kWh",f"연간 {ann_kwh_ai:.1f} kWh")
        c2.metric("고정 60° 대비",f"+{(pow_ai/pow_60-1)*100:.1f}%" if pow_60>0 else "—")
        c3.metric("수직 90° 대비",f"+{(pow_ai/pow_90-1)*100:.1f}%" if pow_90>0 else "—")
        c4.metric("예상 수익",f"{int(pow_ai/1000*kepco_rate):,} 원")

        col_l,col_r = st.columns([3,1])
        with col_l:
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Bar(x=times[mask_day].strftime("%H:%M"),y=ghi_real[mask_day],
                                  name="GHI",marker_color="rgba(255,152,0,0.5)"),secondary_y=False)
            fig.add_trace(go.Scatter(x=times[mask_day].strftime("%H:%M"),y=ai_angles[mask_day],
                                      name="AI 각도",line=dict(color=COLOR_AI,width=3)),secondary_y=True)
            fig.update_yaxes(title_text="GHI (W/m²)",secondary_y=False)
            fig.update_yaxes(title_text="각도 (°)",range=[0,95],secondary_y=True)
            fig.update_layout(height=360,template=PLOT_TEMPLATE,legend=dict(orientation="h",y=1.08))
            st.plotly_chart(fig,use_container_width=True)
        with col_r:
            fig_b = go.Figure(go.Bar(x=["AI","F60°","F90°"],
                y=[pow_ai/1000,pow_60/1000,pow_90/1000],marker_color=[COLOR_AI,COLOR_F60,COLOR_V90],
                text=[f"{v/1000:.3f}" for v in [pow_ai,pow_60,pow_90]],textposition="auto"))
            fig_b.update_layout(height=360,yaxis_title="kWh",template=PLOT_TEMPLATE)
            st.plotly_chart(fig_b,use_container_width=True)

        # 스케줄 테이블
        ghi_d = ghi_real[mask_day]; elev_d = elev[mask_day]
        sf_v = calc_panel_shading_vec(ai_angles[mask_day],elev_d,az[mask_day],half_depth_mm,pitch_mm)
        svf_v = sky_view_factor(ai_angles[mask_day],half_depth_mm,pitch_mm)

        def sf_st(sf,g,e):
            if g<10 or e<=0: return "— (비활성)"
            if sf<0.1: return f"{sf*100:.1f}% 🟢"
            elif sf<0.3: return f"{sf*100:.1f}% 🟡"
            elif sf<0.5: return f"{sf*100:.1f}% 🟠"
            else: return f"{sf*100:.1f}% 🔴"

        df_sch = pd.DataFrame({
            "시간":times[mask_day].strftime("%H:%M").tolist(),
            "AI 각도(°)":ai_angles[mask_day].astype(int).tolist(),
            "GHI":np.round(ghi_d,1).tolist(),
            "음영률":[sf_st(s,g,e) for s,g,e in zip(sf_v,ghi_d,elev_d)],
            "SVF":[f"{s:.2f}" if g>=10 and e>0 else "—" for s,g,e in zip(svf_v,ghi_d,elev_d)]})
        st.dataframe(df_sch,use_container_width=True,hide_index=True)

    # ═══ TAB 1: 학습 데이터 ═══
    with tabs[1]:
        st.subheader("📊 학습 데이터셋 (V15)")
        df_csv = load_training_csv()
        if df_csv is not None:
            tc = "target_angle_v15" if "target_angle_v15" in df_csv.columns else \
                 "target_angle_v14" if "target_angle_v14" in df_csv.columns else "target_angle_v5"
            cv = tc.split("_")[-1].upper()
            st.success(f"✅ {cv} 학습 데이터 | {len(df_csv):,}행")
            dp = df_csv[df_csv["ghi_w_m2"]>10].copy()
            dp["month"] = pd.to_datetime(dp["timestamp"]).dt.month
            mm = {i:n for i,n in zip(range(1,13),["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}
            dp["ms"] = dp["month"].map(mm)
            mo = list(mm.values())
            c1,c2 = st.columns(2)
            with c1:
                fig = px.box(dp,x="ms",y="ghi_w_m2",color="ms",category_orders={"ms":mo},template=PLOT_TEMPLATE)
                fig.update_layout(showlegend=False,height=300,yaxis_title="GHI (W/m²)")
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                fig2 = px.box(dp,x="ms",y=tc,color="ms",category_orders={"ms":mo},template=PLOT_TEMPLATE)
                fig2.update_layout(showlegend=False,height=300,yaxis_title="최적 각도 (°)")
                st.plotly_chart(fig2,use_container_width=True)
        else:
            st.warning("⚠️ CSV 로드 실패")

    # ═══ TAB 2: 피처 중요도 — V15 ═══
    with tabs[2]:
        st.subheader("🎯 피처 중요도 — V15")
        imp_data = {"피처":["doy_cos","ghi_w_m2","hour_cos","hour_sin","temp_actual","doy_sin","cloud_cover"],
                    "Gain":[0.324,0.194,0.139,0.133,0.120,0.078,0.012]}
        df_imp = pd.DataFrame(imp_data).sort_values("Gain",ascending=True)
        fig = go.Figure(go.Bar(x=df_imp["Gain"],y=df_imp["피처"],orientation="h",
            marker_color=[COLOR_AI if g>0.12 else "#90CAF9" if g>0.06 else "#B0BEC5" for g in df_imp["Gain"]],
            text=[f"{g:.3f}" for g in df_imp["Gain"]],textposition="outside"))
        fig.update_layout(height=350,xaxis_title="Gain",xaxis_range=[0,0.40],template=PLOT_TEMPLATE)
        st.plotly_chart(fig,use_container_width=True)

        m1,m2 = st.columns(2)
        with m1:
            st.markdown("""<div class="good-box"><b>MAE = 0.70°</b><br>
            예측-정답 평균 차이 0.70°. 모터 정밀도(±2~3°) 대비 충분.</div>""",unsafe_allow_html=True)
        with m2:
            st.markdown("""<div class="good-box"><b>R² = 0.9966 | RMSE = 1.12°</b><br>
            각도 변동의 99.7%를 모델이 설명. 역대 최고 적합도.</div>""",unsafe_allow_html=True)

    # ═══ TAB 3: 음영 원리 ═══
    with tabs[3]:
        st.subheader("💡 V15 선분교차 음영 원리")
        st.markdown("""<div class="explain-box">
        <b>V15 핵심 변경: Ray-Blade Intersection</b><br>
        기존(V13~V14): 그림자를 수직 투영으로 근사 → 기울어진 발전면 음영 부정확<br>
        V15: 상부 블레이드 끝에서 태양 광선을 쏘아 하부 발전면과의 <b>정확한 교차점</b>을 계산<br><br>
        <b>발전면(Front Face)</b> = 피봇 ~ 바깥끝 (태양전지 부착면)<br>
        <b>음영 방향</b>: 피봇(안쪽)부터 교차점까지 — 태양이 바깥에서 비추므로<br>
        <b>90° 수직</b>: 블레이드가 나란히 서서 발전면을 가리지 않음 → <b>SF=0%</b>
        </div>""", unsafe_allow_html=True)

        elev_ex = st.slider("태양 고도각 (°)",5,80,45,5)
        tilt_ex = st.slider("루버 각도 (°)",15,90,30,5)

        sf_ex = float(calc_panel_shading_vec(tilt_ex,elev_ex,180.0,half_depth_mm,pitch_mm))
        svf_ex = float(sky_view_factor(tilt_ex,half_depth_mm,pitch_mm))

        c1,c2 = st.columns([2,1])
        with c1:
            # Plotly로 단면도 (shadow_comparison 스타일)
            fig = go.Figure()
            n_bl = 4
            pp = 80  # pitch in px
            for i in range(n_bl):
                yc = i*pp
                hd_px = (half_depth_mm/pitch_mm)*pp
                dx = hd_px*np.cos(np.radians(tilt_ex))
                dy = hd_px*np.sin(np.radians(tilt_ex))
                px_piv = (0, yc)
                px_in = (-dx, yc+dy)
                px_out = (dx, yc-dy)
                # Back (벽쪽)
                fig.add_trace(go.Scatter(x=[px_in[0],px_piv[0]],y=[px_in[1],px_piv[1]],
                    mode="lines",line=dict(color="#999",width=3),showlegend=(i==0),
                    name="Back" if i==0 else None))
                # Front Face (발전면)
                fig.add_trace(go.Scatter(x=[px_piv[0],px_out[0]],y=[px_piv[1],px_out[1]],
                    mode="lines",line=dict(color=COLOR_AI,width=6),showlegend=(i==0),
                    name="Front Face" if i==0 else None))
                fig.add_trace(go.Scatter(x=[px_piv[0]],y=[px_piv[1]],mode="markers",
                    marker=dict(size=6,color="red"),showlegend=False))

                # 음영 표시
                if i>0 and sf_ex>0:
                    yc_below = (i-1)*pp
                    piv_b = (0, yc_below)
                    out_b = (dx, yc_below-dy)
                    if sf_ex >= 1.0:
                        fig.add_trace(go.Scatter(x=[piv_b[0],out_b[0]],y=[piv_b[1],out_b[1]],
                            mode="lines",line=dict(color="red",width=8),opacity=0.4,
                            showlegend=(i==1),name=f"Shadow({sf_ex:.0%})" if i==1 else None))
                    else:
                        ix_x = piv_b[0]+sf_ex*(out_b[0]-piv_b[0])
                        ix_y = piv_b[1]+sf_ex*(out_b[1]-piv_b[1])
                        fig.add_trace(go.Scatter(x=[piv_b[0],ix_x],y=[piv_b[1],ix_y],
                            mode="lines",line=dict(color="red",width=8),opacity=0.4,
                            showlegend=(i==1),name=f"Shadow({sf_ex:.0%})" if i==1 else None))

            # 광선
            if elev_ex > 0:
                rdx = -np.cos(np.radians(elev_ex))*60
                rdy = -np.sin(np.radians(elev_ex))*60
                for i in range(1,n_bl):
                    yc = i*pp; hd_px = (half_depth_mm/pitch_mm)*pp
                    ox = hd_px*np.cos(np.radians(tilt_ex))
                    oy = yc - hd_px*np.sin(np.radians(tilt_ex))
                    fig.add_annotation(x=ox+rdx*1.5,y=oy+rdy*1.5,ax=ox-rdx*0.8,ay=oy-rdy*0.8,
                        xref="x",yref="y",axref="x",ayref="y",showarrow=True,
                        arrowhead=2,arrowsize=1.2,arrowcolor="#FF8F00",arrowwidth=2)

            # 벽면
            fig.add_shape(type="line",x0=-5,y0=-30,x1=-5,y1=(n_bl-1)*pp+50,
                          line=dict(color="#555",width=3))

            fig.update_layout(height=420,template=PLOT_TEMPLATE,
                xaxis=dict(range=[-60,100],showgrid=False,zeroline=False,showticklabels=False),
                yaxis=dict(range=[-50,(n_bl-1)*pp+70],showgrid=False,zeroline=False,
                           showticklabels=False,scaleanchor="x"),
                title=f"V15 단면도 — 고도 {elev_ex}° | 루버 {tilt_ex}° | SF={sf_ex:.1%}",
                legend=dict(orientation="h",y=-0.05))
            st.plotly_chart(fig,use_container_width=True)

        with c2:
            st.metric("음영률 (SF)",f"{sf_ex*100:.1f}%",
                delta="양호 ✅" if sf_ex<0.1 else("경미 🟡" if sf_ex<0.3 else "주의 🔴"),delta_color="off")
            st.metric("SVF",f"{svf_ex:.2f}",
                delta="높음 ✅" if svf_ex>0.5 else "낮음 🔴",delta_color="off")
            prot,_,gap = blade_geometry(tilt_ex,half_depth_mm,blade_depth_mm,pitch_mm)
            st.markdown(f"""<div class="explain-box">
            <b>현재 기하학</b><br>
            DEPTH {blade_depth_mm:.0f}mm | PITCH {pitch_mm:.0f}mm<br>
            돌출 {float(prot):.1f}mm | 틈 {float(gap):.1f}mm<br><br>
            <b>V15 음영 계산</b><br>
            선분교차로 정밀 계산<br>
            피봇→교차점 = 음영 영역
            </div>""",unsafe_allow_html=True)

    # ═══ TAB 4: 음영 시각화 — shadow_comparison 스타일 ═══
    with tabs[4]:
        st.subheader("🔥 음영 시각화 — 조건별 단면 비교 (V15)")
        st.markdown("""<div class="explain-box">
        다양한 계절·시간·각도 조건에서의 <b>자기 음영(Self-shading) 패턴</b>을 단면도로 비교합니다.
        <b>파란 선</b>=발전면, <b>빨간 선</b>=음영 영역, <b>주황 화살표</b>=태양 광선
        </div>""",unsafe_allow_html=True)

        conditions = [
            (30,15,0,"동지 09시\ntilt=30° elev=15°"),
            (60,29,0,"동지 정오\ntilt=60° elev=29°"),
            (30,15,0,"동지 16시\ntilt=30° elev=15°"),
            (22,40,0,"하지 09시\ntilt=22° elev=40°"),
            (22,76,0,"하지 정오\ntilt=22° elev=76°"),
            (22,40,0,"하지 16시\ntilt=22° elev=40°"),
            (15,76,0,"하지 정오\ntilt=15° elev=76°"),
            (45,76,0,"하지 정오\ntilt=45° elev=76°"),
            (90,29,0,"동지 정오\ntilt=90° elev=29°"),
        ]

        cols_per_row = 3
        for row_start in range(0, len(conditions), cols_per_row):
            cols = st.columns(cols_per_row)
            for ci, (tilt_c, elev_c, az_c, label_c) in enumerate(conditions[row_start:row_start+cols_per_row]):
                with cols[ci]:
                    sf_c = float(calc_panel_shading_vec(tilt_c,elev_c,180.0+az_c,half_depth_mm,pitch_mm))
                    fig_c = go.Figure()
                    n_b = 4; pp_c = 70
                    for i in range(n_b):
                        yc = i*pp_c
                        hd_px = (half_depth_mm/pitch_mm)*pp_c
                        dx = hd_px*np.cos(np.radians(tilt_c))
                        dy = hd_px*np.sin(np.radians(tilt_c))
                        fig_c.add_trace(go.Scatter(x=[-dx,0],y=[yc+dy,yc],mode="lines",
                            line=dict(color="#999",width=2),showlegend=False))
                        fig_c.add_trace(go.Scatter(x=[0,dx],y=[yc,yc-dy],mode="lines",
                            line=dict(color=COLOR_AI,width=5),showlegend=False))
                        fig_c.add_trace(go.Scatter(x=[0],y=[yc],mode="markers",
                            marker=dict(size=4,color="red"),showlegend=False))
                        if i>0 and sf_c>0:
                            piv_b = (0,(i-1)*pp_c)
                            out_b = (dx,(i-1)*pp_c-dy)
                            t_f = min(sf_c,1.0)
                            ix_x = piv_b[0]+t_f*(out_b[0]-piv_b[0])
                            ix_y = piv_b[1]+t_f*(out_b[1]-piv_b[1])
                            fig_c.add_trace(go.Scatter(
                                x=[piv_b[0],ix_x if sf_c<1 else out_b[0]],
                                y=[piv_b[1],ix_y if sf_c<1 else out_b[1]],
                                mode="lines",line=dict(color="red",width=7),opacity=0.5,showlegend=False))

                    if elev_c>0:
                        rdx=-np.cos(np.radians(elev_c))*40
                        rdy=-np.sin(np.radians(elev_c))*40
                        for i in range(1,n_b):
                            yc=i*pp_c; hd_px=(half_depth_mm/pitch_mm)*pp_c
                            ox=hd_px*np.cos(np.radians(tilt_c))
                            oy=yc-hd_px*np.sin(np.radians(tilt_c))
                            fig_c.add_annotation(x=ox+rdx,y=oy+rdy,ax=ox-rdx*0.5,ay=oy-rdy*0.5,
                                xref="x",yref="y",axref="x",ayref="y",showarrow=True,
                                arrowhead=2,arrowsize=1,arrowcolor="#FF8F00",arrowwidth=1.5)

                    fig_c.add_shape(type="line",x0=-3,y0=-20,x1=-3,y1=(n_b-1)*pp_c+40,
                        line=dict(color="#555",width=2))
                    fig_c.update_layout(height=300,template=PLOT_TEMPLATE,margin=dict(l=5,r=5,t=35,b=5),
                        xaxis=dict(range=[-50,80],showgrid=False,zeroline=False,showticklabels=False,visible=False),
                        yaxis=dict(range=[-30,(n_b-1)*pp_c+50],showgrid=False,zeroline=False,
                                   showticklabels=False,visible=False,scaleanchor="x"),
                        title=dict(text=f"{label_c}<br>SF={sf_c:.0%}",font=dict(size=11)))
                    st.plotly_chart(fig_c,use_container_width=True)

    # ═══ TAB 5: 발전량 비교 ═══
    with tabs[5]:
        st.subheader("⚡ AI vs 고정60° vs 수직90°")
        mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = go.Figure()
        for col,color,name in [("AI",COLOR_AI,"AI 제어"),("고정60°",COLOR_F60,"고정 60°"),("수직90°",COLOR_V90,"수직 90°")]:
            fig.add_trace(go.Bar(x=mn,y=df_monthly[col]*area_scale/1000,name=name,marker_color=color))
        fig.update_layout(barmode="group",height=400,template=PLOT_TEMPLATE,yaxis_title="kWh")
        st.plotly_chart(fig,use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("연간 AI",f"{ann_kwh_ai:.1f} kWh")
        c2.metric("연간 F60°",f"{ann_kwh_60:.1f} kWh",f"AI대비 -{(1-ann_kwh_60/ann_kwh_ai)*100:.1f}%" if ann_kwh_ai>0 else "—")
        c3.metric("연간 F90°",f"{ann_kwh_90:.1f} kWh",f"AI대비 -{(1-ann_kwh_90/ann_kwh_ai)*100:.1f}%" if ann_kwh_ai>0 else "—")

    # ═══ TAB 6: 월별 각도 ═══
    with tabs[6]:
        st.subheader("📅 월별 AI 각도 분포")
        dp2 = df_annual[df_annual["ghi"]>=10].copy()
        dp2["mn"] = dp2["timestamp"].dt.month
        mn_map = {i:n for i,n in zip(range(1,13),["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}
        dp2["ms"] = dp2["mn"].map(mn_map)
        fig = go.Figure()
        for m,ms in mn_map.items():
            fig.add_trace(go.Box(y=dp2[dp2["mn"]==m]["angle_ai"],name=ms,marker_color=COLOR_AI,boxmean=True))
        fig.add_hline(y=ANGLE_MIN,line_dash="dash",line_color="red",annotation_text=f"최소각 {ANGLE_MIN}°")
        fig.update_layout(height=400,yaxis_title="루버 각도 (°)",showlegend=False,template=PLOT_TEMPLATE)
        st.plotly_chart(fig,use_container_width=True)

        v15_ref = {1:62.6,2:55.6,3:42.2,4:25.0,5:23.7,6:22.1,7:23.3,8:25.3,9:38.9,10:51.4,11:60.4,12:61.7}
        ds = df_monthly.copy()
        ds["month_s"] = list(mn_map.values())
        ds = ds.rename(columns={"avg_angle":"시뮬 평균각(°)"})
        ds["V15 참조각"] = [v15_ref[m] for m in range(1,13)]
        st.dataframe(ds[["month_s","시뮬 평균각(°)","V15 참조각","AI","고정60°","수직90°"]].round(1),
                     use_container_width=True,hide_index=True)

    # ═══ TAB 7: 내일 스케줄 ═══
    with tabs[7]:
        st.subheader("🌤️ 내일 예측 스케줄")
        if kma is None: st.error("❌ 기상청 API 실패")
        else: st.success(f"✅ 기상청 연동 | {tomorrow}")
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=times[mask_day].strftime("%H:%M"),y=ghi_real[mask_day],
            name="GHI",marker_color="rgba(255,152,0,0.5)"),secondary_y=False)
        fig.add_trace(go.Scatter(x=times[mask_day].strftime("%H:%M"),y=ai_angles[mask_day],
            name="AI 각도",line=dict(color=COLOR_AI,width=3),mode="lines+markers"),secondary_y=True)
        fig.update_yaxes(title_text="GHI",secondary_y=False)
        fig.update_yaxes(title_text="각도(°)",range=[0,95],secondary_y=True)
        fig.update_layout(height=380,template=PLOT_TEMPLATE)
        st.plotly_chart(fig,use_container_width=True)

    # ═══ TAB 8: 건강진단 ═══
    with tabs[8]:
        st.subheader("🩺 시스템 건강 진단")
        st.info("📌 실측 센서 미연동. 슬라이더로 시뮬레이션.")
        p_pct = st.slider("실측 발전량 비율(%)",10,110,95)
        hr = p_pct/100.0
        w_t,c_t = 90,75
        if hr>=w_t/100: status,color = "✅ NORMAL","green"
        elif hr>=c_t/100: status,color = "⚠️ WARNING","orange"
        else: status,color = "🔴 CRITICAL","red"
        c1,c2,c3 = st.columns(3)
        c1.metric("P_sim",f"{pow_ai/1000:.3f} kWh")
        c2.metric("P_actual",f"{pow_ai/1000*hr:.3f} kWh")
        c3.metric("Health",f"{hr:.0%}",status)
        fig = go.Figure(go.Indicator(mode="gauge+number+delta",value=p_pct,
            delta={"reference":100},gauge={"axis":{"range":[0,110]},"bar":{"color":color},
            "steps":[{"range":[0,c_t],"color":"rgba(244,67,54,0.15)"},
                     {"range":[c_t,w_t],"color":"rgba(255,152,0,0.15)"},
                     {"range":[w_t,110],"color":"rgba(76,175,80,0.15)"}]},
            title={"text":f"상태: {status}"}))
        fig.update_layout(height=300)
        st.plotly_chart(fig,use_container_width=True)

    # ═══ TAB 9: 파라미터 튜닝 ═══
    with tabs[9]:
        st.subheader("🔧 파라미터 민감도 — V15")
        t_loss = st.slider("손실률",0.70,0.95,DEFAULT_LOSS,0.01)

        st.markdown("#### 📐 BLADE_DEPTH → 연간 발전량")
        dr = np.arange(60,181,10)
        pd_list = []
        for bd in dr:
            _,_,_,dm,_ = get_annual_data(last_year,bd/2.0,pitch_mm,capacity_w,unit_count,eff_factor,t_loss)
            pd_list.append(dm["AI"].sum()*area_scale/1000)
        fig1 = go.Figure(go.Scatter(x=dr,y=pd_list,mode="lines+markers",line=dict(color=COLOR_AI,width=2)))
        fig1.add_vline(x=blade_depth_mm,line_dash="dash",line_color=COLOR_F60,annotation_text=f"현재 {blade_depth_mm:.0f}mm")
        fig1.update_layout(height=280,xaxis_title="DEPTH(mm)",yaxis_title="kWh",template=PLOT_TEMPLATE)
        st.plotly_chart(fig1,use_container_width=True)

        st.markdown("#### 📏 피치 → 정오 음영률")
        pr = np.arange(80,201,5)
        sf_list = [float(calc_panel_shading_vec(45,60,180,blade_depth_mm/2,float(p)))*100 for p in pr]
        fig2 = go.Figure(go.Scatter(x=pr,y=sf_list,mode="lines+markers",line=dict(color=COLOR_F60,width=2)))
        fig2.add_vline(x=pitch_mm,line_dash="dash",line_color=COLOR_AI,annotation_text=f"현재 {pitch_mm:.0f}mm")
        fig2.update_layout(height=280,xaxis_title="PITCH(mm)",yaxis_title="음영률%",template=PLOT_TEMPLATE)
        st.plotly_chart(fig2,use_container_width=True)

run_app()
