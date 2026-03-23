# ==============================================================================
# BIPV 통합 관제 시스템 v9.1 — V15 선분교차 물리모델 + 설명 복원
# ==============================================================================
__version__ = "9.1"

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

st.set_page_config(page_title="BIPV AI V15", layout="wide", page_icon="☀️")

try:    KMA_SERVICE_KEY = st.secrets["KMA_SERVICE_KEY"]
except Exception: KMA_SERVICE_KEY = ""
try:    GH_TOKEN = st.secrets["GH_TOKEN"]
except Exception: GH_TOKEN = ""
GH_HEADERS = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}

LAT, LON, TZ = 37.5665, 126.9780, "Asia/Seoul"
NX, NY = 60, 127
DEFAULT_CAPACITY, DEFAULT_EFFICIENCY, DEFAULT_LOSS = 300, 18.7, 0.85
DEFAULT_KEPCO, DEFAULT_UNIT_COUNT, DEFAULT_LOUVER_COUNT = 210, 1, 20
DEFAULT_WIDTH_MM, DEFAULT_HEIGHT_MM, DEFAULT_PITCH_MM = 900.0, 114.0, 114.0
HALF_DEPTH_MM = 57.0
ANGLE_MIN, ANGLE_MAX, ANGLE_NIGHT = 15, 90, 90

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/oopartstiago-debug/260310-BIPVpatenttest/main"
MODEL_URL = f"{GITHUB_RAW_BASE}/bipv_xgboost_model_v15.pkl"
CSV_URL   = f"{GITHUB_RAW_BASE}/bipv_ai_master_data_v15.csv"
XGB_MODEL_FILENAME = "bipv_xgboost_model_v15.pkl"
PLOT_TEMPLATE = "plotly_white"
COLOR_AI, COLOR_F60, COLOR_V90 = "#1976D2", "#F57C00", "#757575"
site = Location(LAT, LON, tz=TZ)

@st.cache_resource
def load_xgb_model():
    if not _XGB_AVAILABLE: return None
    if not os.path.exists(XGB_MODEL_FILENAME):
        try:
            r = requests.get(MODEL_URL, headers=GH_HEADERS, timeout=30)
            if r.status_code == 200:
                with open(XGB_MODEL_FILENAME, "wb") as f: f.write(r.content)
            else: return None
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

def blade_geometry(tilt_deg, half_depth=HALF_DEPTH_MM, blade_depth=DEFAULT_HEIGHT_MM, pitch=DEFAULT_PITCH_MM):
    t = np.radians(np.asarray(tilt_deg, dtype=float))
    return half_depth*np.cos(t), blade_depth*np.sin(t), np.maximum(pitch-blade_depth*np.sin(t), 0.0)

def calc_panel_shading_vec(tilt_deg, solar_elevation, solar_azimuth,
                            half_depth=HALF_DEPTH_MM, pitch=DEFAULT_PITCH_MM, surface_azimuth=180.0):
    tilt=np.asarray(tilt_deg,dtype=float); elev=np.asarray(solar_elevation,dtype=float)
    az=np.asarray(solar_azimuth,dtype=float)
    t_rad=np.radians(tilt); e_rad=np.radians(np.clip(elev,0.1,89.9))
    cos_t,sin_t=np.cos(t_rad),np.sin(t_rad)
    r_x=half_depth*cos_t; r_y=pitch-half_depth*sin_t
    f_ex,f_ey=half_depth*cos_t,-half_depth*sin_t
    az_diff=np.radians(az-surface_azimuth)
    ray_dx=-np.cos(e_rad)*np.cos(az_diff); ray_dy=-np.sin(e_rad)
    denom=f_ex*ray_dy-f_ey*ray_dx
    denom_safe=np.where(np.abs(denom)<1e-12,1e-12,denom)
    t_b=(r_x*ray_dy-r_y*ray_dx)/denom_safe
    t_a=(r_x*f_ey-r_y*f_ex)/denom_safe
    sf=np.where((t_a>0)&(t_b>1.0),1.0,np.where((t_a>0)&(t_b>0)&(t_b<=1.0),t_b,0.0))
    return np.clip(np.where(elev<=0,0.0,sf),0.0,1.0)

def sky_view_factor(tilt_deg, half_depth=HALF_DEPTH_MM, pitch=DEFAULT_PITCH_MM):
    return np.clip(1.0-half_depth*np.cos(np.radians(np.asarray(tilt_deg,dtype=float)))/pitch,0.05,1.0)

def calc_effective_poa_v15(tilt_deg, solar_elevation, solar_azimuth, dni, dhi,
                            half_depth=HALF_DEPTH_MM, pitch=DEFAULT_PITCH_MM):
    tilt=np.asarray(tilt_deg,dtype=float); elev=np.asarray(solar_elevation,dtype=float)
    az=np.asarray(solar_azimuth,dtype=float)
    sf=calc_panel_shading_vec(tilt,elev,az,half_depth,pitch)
    svf=sky_view_factor(tilt,half_depth,pitch)
    poa_dir=np.maximum(pvlib.irradiance.beam_component(surface_tilt=tilt,surface_azimuth=180.0,
        solar_zenith=90.0-elev,solar_azimuth=az,dni=dni),0.0)
    poa_diff=dhi*(1+np.cos(np.radians(tilt)))/2
    return np.maximum(poa_dir*(1-sf*0.7)+poa_diff*svf,0.0)

def predict_angles_xgb(model,times,ghi_real,cloud_series,temp_series,angle_cap_deg):
    n=len(times); h=np.asarray(times.hour,dtype=float); d=np.asarray(times.dayofyear,dtype=float)
    X=np.column_stack([np.sin(2*np.pi*h/24)[:n],np.cos(2*np.pi*h/24)[:n],
        np.sin(2*np.pi*d/365)[:n],np.cos(2*np.pi*d/365)[:n],
        np.asarray(ghi_real,dtype=float).ravel()[:n],
        np.asarray(cloud_series,dtype=float).ravel()[:n],
        np.asarray(temp_series,dtype=float).ravel()[:n]])
    try: pred=model.predict(X)
    except Exception: return None
    pred=np.clip(np.asarray(pred).ravel()[:n],ANGLE_MIN,min(ANGLE_MAX,angle_cap_deg))
    pred[np.asarray(ghi_real).ravel()[:n]<10]=ANGLE_NIGHT
    return pred.astype(float)

def predict_angles_xgb_annual(model,times_y,ghi_y,temp_default=15.0):
    return predict_angles_xgb(model,times_y,ghi_y,np.zeros(len(times_y)),np.full(len(times_y),temp_default),ANGLE_MAX)

def improved_rule_angles(elev_y,ghi_y):
    return np.where(ghi_y<10,float(ANGLE_NIGHT),np.clip(90-elev_y*0.7,ANGLE_MIN,ANGLE_MAX).astype(float))

@st.cache_data(ttl=3600)
def get_kma_forecast():
    decoded_key=urllib.parse.unquote(KMA_SERVICE_KEY)
    base_date=datetime.now().strftime("%Y%m%d"); now_hour=datetime.now().hour
    avail=[2,5,8,11,14,17,20,23]; bti=max([h for h in avail if h<=now_hour] or [23])
    if bti==23 and now_hour<2: base_date=(datetime.now()-timedelta(days=1)).strftime("%Y%m%d")
    url="http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params={"serviceKey":decoded_key,"numOfRows":"1000","dataType":"JSON",
            "base_date":base_date,"base_time":f"{bti:02d}00","nx":NX,"ny":NY}
    try:
        res=requests.get(url,params=params,timeout=10).json()
        items=res["response"]["body"]["items"]["item"]; df=pd.DataFrame(items)
        tom=(datetime.now()+timedelta(days=1)).strftime("%Y%m%d")
        df_t=df[df["fcstDate"]==tom].drop_duplicates(subset=["fcstDate","fcstTime","category"])
        return df_t.pivot(index="fcstTime",columns="category",values="fcstValue"),tom
    except Exception: return None,None

@st.cache_data(ttl=86400)
def get_annual_data(year,half_depth,pitch_mm,capacity_w,unit_count,eff_factor,default_loss,use_xgb=False):
    times_y=pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31 23:00",freq="h",tz=TZ)
    sp=site.get_solarposition(times_y); cs=site.get_clearsky(times_y)
    ghi_y=np.asarray(cs["ghi"].values,dtype=float)
    zen_y,az_y=sp["apparent_zenith"].values,sp["azimuth"].values; elev_y=90.0-zen_y
    dni_y=pvlib.irradiance.dirint(ghi_y,zen_y,times_y).fillna(0).values
    dhi_y=(ghi_y-dni_y*np.cos(np.radians(zen_y))).clip(0)
    if use_xgb:
        model=load_xgb_model()
        angles_ai=predict_angles_xgb_annual(model,times_y,ghi_y) if model else None
        if angles_ai is None: angles_ai=improved_rule_angles(elev_y,ghi_y)
    else: angles_ai=improved_rule_angles(elev_y,ghi_y)
    def energy(ang):
        eff=calc_effective_poa_v15(np.asarray(ang,dtype=float),elev_y,az_y,dni_y,dhi_y,half_depth,pitch_mm)
        return (eff[ghi_y>=10]/1000*capacity_w*unit_count*eff_factor*default_loss).sum()
    wh_ai,wh_60,wh_90=energy(angles_ai),energy(np.full_like(ghi_y,60.0)),energy(np.full_like(ghi_y,90.0))
    df_ann=pd.DataFrame({"timestamp":times_y,"ghi":ghi_y,"zenith":zen_y,"azimuth":az_y,"elevation":elev_y,"angle_ai":angles_ai})
    df_ann["month"]=df_ann["timestamp"].dt.month
    monthly=[]
    for m in range(1,13):
        mm=(df_ann["month"]==m)&(ghi_y>=10)
        def em(ang):
            eff=calc_effective_poa_v15(np.asarray(ang,dtype=float),elev_y[mm],az_y[mm],dni_y[mm],dhi_y[mm],half_depth,pitch_mm)
            return (eff/1000*capacity_w*unit_count*eff_factor*default_loss).sum()
        monthly.append({"month":m,"AI":em(df_ann.loc[mm,"angle_ai"].values),
                        "고정60°":em(np.full(mm.sum(),60.0)),"수직90°":em(np.full(mm.sum(),90.0)),
                        "avg_angle":float(df_ann.loc[mm,"angle_ai"].mean())})
    return wh_ai,wh_60,wh_90,pd.DataFrame(monthly),df_ann

# ==============================================================================
# CSS + 데이터 준비 + UI
# ==============================================================================
st.markdown("""<style>
.stApp{background-color:#F8F9FA}
.explain-box{background:#EEF2FF;border-left:3px solid #3B5BDB;padding:12px 16px;border-radius:6px;margin:10px 0;font-size:.91rem;color:#1A1A2E;line-height:1.65}
.warn-box{background:#FFF3E0;border-left:3px solid #E65100;padding:12px 16px;border-radius:6px;margin:10px 0;font-size:.91rem;color:#3E2000;line-height:1.65}
.good-box{background:#E8F5E9;border-left:3px solid #2E7D32;padding:12px 16px;border-radius:6px;margin:10px 0;font-size:.91rem;color:#1B3A1E;line-height:1.65}
div[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:700}
div[data-testid="stMetricLabel"]{font-size:.85rem;color:#555}
</style>""", unsafe_allow_html=True)

xgb_model = load_xgb_model() if _XGB_AVAILABLE else None
kma, tomorrow = get_kma_forecast()

st.sidebar.title("■ 통합 환경 설정")
if xgb_model: st.sidebar.success("✅ XGBoost V15 모델 로드됨")
else: st.sidebar.warning("⚠️ 규칙 기반 모드 (V15 모델 미로드)")
st.sidebar.subheader("1. 시뮬레이션 날짜")
tomorrow_dt = datetime.strptime(tomorrow,"%Y%m%d") if tomorrow else datetime.now()+timedelta(days=1)
sim_date = st.sidebar.date_input("날짜",tomorrow_dt)
st.sidebar.subheader("2. 블레이드 스펙 (V15)")
st.sidebar.caption("중심축 회전 | 가로(발전면적) × 세로(음영계산) | 피치")
width_mm=st.sidebar.number_input("블레이드 가로(mm)",min_value=100.0,value=DEFAULT_WIDTH_MM,step=100.0)
blade_depth_mm=st.sidebar.number_input("블레이드 세로/DEPTH(mm)",min_value=10.0,value=DEFAULT_HEIGHT_MM,step=1.0)
pitch_mm=st.sidebar.number_input("블레이드 피치(mm)",min_value=10.0,value=DEFAULT_PITCH_MM,step=1.0)
half_depth_mm=blade_depth_mm/2.0
st.sidebar.caption(f"HALF_DEPTH={half_depth_mm:.1f}mm | Pitch:Depth={pitch_mm/blade_depth_mm:.2f}")
louver_count=st.sidebar.number_input("블레이드 개수",min_value=1,value=DEFAULT_LOUVER_COUNT,step=1)
st.sidebar.subheader("3. 패널 스펙")
unit_count=st.sidebar.number_input("설치 유닛 수",min_value=1,value=DEFAULT_UNIT_COUNT)
capacity_w=st.sidebar.number_input("패널 용량(W)",value=DEFAULT_CAPACITY)
target_eff=st.sidebar.number_input("패널 효율(%)",value=DEFAULT_EFFICIENCY,step=0.1)
kepco_rate=st.sidebar.number_input("전기 요금(원/kWh)",value=DEFAULT_KEPCO)
eff_factor=float(target_eff)/DEFAULT_EFFICIENCY
area_scale=(width_mm*blade_depth_mm*louver_count)/(DEFAULT_WIDTH_MM*DEFAULT_HEIGHT_MM*DEFAULT_LOUVER_COUNT)

_sim_d=sim_date.strftime("%Y-%m-%d")
times=pd.date_range(start=f"{_sim_d} 00:00",periods=24,freq="h",tz=TZ)
solpos=site.get_solarposition(times); cs=site.get_clearsky(times)
zen=np.asarray(solpos["apparent_zenith"].values,dtype=float)
az=np.asarray(solpos["azimuth"].values,dtype=float)
elev=np.asarray(solpos["apparent_elevation"].values,dtype=float)
cloud_series=np.zeros(24); temp_series=np.full(24,15.0)
if kma is not None and _sim_d.replace("-","")==tomorrow:
    kma_r=kma.reindex(times.strftime("%H00"))
    if "SKY" in kma.columns: cloud_series=kma_r["SKY"].apply(lambda x:0.0 if x=="1" else(0.5 if x=="3" else 1.0)).fillna(0).astype(float).values
    if "TMP" in kma.columns: temp_series=pd.to_numeric(kma_r["TMP"],errors="coerce").fillna(15.0).values
ghi_real=np.asarray(cs["ghi"].values,dtype=float)*(1.0-cloud_series*0.65)
dni_arr=pvlib.irradiance.dirint(ghi_real,zen,times).fillna(0).values
dhi_arr=(ghi_real-dni_arr*np.cos(np.radians(zen))).clip(0)
cloud_kma=cloud_series*9.0
xgb_angles=None
if xgb_model: xgb_angles=predict_angles_xgb(xgb_model,times,ghi_real,cloud_kma,temp_series,ANGLE_MAX)
ai_angles=xgb_angles if xgb_angles is not None else improved_rule_angles(elev,ghi_real)
angle_mode="XGBoost V15" if xgb_angles is not None else "규칙 기반"
def calc_power_day(angles):
    eff=calc_effective_poa_v15(np.asarray(angles,dtype=float),elev,az,dni_arr,dhi_arr,half_depth_mm,pitch_mm)
    return (eff[ghi_real>=10]/1000*capacity_w*unit_count*eff_factor*DEFAULT_LOSS*area_scale).sum()
pow_ai,pow_60,pow_90=calc_power_day(ai_angles),calc_power_day(np.full(24,60.0)),calc_power_day(np.full(24,90.0))
last_year=sim_date.year-1; use_xgb_ann=xgb_model is not None
wh_ai_y,wh_60_y,wh_90_y,df_monthly,df_annual=get_annual_data(last_year,half_depth_mm,pitch_mm,capacity_w,unit_count,eff_factor,DEFAULT_LOSS,use_xgb_ann)
ann_kwh_ai=wh_ai_y/1000*area_scale; ann_kwh_60=wh_60_y/1000*area_scale; ann_kwh_90=wh_90_y/1000*area_scale
kma_st="✅ 기상청 예보 연동" if (kma is not None and _sim_d.replace("-","")==tomorrow) else "⚠️ 청천 기준"
weather_st="맑음" if np.mean(cloud_series)<0.3 else("구름많음" if np.mean(cloud_series)<0.8 else "흐림")
mask_day=(times.hour>=6)&(times.hour<=19)
mn=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

st.title("☀️ BIPV AI 통합 관제 대시보드")
st.caption(f"v{__version__} (V15 선분교차 물리모델) | {_sim_d} | {weather_st} | {angle_mode} | {kma_st}")
tabs=st.tabs(["🏠 메인","📊 학습데이터","🎯 피처중요도","💡 음영원리","🔥 음영시각화","⚡ 발전량비교","📅 월별각도","🌤️ 내일스케줄","🩺 건강진단","🔧 파라미터튜닝"])

with tabs[0]:
    st.subheader("오늘의 발전 현황")
    st.markdown("""<div class="explain-box"><b>📖 주요 용어</b><br>
    • <b>GHI</b>: 수평면 전일사량 (W/m²). 발전 가능 에너지의 기준값.<br>
    • <b>SVF</b>: 하늘 조망 계수 (0~1). 높을수록 확산광을 많이 받음.<br>
    • <b>음영률(SF)</b>: 상부 블레이드가 하부 발전면을 가리는 비율. V15에서 선분교차로 정밀 계산.</div>""",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("AI 제어 발전량",f"{pow_ai/1000:.3f} kWh",f"연간 {ann_kwh_ai:.1f} kWh")
    c2.metric("고정 60° 대비",f"+{(pow_ai/pow_60-1)*100:.1f}%" if pow_60>0 else "—")
    c3.metric("수직 90° 대비",f"+{(pow_ai/pow_90-1)*100:.1f}%" if pow_90>0 else "—")
    c4.metric("예상 수익",f"{int(pow_ai/1000*kepco_rate):,} 원")
    col_l,col_r=st.columns([3,1])
    with col_l:
        st.subheader("제어 스케줄 (일중)")
        st.markdown("""<div class="explain-box"><b>💡 아침·저녁에 루버가 90°인 이유</b><br>
        GHI &lt; 10 W/m² 구간은 발전 불가. 루버를 수직으로 닫아 기계적 부하를 최소화합니다.</div>""",unsafe_allow_html=True)
        fig=make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=times[mask_day].strftime("%H:%M"),y=ghi_real[mask_day],name="GHI",marker_color="rgba(255,152,0,0.5)"),secondary_y=False)
        fig.add_trace(go.Scatter(x=times[mask_day].strftime("%H:%M"),y=ai_angles[mask_day],name="AI 각도",line=dict(color=COLOR_AI,width=3)),secondary_y=True)
        fig.update_yaxes(title_text="GHI (W/m²)",secondary_y=False); fig.update_yaxes(title_text="각도 (°)",range=[0,95],secondary_y=True)
        fig.update_layout(height=360,template=PLOT_TEMPLATE,legend=dict(orientation="h",y=1.08))
        st.plotly_chart(fig,use_container_width=True)
    with col_r:
        st.subheader("발전량 비교")
        fig_b=go.Figure(go.Bar(x=["AI","F60°","F90°"],y=[pow_ai/1000,pow_60/1000,pow_90/1000],marker_color=[COLOR_AI,COLOR_F60,COLOR_V90],text=[f"{v/1000:.3f}" for v in [pow_ai,pow_60,pow_90]],textposition="auto"))
        fig_b.update_layout(height=360,yaxis_title="kWh",template=PLOT_TEMPLATE)
        st.plotly_chart(fig_b,use_container_width=True)
    st.subheader("시간별 스케줄 테이블")
    ghi_d,elev_d=ghi_real[mask_day],elev[mask_day]
    sf_v=calc_panel_shading_vec(ai_angles[mask_day],elev_d,az[mask_day],half_depth_mm,pitch_mm)
    svf_v=sky_view_factor(ai_angles[mask_day],half_depth_mm,pitch_mm)
    def sf_st(sf,g,e):
        if g<10 or e<=0: return "— (비활성)"
        if sf<0.1: return f"{sf*100:.1f}% 🟢 양호"
        elif sf<0.3: return f"{sf*100:.1f}% 🟡 경미"
        elif sf<0.5: return f"{sf*100:.1f}% 🟠 주의"
        else: return f"{sf*100:.1f}% 🔴 심각"
    df_sch=pd.DataFrame({"시간":times[mask_day].strftime("%H:%M").tolist(),"AI 각도(°)":ai_angles[mask_day].astype(int).tolist(),
        "GHI (W/m²)":np.round(ghi_d,1).tolist(),"음영률":[sf_st(s,g,e) for s,g,e in zip(sf_v,ghi_d,elev_d)],
        "SVF":[f"{s:.2f}" if g>=10 and e>0 else "—" for s,g,e in zip(svf_v,ghi_d,elev_d)]})
    st.dataframe(df_sch,use_container_width=True,hide_index=True)
    st.markdown("""<div class="explain-box"><b>📊 음영률 판정</b><br>
    🟢 0~10% 양호 | 🟡 10~30% 경미 | 🟠 30~50% 주의 | 🔴 50%+ 심각<br>
    <b>SVF</b> 0.5 이상 = 흐린 날에도 확산광 발전 가능</div>""",unsafe_allow_html=True)

with tabs[1]:
    st.subheader("📊 학습 데이터셋 탐색")
    df_csv=load_training_csv()
    if df_csv is not None:
        if "target_angle_v15" in df_csv.columns: tc,cv="target_angle_v15","V15"
        elif "target_angle_v14" in df_csv.columns: tc,cv="target_angle_v14","V14"
        else: tc,cv="target_angle_v5","V5"
        st.success(f"✅ 학습 데이터 로드 ({cv}) | {len(df_csv):,}행 | 2014~2023년")
        st.markdown(f"""<div class="explain-box"><b>📖 학습 변수 설명</b><br>
        • <b>ghi_w_m2</b>: 수평면 전일사량. 핵심 입력값.<br>• <b>cloud_cover</b>: 운량 (0~9).<br>
        • <b>temp_actual</b>: 외기온도.<br>• <b>hour_sin/cos</b>: 시각 순환 표현.<br>
        • <b>doy_sin/cos</b>: 날짜 순환 표현.<br>• <b>{tc}</b>: {cv} 물리모델 최적 각도.</div>""",unsafe_allow_html=True)
        dp=df_csv[df_csv["ghi_w_m2"]>10].copy(); dp["month"]=pd.to_datetime(dp["timestamp"]).dt.month; dp["hour"]=pd.to_datetime(dp["timestamp"]).dt.hour
        mm={i:n for i,n in zip(range(1,13),mn)}; dp["ms"]=dp["month"].map(mm); mo=list(mm.values())
        c1,c2=st.columns(2)
        with c1:
            fig=px.box(dp,x="ms",y="ghi_w_m2",color="ms",category_orders={"ms":mo},template=PLOT_TEMPLATE)
            fig.update_layout(showlegend=False,height=300,yaxis_title="GHI (W/m²)"); st.plotly_chart(fig,use_container_width=True)
            st.caption("여름(6~8월) 중앙값·분산 큼 → 맑은 날+장마 공존")
        with c2:
            fig2=px.box(dp,x="ms",y="cloud_cover",color="ms",category_orders={"ms":mo},template=PLOT_TEMPLATE)
            fig2.update_layout(showlegend=False,height=300,yaxis_title="운량 (0~9)"); st.plotly_chart(fig2,use_container_width=True)
            st.caption("6~7월 장마 구간에 운량 높고 분산 큼")
        c3,c4=st.columns(2)
        with c3:
            fig3=px.box(dp,x="ms",y="temp_actual",color="ms",category_orders={"ms":mo},template=PLOT_TEMPLATE)
            fig3.update_layout(showlegend=False,height=300,yaxis_title="기온 (°C)"); st.plotly_chart(fig3,use_container_width=True)
        with c4:
            fig4=px.box(dp,x="ms",y=tc,color="ms",category_orders={"ms":mo},template=PLOT_TEMPLATE)
            fig4.update_layout(showlegend=False,height=300,yaxis_title="최적 각도 (°)"); st.plotly_chart(fig4,use_container_width=True)
            st.caption("여름 ~22° 최적, 겨울 ~63° 유리")
        st.markdown("---"); st.markdown("**시각·날짜 순환 변수 패턴**")
        c5,c6=st.columns(2)
        with c5:
            ha=dp.groupby("hour")[tc].mean().reset_index()
            fig5=go.Figure(go.Scatter(x=ha["hour"],y=ha[tc],mode="lines+markers",line=dict(color=COLOR_AI,width=2)))
            fig5.update_layout(height=280,xaxis_title="시각(h)",yaxis_title="평균 최적각(°)",template=PLOT_TEMPLATE); st.plotly_chart(fig5,use_container_width=True)
            st.caption("정오 전후 최적각 최저 → 태양 최고점")
        with c6:
            da=dp.groupby("month")[tc].mean().reset_index()
            fig6=go.Figure(go.Scatter(x=da["month"],y=da[tc],mode="lines+markers",line=dict(color=COLOR_F60,width=2)))
            fig6.update_layout(height=280,xaxis_title="월",yaxis_title="평균 최적각(°)",xaxis=dict(tickvals=list(range(1,13)),ticktext=mo),template=PLOT_TEMPLATE); st.plotly_chart(fig6,use_container_width=True)
            st.caption("겨울(1·12월) ~63° 높고 여름(6·7월) ~22° 낮음")
        st.markdown(f"**GHI vs 최적각 ({cv})**")
        sample=dp.sample(min(3000,len(dp)),random_state=42)
        fig7=px.scatter(sample,x="ghi_w_m2",y=tc,color="ms",opacity=0.4,template=PLOT_TEMPLATE,labels={"ghi_w_m2":"GHI",tc:"최적각(°)"})
        fig7.update_layout(height=350); st.plotly_chart(fig7,use_container_width=True)
        st.caption("GHI↑ → 최적각↓. 계절별 클러스터 분리.")
    else: st.warning("⚠️ CSV 로드 실패. GitHub 연결/파일명 확인 필요.")

with tabs[2]:
    st.subheader("🎯 피처 중요도 — V15")
    st.markdown("""<div class="explain-box"><b>📖 피처 중요도</b><br>
    XGBoost가 각 변수를 분기에서 얼마나 활용했는지의 Gain 기여도.</div>""",unsafe_allow_html=True)
    imp={"피처":["doy_cos","ghi_w_m2","hour_cos","hour_sin","temp_actual","doy_sin","cloud_cover"],
         "Gain":[0.324,0.194,0.139,0.133,0.120,0.078,0.012],
         "설명":["계절 위치","수평면 일사량","시각 cos","시각 sin","외기온도","계절 보완","운량"]}
    di=pd.DataFrame(imp).sort_values("Gain",ascending=True)
    fig=go.Figure(go.Bar(x=di["Gain"],y=di["피처"],orientation="h",
        marker_color=[COLOR_AI if g>0.12 else "#90CAF9" if g>0.06 else "#B0BEC5" for g in di["Gain"]],
        text=[f"{g:.3f}" for g in di["Gain"]],textposition="outside",customdata=di["설명"],
        hovertemplate="<b>%{y}</b><br>Gain:%{x:.3f}<br>%{customdata}<extra></extra>"))
    fig.update_layout(height=380,xaxis_title="Gain",xaxis_range=[0,0.40],template=PLOT_TEMPLATE); st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("""<div class="explain-box"><b>🥇 doy_cos (32.4%)</b><br>가장 중요. 겨울(63°) vs 여름(22°)의 계절 차이 포착.</div>
        <div class="explain-box"><b>📅 doy_cos+sin (40.2%)</b><br>계절 합산. 거의 절반의 기여도.</div>""",unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="explain-box"><b>🕐 ghi (19.4%) + hour (27.2%)</b><br>일사량과 시각. 실시간 기상+시간대별 태양 위치.</div>
        <div class="explain-box"><b>🌡️ temp (12.0%) + ☁️ cloud (1.2%)</b><br>기온→효율 영향. 운량은 GHI에 이미 반영.</div>""",unsafe_allow_html=True)
    st.markdown("---")
    m1,m2=st.columns(2)
    with m1: st.markdown("""<div class="good-box"><b>MAE = 0.70°</b><br>예측-정답 평균 0.70°. 모터 정밀도 대비 충분.</div>""",unsafe_allow_html=True)
    with m2: st.markdown("""<div class="good-box"><b>R²=0.9966 | RMSE=1.12°</b><br>99.7% 설명력. 역대 최고.</div>""",unsafe_allow_html=True)

with tabs[3]:
    st.subheader("💡 V15 선분교차 음영 원리")
    st.markdown("""<div class="explain-box"><b>📖 V15 핵심: Ray-Blade Intersection</b><br>
    기존(V13~V14): 수직 투영 근사 → 기울어진 발전면 부정확<br>
    V15: 상부 블레이드 끝→태양 광선→하부 <b>발전면(Front Face)</b> 교차점 정밀 계산<br><br>
    • <b>발전면</b> = 피봇~바깥끝 (태양전지 부착면)<br>• <b>음영 방향</b>: 피봇부터 교차점까지<br>
    • <b>90° 수직</b>: 발전면 서로 가리지 않음 → SF=0%</div>""",unsafe_allow_html=True)
    st.markdown("**블라인드를 떠올려보세요.** 눕히면 빛↑ but 그림자↑, 세우면 그림자↓ but 직사광↓. AI가 매 시간 최적 균형을 찾습니다.")
    elev_ex=st.slider("태양 고도각(°)",5,80,45,5); tilt_ex=st.slider("루버 각도(°)",15,90,30,5)
    sf_ex=float(calc_panel_shading_vec(tilt_ex,elev_ex,180.0,half_depth_mm,pitch_mm))
    svf_ex=float(sky_view_factor(tilt_ex,half_depth_mm,pitch_mm))
    prot_ex,_,gap_ex=blade_geometry(tilt_ex,half_depth_mm,blade_depth_mm,pitch_mm)
    c1,c2=st.columns([2,1])
    with c1:
        fig=go.Figure(); n_bl=4; pp=80
        for i in range(n_bl):
            yc=i*pp; hd_px=(half_depth_mm/pitch_mm)*pp; dx=hd_px*np.cos(np.radians(tilt_ex)); dy=hd_px*np.sin(np.radians(tilt_ex))
            fig.add_trace(go.Scatter(x=[-dx,0],y=[yc+dy,yc],mode="lines",line=dict(color="#999",width=3),showlegend=(i==0),name="Back" if i==0 else None))
            fig.add_trace(go.Scatter(x=[0,dx],y=[yc,yc-dy],mode="lines",line=dict(color=COLOR_AI,width=6),showlegend=(i==0),name="Front Face" if i==0 else None))
            fig.add_trace(go.Scatter(x=[0],y=[yc],mode="markers",marker=dict(size=6,color="red"),showlegend=False))
            if i>0 and sf_ex>0:
                pb=(0,(i-1)*pp); ob=(dx,(i-1)*pp-dy); tf=min(sf_ex,1.0)
                ix_x=pb[0]+tf*(ob[0]-pb[0]); ix_y=pb[1]+tf*(ob[1]-pb[1])
                fig.add_trace(go.Scatter(x=[pb[0],ix_x if sf_ex<1 else ob[0]],y=[pb[1],ix_y if sf_ex<1 else ob[1]],
                    mode="lines",line=dict(color="red",width=8),opacity=0.4,showlegend=(i==1),name=f"Shadow({sf_ex:.0%})" if i==1 else None))
        if elev_ex>0:
            rdx=-np.cos(np.radians(elev_ex))*60; rdy=-np.sin(np.radians(elev_ex))*60
            for i in range(1,n_bl):
                yc=i*pp; hd_px=(half_depth_mm/pitch_mm)*pp; ox=hd_px*np.cos(np.radians(tilt_ex)); oy=yc-hd_px*np.sin(np.radians(tilt_ex))
                fig.add_annotation(x=ox+rdx*1.5,y=oy+rdy*1.5,ax=ox-rdx*0.8,ay=oy-rdy*0.8,xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=2,arrowsize=1.2,arrowcolor="#FF8F00",arrowwidth=2)
        fig.add_shape(type="line",x0=-5,y0=-30,x1=-5,y1=(n_bl-1)*pp+50,line=dict(color="#555",width=3))
        fig.update_layout(height=420,template=PLOT_TEMPLATE,xaxis=dict(range=[-60,100],showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(range=[-50,(n_bl-1)*pp+70],showgrid=False,zeroline=False,showticklabels=False,scaleanchor="x"),
            title=f"V15 단면도 — 고도 {elev_ex}° | 루버 {tilt_ex}° | SF={sf_ex:.1%}",legend=dict(orientation="h",y=-0.05))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.metric("음영률(SF)",f"{sf_ex*100:.1f}%",delta="양호 ✅" if sf_ex<0.1 else("경미 🟡" if sf_ex<0.3 else "주의 🔴"),delta_color="off")
        st.metric("SVF",f"{svf_ex:.2f}",delta="높음 ✅" if svf_ex>0.5 else "낮음 🔴",delta_color="off")
        st.markdown(f"""<div class="explain-box"><b>기하학 정보</b><br>DEPTH {blade_depth_mm:.0f}mm | PITCH {pitch_mm:.0f}mm<br>
        돌출 {float(prot_ex):.1f}mm | 틈 {float(gap_ex):.1f}mm<br><br><b>V15 음영</b><br>선분교차 → 피봇→교차점 = 음영</div>
        <div class="explain-box"><b>💡 AI 전략</b><br>각도↓→직달광↑ but 음영↑<br>각도↑→음영↓ SVF↑ 직달광↓<br>AI=매 시간 순발전량 최대</div>""",unsafe_allow_html=True)

with tabs[4]:
    st.subheader("🔥 음영 시각화 — 조건별 단면 비교 (V15)")
    st.markdown("""<div class="explain-box">다양한 계절·각도 조건의 <b>자기 음영 패턴</b> 단면 비교.<br>
    <b>파란선</b>=발전면 | <b>빨간선</b>=음영 | <b>주황화살표</b>=태양 광선</div>""",unsafe_allow_html=True)
    conds=[(60,15,"동지 09시\ntilt=60° elev=15°"),(63,29,"동지 정오\ntilt=63° elev=29°"),(60,15,"동지 16시\ntilt=60° elev=15°"),
           (22,40,"하지 09시\ntilt=22° elev=40°"),(22,76,"하지 정오\ntilt=22° elev=76°"),(22,40,"하지 16시\ntilt=22° elev=40°"),
           (15,76,"하지 t=15° e=76°"),(45,76,"하지 t=45° e=76°"),(90,29,"동지 t=90° e=29°")]
    for rs in range(0,len(conds),3):
        cols=st.columns(3)
        for ci,(tc2,ec,lc) in enumerate(conds[rs:rs+3]):
            with cols[ci]:
                sfc=float(calc_panel_shading_vec(tc2,ec,180.0,half_depth_mm,pitch_mm))
                fc=go.Figure(); nb=4; pc=70
                for i in range(nb):
                    yc=i*pc; hp=(half_depth_mm/pitch_mm)*pc; dx=hp*np.cos(np.radians(tc2)); dy=hp*np.sin(np.radians(tc2))
                    fc.add_trace(go.Scatter(x=[-dx,0],y=[yc+dy,yc],mode="lines",line=dict(color="#999",width=2),showlegend=False))
                    fc.add_trace(go.Scatter(x=[0,dx],y=[yc,yc-dy],mode="lines",line=dict(color=COLOR_AI,width=5),showlegend=False))
                    fc.add_trace(go.Scatter(x=[0],y=[yc],mode="markers",marker=dict(size=4,color="red"),showlegend=False))
                    if i>0 and sfc>0:
                        pb=(0,(i-1)*pc); ob=(dx,(i-1)*pc-dy); tf=min(sfc,1.0)
                        ix_x=pb[0]+tf*(ob[0]-pb[0]); ix_y=pb[1]+tf*(ob[1]-pb[1])
                        fc.add_trace(go.Scatter(x=[pb[0],ix_x if sfc<1 else ob[0]],y=[pb[1],ix_y if sfc<1 else ob[1]],mode="lines",line=dict(color="red",width=7),opacity=0.5,showlegend=False))
                if ec>0:
                    rdx=-np.cos(np.radians(ec))*40; rdy=-np.sin(np.radians(ec))*40
                    for i in range(1,nb):
                        yc=i*pc; hp=(half_depth_mm/pitch_mm)*pc; ox=hp*np.cos(np.radians(tc2)); oy=yc-hp*np.sin(np.radians(tc2))
                        fc.add_annotation(x=ox+rdx,y=oy+rdy,ax=ox-rdx*0.5,ay=oy-rdy*0.5,xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=2,arrowsize=1,arrowcolor="#FF8F00",arrowwidth=1.5)
                fc.add_shape(type="line",x0=-3,y0=-20,x1=-3,y1=(nb-1)*pc+40,line=dict(color="#555",width=2))
                fc.update_layout(height=300,template=PLOT_TEMPLATE,margin=dict(l=5,r=5,t=35,b=5),
                    xaxis=dict(range=[-50,80],showgrid=False,zeroline=False,showticklabels=False,visible=False),
                    yaxis=dict(range=[-30,(nb-1)*pc+50],showgrid=False,zeroline=False,showticklabels=False,visible=False,scaleanchor="x"),
                    title=dict(text=f"{lc}<br>SF={sfc:.0%}",font=dict(size=11)))
                st.plotly_chart(fc,use_container_width=True)
    st.markdown("""<div class="explain-box"><b>📖 패턴 해석</b><br>
    <b>동지 (elev 15~29°)</b>: 태양 낮아 어떤 각도에서도 음영 거의 없음 → 직달광 최대화(~63°)가 최적<br>
    <b>하지 정오 (elev 76°)</b>: 낮은 각도에서 음영 50%+ → AI가 22° 부근에서 음영과 직달광 균형<br>
    <b>90° 수직</b>: SF=0%이지만 직달광 입사각 커서 발전 효율 낮음</div>""",unsafe_allow_html=True)

with tabs[5]:
    st.subheader("⚡ AI vs 고정60° vs 수직90°")
    if use_xgb_ann: st.success("✅ XGBoost V15 모델 기반 시뮬레이션")
    else: st.info("ℹ️ 규칙 기반 (XGBoost 미로드)")
    fig=go.Figure()
    for col,color,name in [("AI",COLOR_AI,"AI 제어"),("고정60°",COLOR_F60,"고정 60°"),("수직90°",COLOR_V90,"수직 90°")]:
        fig.add_trace(go.Bar(x=mn,y=df_monthly[col]*area_scale/1000,name=name,marker_color=color))
    fig.update_layout(barmode="group",height=400,template=PLOT_TEMPLATE,yaxis_title="발전량(kWh)",legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown(f"""<div class="explain-box"><b>📊 발전량 해석</b><br><br>
    <b>🌞 여름</b>: AI가 음영 고려한 최적각(~22°)으로 고정60° 대비 +7~17%.<br><br>
    <b>❄️ 겨울</b>: V15 최적각이 ~63°로 고정60°에 매우 가까움.
    <b>대시보드는 청천(clearsky) 기준</b>이라 겨울 직달광이 과대 → 60°와 AI 차이가 미미하게 보일 수 있음.
    <b>실측 기상 Colab 결과</b>: AI 621.6kWh vs 고정60° 589.1kWh (<b>+5.5%</b>). 흐린 날 확산광 전략이 핵심.<br><br>
    <b>수직 90°</b>: SF=0%이지만 직달광 입사각이 커서 연중 불리.</div>""",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.metric("연간 AI",f"{ann_kwh_ai:.1f} kWh")
    c2.metric("연간 F60°",f"{ann_kwh_60:.1f} kWh",f"{(ann_kwh_60/ann_kwh_ai-1)*100:+.1f}%" if ann_kwh_ai>0 else "—")
    c3.metric("연간 F90°",f"{ann_kwh_90:.1f} kWh",f"{(ann_kwh_90/ann_kwh_ai-1)*100:+.1f}%" if ann_kwh_ai>0 else "—")
    st.markdown("""<div class="warn-box"><b>⚠️ 대시보드 vs Colab 차이</b><br>
    대시보드=청천 모델. 실측 기상(구름·장마) 포함 Colab 결과: AI +5.5%.
    AI 이득은 흐린 날 확산광 전략에서 발생하며 청천에서는 과소 표시될 수 있습니다.</div>""",unsafe_allow_html=True)
    dc=df_monthly.copy(); dc["AI_c"]=(dc["AI"]*area_scale/1000).cumsum(); dc["F60_c"]=(dc["고정60°"]*area_scale/1000).cumsum(); dc["F90_c"]=(dc["수직90°"]*area_scale/1000).cumsum()
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=mn,y=dc["AI_c"],name="AI",line=dict(color=COLOR_AI,width=3)))
    fig2.add_trace(go.Scatter(x=mn,y=dc["F60_c"],name="고정60°",line=dict(color=COLOR_F60,width=2,dash="dash")))
    fig2.add_trace(go.Scatter(x=mn,y=dc["F90_c"],name="수직90°",line=dict(color=COLOR_V90,width=2,dash="dot")))
    fig2.update_layout(height=320,yaxis_title="누적 kWh",title="연간 누적 발전량",template=PLOT_TEMPLATE,legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fig2,use_container_width=True)
    st.caption("기울기가 가파른 구간=발전 많은 계절. AI 곡선이 위에 있을수록 제어 효과 큼.")

with tabs[6]:
    st.subheader("📅 월별 AI 각도 분포")
    dp2=df_annual[df_annual["ghi"]>=10].copy(); dp2["mn"]=dp2["timestamp"].dt.month
    mn_map={i:n for i,n in zip(range(1,13),mn)}; dp2["ms"]=dp2["mn"].map(mn_map)
    fig=go.Figure()
    for m,ms in mn_map.items(): fig.add_trace(go.Box(y=dp2[dp2["mn"]==m]["angle_ai"],name=ms,marker_color=COLOR_AI,boxmean=True))
    fig.add_hline(y=ANGLE_MIN,line_dash="dash",line_color="red",annotation_text=f"최소각 {ANGLE_MIN}°")
    fig.update_layout(height=420,yaxis_title="루버 각도(°)",showlegend=False,template=PLOT_TEMPLATE)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("""<div class="explain-box"><b>📊 월별 각도 해석 (V15)</b><br><br>
    <b>🌞 여름 ~22°</b>: 태양 고도 높지만 15°까지 눕히면 SF=52%. AI가 음영과 직달광 균형점 선택.<br><br>
    <b>❄️ 겨울 56~63°</b>: 태양 낮아(~29°) 음영 거의 없음. AOI 최소화 위해 60° 부근 최적.<br><br>
    <b>🍂 봄·가을 25~51°</b>: 계절 전환에 따라 유동 조정.</div>""",unsafe_allow_html=True)
    st.subheader("월별 평균 각도 & 발전량")
    v15_ref={1:62.6,2:55.6,3:42.2,4:25.0,5:23.7,6:22.1,7:23.3,8:25.3,9:38.9,10:51.4,11:60.4,12:61.7}
    ds=df_monthly.copy(); ds["month_s"]=list(mn_map.values()); ds=ds.rename(columns={"avg_angle":"시뮬 평균각(°)"})
    ds["V15 참조각"]=[v15_ref[m] for m in range(1,13)]
    st.dataframe(ds[["month_s","시뮬 평균각(°)","V15 참조각","AI","고정60°","수직90°"]].round(1),use_container_width=True,hide_index=True)

with tabs[7]:
    st.subheader("🌤️ 내일 예측 스케줄")
    if kma is None: st.error("❌ 기상청 API 실패. 청천 기준.")
    else: st.success(f"✅ 기상청 연동 | {tomorrow}")
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=times[mask_day].strftime("%H:%M"),y=ghi_real[mask_day],name="예측 GHI",marker_color="rgba(255,152,0,0.5)"),secondary_y=False)
    fig.add_trace(go.Scatter(x=times[mask_day].strftime("%H:%M"),y=ai_angles[mask_day],name="AI 각도",line=dict(color=COLOR_AI,width=3),mode="lines+markers"),secondary_y=True)
    fig.update_yaxes(title_text="GHI",secondary_y=False); fig.update_yaxes(title_text="각도(°)",range=[0,95],secondary_y=True)
    fig.update_layout(height=380,template=PLOT_TEMPLATE,title=f"내일({tomorrow}) 스케줄")
    st.plotly_chart(fig,use_container_width=True)
    sf_tom=calc_panel_shading_vec(ai_angles[mask_day],elev[mask_day],az[mask_day],half_depth_mm,pitch_mm)
    svf_tom=sky_view_factor(ai_angles[mask_day],half_depth_mm,pitch_mm)
    df_tom=pd.DataFrame({"시간":times[mask_day].strftime("%H:%M").tolist(),"예측 GHI":np.round(ghi_real[mask_day],1).tolist(),
        "기온(°C)":np.round(temp_series[mask_day],1).tolist(),"AI 각도(°)":ai_angles[mask_day].astype(int).tolist(),
        "음영률":[f"{s*100:.1f}%" if g>=10 and e>0 else "—" for s,g,e in zip(sf_tom,ghi_real[mask_day],elev[mask_day])],
        "SVF":[f"{s:.2f}" if g>=10 and e>0 else "—" for s,g,e in zip(svf_tom,ghi_real[mask_day],elev[mask_day])]})
    st.dataframe(df_tom,use_container_width=True,hide_index=True)

with tabs[8]:
    st.subheader("🩺 시스템 건강 진단")
    st.markdown("""<div class="explain-box"><b>📖 건강진단</b><br>
    AI 예측 발전량(P_sim) vs 실측(P_actual). 차이로 패널 오염·고장 자동 감지.<br>
    • <b>Health Ratio</b> = P_actual/P_sim (100%=정상)</div>""",unsafe_allow_html=True)
    st.info("📌 실측 센서 미연동. 슬라이더로 시뮬레이션.")
    c1,c2=st.columns(2)
    with c1: p_pct=st.slider("실측 발전량 비율(%)",10,110,95)
    with c2: w_t=st.number_input("WARNING(%)",value=90); c_t=st.number_input("CRITICAL(%)",value=75)
    hr=p_pct/100.0
    if hr>=w_t/100: status,color="✅ NORMAL","green"
    elif hr>=c_t/100: status,color="⚠️ WARNING","orange"
    else: status,color="🔴 CRITICAL","red"
    c1,c2,c3=st.columns(3)
    c1.metric("P_sim",f"{pow_ai/1000:.3f} kWh"); c2.metric("P_actual",f"{pow_ai/1000*hr:.3f} kWh"); c3.metric("Health",f"{hr:.2%}",status)
    fig=go.Figure(go.Indicator(mode="gauge+number+delta",value=p_pct,delta={"reference":100,"valueformat":".1f"},
        gauge={"axis":{"range":[0,110]},"bar":{"color":color},
        "steps":[{"range":[0,c_t],"color":"rgba(244,67,54,0.15)"},{"range":[c_t,w_t],"color":"rgba(255,152,0,0.15)"},{"range":[w_t,110],"color":"rgba(76,175,80,0.15)"}],
        "threshold":{"line":{"color":"gray","width":2},"value":w_t}},title={"text":f"상태: {status}"}))
    fig.update_layout(height=320); st.plotly_chart(fig,use_container_width=True)
    st.markdown("""<div class="explain-box"><b>📊 판정</b><br>
    <b>✅ 90%+</b>: 정상 | <b>⚠️ 75~90%</b>: 오염 의심 | <b>🔴 75%↓</b>: 고장 가능, 즉시 점검</div>""",unsafe_allow_html=True)
    st.caption("📌 실측 연동 후 P_actual 자동 입력 예정 (특허 청구항 4)")

with tabs[9]:
    st.subheader("🔧 파라미터 민감도 — V15")
    st.markdown("파라미터 변화에 따른 발전량·음영률 변화를 확인합니다.")
    t_loss=st.slider("손실률",0.70,0.95,DEFAULT_LOSS,0.01)
    st.markdown("---"); st.markdown("#### 📐 DEPTH → 연간 발전량")
    st.caption("DEPTH↑→돌출↑→음영↑. DEPTH↓→음영↓ but 발전면적↓.")
    dr=np.arange(60,181,10); pdl=[]
    for bd in dr:
        _,_,_,dm,_=get_annual_data(last_year,bd/2.0,pitch_mm,capacity_w,unit_count,eff_factor,t_loss)
        pdl.append(dm["AI"].sum()*area_scale/1000)
    fig1=go.Figure(go.Scatter(x=dr,y=pdl,mode="lines+markers",line=dict(color=COLOR_AI,width=2)))
    fig1.add_vline(x=blade_depth_mm,line_dash="dash",line_color=COLOR_F60,annotation_text=f"현재 {blade_depth_mm:.0f}mm")
    fig1.update_layout(height=300,xaxis_title="DEPTH(mm)",yaxis_title="kWh",template=PLOT_TEMPLATE); st.plotly_chart(fig1,use_container_width=True)
    st.markdown("---"); st.markdown("#### 📏 피치 → 정오 음영률")
    st.caption("피치↑→간격↑→음영↓. 단 설치 면적 증가.")
    pr=np.arange(80,201,5); sfl=[float(calc_panel_shading_vec(45,60,180,blade_depth_mm/2,float(p)))*100 for p in pr]
    fig2=go.Figure(go.Scatter(x=pr,y=sfl,mode="lines+markers",line=dict(color=COLOR_F60,width=2)))
    fig2.add_vline(x=pitch_mm,line_dash="dash",line_color=COLOR_AI,annotation_text=f"현재 {pitch_mm:.0f}mm")
    fig2.add_hline(y=30,line_dash="dot",line_color="green",annotation_text="30% 양호")
    fig2.update_layout(height=300,xaxis_title="PITCH(mm)",yaxis_title="음영률%",template=PLOT_TEMPLATE); st.plotly_chart(fig2,use_container_width=True)
    st.caption("💡 V15: PPO 강화학습 향후 추가 가능 (현재 XGBoost/규칙 기반)")
