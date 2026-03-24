# ==============================================================================
# BIPV 통합 관제 시스템 v9.2
# V15 선분교차 물리모델 | 실측 기상 연간 시뮬 | 비전문가용 설명 강화
# ==============================================================================
__version__ = "9.2"

import os, io, json
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
DC, DE, DL = 300, 18.7, 0.85           # capacity, efficiency, loss
DK, DU, DLC = 210, 1, 20               # kepco, units, louver count
DW, DH, DP = 900.0, 114.0, 114.0       # width, height(depth), pitch
HD = 57.0                               # half depth
AMIN, AMAX, ANIGHT = 15, 90, 90

GH_BASE = "https://raw.githubusercontent.com/oopartstiago-debug/260310-BIPVpatenttest/main"
MODEL_URL = f"{GH_BASE}/bipv_xgboost_model_v15.pkl"
CSV_URL   = f"{GH_BASE}/bipv_ai_master_data_v15.csv"
MODEL_FN  = "bipv_xgboost_model_v15.pkl"
PT = "plotly_white"
C_AI, C_F60, C_V90 = "#1976D2", "#F57C00", "#757575"
site = Location(LAT, LON, tz=TZ)

# ==============================================================================
# 로드
# ==============================================================================
@st.cache_resource
def load_model():
    if not _XGB_AVAILABLE: return None
    if not os.path.exists(MODEL_FN):
        try:
            r = requests.get(MODEL_URL, headers=GH_HEADERS, timeout=30)
            if r.status_code != 200: return None
            with open(MODEL_FN, "wb") as f: f.write(r.content)
        except: return None
    try: return joblib.load(MODEL_FN)
    except: return None

@st.cache_data(ttl=600)  # 10분 캐시 (디버깅용 단축)
def load_csv():
    """CSV 로드 — 디버그 정보 포함"""
    url = CSV_URL
    try:
        r = requests.get(url, headers=GH_HEADERS, timeout=60)
        status = r.status_code
        if status == 404:
            return None, f"HTTP 404: 파일을 찾을 수 없음. URL={url}"
        if status == 401:
            return None, f"HTTP 401: 인증 실패. GH_TOKEN 확인 필요."
        if status != 200:
            return None, f"HTTP {status}: 알 수 없는 오류"
        content = r.text
        if len(content) < 100:
            return None, f"파일 내용 너무 짧음 ({len(content)}자). GitHub LFS?"
        df = pd.read_csv(io.StringIO(content))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Seoul")
        # 디버깅: 실제 컬럼 반환
        cols_info = [c for c in df.columns if "target" in c]
        return df, f"OK | 컬럼: {cols_info} | {len(df)}행"
    except Exception as e:
        return None, f"예외: {str(e)[:200]}"

# ==============================================================================
# V15 물리 — 선분교차
# ==============================================================================
def blade_geo(tilt, hd=HD, bd=DH, p=DP):
    t = np.radians(np.asarray(tilt, dtype=float))
    return hd*np.cos(t), bd*np.sin(t), np.maximum(p-bd*np.sin(t), 0.0)

def panel_sf(tilt, elev, az, hd=HD, p=DP, sa=180.0):
    """V15 선분교차 음영률"""
    tilt=np.asarray(tilt,dtype=float); elev=np.asarray(elev,dtype=float); az=np.asarray(az,dtype=float)
    tr=np.radians(tilt); er=np.radians(np.clip(elev,0.1,89.9))
    ct,st2=np.cos(tr),np.sin(tr)
    rx,ry=hd*ct,p-hd*st2; fx,fy=hd*ct,-hd*st2
    adr=np.radians(az-sa); rdx=-np.cos(er)*np.cos(adr); rdy=-np.sin(er)
    dn=fx*rdy-fy*rdx; dns=np.where(np.abs(dn)<1e-12,1e-12,dn)
    tb=(rx*rdy-ry*rdx)/dns; ta=(rx*fy-ry*fx)/dns
    sf=np.where((ta>0)&(tb>1),1.0,np.where((ta>0)&(tb>0)&(tb<=1),tb,0.0))
    return np.clip(np.where(elev<=0,0.0,sf),0,1)

def svf(tilt, hd=HD, p=DP):
    return np.clip(1-hd*np.cos(np.radians(np.asarray(tilt,dtype=float)))/p,0.05,1)

def eff_poa(tilt, elev, az, dni, dhi, hd=HD, p=DP):
    tilt=np.asarray(tilt,dtype=float); elev=np.asarray(elev,dtype=float); az=np.asarray(az,dtype=float)
    sf=panel_sf(tilt,elev,az,hd,p); s=svf(tilt,hd,p)
    pd2=np.maximum(pvlib.irradiance.beam_component(surface_tilt=tilt,surface_azimuth=180.0,
        solar_zenith=90-elev,solar_azimuth=az,dni=dni),0)
    dd=dhi*(1+np.cos(np.radians(tilt)))/2
    return np.maximum(pd2*(1-sf*0.7)+dd*s,0)

# ==============================================================================
# XGBoost
# ==============================================================================
def predict_xgb(model,times,ghi,cloud,temp,cap):
    n=len(times); h=np.asarray(times.hour,dtype=float); d=np.asarray(times.dayofyear,dtype=float)
    X=np.column_stack([np.sin(2*np.pi*h/24)[:n],np.cos(2*np.pi*h/24)[:n],
        np.sin(2*np.pi*d/365)[:n],np.cos(2*np.pi*d/365)[:n],
        np.asarray(ghi,dtype=float).ravel()[:n],np.asarray(cloud,dtype=float).ravel()[:n],
        np.asarray(temp,dtype=float).ravel()[:n]])
    try: p=model.predict(X)
    except: return None
    p=np.clip(np.asarray(p).ravel()[:n],AMIN,min(AMAX,cap))
    p[np.asarray(ghi).ravel()[:n]<10]=ANIGHT; return p.astype(float)

def predict_annual(model,times,ghi,tmp=15.0):
    return predict_xgb(model,times,ghi,np.zeros(len(times)),np.full(len(times),tmp),AMAX)

def rule_angles(elev,ghi):
    return np.where(ghi<10,float(ANIGHT),np.clip(90-elev*0.7,AMIN,AMAX).astype(float))

# ==============================================================================
# 기상청
# ==============================================================================
@st.cache_data(ttl=3600)
def get_kma():
    dk=urllib.parse.unquote(KMA_SERVICE_KEY)
    bd=datetime.now().strftime("%Y%m%d"); nh=datetime.now().hour
    av=[2,5,8,11,14,17,20,23]; bt=max([h for h in av if h<=nh] or [23])
    if bt==23 and nh<2: bd=(datetime.now()-timedelta(days=1)).strftime("%Y%m%d")
    url="http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    pa={"serviceKey":dk,"numOfRows":"1000","dataType":"JSON","base_date":bd,"base_time":f"{bt:02d}00","nx":NX,"ny":NY}
    try:
        res=requests.get(url,params=pa,timeout=10).json()
        it=res["response"]["body"]["items"]["item"]; df=pd.DataFrame(it)
        tom=(datetime.now()+timedelta(days=1)).strftime("%Y%m%d")
        dt=df[df["fcstDate"]==tom].drop_duplicates(subset=["fcstDate","fcstTime","category"])
        return dt.pivot(index="fcstTime",columns="category",values="fcstValue"),tom
    except: return None,None

# ==============================================================================
# 연간 데이터 — clearsky + cloud factor로 실측 근사
# ==============================================================================
@st.cache_data(ttl=86400)
def get_annual(year,hd,p,cw,uc,ef,dl,use_xgb=False):
    ty=pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31 23:00",freq="h",tz=TZ)
    sp=site.get_solarposition(ty); cs=site.get_clearsky(ty)
    ghi_cs=np.asarray(cs["ghi"].values,dtype=float)
    zn,az2=sp["apparent_zenith"].values,sp["azimuth"].values; el=90-zn

    # 월별 구름 감쇠 계수 (서울 10년 평균 기상 반영)
    # 1~12월: 겨울 맑음(0.15), 장마(0.45), 가을 맑음(0.10)
    cloud_monthly = {1:0.20, 2:0.20, 3:0.25, 4:0.25, 5:0.30,
                     6:0.45, 7:0.50, 8:0.40, 9:0.25, 10:0.15, 11:0.15, 12:0.20}
    months = pd.DatetimeIndex(ty).month
    cloud_factor = np.array([cloud_monthly[m] for m in months])
    ghi_y = ghi_cs * (1.0 - cloud_factor * 0.65)

    dn=pvlib.irradiance.dirint(ghi_y,zn,ty).fillna(0).values
    dh=(ghi_y-dn*np.cos(np.radians(zn))).clip(0)

    if use_xgb:
        md=load_model()
        ai=predict_annual(md,ty,ghi_y) if md else None
        if ai is None: ai=rule_angles(el,ghi_y)
    else: ai=rule_angles(el,ghi_y)

    def en(ang):
        e=eff_poa(np.asarray(ang,dtype=float),el,az2,dn,dh,hd,p)
        return (e[ghi_y>=10]/1000*cw*uc*ef*dl).sum()
    wa,w6,w9=en(ai),en(np.full_like(ghi_y,60.0)),en(np.full_like(ghi_y,90.0))
    da=pd.DataFrame({"timestamp":ty,"ghi":ghi_y,"zenith":zn,"azimuth":az2,"elevation":el,"angle_ai":ai})
    da["month"]=da["timestamp"].dt.month
    ml=[]
    for m in range(1,13):
        mm=(da["month"]==m)&(ghi_y>=10)
        def em(ang):
            e=eff_poa(np.asarray(ang,dtype=float),el[mm],az2[mm],dn[mm],dh[mm],hd,p)
            return (e/1000*cw*uc*ef*dl).sum()
        ml.append({"month":m,"AI":em(da.loc[mm,"angle_ai"].values),
                   "고정60°":em(np.full(mm.sum(),60.0)),"수직90°":em(np.full(mm.sum(),90.0)),
                   "avg_angle":float(da.loc[mm,"angle_ai"].mean())})
    return wa,w6,w9,pd.DataFrame(ml),da

# ==============================================================================
# CSS
# ==============================================================================
st.markdown("""<style>
.stApp{background:#F8F9FA}
.ex{background:#EEF2FF;border-left:3px solid #3B5BDB;padding:12px 16px;border-radius:6px;margin:10px 0;font-size:.91rem;color:#1A1A2E;line-height:1.65}
.wa{background:#FFF3E0;border-left:3px solid #E65100;padding:12px 16px;border-radius:6px;margin:10px 0;font-size:.91rem;color:#3E2000;line-height:1.65}
.gd{background:#E8F5E9;border-left:3px solid #2E7D32;padding:12px 16px;border-radius:6px;margin:10px 0;font-size:.91rem;color:#1B3A1E;line-height:1.65}
div[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:700}
div[data-testid="stMetricLabel"]{font-size:.85rem;color:#555}
</style>""", unsafe_allow_html=True)

# ==============================================================================
# 데이터 준비
# ==============================================================================
mdl = load_model() if _XGB_AVAILABLE else None
kma, tom = get_kma()

st.sidebar.title("■ 통합 환경 설정")
if mdl: st.sidebar.success("✅ XGBoost V15 로드")
else: st.sidebar.warning("⚠️ 규칙 기반 모드")
st.sidebar.subheader("1. 날짜")
tom_dt=datetime.strptime(tom,"%Y%m%d") if tom else datetime.now()+timedelta(days=1)
sim_date=st.sidebar.date_input("날짜",tom_dt)
st.sidebar.subheader("2. 블레이드 (V15)")
st.sidebar.caption("중심축 회전 | Pitch:Depth 1:1")
wm=st.sidebar.number_input("가로(mm)",min_value=100.0,value=DW,step=100.0)
bdm=st.sidebar.number_input("세로/DEPTH(mm)",min_value=10.0,value=DH,step=1.0)
pm=st.sidebar.number_input("피치(mm)",min_value=10.0,value=DP,step=1.0)
hdm=bdm/2; st.sidebar.caption(f"HALF={hdm:.1f}mm | 비율={pm/bdm:.2f}")
lc=st.sidebar.number_input("블레이드 수",min_value=1,value=DLC,step=1)
st.sidebar.subheader("3. 패널")
uc=st.sidebar.number_input("유닛 수",min_value=1,value=DU)
cw=st.sidebar.number_input("용량(W)",value=DC)
te=st.sidebar.number_input("효율(%)",value=DE,step=0.1)
kr=st.sidebar.number_input("요금(원/kWh)",value=DK)
ef=float(te)/DE; asc=(wm*bdm*lc)/(DW*DH*DLC)

sd=sim_date.strftime("%Y-%m-%d")
ts=pd.date_range(start=f"{sd} 00:00",periods=24,freq="h",tz=TZ)
sp=site.get_solarposition(ts); cs=site.get_clearsky(ts)
zn=np.asarray(sp["apparent_zenith"].values,dtype=float)
az=np.asarray(sp["azimuth"].values,dtype=float)
el=np.asarray(sp["apparent_elevation"].values,dtype=float)
cl=np.zeros(24); tp=np.full(24,15.0)
if kma is not None and sd.replace("-","")==tom:
    kr2=kma.reindex(ts.strftime("%H00"))
    if "SKY" in kma.columns: cl=kr2["SKY"].apply(lambda x:0 if x=="1" else(0.5 if x=="3" else 1)).fillna(0).astype(float).values
    if "TMP" in kma.columns: tp=pd.to_numeric(kr2["TMP"],errors="coerce").fillna(15).values
ghi=np.asarray(cs["ghi"].values,dtype=float)*(1-cl*0.65)
dn=pvlib.irradiance.dirint(ghi,zn,ts).fillna(0).values
dh=(ghi-dn*np.cos(np.radians(zn))).clip(0)
ck=cl*9
xa=None
if mdl: xa=predict_xgb(mdl,ts,ghi,ck,tp,AMAX)
ai=xa if xa is not None else rule_angles(el,ghi)
am="XGBoost V15" if xa is not None else "규칙 기반"
def dpow(ang):
    e=eff_poa(np.asarray(ang,dtype=float),el,az,dn,dh,hdm,pm)
    return (e[ghi>=10]/1000*cw*uc*ef*DL*asc).sum()
pa,p6,p9=dpow(ai),dpow(np.full(24,60.0)),dpow(np.full(24,90.0))
ly=sim_date.year-1; ux=mdl is not None

# 연간 발전량: CSV 실측 데이터 우선, 없으면 clearsky 시뮬
df_csv_raw, csv_err_raw = load_csv()

@st.cache_data(ttl=86400)
def get_annual_from_csv(df_src, hd, p, cw_p, uc_p, ef_p, dl_p):
    """CSV 실측 기상 데이터로 발전량 직접 계산 — Colab과 동일 결과"""
    df2 = df_src.copy()
    df2["ts"] = pd.to_datetime(df2["timestamp"])
    # 2023년 (또는 가장 최근 1년) 필터
    max_year = df2["ts"].dt.year.max()
    df2 = df2[df2["ts"].dt.year == max_year]
    
    ghi2 = df2["ghi_w_m2"].values
    mask = ghi2 >= 10  # numpy boolean 배열
    el2 = df2["solar_elevation"].values
    az2 = df2["solar_azimuth"].values
    dn2 = df2["dni"].values
    dh2 = df2["dhi"].values
    
    # AI 각도: 모델 예측
    model = load_model()
    if model is not None:
        h2 = df2["ts"].dt.hour.values.astype(float)
        d2 = df2["ts"].dt.dayofyear.values.astype(float)
        X2 = np.column_stack([np.sin(2*np.pi*h2/24), np.cos(2*np.pi*h2/24),
            np.sin(2*np.pi*d2/365), np.cos(2*np.pi*d2/365),
            ghi2, df2["cloud_cover"].values, df2["temp_actual"].values])
        ai2 = np.clip(model.predict(X2), AMIN, AMAX)
        ai2[ghi2 < 10] = ANIGHT
    else:
        ai2 = rule_angles(el2, ghi2)
    
    def en(ang):
        e = eff_poa(np.asarray(ang,dtype=float), el2, az2, dn2, dh2, hd, p)
        return (e[mask]/1000*cw_p*uc_p*ef_p*dl_p).sum()
    
    wa2, w62, w92 = en(ai2), en(np.full(len(el2), 60.0)), en(np.full(len(el2), 90.0))
    
    da2 = pd.DataFrame({
        "timestamp": df2["ts"].values,
        "ghi": ghi2,
        "zenith": df2["solar_zenith"].values if "solar_zenith" in df2.columns else 90.0 - el2,
        "azimuth": az2,
        "elevation": el2,
        "angle_ai": ai2
    })
    da2["month"] = pd.to_datetime(da2["timestamp"]).dt.month
    
    months_arr = pd.to_datetime(da2["timestamp"]).dt.month.values
    
    ml2 = []
    for m in range(1,13):
        mm = (months_arr == m) & mask
        if mm.sum() == 0:
            ml2.append({"month":m,"AI":0,"고정60°":0,"수직90°":0,"avg_angle":0})
            continue
        def em(ang):
            e = eff_poa(np.asarray(ang,dtype=float), el2[mm], az2[mm], dn2[mm], dh2[mm], hd, p)
            return (e/1000*cw_p*uc_p*ef_p*dl_p).sum()
        ml2.append({"month":m,"AI":em(ai2[mm]),
                    "고정60°":em(np.full(mm.sum(),60.0)),"수직90°":em(np.full(mm.sum(),90.0)),
                    "avg_angle":float(ai2[mm].mean())})
    return wa2, w62, w92, pd.DataFrame(ml2), da2, max_year

# CSV 로드 성공 → 실측 기반, 실패 → clearsky 시뮬
if df_csv_raw is not None and "solar_elevation" in df_csv_raw.columns:
    wa,w6,w9,dmo,dan,data_year = get_annual_from_csv(df_csv_raw, hdm, pm, cw, uc, ef, DL)
    annual_source = f"실측 기상 ({data_year}년)"
else:
    wa,w6,w9,dmo,dan = get_annual(ly,hdm,pm,cw,uc,ef,DL,ux)
    annual_source = f"청천 모델 근사 ({ly}년)"

ka=wa/1000*asc; k6=w6/1000*asc; k9=w9/1000*asc
ks="✅ 기상청 연동" if (kma is not None and sd.replace("-","")==tom) else "⚠️ 청천 기준"
ws="맑음" if np.mean(cl)<0.3 else("구름많음" if np.mean(cl)<0.8 else "흐림")
md2=(ts.hour>=6)&(ts.hour<=19)
mn=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ==============================================================================
# 탭
# ==============================================================================
st.title("☀️ BIPV AI 통합 관제 대시보드")
st.caption(f"v{__version__} (V15 선분교차) | {sd} | {ws} | {am} | {ks}")
tabs=st.tabs(["🏠 메인","📊 학습데이터","🎯 피처중요도","💡 음영원리","🔥 음영시각화","⚡ 발전량비교","📅 월별각도","🌤️ 내일스케줄","🩺 건강진단","🔧 파라미터튜닝"])

# ═══ 메인 ═══
with tabs[0]:
    st.subheader("오늘의 발전 현황")
    st.markdown("""<div class="ex"><b>📖 이 시스템은 무엇인가요?</b><br><br>
    건물 외벽에 설치된 <b>루버형 태양광 패널(BIPV)</b>의 각도를 AI가 자동으로 제어하여
    <b>발전량을 최대화</b>하는 시스템입니다.<br><br>
    <b>왜 각도 제어가 필요한가요?</b><br>
    태양은 계절·시간에 따라 위치가 변합니다. 루버를 고정하면 특정 시간에만 효율적이지만,
    AI가 매 시간 최적 각도를 찾으면 연간 발전량이 크게 증가합니다.<br><br>
    <b>주요 용어</b><br>
    • <b>GHI</b>: 수평면 전일사량 (W/m²) — 태양이 보내는 에너지 총량<br>
    • <b>SVF</b>: 하늘 조망 계수 (0~1) — 루버 사이로 보이는 하늘 비율. 흐린 날 확산광 수집에 중요<br>
    • <b>음영률(SF)</b>: 위 블레이드가 아래 발전면을 가리는 비율. 낮을수록 좋음
    </div>""",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("AI 발전량",f"{pa/1000:.3f} kWh",f"연간 {ka:.1f} kWh")
    c2.metric("vs 고정60°",f"+{(pa/p6-1)*100:.1f}%" if p6>0 else "—")
    c3.metric("vs 수직90°",f"+{(pa/p9-1)*100:.1f}%" if p9>0 else "—")
    c4.metric("예상 수익",f"{int(pa/1000*kr):,} 원")
    cl2,cr=st.columns([3,1])
    with cl2:
        st.subheader("AI 제어 스케줄")
        st.markdown("""<div class="ex"><b>💡 아침·저녁에 루버가 90°인 이유</b><br>
        태양 에너지(GHI)가 10 W/m² 미만이면 발전이 불가합니다.
        이 시간에는 루버를 수직(90°)으로 닫아 <b>바람·먼지로부터 패널을 보호</b>합니다.
        GHI가 10을 넘으면 AI가 즉시 최적 각도로 전환합니다.</div>""",unsafe_allow_html=True)
        fg=make_subplots(specs=[[{"secondary_y":True}]])
        fg.add_trace(go.Bar(x=ts[md2].strftime("%H:%M"),y=ghi[md2],name="GHI",marker_color="rgba(255,152,0,0.5)"),secondary_y=False)
        fg.add_trace(go.Scatter(x=ts[md2].strftime("%H:%M"),y=ai[md2],name="AI 각도",line=dict(color=C_AI,width=3)),secondary_y=True)
        fg.update_yaxes(title_text="GHI (W/m²)",secondary_y=False); fg.update_yaxes(title_text="각도(°)",range=[0,95],secondary_y=True)
        fg.update_layout(height=360,template=PT,legend=dict(orientation="h",y=1.08)); st.plotly_chart(fg,use_container_width=True)
    with cr:
        st.subheader("일일 발전량")
        fb=go.Figure(go.Bar(x=["AI","F60°","F90°"],y=[pa/1000,p6/1000,p9/1000],marker_color=[C_AI,C_F60,C_V90],
            text=[f"{v/1000:.3f}" for v in [pa,p6,p9]],textposition="auto"))
        fb.update_layout(height=360,yaxis_title="kWh",template=PT); st.plotly_chart(fb,use_container_width=True)
    st.subheader("시간별 상세 테이블")
    gd,ed=ghi[md2],el[md2]
    sv=panel_sf(ai[md2],ed,az[md2],hdm,pm); svv=svf(ai[md2],hdm,pm)
    def sfs(s,g,e):
        if g<10 or e<=0: return "— (비활성)"
        if s<0.1: return f"{s*100:.1f}% 🟢 양호"
        elif s<0.3: return f"{s*100:.1f}% 🟡 경미"
        elif s<0.5: return f"{s*100:.1f}% 🟠 주의"
        else: return f"{s*100:.1f}% 🔴 심각"
    ds=pd.DataFrame({"시간":ts[md2].strftime("%H:%M").tolist(),"AI각도(°)":ai[md2].astype(int).tolist(),
        "GHI":np.round(gd,1).tolist(),"음영률":[sfs(s,g,e) for s,g,e in zip(sv,gd,ed)],
        "SVF":[f"{s:.2f}" if g>=10 and e>0 else "—" for s,g,e in zip(svv,gd,ed)]})
    st.dataframe(ds,use_container_width=True,hide_index=True)
    st.markdown("""<div class="ex"><b>📊 음영률 상태 의미</b><br>
    🟢 <b>0~10%</b>: 거의 그림자 없음. 발전 손실 미미<br>
    🟡 <b>10~30%</b>: 약간의 그림자. 손실 5~15%<br>
    🟠 <b>30~50%</b>: 주의 필요. AI가 각도 조정으로 최소화 중<br>
    🔴 <b>50%+</b>: 발전면 절반 이상 가려짐. 하지만 이 시간대는 보통 GHI 자체가 낮아 실질 영향 제한적</div>""",unsafe_allow_html=True)

# ═══ 학습데이터 ═══
with tabs[1]:
    st.subheader("📊 XGBoost 학습 데이터셋")
    st.markdown("""<div class="ex"><b>📖 학습 데이터란?</b><br>
    AI 모델이 '이런 날씨·시간에는 이 각도가 최적이다'를 배우기 위한 <b>과거 10년(2014~2023) 기상 데이터</b>입니다.
    기상청 실측 관측값을 바탕으로 물리 시뮬레이션을 통해 각 시간대의 최적 각도를 미리 계산해 놓은 것입니다.</div>""",unsafe_allow_html=True)
    df_csv = df_csv_raw
    csv_err = csv_err_raw
    # 디버깅 표시
    if csv_err:
        st.sidebar.info(f"CSV: {csv_err}")
    if df_csv is not None:
        target_cols = [c for c in df_csv.columns if "target" in c]
        st.sidebar.info(f"CSV 타겟 컬럼: {target_cols}")
        if "target_angle_v15" in df_csv.columns: tc,cv="target_angle_v15","V15"
        elif "target_angle_v14" in df_csv.columns: tc,cv="target_angle_v14","V14"
        else: tc,cv="target_angle_v5","V5"
        st.success(f"✅ {cv} 학습 데이터 | {len(df_csv):,}행 | 2014~2023년")
        st.markdown(f"""<div class="ex"><b>📖 각 변수의 의미</b><br><br>
        <b>입력 변수 (AI가 보는 정보)</b><br>
        • <b>ghi_w_m2</b>: 수평면 전일사량 — 현재 태양이 보내는 에너지 총량. 가장 중요한 입력값.<br>
        • <b>cloud_cover (0~9)</b>: 하늘의 구름 양. 0=완전 맑음, 9=완전 흐림. GHI를 얼마나 감쇠시키는지 반영.<br>
        • <b>temp_actual</b>: 외기 온도. 패널은 고온에서 효율이 떨어지는 특성(온도계수 약 -0.4%/°C)이 있음.<br>
        • <b>hour_sin/cos</b>: 현재 시각을 수학적으로 표현한 값. AI가 "오전인지 오후인지"를 구분하는 데 사용.<br>
        • <b>doy_sin/cos</b>: 현재 날짜(1~365일)를 수학적으로 표현. AI가 "여름인지 겨울인지"를 구분.<br><br>
        <b>출력 (AI가 예측하는 값)</b><br>
        • <b>{tc}</b>: 해당 시간에 발전량을 최대화하는 <b>최적 루버 각도</b>. 물리 시뮬레이션으로 미리 계산된 정답.</div>""",unsafe_allow_html=True)
        dp=df_csv[df_csv["ghi_w_m2"]>10].copy(); dp["month"]=pd.to_datetime(dp["timestamp"]).dt.month
        dp["hour"]=pd.to_datetime(dp["timestamp"]).dt.hour
        mm2={i:n for i,n in zip(range(1,13),mn)}; dp["ms"]=dp["month"].map(mm2); mo=list(mm2.values())
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**GHI 월별 분포** — 태양 에너지가 계절별로 얼마나 다른가?")
            fg=px.box(dp,x="ms",y="ghi_w_m2",color="ms",category_orders={"ms":mo},template=PT)
            fg.update_layout(showlegend=False,height=300,yaxis_title="GHI (W/m²)"); st.plotly_chart(fg,use_container_width=True)
            st.caption("여름(6~8월)은 일사량이 높지만 장마로 변동도 큼. 겨울은 일사량 자체가 적음.")
        with c2:
            st.markdown("**운량 월별 분포** — 구름이 발전에 미치는 영향")
            fg2=px.box(dp,x="ms",y="cloud_cover",color="ms",category_orders={"ms":mo},template=PT)
            fg2.update_layout(showlegend=False,height=300,yaxis_title="운량 (0~9)"); st.plotly_chart(fg2,use_container_width=True)
            st.caption("6~7월 장마철 운량↑ → 직달광 감소 → AI가 확산광 전략으로 전환해야 하는 구간")
        c3,c4=st.columns(2)
        with c3:
            st.markdown("**기온 월별 분포** — 패널 효율에 미치는 영향")
            fg3=px.box(dp,x="ms",y="temp_actual",color="ms",category_orders={"ms":mo},template=PT)
            fg3.update_layout(showlegend=False,height=300,yaxis_title="°C"); st.plotly_chart(fg3,use_container_width=True)
            st.caption("여름 30°C+ → 패널 효율 약 3~4% 하락. 겨울 저온은 효율에 유리하지만 일사량 부족.")
        with c4:
            st.markdown(f"**최적 각도 월별 분포 ({cv})** — AI가 배우는 정답")
            fg4=px.box(dp,x="ms",y=tc,color="ms",category_orders={"ms":mo},template=PT)
            fg4.update_layout(showlegend=False,height=300,yaxis_title="최적 각도(°)"); st.plotly_chart(fg4,use_container_width=True)
            st.caption("여름: ~22° (루버를 눕혀 직달광 수집). 겨울: ~63° (태양이 낮아 루버를 세움)")
        st.markdown("---")
        c5,c6=st.columns(2)
        with c5:
            st.markdown("**시간대별 최적각 패턴** — 하루 중 언제 각도가 가장 낮은가?")
            ha=dp.groupby("hour")[tc].mean().reset_index()
            fg5=go.Figure(go.Scatter(x=ha["hour"],y=ha[tc],mode="lines+markers",line=dict(color=C_AI,width=2)))
            fg5.update_layout(height=280,xaxis_title="시각(h)",yaxis_title="평균 최적각(°)",template=PT); st.plotly_chart(fg5,use_container_width=True)
            st.caption("정오(12시) 전후에 최적각이 가장 낮음 = 태양이 가장 높이 떠서 루버를 최대한 눕힘")
        with c6:
            st.markdown("**월별 최적각 패턴** — 계절에 따라 얼마나 달라지는가?")
            da2=dp.groupby("month")[tc].mean().reset_index()
            fg6=go.Figure(go.Scatter(x=da2["month"],y=da2[tc],mode="lines+markers",line=dict(color=C_F60,width=2)))
            fg6.update_layout(height=280,xaxis_title="월",yaxis_title="평균 최적각(°)",
                xaxis=dict(tickvals=list(range(1,13)),ticktext=mo),template=PT); st.plotly_chart(fg6,use_container_width=True)
            st.caption("여름↔겨울 각도 차이 약 40°. 이 큰 변화를 AI가 자동으로 추적합니다.")
        st.markdown(f"**GHI vs 최적각** — 일사량과 각도의 관계")
        samp=dp.sample(min(3000,len(dp)),random_state=42)
        fg7=px.scatter(samp,x="ghi_w_m2",y=tc,color="ms",opacity=0.4,template=PT,
                       labels={"ghi_w_m2":"GHI","ms":"월",tc:"최적각(°)"})
        fg7.update_layout(height=350); st.plotly_chart(fg7,use_container_width=True)
        st.caption("GHI가 높을수록(태양이 강할수록) 최적각이 낮아지는 경향. 계절별로 뚜렷한 클러스터가 형성됨.")
    else:
        st.warning(f"⚠️ CSV 로드 실패")
        if csv_err:
            st.error(f"원인: {csv_err}")
            st.info("💡 GitHub 레포에서 파일명이 정확한지, LFS 파일이 아닌지 확인해주세요.")

# ═══ 피처중요도 ═══
with tabs[2]:
    st.subheader("🎯 피처 중요도 — AI가 무엇을 가장 중요하게 보는가?")
    st.markdown("""<div class="ex"><b>📖 피처 중요도란?</b><br>
    AI 모델(XGBoost)이 루버 각도를 결정할 때 <b>어떤 정보를 가장 많이 참고하는지</b>를 수치화한 것입니다.
    마치 사람이 "오늘 날씨가 어떤지"를 가장 먼저 확인하듯, AI도 특정 변수를 더 중요하게 봅니다.<br><br>
    <b>Gain</b>: 해당 변수가 예측 정확도를 얼마나 향상시켰는지의 누적 기여도. 높을수록 중요.</div>""",unsafe_allow_html=True)
    imp={"피처":["doy_cos","ghi_w_m2","hour_cos","hour_sin","temp_actual","doy_sin","cloud_cover"],
         "Gain":[0.324,0.194,0.139,0.133,0.120,0.078,0.012],
         "의미":["계절 위치 (겨울/여름 구분)","현재 일사량 세기","시각 (오전/오후)","시각 보완",
                 "외기 온도","계절 보완","구름양 (GHI에 이미 반영)"]}
    di=pd.DataFrame(imp).sort_values("Gain",ascending=True)
    fg=go.Figure(go.Bar(x=di["Gain"],y=di["피처"],orientation="h",
        marker_color=[C_AI if g>0.12 else "#90CAF9" if g>0.06 else "#B0BEC5" for g in di["Gain"]],
        text=[f"{g:.3f}" for g in di["Gain"]],textposition="outside",customdata=di["의미"],
        hovertemplate="<b>%{y}</b><br>Gain: %{x:.3f}<br>%{customdata}<extra></extra>"))
    fg.update_layout(height=380,xaxis_title="Gain (기여도)",xaxis_range=[0,0.40],template=PT); st.plotly_chart(fg,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("""<div class="ex"><b>🥇 1위: 계절 정보 (doy_cos 32.4%)</b><br>
        AI가 가장 중요하게 보는 건 <b>"지금이 여름인지 겨울인지"</b>입니다.
        여름에는 태양이 높아 루버를 눕히고(22°), 겨울에는 낮아 세워야(63°) 하기 때문입니다.
        계절 변수 합산(doy_cos+sin)은 <b>전체의 40%</b>를 차지합니다.</div>""",unsafe_allow_html=True)
        st.markdown("""<div class="ex"><b>🥈 2위: 일사량 (ghi 19.4%)</b><br>
        "지금 햇빛이 얼마나 강한가"도 핵심입니다. 같은 여름이라도 맑은 날과 흐린 날의 최적 각도가 다릅니다.
        맑으면 직달광 최대화, 흐리면 확산광 전략으로 전환합니다.</div>""",unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="ex"><b>🥉 3위: 시각 (hour_sin+cos 27.2%)</b><br>
        "아침인지 정오인지 저녁인지"가 세 번째로 중요합니다.
        정오에는 태양이 가장 높아 낮은 각도가 최적이지만, 아침·저녁에는 높은 각도가 유리합니다.</div>""",unsafe_allow_html=True)
        st.markdown("""<div class="ex"><b>기타: 온도 (12%) + 구름 (1.2%)</b><br>
        온도는 패널 효율에 영향. 구름은 이미 GHI에 반영되어 있어 별도 기여도가 매우 낮습니다.</div>""",unsafe_allow_html=True)
    st.markdown("---")
    m1,m2=st.columns(2)
    with m1:
        st.markdown("""<div class="gd"><b>MAE = 0.70° (평균 오차)</b><br>
        AI 예측값과 실제 최적값의 차이가 평균 <b>0.70°</b>에 불과합니다.
        루버 제어 모터의 물리적 정밀도가 ±2~3°인 점을 고려하면, 실제 제어에 전혀 지장 없는 수준입니다.</div>""",unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="gd"><b>R² = 0.9966 (설명력 99.7%)</b><br>
        AI가 루버 각도 변동의 <b>99.7%</b>를 정확히 설명합니다.
        나머지 0.3%는 순간적 기상 변동(돌풍, 급변하는 구름)에 의한 것으로, 예측이 본질적으로 어려운 영역입니다.</div>""",unsafe_allow_html=True)

# ═══ 음영원리 ═══
with tabs[3]:
    st.subheader("💡 루버 음영의 원리 (V15 선분교차)")
    st.markdown("""<div class="ex"><b>📖 자기 음영(Self-shading)이란?</b><br><br>
    루버는 여러 장의 블레이드가 나란히 배열된 구조입니다.
    태양이 비추면 <b>위쪽 블레이드가 아래쪽 블레이드에 그림자를 드리울 수</b> 있습니다.
    이 그림자가 태양전지(발전면) 위에 떨어지면 발전량이 감소합니다.<br><br>
    <b>V15의 혁신: 선분교차(Ray-Blade Intersection)</b><br>
    이전 버전(V13~V14)은 그림자를 단순한 수학 공식으로 근사했지만,
    V15는 <b>실제 태양 광선의 경로를 추적하여 발전면 위의 정확한 그림자 위치를 계산</b>합니다.
    마치 CAD에서 정밀하게 선분의 교차점을 구하는 것과 같습니다.<br><br>
    <b>핵심 포인트</b><br>
    • 발전면 = 블레이드의 <b>바깥쪽 면</b> (피봇~바깥끝, 태양전지가 부착된 면)<br>
    • 그림자는 피봇(안쪽)에서부터 바깥끝 방향으로 채워짐<br>
    • 90° 수직이면 → 블레이드가 나란히 서서 서로의 발전면을 가리지 않음 → <b>SF=0%</b></div>""",unsafe_allow_html=True)

    st.markdown("#### 인터랙티브 시뮬레이터")
    st.markdown("슬라이더를 조절하여 태양 고도와 루버 각도에 따른 음영 변화를 실시간으로 확인하세요.")
    eex=st.slider("☀️ 태양 고도각 (°) — 높을수록 태양이 높이 떠 있음",5,80,45,5)
    tex=st.slider("📐 루버 각도 (°) — 낮을수록 루버가 눕혀짐",15,90,30,5)
    sfx=float(panel_sf(tex,eex,180,hdm,pm)); svx=float(svf(tex,hdm,pm))
    prx,_,gpx=blade_geo(tex,hdm,bdm,pm)
    c1,c2=st.columns([2,1])
    with c1:
        fg=go.Figure(); nb=4; pp=80
        for i in range(nb):
            yc=i*pp; hp=(hdm/pm)*pp; dx=hp*np.cos(np.radians(tex)); dy=hp*np.sin(np.radians(tex))
            fg.add_trace(go.Scatter(x=[-dx,0],y=[yc+dy,yc],mode="lines",line=dict(color="#999",width=3),showlegend=(i==0),name="벽쪽 면" if i==0 else None))
            fg.add_trace(go.Scatter(x=[0,dx],y=[yc,yc-dy],mode="lines",line=dict(color=C_AI,width=6),showlegend=(i==0),name="발전면(Front)" if i==0 else None))
            fg.add_trace(go.Scatter(x=[0],y=[yc],mode="markers",marker=dict(size=6,color="red"),showlegend=False))
            if i>0 and sfx>0:
                pb=(0,(i-1)*pp); ob=(dx,(i-1)*pp-dy); tf=min(sfx,1)
                ix2=pb[0]+tf*(ob[0]-pb[0]); iy2=pb[1]+tf*(ob[1]-pb[1])
                fg.add_trace(go.Scatter(x=[pb[0],ix2 if sfx<1 else ob[0]],y=[pb[1],iy2 if sfx<1 else ob[1]],
                    mode="lines",line=dict(color="red",width=8),opacity=0.4,showlegend=(i==1),name=f"음영({sfx:.0%})" if i==1 else None))
        if eex>0:
            rdx=-np.cos(np.radians(eex))*60; rdy=-np.sin(np.radians(eex))*60
            for i in range(1,nb):
                yc=i*pp; hp=(hdm/pm)*pp; ox=hp*np.cos(np.radians(tex)); oy=yc-hp*np.sin(np.radians(tex))
                fg.add_annotation(x=ox+rdx*1.5,y=oy+rdy*1.5,ax=ox-rdx*0.8,ay=oy-rdy*0.8,xref="x",yref="y",axref="x",ayref="y",
                    showarrow=True,arrowhead=2,arrowsize=1.2,arrowcolor="#FF8F00",arrowwidth=2)
        fg.add_shape(type="line",x0=-5,y0=-30,x1=-5,y1=(nb-1)*pp+50,line=dict(color="#555",width=3))
        fg.update_layout(height=420,template=PT,xaxis=dict(range=[-60,100],showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(range=[-50,(nb-1)*pp+70],showgrid=False,zeroline=False,showticklabels=False,scaleanchor="x"),
            title=f"단면도 — 고도 {eex}° | 루버 {tex}° | 음영 {sfx:.1%}",legend=dict(orientation="h",y=-0.05))
        st.plotly_chart(fg,use_container_width=True)
    with c2:
        st.metric("음영률",f"{sfx*100:.1f}%",delta="양호 ✅" if sfx<0.1 else("경미 🟡" if sfx<0.3 else "주의 🔴"),delta_color="off")
        st.metric("SVF",f"{svx:.2f}",delta="높음 ✅" if svx>0.5 else "낮음 🔴",delta_color="off")
        st.markdown(f"""<div class="ex"><b>현재 설정값</b><br>
        블레이드 DEPTH: {bdm:.0f}mm<br>PITCH: {pm:.0f}mm<br>
        돌출: {float(prx):.1f}mm | 틈: {float(gpx):.1f}mm<br><br>
        <b>💡 AI의 딜레마</b><br>
        • 각도를 낮추면 → 직달광↑ but 음영↑<br>
        • 각도를 높이면 → 음영↓ but 직달광↓<br>
        AI는 매 시간 이 균형점에서 <b>순발전량이 최대</b>인 각도를 선택합니다.</div>""",unsafe_allow_html=True)

# ═══ 음영시각화 ═══
with tabs[4]:
    st.subheader("🔥 조건별 음영 패턴 비교 (V15 선분교차)")
    st.markdown("""<div class="ex"><b>📖 이 차트의 의미</b><br>
    서울 기준 동지(겨울)와 하지(여름)의 다양한 시간·각도 조건에서 <b>블레이드 단면의 그림자 패턴</b>을 비교합니다.<br>
    <b>파란선</b>=발전면(태양전지) | <b>빨간선</b>=그림자 영역 | <b>주황화살표</b>=태양광선 방향<br><br>
    <b>핵심 관찰 포인트</b>: 같은 각도라도 태양 높이에 따라 음영이 완전히 달라집니다.</div>""",unsafe_allow_html=True)
    cds=[(60,15,"동지 09시\ntilt=60° elev=15°"),(63,29,"동지 정오\ntilt=63° elev=29°"),(60,15,"동지 16시\ntilt=60° elev=15°"),
         (22,40,"하지 09시\ntilt=22° elev=40°"),(22,76,"하지 정오\ntilt=22° elev=76°"),(22,40,"하지 16시\ntilt=22° elev=40°"),
         (15,76,"하지 t=15° e=76°\n(최소각)"),(45,76,"하지 t=45° e=76°"),(90,29,"동지 t=90° e=29°\n(수직)")]
    for rs in range(0,len(cds),3):
        cols=st.columns(3)
        for ci,(t2,ec,lc) in enumerate(cds[rs:rs+3]):
            with cols[ci]:
                sfc=float(panel_sf(t2,ec,180,hdm,pm))
                fc=go.Figure(); nb=4; pc=70
                for i in range(nb):
                    yc=i*pc; hp=(hdm/pm)*pc; dx=hp*np.cos(np.radians(t2)); dy=hp*np.sin(np.radians(t2))
                    fc.add_trace(go.Scatter(x=[-dx,0],y=[yc+dy,yc],mode="lines",line=dict(color="#999",width=2),showlegend=False))
                    fc.add_trace(go.Scatter(x=[0,dx],y=[yc,yc-dy],mode="lines",line=dict(color=C_AI,width=5),showlegend=False))
                    fc.add_trace(go.Scatter(x=[0],y=[yc],mode="markers",marker=dict(size=4,color="red"),showlegend=False))
                    if i>0 and sfc>0:
                        pb=(0,(i-1)*pc); ob=(dx,(i-1)*pc-dy); tf=min(sfc,1)
                        ix2=pb[0]+tf*(ob[0]-pb[0]); iy2=pb[1]+tf*(ob[1]-pb[1])
                        fc.add_trace(go.Scatter(x=[pb[0],ix2 if sfc<1 else ob[0]],y=[pb[1],iy2 if sfc<1 else ob[1]],mode="lines",line=dict(color="red",width=7),opacity=0.5,showlegend=False))
                if ec>0:
                    rdx=-np.cos(np.radians(ec))*40; rdy=-np.sin(np.radians(ec))*40
                    for i in range(1,nb):
                        yc=i*pc; hp=(hdm/pm)*pc; ox=hp*np.cos(np.radians(t2)); oy=yc-hp*np.sin(np.radians(t2))
                        fc.add_annotation(x=ox+rdx,y=oy+rdy,ax=ox-rdx*0.5,ay=oy-rdy*0.5,xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=2,arrowsize=1,arrowcolor="#FF8F00",arrowwidth=1.5)
                fc.add_shape(type="line",x0=-3,y0=-20,x1=-3,y1=(nb-1)*pc+40,line=dict(color="#555",width=2))
                fc.update_layout(height=300,template=PT,margin=dict(l=5,r=5,t=35,b=5),
                    xaxis=dict(range=[-50,80],showgrid=False,zeroline=False,visible=False),
                    yaxis=dict(range=[-30,(nb-1)*pc+50],showgrid=False,zeroline=False,visible=False,scaleanchor="x"),
                    title=dict(text=f"{lc}<br>SF={sfc:.0%}",font=dict(size=11)))
                st.plotly_chart(fc,use_container_width=True)
    st.markdown("""<div class="ex"><b>📊 패턴 해석 — 왜 이렇게 되는가?</b><br><br>
    <b>❄️ 동지 (태양 고도 15~29°)</b><br>
    태양이 낮게 비추면 광선이 블레이드 사이를 비스듬히 통과합니다.
    Pitch=Depth=114mm(1:1 비율)에서는 이 조건에서 <b>음영이 거의 발생하지 않습니다</b>.
    따라서 겨울에는 직달광 입사각(AOI)을 최소화하는 ~63°가 최적입니다.<br><br>
    <b>☀️ 하지 정오 (태양 고도 76°)</b><br>
    태양이 거의 머리 위에서 비추면, 위 블레이드가 아래 발전면에 <b>큰 그림자</b>를 드리웁니다.
    tilt=15°에서 SF=52%, tilt=22°에서도 SF=51%. 이때 AI는 음영 손실과 직달광 이득을 비교하여
    <b>22° 부근의 균형점</b>을 선택합니다.<br><br>
    <b>수직 90° (동지 정오)</b><br>
    블레이드가 완전히 수직이면 서로의 발전면을 전혀 가리지 않아 <b>SF=0%</b>.
    하지만 직달광 입사각이 커서 발전 효율 자체가 낮습니다. 따라서 최적이 아닙니다.</div>""",unsafe_allow_html=True)

# ═══ 발전량비교 ═══
with tabs[5]:
    st.subheader("⚡ AI vs 고정60° vs 수직90° 발전량 비교")
    if ux: st.success(f"✅ XGBoost V15 | 데이터: {annual_source}")
    else: st.info(f"ℹ️ 규칙 기반 | 데이터: {annual_source}")
    st.markdown("""<div class="ex"><b>📖 이 그래프는 무엇을 보여주나요?</b><br>
    세 가지 루버 운전 방식의 <b>월별 발전량</b>을 비교합니다:<br>
    • <b>AI 제어</b>: XGBoost 모델이 매 시간 최적 각도를 예측하여 제어<br>
    • <b>고정 60°</b>: 일년 내내 60°로 고정 (서울 위도 기준 일반적 설치각)<br>
    • <b>수직 90°</b>: 일년 내내 수직 고정<br><br>
    연간 시뮬레이션은 <b>서울 10년 평균 기상 패턴</b>(월별 구름 감쇠 포함)을 적용하여
    청천 모델의 한계를 보완했습니다.</div>""",unsafe_allow_html=True)
    fg=go.Figure()
    for col,color,name in [("AI",C_AI,"AI 제어"),("고정60°",C_F60,"고정 60°"),("수직90°",C_V90,"수직 90°")]:
        fg.add_trace(go.Bar(x=mn,y=dmo[col]*asc/1000,name=name,marker_color=color))
    fg.update_layout(barmode="group",height=400,template=PT,yaxis_title="발전량(kWh)",legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fg,use_container_width=True)
    st.markdown(f"""<div class="ex"><b>📊 월별 해석</b><br><br>
    <b>🌞 여름 (5~8월)</b>: AI가 음영을 고려한 최적각(~22°)으로 고정60° 대비 <b>+7~17%</b> 발전량 달성.
    이 구간이 AI 제어의 가장 큰 이득 구간입니다.<br><br>
    <b>❄️ 겨울 (11~2월)</b>: V15 최적각이 ~63°로 고정60°에 매우 가까움.
    겨울에는 음영이 거의 없어 직달광 입사각(AOI) 최소화가 핵심이며, 60°와 63°의 차이는 미미합니다.
    대신 <b>흐린 날 AI가 확산광 전략(높은 각도→SVF↑)으로 전환</b>하여 소폭 우위를 점합니다.<br><br>
    <b>Colab 실측 기상 검증</b>: AI 621.6kWh vs 고정60° 589.1kWh (<b>+5.5%</b>).</div>""",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.metric("연간 AI",f"{ka:.1f} kWh")
    c2.metric("연간 F60°",f"{k6:.1f} kWh",f"{(k6/ka-1)*100:+.1f}%" if ka>0 else "—")
    c3.metric("연간 F90°",f"{k9:.1f} kWh",f"{(k9/ka-1)*100:+.1f}%" if ka>0 else "—")
    dc=dmo.copy(); dc["ac"]=(dc["AI"]*asc/1000).cumsum(); dc["fc"]=(dc["고정60°"]*asc/1000).cumsum(); dc["vc"]=(dc["수직90°"]*asc/1000).cumsum()
    fg2=go.Figure()
    fg2.add_trace(go.Scatter(x=mn,y=dc["ac"],name="AI",line=dict(color=C_AI,width=3)))
    fg2.add_trace(go.Scatter(x=mn,y=dc["fc"],name="고정60°",line=dict(color=C_F60,width=2,dash="dash")))
    fg2.add_trace(go.Scatter(x=mn,y=dc["vc"],name="수직90°",line=dict(color=C_V90,width=2,dash="dot")))
    fg2.update_layout(height=320,yaxis_title="누적 kWh",title="연간 누적 발전량",template=PT,legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fg2,use_container_width=True)
    st.markdown("""<div class="ex"><b>누적 곡선 읽는 법</b><br>
    기울기가 가파른 구간 = 발전량이 많은 계절. AI 곡선이 다른 곡선보다 위에 있을수록 제어 효과가 큽니다.
    특히 <b>5~8월 구간에서 AI와 고정60°의 간격이 벌어지는 것</b>이 핵심 이득 구간입니다.</div>""",unsafe_allow_html=True)

# ═══ 월별각도 ═══
with tabs[6]:
    st.subheader("📅 월별 AI 제어 각도 분포")
    st.markdown("""<div class="ex"><b>📖 이 차트는 무엇을 보여주나요?</b><br>
    AI가 각 월에 실제로 선택한 루버 각도의 <b>분포(박스플롯)</b>입니다.
    박스 안의 선은 중앙값, 박스 범위는 25~75% 구간, 점은 극단값입니다.</div>""",unsafe_allow_html=True)
    dp2=dan[dan["ghi"]>=10].copy(); dp2["mn"]=dp2["timestamp"].dt.month
    mn2={i:n for i,n in zip(range(1,13),mn)}; dp2["ms"]=dp2["mn"].map(mn2)
    fg=go.Figure()
    for m,ms in mn2.items(): fg.add_trace(go.Box(y=dp2[dp2["mn"]==m]["angle_ai"],name=ms,marker_color=C_AI,boxmean=True))
    fg.add_hline(y=AMIN,line_dash="dash",line_color="red",annotation_text=f"최소각 {AMIN}°")
    fg.update_layout(height=420,yaxis_title="루버 각도(°)",showlegend=False,template=PT)
    st.plotly_chart(fg,use_container_width=True)
    st.markdown("""<div class="ex"><b>📊 계절별 해석</b><br><br>
    <b>🌞 여름 (6~8월) — ~22°</b><br>
    태양 고도가 높아(최대 76°) 루버를 눕히고 싶지만, 15°까지 눕히면 <b>음영 52%</b>가 발생합니다.
    AI는 음영과 직달광의 균형점인 22° 부근을 선택합니다.<br><br>
    <b>❄️ 겨울 (12~2월) — 56~63°</b><br>
    태양 최대 고도가 ~29°로 낮아 <b>어떤 각도에서도 음영이 거의 없습니다</b>.
    이때는 직달광 입사각(AOI)을 최소화하는 것이 핵심이며, 태양 고도 29°에서 AOI가 최소인 각도가 ~61°입니다.<br><br>
    <b>🍂 봄·가을 — 25~51°</b><br>
    태양 고도가 여름↔겨울 사이를 오가며 각도도 점진적으로 전환됩니다.</div>""",unsafe_allow_html=True)
    st.subheader("월별 평균 각도 & 발전량")
    v15r={1:62.6,2:55.6,3:42.2,4:25.0,5:23.7,6:22.1,7:23.3,8:25.3,9:38.9,10:51.4,11:60.4,12:61.7}
    ds2=dmo.copy(); ds2["month_s"]=list(mn2.values()); ds2=ds2.rename(columns={"avg_angle":"시뮬 평균각(°)"})
    ds2["V15 참조각"]=[v15r[m] for m in range(1,13)]
    st.dataframe(ds2[["month_s","시뮬 평균각(°)","V15 참조각","AI","고정60°","수직90°"]].round(1),use_container_width=True,hide_index=True)

# ═══ 내일스케줄 ═══
with tabs[7]:
    st.subheader("🌤️ 내일 예측 스케줄")
    if kma is None: st.error("❌ 기상청 API 실패")
    else: st.success(f"✅ 기상청 연동 | {tom}")
    fg=make_subplots(specs=[[{"secondary_y":True}]])
    fg.add_trace(go.Bar(x=ts[md2].strftime("%H:%M"),y=ghi[md2],name="GHI",marker_color="rgba(255,152,0,0.5)"),secondary_y=False)
    fg.add_trace(go.Scatter(x=ts[md2].strftime("%H:%M"),y=ai[md2],name="AI 각도",line=dict(color=C_AI,width=3),mode="lines+markers"),secondary_y=True)
    fg.update_yaxes(title_text="GHI",secondary_y=False); fg.update_yaxes(title_text="각도(°)",range=[0,95],secondary_y=True)
    fg.update_layout(height=380,template=PT,title=f"내일({tom}) 스케줄")
    st.plotly_chart(fg,use_container_width=True)
    sft=panel_sf(ai[md2],el[md2],az[md2],hdm,pm); svt=svf(ai[md2],hdm,pm)
    dt=pd.DataFrame({"시간":ts[md2].strftime("%H:%M").tolist(),"GHI":np.round(ghi[md2],1).tolist(),
        "기온(°C)":np.round(tp[md2],1).tolist(),"AI각도(°)":ai[md2].astype(int).tolist(),
        "음영률":[f"{s*100:.1f}%" if g>=10 and e>0 else "—" for s,g,e in zip(sft,ghi[md2],el[md2])],
        "SVF":[f"{s:.2f}" if g>=10 and e>0 else "—" for s,g,e in zip(svt,ghi[md2],el[md2])]})
    st.dataframe(dt,use_container_width=True,hide_index=True)

# ═══ 건강진단 ═══
with tabs[8]:
    st.subheader("🩺 시스템 건강 진단")
    st.markdown("""<div class="ex"><b>📖 건강진단이란?</b><br>
    AI가 예측한 발전량(P_sim)과 실제 센서 측정값(P_actual)을 비교하여
    <b>패널 오염·고장·케이블 이상</b> 등을 자동으로 감지하는 기능입니다.<br><br>
    • <b>P_sim</b>: "이 날씨면 이만큼 발전돼야 한다"는 AI 예측값<br>
    • <b>P_actual</b>: 인버터에서 실제로 측정된 값<br>
    • <b>Health Ratio</b>: P_actual ÷ P_sim. 100% = 완벽 정상. 낮을수록 이상</div>""",unsafe_allow_html=True)
    st.info("📌 현재 실측 센서 미연동. 슬라이더로 시나리오 체험 가능.")
    c1,c2=st.columns(2)
    with c1: pp2=st.slider("실측 발전량 비율(%)",10,110,95)
    with c2: wt=st.number_input("WARNING(%)",value=90); ct2=st.number_input("CRITICAL(%)",value=75)
    hr=pp2/100
    if hr>=wt/100: st2,co="✅ NORMAL","green"
    elif hr>=ct2/100: st2,co="⚠️ WARNING","orange"
    else: st2,co="🔴 CRITICAL","red"
    c1,c2,c3=st.columns(3)
    c1.metric("P_sim",f"{pa/1000:.3f} kWh"); c2.metric("P_actual",f"{pa/1000*hr:.3f} kWh"); c3.metric("Health",f"{hr:.2%}",st2)
    fg=go.Figure(go.Indicator(mode="gauge+number+delta",value=pp2,delta={"reference":100,"valueformat":".1f"},
        gauge={"axis":{"range":[0,110]},"bar":{"color":co},
        "steps":[{"range":[0,ct2],"color":"rgba(244,67,54,0.15)"},{"range":[ct2,wt],"color":"rgba(255,152,0,0.15)"},{"range":[wt,110],"color":"rgba(76,175,80,0.15)"}],
        "threshold":{"line":{"color":"gray","width":2},"value":wt}},title={"text":f"상태: {st2}"}))
    fg.update_layout(height=320); st.plotly_chart(fg,use_container_width=True)
    st.markdown("""<div class="ex"><b>📊 판정 기준 상세</b><br><br>
    <b>✅ 90% 이상 (NORMAL)</b>: 정상. P_sim과 P_actual이 거의 일치. 추가 조치 불필요.<br><br>
    <b>⚠️ 75~90% (WARNING)</b>: 패널 오염 의심. 먼지·새 배설물·낙엽 등으로 인한 10~25% 손실.
    <b>권장 조치</b>: 패널 표면 청소, 육안 점검.<br><br>
    <b>🔴 75% 미만 (CRITICAL)</b>: 심각한 성능 저하. 단순 오염을 넘어 <b>셀 파손·케이블 불량·인버터 고장</b> 가능성.
    <b>권장 조치</b>: 즉시 전문 점검. 연결 상태, 인버터 출력, 셀 표면 확인.</div>""",unsafe_allow_html=True)
    st.caption("📌 실측 센서 연동 후 P_actual 자동 입력 예정 (특허 청구항 4 대상)")

# ═══ 파라미터 ═══
with tabs[9]:
    st.subheader("🔧 파라미터 민감도 — V15 물리모델 기반")
    st.markdown("""<div class="ex"><b>📖 파라미터 민감도란?</b><br>
    블레이드의 물리적 치수(깊이, 간격 등)를 바꿨을 때 <b>발전량과 음영이 어떻게 변하는지</b>를 확인하는 도구입니다.
    제품 설계 시 최적의 물리 사양을 찾는 데 활용됩니다.</div>""",unsafe_allow_html=True)
    tl=st.slider("시스템 손실률",0.70,0.95,DL,0.01)
    st.markdown("---")
    st.markdown("#### 📐 블레이드 DEPTH 변화 → 연간 발전량")
    st.markdown("""<div class="ex"><b>DEPTH란?</b> 블레이드의 세로 길이(벽에서 바깥으로의 깊이)입니다.<br>
    • DEPTH↑ → 발전면적 증가(+) but 돌출↑ → 음영 증가(-)<br>
    • DEPTH↓ → 음영 감소(+) but 발전면적 감소(-)<br>
    이 상충 관계에서 최적점이 존재합니다.</div>""",unsafe_allow_html=True)
    dr=np.arange(60,181,10); pdl=[]
    for bd in dr:
        _,_,_,dm,_=get_annual(ly,bd/2,pm,cw,uc,ef,tl)
        pdl.append(dm["AI"].sum()*asc/1000)
    fg1=go.Figure(go.Scatter(x=dr,y=pdl,mode="lines+markers",line=dict(color=C_AI,width=2)))
    fg1.add_vline(x=bdm,line_dash="dash",line_color=C_F60,annotation_text=f"현재 {bdm:.0f}mm")
    fg1.update_layout(height=300,xaxis_title="DEPTH(mm)",yaxis_title="연간 발전량(kWh)",template=PT)
    st.plotly_chart(fg1,use_container_width=True)
    st.markdown("""<div class="ex"><b>📊 그래프 해석</b><br>
    그래프가 우상향하다가 꺾이는 지점이 있다면 그것이 <b>발전면적 이득 vs 음영 손실의 균형점</b>입니다.
    현재 114mm 지점이 이 균형 부근에 있는지 확인하세요.
    그래프가 계속 우상향이면 현재 Pitch에서는 DEPTH를 키워도 음영 손실보다 면적 이득이 큰 것입니다.</div>""",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📏 피치 변화 → 정오 기준 음영률")
    st.markdown("""<div class="ex"><b>피치란?</b> 블레이드 간 수직 간격입니다.<br>
    • 피치↑ → 블레이드 간격 넓어짐 → 음영↓ but 같은 벽면에 블레이드 수 감소<br>
    • 피치↓ → 블레이드 수 증가 → 전체 면적↑ but 음영↑<br>
    현재 Pitch:Depth = 1:1 설정이 LX Hausys 제품 기준입니다.</div>""",unsafe_allow_html=True)
    pr=np.arange(80,201,5); sfl=[float(panel_sf(45,60,180,bdm/2,float(p)))*100 for p in pr]
    fg2=go.Figure(go.Scatter(x=pr,y=sfl,mode="lines+markers",line=dict(color=C_F60,width=2)))
    fg2.add_vline(x=pm,line_dash="dash",line_color=C_AI,annotation_text=f"현재 {pm:.0f}mm")
    fg2.add_hline(y=30,line_dash="dot",line_color="green",annotation_text="30% 양호 기준")
    fg2.update_layout(height=300,xaxis_title="PITCH(mm)",yaxis_title="음영률(%)",template=PT)
    st.plotly_chart(fg2,use_container_width=True)
    st.markdown("""<div class="ex"><b>📊 그래프 해석</b><br>
    피치가 넓어질수록 음영률이 낮아집니다. <b>30% 이하(초록 점선)</b>면 양호한 수준입니다.
    현재 114mm에서의 음영률을 확인하고, 설치 환경에 맞는 최적 피치를 결정하세요.
    단, 피치를 넓히면 같은 벽면에 설치할 수 있는 블레이드 수가 줄어들어 전체 발전량이 감소할 수 있습니다.</div>""",unsafe_allow_html=True)
    st.caption("💡 V15: PPO 강화학습을 향후 실시예로 추가 가능 (현재 XGBoost/규칙 기반)")
