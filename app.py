# ==============================================================================
# BIPV 통합 관제 시스템 v11.0
# 물리 v3 (physics_v3.py) | 모델 v17 | 실측 기상 10년 평균 연간 시뮬
# ==============================================================================
# v11.0 변경사항 (근거: ADVERSARIAL_REVIEW_RESULTS.md 2026-06-11 적대검토 + v17):
#   - 실기하 적용 (도면 1043A): 현=피치=97.5mm (구 57/114 placeholder 폐기)
#   - physics_v3: VF 현/피치 의미론 교정 + PV 띠 중앙(83/97.5) + Martin-Ruiz IAM + albedo 0.15
#   - 라벨 target_angle_v17 재생성 → XGBoost v17 재학습 (최적 고원 75~88°, 중앙 83°)
#   - model 지표 → model_metrics_v17.json (홀드아웃 실측)
# ==============================================================================
__version__ = "11.0"

import os, io, json
import numpy as np
import pandas as pd
import requests, urllib.parse
import streamlit as st
import streamlit.components.v1 as components
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

st.set_page_config(page_title="BIPV AI v17", layout="wide", page_icon="☀️")

try:    KMA_SERVICE_KEY = st.secrets["KMA_SERVICE_KEY"]
except Exception: KMA_SERVICE_KEY = ""
try:    GH_TOKEN = st.secrets["GH_TOKEN"]
except Exception: GH_TOKEN = ""
GH_HEADERS = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}

LAT, LON, TZ = 37.5665, 126.9780, "Asia/Seoul"
NX, NY = 60, 127
DC, DE, DL = 300, 18.7, 0.85           # capacity, efficiency, loss
DK, DU, DLC = 210, 1, 20               # kepco, units, louver count
DW, DCH, DP = 900.0, 97.5, 97.5        # width, chord(현), pitch — 도면 1043A 실기하
AMIN, AMAX, ANIGHT = 15, 90, 90

GH_BASE = "https://raw.githubusercontent.com/oopartstiago-debug/260310-BIPVpatenttest/main"
MODEL_URL = f"{GH_BASE}/bipv_xgboost_model_v17.pkl"
CSV_URL   = f"{GH_BASE}/bipv_ai_master_data_v15.csv"
MODEL_FN  = "bipv_xgboost_model_v17.pkl"
PT = "plotly_dark"
C_AI, C_F60, C_V90 = "#8b5cf6", "#f97316", "#64748b"
C_GHI = "rgba(234,179,8,0.5)"
DARK_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.6)", font=dict(color="#e2e8f0"))
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

@st.cache_data(ttl=600)
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
        cols_info = [c for c in df.columns if "target" in c]
        return df, f"OK | 컬럼: {cols_info} | {len(df)}행"
    except Exception as e:
        return None, f"예외: {str(e)[:200]}"

# ==============================================================================
# 물리 v3 — physics_v3.py 단일 소스 (라벨 생성 train_v17.py 와 동일 코드 경로)
# ==============================================================================
from physics_v3 import eff_poa

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
# 연간 데이터 — clearsky + cloud factor로 실측 근사 (폴백용)
# ==============================================================================
@st.cache_data(ttl=86400)
def get_annual(year,c,p,cw,uc,ef,dl,use_xgb=False):
    ty=pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31 23:00",freq="h",tz=TZ)
    sp=site.get_solarposition(ty); cs=site.get_clearsky(ty)
    ghi_cs=np.asarray(cs["ghi"].values,dtype=float)
    zn,az2=sp["apparent_zenith"].values,sp["azimuth"].values; el=90-zn

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
        e=eff_poa(np.asarray(ang,dtype=float),el,az2,dn,dh,c,p)
        return (e[ghi_y>=10]/1000*cw*uc*ef*dl).sum()
    wa,w6,w9=en(ai),en(np.full_like(ghi_y,60.0)),en(np.full_like(ghi_y,90.0))
    da=pd.DataFrame({"timestamp":ty,"ghi":ghi_y,"zenith":zn,"azimuth":az2,"elevation":el,"angle_ai":ai})
    da["month"]=da["timestamp"].dt.month
    ml=[]
    for m in range(1,13):
        mm=(da["month"]==m)&(ghi_y>=10)
        def em(ang):
            e=eff_poa(np.asarray(ang,dtype=float),el[mm],az2[mm],dn[mm],dh[mm],c,p)
            return (e/1000*cw*uc*ef*dl).sum()
        ml.append({"month":m,"AI":em(da.loc[mm,"angle_ai"].values),
                   "고정60°":em(np.full(mm.sum(),60.0)),"수직90°":em(np.full(mm.sum(),90.0)),
                   "avg_angle":float(da.loc[mm,"angle_ai"].mean())})
    return wa,w6,w9,pd.DataFrame(ml),da

# ==============================================================================
# CSS — minimal width override; theme/colors are set in .streamlit/config.toml
# ==============================================================================
st.markdown("""<style>
.block-container{padding-top:1.0rem;padding-bottom:0.5rem;max-width:none}
</style>""", unsafe_allow_html=True)

# ==============================================================================
# 데이터 준비
# ==============================================================================
mdl = load_model() if _XGB_AVAILABLE else None
kma, tom = get_kma()

st.sidebar.title("BIPV · 설정")
tom_dt=datetime.strptime(tom,"%Y%m%d") if tom else datetime.now()+timedelta(days=1)
sim_date=st.sidebar.date_input("날짜",tom_dt)

with st.sidebar.expander("고급 설정", expanded=False):
    st.subheader("2. 블레이드 (도면 1043A)")
    st.caption("중심축 회전 | 현≈피치 (닫힘 연속면)")
    wm=st.number_input("가로(mm)",min_value=100.0,value=DW,step=100.0)
    cm=st.number_input("현/폭 CHORD(mm)",min_value=10.0,value=DCH,step=0.5)
    pm=st.number_input("피치(mm)",min_value=10.0,value=DP,step=0.5)
    st.caption(f"현/피치 비율={cm/pm:.2f}")
    lc=st.number_input("블레이드 수",min_value=1,value=DLC,step=1)
    st.subheader("3. 패널")
    uc=st.number_input("유닛 수",min_value=1,value=DU)
    cw=st.number_input("용량(W)",value=DC)
    te=st.number_input("효율(%)",value=DE,step=0.1)
    kr=st.number_input("요금(원/kWh)",value=DK)
ef=float(te)/DE; asc=(wm*cm*lc)/(DW*DCH*DLC)

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
am="XGBoost v17" if xa is not None else "규칙 기반"
def dpow(ang):
    e=eff_poa(np.asarray(ang,dtype=float),el,az,dn,dh,cm,pm)
    return (e[ghi>=10]/1000*cw*uc*ef*DL*asc).sum()
pa,p6,p9=dpow(ai),dpow(np.full(24,60.0)),dpow(np.full(24,90.0))
ly=sim_date.year-1; ux=mdl is not None

# 연간 발전량: CSV 실측 데이터 우선, 없으면 clearsky 시뮬
df_csv_raw, csv_err_raw = load_csv()

# ==============================================================================
# ★ v9.3 변경: 10년 전체 평균으로 연간 발전량 계산 (Colab V15 일치)
# ==============================================================================
@st.cache_data(ttl=86400)
def get_annual_from_csv(df_src, c, p, cw_p, uc_p, ef_p, dl_p):
    """CSV 실측 기상 데이터로 발전량 직접 계산 — 10년 평균 (Colab V15 동일)"""
    df2 = df_src.copy()
    df2["ts"] = pd.to_datetime(df2["timestamp"])

    # ── v9.3: 전체 연도 사용, 연수로 나누어 연평균 ──
    years = sorted(df2["ts"].dt.year.unique())
    n_years = len(years)
    year_range = f"{years[0]}~{years[-1]}" if n_years > 1 else str(years[0])

    ghi2 = df2["ghi_w_m2"].values
    mask = ghi2 >= 10
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

    # ── v9.3: 전체 합산 / 연수 = 연평균 ──
    def en(ang):
        e = eff_poa(np.asarray(ang,dtype=float), el2, az2, dn2, dh2, c, p)
        return (e[mask]/1000*cw_p*uc_p*ef_p*dl_p).sum() / n_years

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
        # ── v9.3: 월별도 연수로 나누기 ──
        def em(ang):
            e = eff_poa(np.asarray(ang,dtype=float), el2[mm], az2[mm], dn2[mm], dh2[mm], c, p)
            return (e/1000*cw_p*uc_p*ef_p*dl_p).sum() / n_years
        ml2.append({"month":m,"AI":em(ai2[mm]),
                    "고정60°":em(np.full(mm.sum(),60.0)),"수직90°":em(np.full(mm.sum(),90.0)),
                    "avg_angle":float(ai2[mm].mean())})
    return wa2, w62, w92, pd.DataFrame(ml2), da2, year_range

# ── v9.3: 호출부 수정 ──
if df_csv_raw is not None and "solar_elevation" in df_csv_raw.columns:
    wa,w6,w9,dmo,dan,year_range = get_annual_from_csv(df_csv_raw, cm, pm, cw, uc, ef, DL)
    annual_source = f"실측 기상 {year_range} 평균"
    _dbg_years = sorted(pd.to_datetime(df_csv_raw["timestamp"]).dt.year.unique())
else:
    wa,w6,w9,dmo,dan = get_annual(ly,cm,pm,cw,uc,ef,DL,ux)
    annual_source = f"청천 모델 근사 ({ly}년)"

ka=wa/1000*asc; k6=w6/1000*asc; k9=w9/1000*asc
ks="✅ 기상청 연동" if (kma is not None and sd.replace("-","")==tom) else "⚠️ 청천 기준"
ws="맑음" if np.mean(cl)<0.3 else("구름많음" if np.mean(cl)<0.8 else "흐림")
md2=(ts.hour>=6)&(ts.hour<=19)
mn=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ==============================================================================
# Single-page render — embed main.html once with the full data dict
# ==============================================================================
from pathlib import Path

if df_csv_raw is not None:
    _yrs = sorted(pd.to_datetime(df_csv_raw["timestamp"]).dt.year.unique())
    _yr_range = f"{_yrs[0]}–{_yrs[-1]}"
    _train_rows = int(len(df_csv_raw))
else:
    _yr_range = "2014–2023"
    _train_rows = 44225

# 월별 AI 우위 (고정 60° 대비 추가 kWh) — dmo 는 get_annual_from_csv() 또는 get_annual() 결과
_mo_AI    = [round(float(v) * asc / 1000, 1) for v in dmo["AI"].values]
_mo_F60   = [round(float(v) * asc / 1000, 1) for v in dmo["고정60°"].values]
_mo_extra = [round(a - b, 1) for a, b in zip(_mo_AI, _mo_F60)]

# 월간 uplift: 라벨이 "이번 달 성과"이므로 (ka/k6 - 1)*100 의 연간 평균 uplift rate 사용
# (월간 데이터가 없어도 비율은 연간/월간이 비슷)
_uplift_pct = round((ka/k6 - 1) * 100, 1) if k6 > 0 else 0.0
_month_kwh  = round((ka - k6) / 12, 1)  # 월평균 추가 kWh

# ── 모델 지표: model_metrics_v17.json (train_v17.py 홀드아웃 실측) ──
try:
    with open("model_metrics_v17.json", encoding="utf-8") as _f:
        _met = json.load(_f)
except Exception:
    _met = {"r2_day": 0.9665, "mae_day": 0.6}  # v17 학습 시점 홀드아웃 값 (json 부재 시)

# ── feature importance: 로드된 모델에서 동적 산출 (predict_xgb 입력 순서) ──
_FEAT_ORDER = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "ghi_w_m2", "cloud_cover", "temp_actual"]
_FEAT_KO = {
    "doy_cos": "계절 위치 (cos)", "doy_sin": "계절 (sin)",
    "hour_cos": "시간 (cos)", "hour_sin": "시간 (sin)",
    "ghi_w_m2": "GHI 일사량", "cloud_cover": "운량", "temp_actual": "외기 온도",
}
_FEAT_MEANING = {
    "doy_cos": "1년 주기에서 현재가 겨울인지 여름인지 — 계절별 태양 궤적 차이.",
    "doy_sin": "춘분 · 추분 지점 식별 (doy_cos 보완).",
    "hour_cos": "24시간 주기 중 오전 / 오후 구분 — 태양이 동쪽인지 서쪽인지.",
    "hour_sin": "24시간 주기 중 정오 부근인지 새벽인지 (hour_cos 보완).",
    "ghi_w_m2": "단위 면적당 들어오는 태양 에너지(W/m²) — 현재 발전 잠재력 직접 신호.",
    "cloud_cover": "구름량 — 직달/확산 비율을 바꿔 최적각에 약하게 기여.",
    "temp_actual": "외기 온도 — 최적각과 물리적 인과 없음(계절·시각과의 상관으로 잡힘).",
}
def _importance_list(model):
    try:
        gains = [float(v) for v in model.feature_importances_]
    except Exception:
        gains = [float(_met.get("importance", {}).get(f, 0.0)) for f in _FEAT_ORDER]
    items = sorted(zip(_FEAT_ORDER, gains), key=lambda x: -x[1])
    return [{"f": f, "gain": round(g, 3), "ko": _FEAT_KO[f], "meaning": _FEAT_MEANING[f]}
            for f, g in items]

bipv_data = {
    "address": "서울시 마곡동 1호기",
    "date": sd,
    "system_version": am,
    "weather_mode": "실시간 기상 연동" if (kma is not None and sd.replace("-","") == tom) else "맑음 추정 모드",
    "app_version": __version__,

    "thesis": {
        "headline": "음영을 피하면서 최대 발전. AI가 매 시간 결정하는 루버 각도.",
        "subhead":  f"서울시 마곡동 1호기에서 5분마다 자동 제어 중. {am} 모델, 기상청 10년 시계열로 학습.",
    },

    "this_month": {
        "uplift_pct":           _uplift_pct,
        "kwh_extra":            _month_kwh,
        "annual_kwh_extra":     round(ka - k6, 1),
        "annual_revenue_extra": int((ka - k6) * kr),
    },
    "today": {
        "ai_kwh":            round(pa / 1000, 2),
        "ai_revenue":        int(pa / 1000 * kr),
        "uplift_vs_f60_kwh": round((pa - p6) / 1000, 2),
    },

    "schedule_24h": {
        "hours":    [t.strftime("%H") for t in ts[md2]],
        "ghi":      ghi[md2].round(1).tolist(),
        "ai_angle": ai[md2].astype(int).tolist(),
    },

    # 3D 하루 시뮬레이터용 24h 시계열 (실제 태양위치·일사량·AI각)
    "sim_day": {
        "hour":      [int(t.hour) for t in ts],
        "elevation": [round(float(v), 2) for v in el],
        "azimuth":   [round(float(v), 2) for v in az],
        "ghi":       [round(float(v), 1) for v in ghi],
        "dni":       [round(float(v), 1) for v in dn],
        "dhi":       [round(float(v), 1) for v in dh],
        "ai_angle":  [int(round(float(v))) for v in ai],
    },

    "compare": {
        "AI":  round(pa / 1000, 3),
        "F60": round(p6 / 1000, 3),
        "F90": round(p9 / 1000, 3),
    },

    # v17 실기하 (JS 측 panel_sf / svf 라이브 계산용) — 도면 1043A
    "geom": {
        "c": float(cm),     # 현(chord, mm)
        "p": float(pm),     # 피치 (mm)
    },

    # 현재 블레이드 사양 (Section 06 spec card)
    "blade": {
        "chord":    round(float(cm), 1),
        "pitch":    round(float(pm), 1),
        "width":    int(wm),
        "count":    int(lc),
        "capacity": int(cw),
    },

    "model": {
        "r2": _met.get("r2_day"),
        "mae_deg": _met.get("mae_day"),
        "energy_regret_pct": _met.get("energy_regret_pct"),
        "training_rows": _train_rows,
        "year_range": _yr_range,
        "n_features": 7,
        "cv": _met.get("holdout", "시간순 홀드아웃 20%"),
        "motor_precision_deg": 2,
        "feature_names_korean": _FEAT_KO,
        "importance": _importance_list(mdl),
    },

    "ops": {
        "model_loaded":   bool(mdl),
        "csv_loaded":     df_csv_raw is not None,
        "weather_api_ok": kma is not None,
    },

    # 월별 AI 우위 — Section 06 monthly chart
    "monthly_extra": {
        "months": list(range(1, 13)),
        "AI":     _mo_AI,
        "F60":    _mo_F60,
        "extra":  _mo_extra,
    },

    "glossary": {
        "GHI": "수평면 전일사량(W/m²) — 단위 면적당 받는 태양 에너지",
        "SF":  "음영률 — 인접 블레이드가 발전면을 가리는 비율 (낮을수록 좋음)",
        "SVF": "하늘 조망 계수(0~1) — 루버 사이로 보이는 하늘 비율, 확산광 수집 지표",
        "R²":  "결정계수 — 모델이 데이터 분산을 설명하는 비율 (1에 가까울수록 좋음)",
        "MAE": "평균 절대 오차 — 예측 각도와 실제 최적각의 평균 차이",
    },
}

_html = Path("main.html").read_text(encoding="utf-8").replace(
    "{{BIPV_DATA}}", json.dumps(bipv_data, ensure_ascii=False)
)
components.html(_html, height=6800, scrolling=False)
