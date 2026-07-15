# TOU (time-of-use) revenue: does elevation-only tilt tracking shift generation
# into expensive hours vs a fixed angle optimized for the SAME objective (revenue)?
import numpy as np, pandas as pd
from physics_v3 import eff_poa

df = pd.read_csv('bipv_ai_master_data_v17.csv', parse_dates=['timestamp'])
elev = df['solar_elevation'].values
az   = df['solar_azimuth'].values
dni  = df['dni'].values
dhi  = df['dhi'].values
hour = df['timestamp'].dt.hour.values
month= df['timestamp'].dt.month.values

TILTS = np.arange(0, 90.001, 1.0)  # deg grid

# ---- KEPCO 산업용(을) 고압A 선택II 계시별 요금 (원/kWh, 2024 대략) ----
# 여름(6-8): 경101 / 중147 / 최230   (peak/off = 2.28x)
# 봄가을(3-5,9-10): 경 96 / 중106 / 최147
# 겨울(11-2): 경104 / 중141 / 최193
def price_row(h, m):
    # returns 원/kWh for hour h, month m  (KEPCO 계시 구분)
    if m in (6,7,8):      s='summer'
    elif m in (11,12,1,2):s='winter'
    else:                 s='spring'
    if s=='summer':
        # 최대: 10-12,13-17 ; 중간:9-10,12-13,17-23 ; 경:23-9
        if (10<=h<12) or (13<=h<17): return 230.0
        if (9<=h<10) or (12<=h<13) or (17<=h<23): return 147.0
        return 101.0
    if s=='winter':
        # 최대:10-12,17-20,22-23 ; 중간:9-10,12-17,20-22 ; 경:23-9
        if (10<=h<12) or (17<=h<20) or (22<=h<23): return 193.0
        if (9<=h<10) or (12<=h<17) or (20<=h<22): return 141.0
        return 104.0
    # spring/fall
    if (10<=h<12) or (13<=h<17): return 147.0
    if (9<=h<10) or (12<=h<13) or (17<=h<23): return 106.0
    return 96.0

price = np.array([price_row(h,m) for h,m in zip(hour,month)])
flat  = np.full_like(price, price.mean())  # flat-price control

def analyze(sa, label):
    # gen[i, t] : eff_poa at hour i under tilt t  -> build per tilt (vector over hours)
    G = np.empty((len(TILTS), len(elev)))
    for k,t in enumerate(TILTS):
        G[k] = eff_poa(t, elev, az, dni, dhi, sa=sa)
    # TRACKING: per hour pick tilt maximizing gen (== maximizing gen*price since price>0 scalar)
    track_gen = G.max(axis=0)
    # FIXED optimized for a given price vector p: argmax_t sum(G[t]*p)
    def fixed_for(p):
        rev = (G * p[None,:]).sum(axis=1)
        k = rev.argmax()
        return TILTS[k], G[k]
    out={}
    for pname, p in (('TOU',price),('flat',flat)):
        ft, fgen = fixed_for(p)
        track_rev = (track_gen*p).sum()
        fix_rev   = (fgen*p).sum()
        out[pname]=dict(fixed_tilt=ft,
                        gain_pct=100*(track_rev/fix_rev-1),
                        track_rev=track_rev, fix_rev=fix_rev)
    # also: fixed optimized for kWh (flat) then valued at TOU  == naive fixed
    ft_kwh, fgen_kwh = fixed_for(flat)
    naive_rev = (fgen_kwh*price).sum()
    track_rev_tou = (track_gen*price).sum()
    print(f"\n=== {label} (sa={sa}) ===")
    print(f"  fixed tilt  (opt for kWh)     : {ft_kwh:.0f} deg")
    print(f"  fixed tilt  (opt for TOU rev) : {out['TOU']['fixed_tilt']:.0f} deg")
    print(f"  TRACKING vs FAIR fixed(TOU)   : +{out['TOU']['gain_pct']:.2f} %  <-- headline")
    print(f"  TRACKING vs fixed (flat/kWh)  : +{out['flat']['gain_pct']:.2f} %  (kWh-tracking gain ref)")
    print(f"  [unfair] tracking vs naive kWh-fixed valued@TOU: +{100*(track_rev_tou/naive_rev-1):.2f} %")
    return out['TOU']['gain_pct']

res={}
for sa,lab in ((90,'EAST'),(180,'SOUTH'),(270,'WEST')):
    res[lab]=analyze(sa,lab)

print("\n--- SUMMARY: tracking gain vs same-objective(TOU) optimal fixed ---")
for lab,g in res.items():
    print(f"  {lab:6s}: +{g:.2f} %")
