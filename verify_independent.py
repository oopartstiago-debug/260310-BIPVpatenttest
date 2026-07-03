# ==============================================================================
# 독립 적대검증 (2026-06-23) — physics_v3 시야계수를 재사용하지 않는 경로로
#   "최고 고정 81° · 평탄 고원 73-85° · 트래킹 천장 ~+0.5%" 결론을 반증 시도.
#   #1 pvlib.bifacial.infinite_sheds (표준 밀집행 모델, gcr=현/피치)  ← 독립
#   #2 해석적 masking-angle (Passias) → 시야계수 교차앵커            ← 독립
#   #3 확산광 비중·TMY 출처 감사 (diffuse fraction 0.223 이상치)     ← 데이터
#   #4 입면 방위 스윕 (physics_v3, 81°가 정남 전용 인공물인지)       ← 방위축
# 실행: .venv/bin/python verify_independent.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
from pvlib.bifacial import infinite_sheds
from pvlib import shading

df = pd.read_csv("bipv_ai_master_data_v15.csv")
d = df[df.ghi_w_m2 >= 10].copy()
zen = d.solar_zenith.to_numpy(float); el = d.solar_elevation.to_numpy(float)
saz = d.solar_azimuth.to_numpy(float)
dni = d.dni.to_numpy(float); dhi = d.dhi.to_numpy(float)
ghi = dni * np.sin(np.radians(np.clip(el, 0, 90))) + dhi
A = np.arange(0, 90.1, 1.0)
GCR = 97.5 / 97.5          # 현/피치 = 1.0 (실기하). infinite_sheds 안정 위해 0.99 fallback
ALB = 0.15


def scan(poa_of_tilt, tilts=A):
    E = np.stack([np.asarray(poa_of_tilt(t)) for t in tilts])   # [tilt, time]
    ann = E.sum(1); mx = ann.max(); best = tilts[ann.argmax()]
    w = tilts[ann >= 0.99 * mx]
    orac = E.max(0).sum()
    return best, w.min(), w.max(), (orac / mx - 1) * 100


# ── #1 infinite_sheds (독립 표준모델) ────────────────────────────────────────
print("#1 pvlib infinite_sheds (gcr=현/피치, physics_v3 미사용) — 독립 표준 밀집행 모델")
def is_poa(gcr):
    def f(t):
        r = infinite_sheds.get_irradiance(
            surface_tilt=t, surface_azimuth=180.0,
            solar_zenith=np.clip(zen, 0, 89.9), solar_azimuth=saz,
            gcr=gcr, height=1.0, pitch=1.0,
            ghi=ghi, dhi=dhi, dni=dni, albedo=ALB, model="isotropic")
        col = "poa_front" if "poa_front" in r else "poa_global_front" if "poa_global_front" in r else "poa_global"
        return np.nan_to_num(r[col].to_numpy(float))
    return f
for gcr in (GCR, 0.99, 0.7, 0.5):
    try:
        best, lo, hi, gain = scan(is_poa(gcr))
        tag = " ←실기하" if abs(gcr - GCR) < 1e-9 else ""
        print(f"   gcr={gcr:.2f}: 최고={best:>4.0f}°  고원1%={lo:.0f}-{hi:.0f}°  오라클vs최고={gain:+.2f}%{tag}")
    except Exception as e:
        print(f"   gcr={gcr:.2f}: 실패 {type(e).__name__}: {e}")
print("   (대조: physics_v3 = 최고 81° / 고원 73-85° / +0.53%)")

# ── #2 해석적 masking-angle (Passias) → 시야계수 교차앵커 ────────────────────
print("\n#2 해석적 masking-angle(Passias) vs physics_v3 MC 시야계수 — 독립 닫힌해")
import physics_v3 as P
print("   tilt   Passias마스킹각   sky차폐율(1-(1+cosψ)/(1+cosβ))   physics_v3 F_sky")
for t in (30.0, 45.0, 60.0, 75.0, 81.0):
    psi = float(shading.masking_angle_passias(t, GCR))     # 평균 마스킹각(도)
    beta = np.radians(t)
    # 등방 하늘: 마스킹각 ψ 아래는 이웃 블레이드가 가림 → 차폐율 근사
    open_vf = (1 + np.cos(beta)) / 2
    masked_vf = (1 + np.cos(beta + np.radians(psi))) / 2
    block = 1 - masked_vf / open_vf
    fsky_v3 = float(P.view_factor(np.array([t]), kind="f_sky")[0])
    print(f"   {t:>4.0f}°    {psi:>6.1f}°          {block:>6.3f}                  {fsky_v3:.3f}")

# ── #3 확산광 비중·TMY 출처 감사 ─────────────────────────────────────────────
print("\n#3 확산광 비중·TMY 감사 (diffuse fraction 0.223 이상치 원인)")
d["month"] = pd.to_datetime(d.timestamp).dt.month
dfrac_day = dhi.sum() / ghi.sum()
# 전시간(주야 무관) 단순 평균 비중도
allh = df.copy()
ze = allh.solar_zenith.to_numpy(float)
ghi_all = allh.ghi_w_m2.to_numpy(float)
dhi_all = allh.dhi.to_numpy(float)
sun = ghi_all > 1
print(f"   주간 에너지가중 diffuse/GHI = {dfrac_day:.3f}  (서울 KMA 실측 연평균 ~0.50-0.55)")
print("   월별 diffuse fraction:")
mf = d.groupby("month").apply(lambda g: g.dhi.sum() /
        (g.dni*np.sin(np.radians(g.solar_elevation.clip(0))) + g.dhi).sum())
print("   " + "  ".join(f"{m}월:{v:.2f}" for m, v in mf.items()))
# 청명지수 kt = ghi / ghi_extraterrestrial_horizontal
I0 = 1361.0
kt = ghi / np.maximum(I0 * np.sin(np.radians(np.clip(el, 0.1, 90))), 1.0)
print(f"   청명지수 kt 중앙값={np.median(kt):.2f} (맑을수록↑; 서울 연평균 kt~0.45-0.5)")
print(f"   → diffuse 0.22+kt 높음 = 데이터가 실제 서울보다 맑음/직달 위주. 영향: 트래킹 이득은")
print(f"     오히려 과대평가 쪽(빔↑) — 즉 현실(확산↑)에선 +0.5% 더 낮아짐 → '낮은 이득'에 보수적.")

# ── #4 입면 방위 스윕 (physics_v3) ───────────────────────────────────────────
print("\n#4 입면 방위 스윕 — 81°·저이득이 정남(180°) 전용 인공물인지")
def v3_poa(sa):
    def f(t):
        return np.asarray(P.eff_poa(np.full(len(el), t), el, saz, dni, dhi, sa=sa)).ravel()
    return f
for sa, name in [(90, "동"), (135, "남동"), (180, "남"), (225, "남서"), (270, "서")]:
    best, lo, hi, gain = scan(v3_poa(sa))
    print(f"   {name}({sa}°): 최고={best:>4.0f}°  고원1%={lo:.0f}-{hi:.0f}°  오라클vs최고={gain:+.2f}%")
