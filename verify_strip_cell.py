# ==============================================================================
# 셀 폭(83mm 하프셀) < 블레이드 현(97.5mm) — "중요한 분기점" 정밀 재검증
#   (2026-07-01, 사용자 지적: 블레이드 폭≠셀 폭, 이게 결론을 뒤집나?)
#
# 확정 기하 (사용자 2026-07-01):
#   · M6 하프셀 = 166 × 83 mm.  83mm=짧은변=현(97.5) 방향, 166mm=장변=길이 방향
#   · 블레이드당 5셀 → 현 방향 '1열'(83mm), 길이 방향 5셀 직렬
#   · ⇒ 자기음영(현방향 밴드)은 길이방향 5셀에 '균일'하게 걸림
#
# 검증 축:
#   A. 마진 효과   — STRIP_FRAC 1.0(순진, 셀=현) vs 0.851(실제 83/97.5)
#   B. 배치 민감도 — PV 띠 중앙 vs 안쪽(피벗) vs 바깥쪽
#   C. 회로 토폴로지 — 선형(현모델) vs 바이패스 절벽 스펙트럼(1/1, 1/2, 1/3, full)
#                      : 절벽이 물리적으로 성립하는지 + 성립한다면 결론이 얼마나 움직이나
#   각 시나리오: 연 최고고정각 / 오라클 천장(vs최고고정) / 실제 XGBoost 모델 이득
# 실행: .venv/bin/python verify_strip_cell.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P
from physics_v2 import panel_sf

EL, AZ, DNI, DHI = "solar_elevation", "solar_azimuth", "dni", "dhi"
df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df[EL] > 0].copy()
el = dd[EL].to_numpy(float); az = dd[AZ].to_numpy(float)
dni = dd[DNI].to_numpy(float); dhi = dd[DHI].to_numpy(float)
N = len(dd)
TILTS = np.arange(0, 90.1, 1.0)
T15 = np.arange(15, 90.1, 1.0)
C = P.CHORD; PIT = P.PITCH
FRAC_REAL = 83.0 / 97.5
print(f"주간 {N}행 · 현/피치 {C}/{PIT} · 실제 STRIP_FRAC=83/97.5={FRAC_REAL:.3f} "
      f"(마진 각 {(97.5-83)/2:.2f}mm)\n")


# ── 공통: 유효 POA 를 'retain(음영→직달 잔존율)' 함수로 파라미터화 ────────────
# eff_poa 의 직달항만 교체: bp·iam·retain(sf).  확산/지면항은 v3 그대로.
def poa_custom(tilt, retain_fn):
    tilt = np.asarray(tilt, float)
    sf = panel_sf(tilt, el, az, hd=C, p=PIT)
    aoi = pvlib.irradiance.aoi(tilt, 180.0, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    iam_b = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0.0)
    sky, grd = pvlib.iam.martin_ruiz_diffuse(np.atleast_1d(tilt).ravel(), a_r=P.A_R)
    iam_sky = np.asarray(sky, float).reshape(np.shape(tilt))
    iam_grd = np.asarray(grd, float).reshape(np.shape(tilt))
    ghi_est = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    return np.maximum(bp * iam_b * retain_fn(sf)
                      + dhi * P.view_factor(tilt, C, PIT, "f_sky") * iam_sky
                      + ghi_est * P.ALBEDO * P.view_factor(tilt, C, PIT, "f_grd") * iam_grd, 0)


def scan(retain_fn):
    """연 최고고정각·1%고원·오라클천장(vs최고고정)·모델이득 반환."""
    E = np.stack([poa_custom(np.full(N, t), retain_fn) for t in TILTS])  # [tilt, time]
    ann = E.sum(1); mx = ann.max(); best = TILTS[ann.argmax()]
    w = TILTS[ann >= 0.99 * mx]
    # 오라클: 15~90 argmax (라벨 정의)
    E15 = np.stack([poa_custom(np.full(N, t), retain_fn) for t in T15])
    orac_e = E15.max(0).sum()
    gain_orac = (orac_e / mx - 1) * 100
    return best, w.min(), w.max(), gain_orac, mx, E15


# ── retain 함수들 ────────────────────────────────────────────────────────────
def make_strip(lo, frac):
    """선형 PV 띠 음영: 그림자 [0,sf] 안쪽부터, 띠 [lo,lo+frac]."""
    return lambda sf: 1 - np.clip(np.minimum(sf, lo + frac) - lo, 0, None) / frac


def make_cliff(lo, frac, nsub):
    """바이패스 절벽 스펙트럼: 현방향 셀 1열을 nsub 서브스트링(바이패스 단위)으로 나눔.
       그림자가 서브스트링 경계를 넘을 때마다 그 서브스트링 통째 우회(계단).
       nsub=1: 셀 하나라도 부분음영이면 전체 우회(최악) / nsub→∞: 선형 수렴.
       ★물리 주석: 자기음영은 길이방향 5셀에 '균일'하므로 실제로는 스트링 내
       전류매칭(바이패스 미발동)=선형. 이 절벽들은 '만약 걸린다면'의 상한 바운드."""
    edges = lo + np.linspace(0, frac, nsub + 1)  # 서브스트링 경계 (현 좌표)

    def retain(sf):
        sf = np.asarray(sf, float)
        # 각 서브스트링: 그림자가 그 구간에 조금이라도 걸치면 통째 손실
        lost = np.zeros_like(sf)
        for k in range(nsub):
            hit = sf > edges[k]            # 그림자 앞끝이 서브스트링 시작을 넘었나
            lost += hit.astype(float) / nsub
        return np.clip(1 - lost, 0, 1)
    return retain


LO_MID = (1 - FRAC_REAL) / 2
LO_INNER = 0.0                  # PV 띠가 피벗(안쪽) 끝에 붙음 → 그림자 즉시 강타
LO_OUTER = 1 - FRAC_REAL       # PV 띠가 바깥 끝 → 그림자가 마진 다 먹고서야

def report(name, retain_fn):
    best, lo, hi, g, mx, _ = scan(retain_fn)
    # 실제 XGBoost 모델 이득
    try:
        import joblib
        model = joblib.load("bipv_xgboost_model_v17.pkl")
        FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "ghi_w_m2", "cloud_cover", "temp_actual"]
        pred = np.clip(model.predict(dd[FEATS].to_numpy(float)), 15, 90).astype(float)
        e_model = poa_custom(pred, retain_fn).sum()
        gm = (e_model / mx - 1) * 100
    except Exception:
        gm = float("nan")
    print(f"  {name:<34} 최고={best:>4.0f}°  고원1%={lo:>3.0f}-{hi:<3.0f}  "
          f"오라클vs최고={g:+6.2f}%  모델vs최고={gm:+6.2f}%")


print("═" * 92)
print("A+B. 마진(STRIP_FRAC) × 배치 — 전부 '선형' 회로 (현 모델의 물리)")
print("─" * 92)
report("[기준] frac=1.0 (마진무시, 셀=현)", make_strip(0.0, 1.0))
report("frac=0.851 · 중앙배치  ★현모델", make_strip(LO_MID, FRAC_REAL))
report("frac=0.851 · 안쪽(피벗)배치", make_strip(LO_INNER, FRAC_REAL))
report("frac=0.851 · 바깥쪽배치", make_strip(LO_OUTER, FRAC_REAL))
print()
print("═" * 92)
print("C. 회로 토폴로지 — 선형 vs 바이패스 절벽 스펙트럼 (frac=0.851 중앙 고정)")
print("   ※ 자기음영은 길이방향 5셀에 균일 → 실제=전류매칭=선형. 아래 절벽은 상한 바운드.")
print("─" * 92)
report("선형 (현 모델, 셀 균일음영)", make_strip(LO_MID, FRAC_REAL))
report("절벽 nsub=3 (하프컷 3-바이패스 가정)", make_cliff(LO_MID, FRAC_REAL, 3))
report("절벽 nsub=2 (2-바이패스)", make_cliff(LO_MID, FRAC_REAL, 2))
report("절벽 nsub=1 (최악: 부분음영→전손실)", make_cliff(LO_MID, FRAC_REAL, 1))
print()
print("해석 가이드:")
print("  · A+B 행들이 서로 가까우면 → 마진/배치는 결론(최고각·천장)에 미미 = 이미 잘 반영됨.")
print("  · C 에서 선형↔절벽이 크게 벌어지면 → 셀 배선방향(절벽 성립 여부)이 진짜 분기점.")
print("    벌어지지 않으면 → 절벽이어도 결론 불변(음영시간 자체가 이득의 지배항 아님).")
