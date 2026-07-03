# ==============================================================================
# 음영 회피 방안 종합 검토 (2026-07-02, 사용자: 다른 방안 있는지 확정)
#   확정 제약: 피치=현(밀폐 딱맞물림, gcr=1.0) · 셀 83mm · 그림자=상단부터
#   실기하: 현 114mm, 현 배치 상24/하7 = 70°/72%자유
#
# 검토 방안:
#   A. 셀 하단끝 배치 (하부마진 0)         — 실링 대가로 상단마진 최대
#   B. 블레이드 폭(=피치) 최적화            — 넓히면 마진↑(효율↑) but 입면 PV밀도↓
#      ★기준 = 입면 세로 단위당 발전(=셀효율 × 83/현), gcr=1 이라 블레이드가 입면 꽉 채움
#   C. 양면 셀 (뒤 실외기실 반사)           — backspace 결과 재확인
#   D. (정성) 위 블레이드 투과상단·셀 2열 등
# 실행: .venv/bin/python verify_shadow_options.py
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P
from physics_v2 import panel_sf
from train_v17 import view_factors_mc

df = pd.read_csv("bipv_ai_master_data_v17.csv")
dd = df[df.solar_elevation > 0]
el = dd.solar_elevation.to_numpy(float); az = dd.solar_azimuth.to_numpy(float)
dni = dd.dni.to_numpy(float); dhi = dd.dhi.to_numpy(float); N = len(dd)
T = np.arange(0, 90.1, 1.0)
CELL = 83.0


def poa_free(t):
    t = np.asarray(t, float); b = np.radians(t)
    aoi = pvlib.irradiance.aoi(t, 180, 90 - el, az)
    bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
    ib = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0)
    sk, gd = pvlib.iam.martin_ruiz_diffuse(np.atleast_1d(t).ravel(), a_r=P.A_R)
    ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
    return np.maximum(bp * ib + dhi * (1 + np.cos(b)) / 2 * np.asarray(sk).reshape(t.shape)
                      + ghi * P.ALBEDO * (1 - np.cos(b)) / 2 * np.asarray(gd).reshape(t.shape), 0)


Ef = max(poa_free(np.full(N, t)).sum() for t in T)


def cell_eff(chord, mtop, bifacial_gain=0.0):
    """현=피치=chord(밀폐 gcr1), 셀83 상단마진 mtop 배치. 최적각서 셀 면적당 효율(%자유)."""
    fr = CELL / chord; lo = mtop / chord
    fs = {t: view_factors_mc(t, 1.0)[0] for t in T}     # gcr=1 고정
    fg = {t: view_factors_mc(t, 1.0)[1] for t in T}

    def E(ts):
        t = np.full(N, ts)
        sf = panel_sf(t, el, az, hd=chord, p=chord)
        ov = P.strip_shade(sf, lo, fr)
        aoi = pvlib.irradiance.aoi(t, 180, 90 - el, az)
        bp = np.maximum(dni * np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
        ib = np.where(aoi < 90, pvlib.iam.martin_ruiz(np.clip(aoi, 0, 89.999), a_r=P.A_R), 0)
        sk, gd = pvlib.iam.martin_ruiz_diffuse(np.array([ts]), a_r=P.A_R)
        ghi = dni * np.maximum(np.sin(np.radians(el)), 0) + dhi
        front = bp * ib * (1 - ov) + dhi * fs[ts] * float(np.ravel(sk)[0]) + ghi * P.ALBEDO * fg[ts] * float(np.ravel(gd)[0])
        return np.maximum(front, 0).sum() * (1 + bifacial_gain)
    Es = np.array([E(t) for t in T]); bi = int(Es.argmax())
    return T[bi], Es[bi] / Ef


print(f"자유패널=100% · 셀 83mm · 밀폐(피치=현) · 그림자 상단부터\n")
print("═" * 74)
print("A. 셀 배치 (현 114 고정) — 하부마진 제약별")
print("─" * 74)
for mbot in (7, 0):
    mtop = 114 - CELL - mbot
    t, e = cell_eff(114, mtop)
    tag = "★현 배치" if mbot == 7 else "셀 하단끝(실링 대가)"
    print(f"  상{mtop:.0f}/하{mbot} mm   최적={t:>3.0f}°   면적당 {e*100:.1f}%자유   {tag}")

print()
print("═" * 74)
print("B. 블레이드 폭(=피치) 최적화 — 입면 세로 단위당 발전밀도")
print("   밀도 = 셀면적당효율 × (83/현)   [gcr=1 이라 블레이드가 입면 꽉 채움]")
print("─" * 74)
print("  현=피치   상단마진   최적각   면적당(%자유)   입면발전밀도")
rows = []
for chord in (90, 97, 105, 114, 125, 140, 160):
    mtop = chord - CELL - 7      # 하부 7 고정
    if mtop < 0: continue
    t, e = cell_eff(chord, mtop)
    dens = e * (CELL / chord)
    rows.append((chord, dens))
    tag = "★현" if chord == 114 else ""
    print(f"  {chord:>4.0f}mm    {mtop:>4.0f}mm     {t:>3.0f}°      {e*100:>5.1f}%       {dens:.3f} {tag}")
bc = max(rows, key=lambda x: x[1])
cur = [d for c, d in rows if c == 114][0]
print(f"\n  → 입면 발전밀도 최대 = 현 {bc[0]}mm ({bc[1]:.3f}), 현114 대비 {(bc[1]/cur-1)*100:+.1f}%")

print()
print("═" * 74)
print("C. 양면 셀 (뒤 실외기실 내부 반사, 후면 수광률 가정)")
print("─" * 74)
for bg in (0.0, 0.05, 0.10, 0.15):
    t, e = cell_eff(114, 24, bifacial_gain=bg)
    print(f"  후면이득 +{bg*100:>2.0f}%   최적={t:>3.0f}°   면적당 {e*100:.1f}%자유")
print("  (후면이득=후면수광/전면. 실외기실 어두우면 0~5%, 밝은반사도장 10~15%)")

print()
print("═" * 74)
print("정성 방안 (정량 부적합/저가치)")
print("─" * 74)
print("  D1 위 블레이드 상단 투과화: 아래 그림자↓ but 위 블레이드 발전↓ (제로섬, 순이득 미미)")
print("  D2 셀 현방향 2열: 현 114에 83 1열이 한계(2열=166>114 불가). 저가치.")
print("  D3 블레이드 계단식 엇배치: 밀폐 파괴 → 사용자 제약 위반. 불가.")
print("  D4 상단 반사판→셀 유도: 구조 복잡·입사각 손실, 실측 없이 과설계.")
