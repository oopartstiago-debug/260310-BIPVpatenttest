# ==============================================================================
# 트리 A — 가변 피치 정량 (2026-07-03)
#   근본 질문: "발전시간엔 벌리고(gcr↓), 밤엔 밀폐" — 텔레스코픽 A1의 발전 이득 상한은?
#   현 기하: chord(현)=114, 밀폐피치 p0=97.5 → gcr0=1.169 (겹침 16.5mm).
#   개방: 피치를 p0→pmax 로 벌림(gcr↓). 밤엔 다시 p0 로 밀폐(발전 0이라 발전량 무관).
#
#   ★두 프레이밍 (정직하게 둘 다):
#     [공간당] 실외기실 개구부 크기 고정 → 벌리면 블레이드 수 ∝ 1/p 로 감소.
#              발전밀도 D = (chord/p)·eff_poa  (블레이드 수 × 블레이드당 POA, chord 상수).
#              ← 이 제품(고정 개구부)의 정직한 기준.
#     [면적당] 블레이드당 POA = eff_poa 자체. 파사드 공간이 남아돌 때의 상한.
#
#   baseline = 현 제품: 밀폐(p0) + 최고 고정 tilt.
#   비교군   = 가변피치 + tilt 트래킹 (매 스텝 tilt·p 자유선택, p∈[p0, pmax]).
#   교차검증 = pvlib infinite_sheds (독립 표준, 공간당은 poa_front×gcr).
# ==============================================================================
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import physics_v3 as P

CHORD = 114.0          # 현 (도면 1043A)
P0 = 97.5              # 밀폐 피치 → gcr0 = 114/97.5 = 1.169
ALB = P.ALBEDO

df = pd.read_csv("bipv_ai_master_data_v15_realistic.csv")   # TMY 현실화 (구름 반영)
d = df[df.ghi_w_m2 >= 10].copy()
el = d.solar_elevation.to_numpy(float); saz = d.solar_azimuth.to_numpy(float)
dni = d.dni.to_numpy(float); dhi = d.dhi.to_numpy(float)
cc = d.cloud_cover.to_numpy(float)
N = len(d)

# 피치 후보 (ratio = chord/p, VF 표 0.30~1.20 안). p0=97.5 는 gcr 1.169.
PITCHES = np.array([97.5, 105, 114, 125, 137, 152, 171, 195, 228, 285, 380])
TILTS = np.arange(0.0, 90.1, 3.0)

# eff_poa[pitch_idx, tilt_idx, timestep]  (블레이드당 POA)
print(f"계산: {len(PITCHES)} 피치 × {len(TILTS)} tilt × {N} 스텝 = {len(PITCHES)*len(TILTS)} eff_poa 호출 ...")
POA = np.empty((len(PITCHES), len(TILTS), N))
for ip, p in enumerate(PITCHES):
    for it, t in enumerate(TILTS):
        POA[ip, it] = np.asarray(P.eff_poa(np.full(N, t), el, saz, dni, dhi, c=CHORD, p=p, albedo=ALB)).ravel()

gcr = CHORD / PITCHES                        # 블레이드 밀도 가중 (공간당)
D = POA * gcr[:, None, None]                 # 공간당 발전밀도 = (chord/p)·eff_poa

def annual(mask=None):
    sel = slice(None) if mask is None else mask
    # --- baseline: 밀폐 p0 + 최고 고정 tilt (공간당) ---
    d0 = D[0][:, sel]                         # p0 (index 0)
    base_ann = d0.sum(1); base = base_ann.max(); base_t = TILTS[base_ann.argmax()]
    # --- 1. tilt 트래킹만, 밀폐 p0 (공간당) : 매 스텝 tilt 최적, p=p0 ---
    trk_dense = D[0][:, sel].max(0).sum()
    # --- 2. 가변피치+tilt 트래킹 (공간당) : pmax 별로 p∈[p0,pmax] 허용 ---
    #     per timestep max over (tilt, p<=pmax)
    facade = {}
    for k in range(len(PITCHES)):            # pmax = PITCHES[k]
        best_per_t = D[:k+1][:, :, sel].reshape(-1, D[:, :, sel].shape[-1]).max(0)  # over pitch<=pmax & tilt
        facade[PITCHES[k]] = best_per_t.sum()
    # --- 3. 면적당(블레이드당) 상한: eff_poa (gcr 가중 없음) ---
    poa0 = POA[0][:, sel]; blade_base = poa0.sum(1).max()
    blade = {}
    for k in range(len(PITCHES)):
        best_per_t = POA[:k+1][:, :, sel].reshape(-1, POA[:, :, sel].shape[-1]).max(0)
        blade[PITCHES[k]] = best_per_t.sum()
    return base, base_t, trk_dense, facade, blade, blade_base

def show(title, mask=None):
    base, base_t, trk_dense, facade, blade, blade_base = annual(mask)
    print(f"\n===== {title} =====")
    print(f"baseline(현 제품) = 밀폐 p0=97.5(gcr1.169) + 최고고정 tilt={base_t:.0f}°")
    print(f"  tilt 트래킹만(밀폐 유지)          : {(trk_dense/base-1)*100:+.2f}%   ← 피치 고정, 각도만")
    print(f"\n  {'pmax(mm)':>9}{'gcr':>7}{'개방gap':>8} │ {'공간당(정직)':>12} │ {'면적당(상한)':>12}")
    print(f"  {'':>9}{'':>7}{'':>8} │ {'vs 현제품':>12} │ {'블레이드당':>12}")
    print("  " + "-"*56)
    for p in PITCHES:
        fac = (facade[p]/base-1)*100
        bl  = (blade[p]/blade_base-1)*100
        tag = "  ← 밀폐(개방없음)" if p==P0 else ""
        print(f"  {p:>9.0f}{CHORD/p:>7.2f}{p-P0:>7.0f}mm │ {fac:>+11.2f}% │ {bl:>+11.2f}%{tag}")
    return base, facade, blade, blade_base

show("전체 연간 (TMY 현실화)")
show("흐림만 (cloud_cover >= 0.7)", cc >= 0.7)
show("맑음만 (cloud_cover <= 0.3)", cc <= 0.3)

print("\n주해:")
print(" · [공간당]=실외기실 개구부 고정 → 벌리면 블레이드 수↓, 이게 이 제품의 정직한 기준.")
print(" · [면적당]=파사드 공간이 남을 때만 유효한 상한 (블레이드당 POA).")
print(" · 밤 밀폐는 발전 0이라 발전량엔 무관 — 가변피치의 순가치는 '주간 개방 이득'.")
print(" · TMY 현실화도 여전히 낙관(GHI1510·diffuse0.44) → 절대 이득은 상단 가능.")
