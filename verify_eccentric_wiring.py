# ==============================================================================
# 편심이 우리 밀집 기하(gcr1.169)에서 이득 나는가 — 셀 배선 의존성 결판 (2026-07-08)
#   사용자 지적: "HDC도 밀폐 밀집 루버다(성긴 gcr0.4 아님). 같은 루버면 결론이 달라질 수 없다."
#   → Fable "성긴 기하" 추론 폐기. 진짜 갈림 = 셀 배선(선형 vs 비선형)인지 직접 판정.
#
#   우리 실기하(현114·피치97.5·겹침)로 편심(특허 +0.24/0/-0.11) vs 균일:
#     ① 편심이 3장 그림자를 실제로 "몰아주는가"(집중) — 밀집서도 되나 (기하 질문)
#     ② 셀 배선별 발전: retain=max(0,1-k·sf), k=1(선형=우리) / k=2,3(비선형=그림자방향 직렬)
#        → 몰아주기(편심)가 각 배선서 이득인가 (전기 질문)
#   핵심: 밀집이면 편심 이득이 0인가(우리 물리), 아니면 배선만 바꾸면 우리도 이득인가.
#   실행: .venv/bin/python verify_eccentric_wiring.py
# ==============================================================================
import numpy as np, pandas as pd

C, PIT = 114.0, 97.5   # 우리 실기하 (밀폐 겹침, gcr=1.169)


def sf_per_blade(tilt, elev, az, ecc, sa=180.0, n=60):
    """3장 반복 그룹, 블레이드별 자기음영률 [sf0,sf1,sf2] 광선추적."""
    tr = np.radians(tilt); er = np.radians(np.clip(elev, 0.1, 89.9))
    d = np.array([np.cos(er) * np.cos(np.radians(az - sa)), np.sin(er)])
    half = C / 2
    dirv = np.array([np.cos(tr), -np.sin(tr)])

    def endpoints(j, ecc_j):
        cy = j * PIT; axis = np.array([0.0, cy + ecc_j * C])
        top = axis + (half - ecc_j * C) * dirv
        bot = axis + (-half - ecc_j * C) * dirv
        return top, bot

    blades = [endpoints(g * 3 + j, ej) for g in range(-2, 3) for j, ej in enumerate(ecc)]

    def hit(px, py):
        p = np.array([px, py])
        for (top, bot) in blades:
            if abs(top[1] - py) < 1e-6 and abs(bot[1] - py) < 1e-6: continue
            e = bot - top; q = top - p
            den = d[0] * e[1] - d[1] * e[0]
            if abs(den) < 1e-12: continue
            t = (q[0] * e[1] - q[1] * e[0]) / den
            s = (q[0] * d[1] - q[1] * d[0]) / den
            if t > 1e-6 and 0 <= s <= 1: return True
        return False

    out = []
    for j, ej in enumerate(ecc):
        top, bot = endpoints(j, ej)
        sh = sum(1 for s in np.linspace(0.02, 0.98, n) if elev > 0 and hit(*(top + s * (bot - top))))
        out.append(sh / n)
    return np.array(out)


ECC_UNI = (0.0, 0.0, 0.0)
ECC_PAT = (0.24, 0.0, -0.11)   # 특허 편심 (상24%/중앙/하11%)

# 데이터 — 빔가중 연간 (밀집서 최적 운용각 78° 고정; 저각 40/50°도 참고)
df = pd.read_csv("bipv_ai_master_data_v17.csv"); dd = df[df.solar_elevation > 5].copy()
# 빔 세기로 가중 표본 (전수 44k는 순수파이썬 광선추적서 과함 → 빔 상위 가중 800 표본)
import numpy as _np
aoi_cos = _np.maximum(_np.cos(_np.radians(_np.abs(dd.solar_azimuth - 180))) * _np.cos(_np.radians(90 - dd.solar_elevation)), 0)
beam = dd.dni.to_numpy(float) * aoi_cos
dd = dd.assign(_beam=beam)
samp = dd.sort_values("_beam", ascending=False).head(1600).iloc[::2]   # 빔 상위 표본 800
el = samp.solar_elevation.to_numpy(float); az = samp.solar_azimuth.to_numpy(float)
w = samp._beam.to_numpy(float); w = w / w.sum()

for TILT in (78, 50, 40):
    su = np.zeros(3); sp = np.zeros(3)
    for i in range(len(el)):
        su += w[i] * sf_per_blade(TILT, el[i], az[i], ECC_UNI)
        sp += w[i] * sf_per_blade(TILT, el[i], az[i], ECC_PAT)
    print(f"\n{'═'*70}\n운영각 tilt={TILT}° (빔가중 표본 {len(el)}, 우리 밀집 gcr{C/PIT:.3f})")
    print(f"  블레이드별 그림자 [b0,b1,b2]:  균일 {np.round(su,3)}  →  편심 {np.round(sp,3)}")
    print(f"  그림자 총합(∝면적):  균일 {su.sum():.3f}  편심 {sp.sum():.3f}  (Jensen: 편심이 총량 못 줄임)")
    print(f"  분산(집중도):  균일 {su.var():.4f}  편심 {sp.var():.4f}  {'← 편심이 몰아줌' if sp.var()>su.var()+1e-4 else '← 집중 안 생김(밀집서 재분배만)'}")
    print(f"  {'─'*66}")
    print(f"  셀 배선별 발전 (retain=max(0,1-k·sf), 편심/균일 비):")
    for k, lbl in [(1.0, "k=1 선형(우리: 현방향1열·길이균일음영)"),
                   (2.0, "k=2 약비선형"),
                   (3.0, "k=3 강비선형(그림자방향 직렬=HDC형)")]:
        gu = np.maximum(0, 1 - k * su).mean(); gp = np.maximum(0, 1 - k * sp).mean()
        gain = (gp / gu - 1) * 100 if gu > 0 else float('nan')
        print(f"    {lbl:38}  균일 {gu:.3f}  편심 {gp:.3f}  → {gain:+.1f}%")
print(f"\n{'═'*70}")
print("판정 기준: ①편심이 밀집서 그림자를 몰아주나(분산↑)? ②몰아주면 어느 배선서 이득?")
print("  선형(k=1)서 편심≈0 이고 강비선형(k=3)서만 이득이면 → 차이는 '배선'이지 '루버'가 아님.")
print("  우리 배선=선형(codex검증) → 이득 0. 단 배선 바꾸면(k=3) 우리도 이득 가능 = 설계선택.")
