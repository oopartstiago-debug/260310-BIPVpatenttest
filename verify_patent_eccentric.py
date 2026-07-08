# ==============================================================================
# KR102460661B1 편심 회전축 구조가 우리 제품에 영향 주는지 검증 (2026-07-02)
#   특허: 3장 그룹 회전축 수직 편심(제1 상측 24%, 제3 하측 11%, 제2 중앙고정)
#         → 자기음영 격자 분산. "여름 40°각 +16.8%" 주장(실측 미검증).
#   우리: 현114·피치97.5(gcr1.169) 균일, 최적 80° 고각 운용.
#
#   검증: 광선추적으로 (a)우리 균일 기하 각도별 자기음영 sf,
#         (b)특허 편심 3장 그룹 자기음영 → 각도별 저감량,
#         (c)연간 발전·최적각 변화 (편심 이득이 우리 최적점 80°에 유의미한가)
# 실행: .venv/bin/python verify_patent_eccentric.py
#
# ★2026-07-08 Fable 적대검증 정정 (원문 정독 후 — 이 스크립트 해석문은 구판):
#   ①각도 관례: 특허 회전각=수직(닫힘) 기준 → 특허 40° = 우리 tilt 50°. 구판 해석
#     "특허는 40~60° 저각 운용에서만 유효"는 관례 혼동 — 특허는 자기 기하서 전 각도 이득 주장.
#   ②특허 기하 ≠ 우리: 성능표 역산 gcr≈0.4 성긴 배치(우리 밀폐 1.169는 피팅 배제, 3배 차).
#   ③선형 지표로는 편심 이득 수학적 불가(Jensen: 간격 재분배는 평균음영 못 줄임, 전 조합 재현 실패).
#     특허 +16.8~56%는 그림자 집중([25,25,25]→[49,36,0]%)이 다중 셀스트링 부분음영 비선형을
#     탈 때만 성립. 우리 셀=현방향 단일 스트립·음영 선형(codex 검증) → 메커니즘 자체 무적용.
#   ④아래 ecc 구현은 중심고정(-7%p@80°는 그 산물). 특허 정독=축고정 → 우리 기하서 효과 0.0%.
#     어느 구현이든 "이득 없음" 결론 동일, 수치만 구현 의존.
# ==============================================================================
import numpy as np, pandas as pd, pvlib
import physics_v3 as P

C, PIT = 114.0, 97.5   # 우리 실기하

def sf_group(tilt, elev, az, ecc, sa=180.0, n=60):
    """3장 반복 그룹 광선추적 자기음영률(그룹 평균). ecc=(e1,e2,e3) 편심(현 비율, 축이 블레이드중심 위=+).
       균일=ecc(0,0,0). 특허=대략 (+0.24, 0, -0.11)."""
    tr = np.radians(tilt); er = np.radians(np.clip(elev, 0.1, 89.9))
    d = np.array([np.cos(er)*np.cos(np.radians(az-sa)), np.sin(er)])  # 점→태양(위로), shadow_dir.py 검증방향
    half = C/2
    # 블레이드 j의 기하중심 y = j*PIT (균일 배치), 축 = center + ecc_j*C
    # 회전 방향 단위(2D 프로파일): 블레이드는 tilt로 기움. dir=(cos tr, -sin tr)
    dirv = np.array([np.cos(tr), -np.sin(tr)])
    def endpoints(j, ecc_j):
        cy = j*PIT; axis = np.array([0.0, cy + ecc_j*C])
        top = axis + (half - ecc_j*C)*dirv      # 상단(+half from center)
        bot = axis + (-half - ecc_j*C)*dirv     # 하단
        return top, bot, axis
    # 반복 단위 3장(j=0,1,2), 이웃 그룹 ±2단위까지
    blades = []
    for g in range(-2, 3):
        for j,ej in enumerate(ecc):
            blades.append(endpoints(g*3+j, ej))
    def seg_hit(px, py):
        p = np.array([px,py])
        for (top,bot,ax) in blades:
            if abs(top[1]-py)<1e-6 and abs(bot[1]-py)<1e-6: continue
            e = bot-top; q = top-p
            den = d[0]*e[1]-d[1]*e[0]
            if abs(den)<1e-12: continue
            t = (q[0]*e[1]-q[1]*e[0])/den
            s = (q[0]*d[1]-q[1]*d[0])/den
            if t>1e-6 and 0<=s<=1: return True
        return False
    # 대상 = 중앙 그룹 3장 각각의 현 위 샘플, 음영 비율(그룹 평균)
    shaded=0; tot=0
    for j,ej in enumerate(ecc):
        top,bot,ax = endpoints(j,ej)
        for s in np.linspace(0.02,0.98,n):
            pt = top + s*(bot-top)
            tot+=1
            if elev>0 and seg_hit(pt[0],pt[1]): shaded+=1
    return shaded/tot if tot else 0.0

# 데이터
df=pd.read_csv("bipv_ai_master_data_v17.csv"); dd=df[df.solar_elevation>0]
el=dd.solar_elevation.to_numpy(float);az=dd.solar_azimuth.to_numpy(float)
dni=dd.dni.to_numpy(float);dhi=dd.dhi.to_numpy(float);N=len(dd)

from physics_v2 import panel_sf
print("(검증) 균일 sf_group vs 정식 panel_sf (일치해야 신뢰)")
print("    tilt   sf_group   panel_sf")
for t in (40,60,80):
    g=sf_group(t,50,150,(0,0,0)); ps=float(panel_sf(np.array([float(t)]),np.array([50.0]),np.array([150.0]),hd=C,p=PIT)[0])
    print(f"    {t:>3}°   {g:.3f}     {ps:.3f}")
print()
print("(a) 각도별 자기음영률 sf — 우리 균일 기하 (여름정오 elev76, 저각 elev25)")
print("    tilt   여름76°   저각25°   ← 우리 최적은 80°(고각)")
for t in (40,50,60,70,80,90):
    print(f"    {t:>3}°    {sf_group(t,76,180,(0,0,0)):.3f}    {sf_group(t,25,180,(0,0,0)):.3f}")

print("\n(b) 편심(특허) vs 균일(우리) 자기음영 저감 — 여름 elev76")
print("    tilt   균일sf   편심sf   저감")
for t in (40,50,60,70,80):
    u=sf_group(t,76,180,(0,0,0)); e=sf_group(t,76,180,(0.24,0,-0.11))
    print(f"    {t:>3}°   {u:.3f}   {e:.3f}   {(u-e):+.3f} ({(1-e/u)*100 if u>0 else 0:+.0f}%)")

print("\n(c) 최적각에서 자기음영이 이미 작은가? (우리 운용점)")
for t in (80,):
    print(f"    tilt {t}° 여름정오 sf={sf_group(t,76,180,(0,0,0)):.3f} → 편심 이득 상한 = 이 값×직달비중")
print("\n해석(2026-07-08 정정): 편심 이득은 성긴 기하(gcr≈0.4)+다중 스트링 부분음영 비선형 전제.")
print("      밀폐 gcr1.169+선형 단일스트립 셀엔 어느 각도서도 이전 불가(축고정 구현 시 효과 0.0%).")
print("      (구판 '40~60° 저각서만 유효'는 각도관례 혼동 — 헤더 ★2026-07-08 블록 참조)")
