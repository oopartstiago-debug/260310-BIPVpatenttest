# ==============================================================================
# AI Scientist 공격 루프 — v17 결론("최적 고원 79~84°, AI vs최고 +1%대")의 재귀 적대검증
# 실행: .venv/bin/python loop_attack.py [--no-llm]
# 구조 (rnd-model critique/conductor 패턴):
#   라운드마다: ① 비평가(LLM qwen3.5:35b, 폴백=정적 공격은행)가 "결론을 바꿀 가정" 제안
#              ② 민감도 후크에 매핑 → 연간 사다리 재계산 → 결정 변동성 판정
#              ③ 매핑 불가 제안 = human-gate 목록 (자동화로 못 재는 물리 — 정직 보고)
#   종료: 신규 material 발견 0 라운드 2연속 (loop-until-dry) 또는 max 4 라운드
#   원장: loop_ledger_v17.jsonl (append, 감사가능)
# Material 기준: |Δ(AI vs 최고고정)| > 1.0%p 또는 최고고정각 이동 > 5° 또는 중앙최적 이동 > 5°
# ==============================================================================
import json
import sys
import urllib.request
import numpy as np
import pandas as pd

import physics_v3 as P3

ANGLES = np.arange(15, 91, 1.0)
LLM_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen3.5:35b"
USE_LLM = "--no-llm" not in sys.argv

df = pd.read_csv("bipv_ai_master_data_v17.csv")
day = df[df["ghi_w_m2"] >= 10]
EL = day["solar_elevation"].to_numpy(float); AZ = day["solar_azimuth"].to_numpy(float)
DNI = day["dni"].to_numpy(float); DHI = day["dhi"].to_numpy(float)


# ── 민감도 후크: 이름 → eff_poa 파라미터 오버라이드 (전부 physics_v3 파라미터) ──
HOOKS = {
    "알베도 0.0 (지면반사 전무)":   {"albedo": 0.0},
    "알베도 0.30 (밝은 지면)":      {"albedo": 0.30},
    "IAM a_r 0.12 (AR코팅 유리)":   {"a_r": 0.12},
    "IAM a_r 0.21 (오염/저급 유리)": {"a_r": 0.21},
    "PV띠 바깥 배치":               {"strip_lo": 1 - P3.STRIP_FRAC},
    "PV띠 안쪽 배치":               {"strip_lo": 0.0},
    "현/피치 0.85 (현 과대평가시)":  {"c": 0.85 * P3.PITCH},
    "현/피치 1.05 (겹침 가스켓)":    {"c": 1.05 * P3.PITCH},
    "동향 설치 (az 90)":            {"sa": 90.0},
    "서향 설치 (az 270)":           {"sa": 270.0},
    "하늘 이방성 (회절→빔 채널)":     "ANISO",   # 특수 구현
}
KEYMAP = {  # 비평가 제안 키워드 → 후크
    "알베도": ["알베도 0.0 (지면반사 전무)", "알베도 0.30 (밝은 지면)"],
    "지면": ["알베도 0.0 (지면반사 전무)", "알베도 0.30 (밝은 지면)"],
    "반사손실": ["IAM a_r 0.12 (AR코팅 유리)", "IAM a_r 0.21 (오염/저급 유리)"],
    "iam": ["IAM a_r 0.12 (AR코팅 유리)", "IAM a_r 0.21 (오염/저급 유리)"],
    "입사각": ["IAM a_r 0.12 (AR코팅 유리)", "IAM a_r 0.21 (오염/저급 유리)"],
    "셀 위치": ["PV띠 바깥 배치", "PV띠 안쪽 배치"],
    "띠": ["PV띠 바깥 배치", "PV띠 안쪽 배치"],
    "마진": ["PV띠 바깥 배치", "PV띠 안쪽 배치"],
    "현": ["현/피치 0.85 (현 과대평가시)", "현/피치 1.05 (겹침 가스켓)"],
    "피치": ["현/피치 0.85 (현 과대평가시)", "현/피치 1.05 (겹침 가스켓)"],
    "방위": ["동향 설치 (az 90)", "서향 설치 (az 270)"],
    "동향": ["동향 설치 (az 90)"], "서향": ["서향 설치 (az 270)"],
    "이방성": ["하늘 이방성 (회절→빔 채널)"], "perez": ["하늘 이방성 (회절→빔 채널)"],
    "확산": ["하늘 이방성 (회절→빔 채널)"],
}
ASSUMPTIONS = ("등방 하늘(검증:이방성 ±0.2%p), 알베도 0.15 고정, IAM Martin-Ruiz a_r=0.16, "
               "PV띠 중앙(도면+사용자 확정), 현=피치=97.5(도면), 남향, 평판 블레이드, "
               "블레이드간 반사 무시, 온도/적설/오염 무시, 선형 retain(전류매칭)")


def annual(label, **over):
    """연간 사다리: physics_v3 파라미터 오버라이드 → 결정 지표."""
    kw = {k: v for k, v in over.items() if k != "aniso"}
    T = ANGLES[:, None]
    E = P3.eff_poa(T, EL[None, :], AZ[None, :], DNI[None, :], DHI[None, :], **kw)
    if over.get("aniso"):
        import pvlib
        dni_ext = np.asarray(pvlib.irradiance.get_extra_radiation(
            pd.to_datetime(day["timestamp"]).dt.dayofyear), dtype=float)
        ai = np.clip(DNI / dni_ext, 0, 1)
        sa = kw.get("sa", 180.0)
        sf = P3.panel_sf(T, EL[None, :], AZ[None, :], hd=kw.get("c", P3.CHORD), p=kw.get("p", P3.PITCH), sa=sa)
        ov = P3.strip_shade(sf, kw.get("strip_lo", P3.STRIP_LO))
        aoi = pvlib.irradiance.aoi(T, sa, 90 - EL[None, :], AZ[None, :])
        rb = np.maximum(np.cos(np.radians(np.clip(aoi, 0, 90))), 0)
        cs = DHI[None, :] * ai[None, :] * rb / np.maximum(np.sin(np.radians(EL))[None, :], 0.087)
        fsky = P3.view_factor(T, kw.get("c", P3.CHORD), kw.get("p", P3.PITCH), "f_sky")
        E = E - DHI[None, :] * ai[None, :] * fsky + cs * (1 - ov)   # 등방분 회절몫을 빔채널로 이동
        E = np.maximum(E, 0)
    idx = np.argmax(E, axis=0)
    e_ai = E[idx, np.arange(E.shape[1])].sum()
    sums = E.sum(axis=1)
    best_j = int(np.argmax(sums))
    return {"label": label, "median_opt": float(np.median(ANGLES[idx])),
            "best_fixed": float(ANGLES[best_j]),
            "ai_vs_best": round(float(e_ai / sums[best_j] - 1) * 100, 2),
            "ai_vs_60": round(float(e_ai / sums[int(np.where(ANGLES == 60)[0][0])] - 1) * 100, 2),
            "ai_vs_90": round(float(e_ai / sums[int(np.where(ANGLES == 90)[0][0])] - 1) * 100, 2)}


def run_hook(name):
    spec = HOOKS[name]
    return annual(name, aniso=True) if spec == "ANISO" else annual(name, **spec)


def material(r, base):
    return (abs(r["ai_vs_best"] - base["ai_vs_best"]) > 1.0
            or abs(r["best_fixed"] - base["best_fixed"]) > 5
            or abs(r["median_opt"] - base["median_opt"]) > 5)


def llm_attacks(tested):
    """비평가 LLM: 결론을 바꿀 수 있는 가정 공격 제안. 실패 시 None → 정적 은행 폴백."""
    prompt = (
        "너는 BIPV 루버(블레이드 현=피치=97.5mm, 외벽 개구부 충전, 남향, 서울) 발전물리 모델의 "
        f"적대적 비평가다. 현재 모델 가정: {ASSUMPTIONS}. "
        f"이미 공격한 항목: {sorted(tested) or '없음'}. "
        "모델 결론 '최적 tilt 고원 79~84°, AI 제어 vs 최고고정각 +1%대'를 뒤집을 수 있는 "
        "새로운 공격 가설을 최대 6개, 중복 없이 제안하라. "
        'JSON만 출력: {"attacks":[{"가설":"...","키워드":["..."]}]}')
    body = json.dumps({"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "think": False, "format": "json",
                       "options": {"temperature": 0.4}}).encode()
    try:
        req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(json.load(r)["message"]["content"])
        return out.get("attacks") or None
    except Exception as e:
        print(f"  (LLM 비평가 불가 → 정적 은행 폴백: {type(e).__name__})")
        return None


STATIC_BANK = [{"가설": f"{k} 가정 공격", "키워드": [k]} for k in
               ["알베도", "iam", "셀 위치", "현", "방위", "이방성"]]

print("=" * 78)
print(f"AI Scientist 공격 루프 — v17 (비평가: {'LLM ' + LLM_MODEL if USE_LLM else '정적 은행'})")
print("=" * 78)
base = annual("baseline (v17 기본)")
print(f" 기준: 중앙최적 {base['median_opt']:.0f}° 최고고정 {base['best_fixed']:.0f}° "
      f"AI vs최고 +{base['ai_vs_best']}% vs60 +{base['ai_vs_60']}% vs90 +{base['ai_vs_90']}%")

tested, findings, human_gate, dry = set(), [], [], 0
ledger = open("loop_ledger_v17.jsonl", "a")
for rnd in range(1, 5):
    print(f"\n── 라운드 {rnd} ──")
    attacks = (llm_attacks(tested) if USE_LLM else None) or STATIC_BANK
    new_mat = 0
    rec = {"round": rnd, "attacks": attacks, "results": [], "human_gate": []}
    for a in attacks:
        kws = [str(k).lower() for k in a.get("키워드", [])] + [str(a.get("가설", "")).lower()]
        hooks = sorted({h for kw in kws for key, hs in KEYMAP.items() if key in " ".join(kws) for h in hs})
        if not hooks:
            tag = str(a.get("가설", ""))[:60]
            if tag and tag not in [h[:60] for h in human_gate]:
                human_gate.append(tag)
                rec["human_gate"].append(tag)
                print(f"  [human-gate] {tag} (자동 정량화 불가 — 수동 검토 필요)")
            continue
        for h in hooks:
            if h in tested:
                continue
            tested.add(h)
            r = run_hook(h)
            mat = material(r, base)
            new_mat += mat
            verdict = "★MATERIAL" if mat else "견고"
            print(f"  [{verdict}] {h}: 최적 {r['median_opt']:.0f}°/최고고정 {r['best_fixed']:.0f}° "
                  f"AI vs최고 +{r['ai_vs_best']}% vs90 +{r['ai_vs_90']}%")
            rec["results"].append({**r, "material": bool(mat)})
            if mat:
                findings.append(r)
    ledger.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dry = dry + 1 if new_mat == 0 else 0
    print(f"  라운드 {rnd}: 신규 material {new_mat}건 (dry {dry}/2)")
    if dry >= 2:
        print("\n★ 수렴: 2라운드 연속 신규 발견 0 — 루프 종료")
        break
ledger.close()

print("\n" + "=" * 78)
print(f"종합: 후크 {len(tested)}개 시험, material {len(findings)}건, human-gate {len(human_gate)}건")
for f_ in findings:
    print(f"  ★ {f_['label']}: 최고고정 {f_['best_fixed']:.0f}° (기준 {base['best_fixed']:.0f}°), "
          f"AI vs최고 +{f_['ai_vs_best']}%")
for h in human_gate:
    print(f"  ⚠ human-gate: {h}")
with open("loop_attack_summary.json", "w") as f:
    json.dump({"baseline": base, "tested": sorted(tested), "material": findings,
               "human_gate": human_gate}, f, ensure_ascii=False, indent=1)
print("saved → loop_attack_summary.json / loop_ledger_v17.jsonl")
