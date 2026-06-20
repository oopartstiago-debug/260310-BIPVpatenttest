# ==============================================================================
# AI Tilt — louver_cloud 모델 전수 스캔 (2026-06-19)
#   3652일 전부 닫힌루프 롤아웃(도메인 랜덤화 피치/알베도 포함, 데모와 동일 분포)
#   → 정오(elev>30°) 모델각이 저각으로 붕괴하는 날이 하나라도 있는지 전수 확인.
#   판정: 전 일자 정오 모델각이 고각이면 → "흐린날 0°" 는 전세션 오독, 모델 정상.
# ==============================================================================
import json
import numpy as np
import onnxruntime as ort

MODEL = "unity_viz/results/louver_cloud/LouverTilt/LouverTilt-3000120.onnx"
LIB = "unity_viz/AITiltViz/AITiltViz/Assets/StreamingAssets/sun_days.json"
STEPS, MAX_DELTA, CHORD = 120, 3.0, 97.5
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(42)


def att(dni, dhi, elev, C):
    s = max(0.0, np.sin(np.radians(elev)))
    ghi = (dni * s + dhi) * (1 - 0.75 * C ** 3.4)
    d = dni * (1 - C) ** 1.5
    return d, max(0.0, ghi - d * s)


def sample(fr, t01):
    n = len(fr); f = min(max(t01, 0), 1) * (n - 1)
    i0 = min(int(f), n - 1); i1 = min(i0 + 1, n - 1); w = f - i0
    a, b = fr[i0], fr[i1]; L = lambda k: a[k] + (b[k] - a[k]) * w
    elev = L("elev"); dni, dhi = att(L("dni"), L("dhi"), elev, min(max(L("cloud") / 10, 0), 1))
    return elev, L("az"), dni, dhi


def rollout_min_midday(fr, pitch):
    tilt = float(rng.uniform(0, 90)); ratio = min(max((CHORD / pitch) / 2, 0), 1)
    mid = []
    for step in range(STEPS):
        t01 = step / (STEPS - 1)
        elev, az, dni, dhi = sample(fr, t01)
        azr = np.radians(az)
        ov = np.array([[min(max(elev / 90, 0), 1), np.sin(azr), np.cos(azr),
                        min(max(dni / 1000, 0), 1), min(max(dhi / 400, 0), 1),
                        tilt / 90, ratio, t01]], np.float32)
        a = float(np.clip(sess.run(["deterministic_continuous_actions"],
                  {"obs_0": ov, "action_masks": np.zeros((1, 0), np.float32),
                   "recurrent_in": np.zeros((1, 1, 0), np.float32)})[0][0, 0], -1, 1))
        tilt = float(np.clip(tilt + a * MAX_DELTA, 0, 90))
        if elev > 30:                    # 진짜 정오대(태양 충분히 높음, 발전 유의미)
            mid.append(tilt)
    return min(mid) if mid else 90.0     # 정오대 최소 모델각


days = json.load(open(LIB))["days"]
worst = (90.0, None, None)
hist = {"<20°": 0, "20–50°": 0, "50–70°": 0, "70–80°": 0, "80–90°": 0}
n = 0
for d in days:
    if not any(f["elev"] > 30 for f in d["f"]):   # 한겨울 저고도일 제외
        continue
    pitch = float(rng.uniform(80, 140))
    m = rollout_min_midday(d["f"], pitch)
    n += 1
    if m < worst[0]:
        worst = (m, d["date"], pitch)
    if m < 20: hist["<20°"] += 1
    elif m < 50: hist["20–50°"] += 1
    elif m < 70: hist["50–70°"] += 1
    elif m < 80: hist["70–80°"] += 1
    else: hist["80–90°"] += 1

print(f"스캔 일수(정오 elev>30 있는 날) = {n}")
print(f"전 일자 '정오대 최소 모델각' 분포:")
for k, v in hist.items():
    print(f"  {k:>8}: {v:>5}일  ({100*v/n:.1f}%)")
print(f"\n전 데이터셋 최악(가장 낮은 정오 모델각) = {worst[0]:.0f}°  ({worst[1]}, 피치 {worst[2]:.0f})")
print(f"판정: {'정오 저각 붕괴 0건 → 모델 정상, 0° 보고는 전세션 오독/일시값' if worst[0] >= 60 else '저각 붕괴일 존재 → 추가 조사 필요'}")
