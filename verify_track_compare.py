# ==============================================================================
# AI Tilt — 신/구 모델 비교(동일 조건): 에너지 추종률 + 아침지각/오후오버슛 비대칭
#   목적: oracle 추종 셰이핑 보상(wTrack) 재학습이 아침 모터램프 지각 ↔ 오후
#         오버슛 비대칭을 실제로 줄였는지 정량 검증. 같은 rng(42)로 시작각·피치·날을
#         동일 추출 → 두 모델을 공정 비교.
# ==============================================================================
import json
import numpy as np
import onnxruntime as ort
from physics_v3 import eff_poa

LIB = "unity_viz/AITiltViz/AITiltViz/Assets/StreamingAssets/sun_days.json"
STEPS, MAX_DELTA, CHORD, ALB = 120, 3.0, 97.5, 0.15
TILTS = np.arange(0, 90.1, 1.0)
MODELS = {
    "구 louver_cloud": "unity_viz/results/louver_cloud/LouverTilt/LouverTilt-3000120.onnx",
    "신 louver_track": "unity_viz/results/louver_track/LouverTilt/LouverTilt-3000120.onnx",
}


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
    return elev, L("az"), dni, dhi, L("h"), min(max(L("cloud") / 10, 0), 1)


def rollout(sess, fr, pitch, start_tilt, acc):
    tilt = start_tilt; ratio = min(max((CHORD / pitch) / 2, 0), 1)
    cp = co = 0.0
    for step in range(STEPS):
        t01 = step / (STEPS - 1)
        elev, az, dni, dhi, hr, C = sample(fr, t01)
        azr = np.radians(az)
        ov = np.array([[min(max(elev / 90, 0), 1), np.sin(azr), np.cos(azr),
                        min(max(dni / 1000, 0), 1), min(max(dhi / 400, 0), 1),
                        tilt / 90, ratio, t01]], np.float32)
        a = float(np.clip(sess.run(["deterministic_continuous_actions"],
                  {"obs_0": ov, "action_masks": np.zeros((1, 0), np.float32),
                   "recurrent_in": np.zeros((1, 1, 0), np.float32)})[0][0, 0], -1, 1))
        tilt = float(np.clip(tilt + a * MAX_DELTA, 0, 90))
        poa = np.asarray(eff_poa(TILTS, elev, az, dni, dhi, c=CHORD, p=pitch, albedo=ALB)).ravel()
        pm = float(np.asarray(eff_poa(tilt, elev, az, dni, dhi, c=CHORD, p=pitch, albedo=ALB)).ravel()[0])
        cp += pm; co += poa.max()
        ot = TILTS[poa.argmax()]; h = hr
        if elev > 10 and 7 <= h <= 10:           # 아침: |지각| 절대오차
            acc["m_sum"] += abs(tilt - ot); acc["m_n"] += 1
        if elev > 10 and 15 <= h <= 18:          # 오후: 오버슛(부호 있는 model-oracle)
            acc["a_sum"] += (tilt - ot); acc["a_abs"] += abs(tilt - ot); acc["a_n"] += 1
    return 100 * cp / co if co > 1e-3 else 100.0


def run(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)               # 두 모델 동일 추출 보장
    days = json.load(open(LIB))["days"]
    acc = dict(m_sum=0.0, m_n=0, a_sum=0.0, a_abs=0.0, a_n=0)
    tracks, worst = [], (101.0, None)
    for d in days:
        if not any(f["elev"] > 30 for f in d["f"]):
            continue
        pitch = float(rng.uniform(80, 140))       # ← 구 스크립트와 동일 순서
        start = float(rng.uniform(0, 90))         # 시작각도 동일 추출
        tr = rollout(sess, d["f"], pitch, start, acc)
        tracks.append(tr)
        if tr < worst[0]:
            worst = (tr, d["date"])
    t = np.array(tracks)
    return dict(
        n=len(t), mean=t.mean(), med=np.median(t), mn=t.min(),
        p5=np.percentile(t, 5), ge90=100 * (t >= 90).mean(), ge95=100 * (t >= 95).mean(),
        worst=worst[0], worst_day=worst[1],
        morn_lag=acc["m_sum"] / max(1, acc["m_n"]),
        aft_overshoot=acc["a_sum"] / max(1, acc["a_n"]),
        aft_abs=acc["a_abs"] / max(1, acc["a_n"]),
    )


res = {k: run(v) for k, v in MODELS.items()}
hdr = f"{'지표':<22}" + "".join(f"{k:>18}" for k in MODELS)
print(hdr); print("-" * len(hdr))
rows = [
    ("스캔 일수", "n", "{:.0f}"),
    ("추종률 평균 %", "mean", "{:.2f}"),
    ("추종률 중앙값 %", "med", "{:.2f}"),
    ("추종률 최소 %", "mn", "{:.2f}"),
    ("추종률 5%분위 %", "p5", "{:.2f}"),
    ("90%↑ 비율", "ge90", "{:.1f}"),
    ("95%↑ 비율", "ge95", "{:.1f}"),
    ("최악일 추종 %", "worst", "{:.1f}"),
    ("아침 |지각| 평균°", "morn_lag", "{:.2f}"),
    ("오후 오버슛(부호)°", "aft_overshoot", "{:.2f}"),
    ("오후 |오차| 평균°", "aft_abs", "{:.2f}"),
]
for label, key, fmt in rows:
    print(f"{label:<22}" + "".join(f"{fmt.format(res[k][key]):>18}" for k in MODELS))
print(f"\n최악일: " + " / ".join(f"{k}={res[k]['worst_day']}" for k in MODELS))
