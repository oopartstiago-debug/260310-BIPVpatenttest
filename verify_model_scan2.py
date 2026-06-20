# ==============================================================================
# AI Tilt — louver_cloud 에너지 추종률 전수 스캔 + 최악일 추적 (2026-06-19)
#   각 일자 닫힌루프 롤아웃(도메인 랜덤화) → 하루 누적 POA / oracle POA = 추종률.
#   순간 각도가 아니라 '발전 에너지' 기준 판정(평탄지형이라 각도 ±는 무해할 수 있음).
# ==============================================================================
import json
import numpy as np
import onnxruntime as ort
from physics_v3 import eff_poa

MODEL = "unity_viz/results/louver_cloud/LouverTilt/LouverTilt-3000120.onnx"
LIB = "unity_viz/AITiltViz/AITiltViz/Assets/StreamingAssets/sun_days.json"
STEPS, MAX_DELTA, CHORD, ALB = 120, 3.0, 97.5, 0.15
TILTS = np.arange(0, 90.1, 1.0)
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
    return elev, L("az"), dni, dhi, L("h"), min(max(L("cloud") / 10, 0), 1)


def rollout(fr, pitch, trace=False):
    tilt = float(rng.uniform(0, 90)); ratio = min(max((CHORD / pitch) / 2, 0), 1)
    cp = co = 0.0; rows = []
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
        if trace:
            rows.append((hr, elev, C, dni, dhi, tilt, TILTS[poa.argmax()]))
    return (100 * cp / co if co > 1e-3 else 100.0), rows


days = json.load(open(LIB))["days"]
pitches = {}
tracks = []
worst = (101.0, None, None)
for d in days:
    if not any(f["elev"] > 30 for f in d["f"]):
        continue
    pitch = float(rng.uniform(80, 140)); pitches[d["date"]] = pitch
    tr, _ = rollout(d["f"], pitch)
    tracks.append(tr)
    if tr < worst[0]:
        worst = (tr, d["date"], pitch)

tracks = np.array(tracks)
print(f"스캔 {len(tracks)}일 — 하루 에너지 추종률(POA/oracle):")
print(f"  평균 {tracks.mean():.2f}%  중앙값 {np.median(tracks):.2f}%  최소 {tracks.min():.2f}%  5%분위 {np.percentile(tracks,5):.2f}%")
print(f"  98%↑: {100*(tracks>=98).mean():.1f}%일   95%↑: {100*(tracks>=95).mean():.1f}%일   90%↑: {100*(tracks>=90).mean():.1f}%일")

print(f"\n최악 추종일 = {worst[1]}  추종률 {worst[0]:.1f}%  (피치 {worst[2]:.0f})  — 시간별:")
_, rows = rollout(days_by := next(d for d in days if d["date"] == worst[1])["f"], worst[2], trace=True)
print(f"{'시':>4} {'elev':>6} {'cloud':>6} {'DNI':>6} {'DHI':>6} {'모델각':>7} {'oracle각':>9} {'Δ':>5}")
seen = set()
for hr, elev, C, dni, dhi, mt, ot in rows:
    h = int(round(hr))
    if h in seen or elev < 5:
        continue
    seen.add(h)
    print(f"{h:>4d} {elev:>6.1f} {C:>6.2f} {dni:>6.0f} {dhi:>6.0f} {mt:>6.0f}° {ot:>8.0f}° {mt-ot:>4.0f}°")
