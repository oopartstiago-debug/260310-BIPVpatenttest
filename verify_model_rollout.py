# ==============================================================================
# AI Tilt — louver_cloud 모델 닫힌루프 롤아웃 (2026-06-19)
#   실제 onnx 정책을 LouverAgent 와 1:1 동일하게 하루(120스텝) 돌려
#   시간별 '모델 예측각' vs 'oracle 최적각' 비교 → 흐린날 0° 가 실재하는지 확정.
#   실행: <mlagents env python> verify_model_rollout.py
# ==============================================================================
import json
import numpy as np
import onnxruntime as ort
from physics_v3 import eff_poa, CHORD, ALBEDO

MODEL = "unity_viz/results/louver_cloud/LouverTilt/LouverTilt-3000120.onnx"
LIB = "unity_viz/AITiltViz/AITiltViz/Assets/StreamingAssets/sun_days.json"
TILTS = np.arange(0, 90.1, 1.0)
STEPS_PER_DAY = 120          # LouverAgent.stepsPerDay
MAX_DELTA = 3.0              # 모터 속도 한계 deg/step
PITCH = 97.5                 # 실하드웨어 nominal (도메인 랜덤화 OFF로 공정 평가)

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])


def att(dni_cs, dhi_cs, elev, C):
    """SolarDayEnv.AttenuateByCloud 1:1."""
    s = max(0.0, np.sin(np.radians(elev)))
    ghi = (dni_cs * s + dhi_cs) * (1 - 0.75 * C ** 3.4)
    dni = dni_cs * (1 - C) ** 1.5
    return dni, max(0.0, ghi - dni * s)


def sample(frames, t01):
    """SolarDayEnv.Sample — 프레임 선형보간 + 구름감쇠."""
    n = len(frames)
    f = min(max(t01, 0.0), 1.0) * (n - 1)
    i0 = min(int(f), n - 1)
    i1 = min(i0 + 1, n - 1)
    fr = f - i0
    a, b = frames[i0], frames[i1]
    lerp = lambda k: a[k] + (b[k] - a[k]) * fr
    elev, az = lerp("elev"), lerp("az")
    dni_cs, dhi_cs = lerp("dni"), lerp("dhi")
    C = min(max(lerp("cloud") / 10.0, 0.0), 1.0)
    dni, dhi = att(dni_cs, dhi_cs, elev, C)
    return dict(elev=elev, az=az, dni=dni, dhi=dhi, cloud=C, hour=lerp("h"))


def obs_vec(s, tilt, t01, pitch=PITCH):
    """LouverAgent.CollectObservations 1:1 (8차원)."""
    azr = np.radians(s["az"])
    return np.array([[
        min(max(s["elev"] / 90.0, 0), 1),
        np.sin(azr),
        np.cos(azr),
        min(max(s["dni"] / 1000.0, 0), 1),
        min(max(s["dhi"] / 400.0, 0), 1),
        tilt / 90.0,
        min(max((CHORD / pitch) / 2.0, 0), 1),
        t01,
    ]], dtype=np.float32)


def oracle(s, pitch=PITCH):
    poa = np.asarray(eff_poa(TILTS, s["elev"], s["az"], s["dni"], s["dhi"],
                             c=CHORD, p=pitch, albedo=ALBEDO)).ravel()
    return TILTS[poa.argmax()], poa


def rollout(frames, pitch=PITCH, start_tilt=45.0):
    tilt = start_tilt
    rows, cum_poa, cum_ora = [], 0.0, 0.0
    for step in range(STEPS_PER_DAY):
        t01 = step / (STEPS_PER_DAY - 1)
        s = sample(frames, t01)
        ov = obs_vec(s, tilt, t01, pitch)
        out = sess.run(["deterministic_continuous_actions"],
                       {"obs_0": ov,
                        "action_masks": np.zeros((1, 0), np.float32),
                        "recurrent_in": np.zeros((1, 1, 0), np.float32)})[0]
        a = float(np.clip(out[0, 0], -1, 1))
        tilt = float(np.clip(tilt + a * MAX_DELTA, 0, 90))
        ot, poa = oracle(s, pitch)
        poa_now = float(np.asarray(eff_poa(tilt, s["elev"], s["az"], s["dni"], s["dhi"],
                                           c=CHORD, p=pitch, albedo=ALBEDO)).ravel()[0])
        cum_poa += poa_now
        cum_ora += poa.max()
        rows.append((s["hour"], s["elev"], s["cloud"], s["dni"], s["dhi"], tilt, ot))
    track = 100 * cum_poa / cum_ora if cum_ora > 1e-3 else 0
    return rows, track


def show(frames, label, pitch=PITCH):
    rows, track = rollout(frames, pitch)
    print(f"\n{'='*64}\n({label})  추종률(하루누적)={track:.1f}%   피치={pitch}\n{'='*64}")
    print(f"{'시':>5} {'elev':>6} {'cloud':>6} {'DNI':>6} {'DHI':>6} {'모델각':>7} {'oracle각':>9} {'Δ':>6}")
    seen = set()
    for h, elev, C, dni, dhi, mt, ot in rows:
        hr = int(round(h))
        if hr in seen or elev < 5:
            continue
        seen.add(hr)
        flag = "  ⚠️" if abs(mt - ot) > 20 else ""
        print(f"{hr:>5d} {elev:>6.1f} {C:>6.2f} {dni:>6.0f} {dhi:>6.0f} {mt:>6.0f}° {ot:>8.0f}° {mt-ot:>5.0f}°{flag}")


days = json.load(open(LIB))["days"]
mean_cloud = lambda d: np.mean([f["cloud"] for f in d["f"] if f["elev"] > 5] or [0]) / 10
summer = sorted([d for d in days if d["date"][5:7] in ("06", "07", "08")], key=mean_cloud)
clear, cloudy = summer[0], summer[-1]
print(f"맑은날={clear['date']}(구름{mean_cloud(clear):.2f})  흐린날={cloudy['date']}(구름{mean_cloud(cloudy):.2f})")
show(clear["f"], f"맑은 여름날 {clear['date']}")
show(cloudy["f"], f"흐린 여름날 {cloudy['date']}")
