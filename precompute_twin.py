# ==============================================================================
# AI Tilt — 디지털 트윈 궤적 사전계산 (교육용 인터랙티브 트윈 데이터)
#   학습된 RL 정책(louver_track)을 대표일에 실제로 롤아웃해, 브라우저가 재생할
#   궤적을 뽑는다. 관측 인코딩/물리는 verify_track_compare.py 와 동일(추측 없음).
#   출력: RL 각도 vs 물리최적(oracle) vs 최적고정각 + 시간별 발전량.
#
#   ★source="simulation": 아직 실측 미연결. inverter-rnd 현장 데이터가 붙으면
#     같은 스키마의 measured 트랙을 추가(Phase 2 sim-to-real). 지금은 슬롯만.
#
#   실행:  /opt/homebrew/Caskroom/miniconda/base/envs/mlagents/bin/python precompute_twin.py
# ==============================================================================
import json
import sys
import numpy as np
import onnxruntime as ort

sys.path.insert(0, ".")
from physics_v3 import eff_poa

LIB = "unity_viz/AITiltViz/AITiltViz/Assets/StreamingAssets/sun_days.json"
MODEL = "unity_viz/results/louver_track/LouverTilt/LouverTilt-3000120.onnx"
OUT_JSON = "static/twin_trajectories.json"

STEPS, MAX_DELTA, CHORD, ALB, PITCH = 120, 3.0, 97.5, 0.15, 110.0
TILTS = np.arange(0, 90.1, 1.0)
RATIO = min(max((CHORD / PITCH) / 2, 0), 1)
# 1세대 고정형의 현실적 기준 = 연중 단일 고정각(프로젝트 확정 best fixed ≈ 81°).
# (그날 사후최적 고정각은 실제로 못 누리는 비현실적 기준이라 쓰지 않음.)
FIXED_ANNUAL = 81.0


def att(dni, dhi, elev, C):
    s = max(0.0, np.sin(np.radians(elev)))
    ghi = (dni * s + dhi) * (1 - 0.75 * C ** 3.4)
    d = dni * (1 - C) ** 1.5
    return d, max(0.0, ghi - d * s)


def sample(fr, t01):
    n = len(fr); f = min(max(t01, 0), 1) * (n - 1)
    i0 = min(int(f), n - 1); i1 = min(i0 + 1, n - 1); w = f - i0
    a, b = fr[i0], fr[i1]; L = lambda k: a[k] + (b[k] - a[k]) * w
    elev = L("elev")
    dni, dhi = att(L("dni"), L("dhi"), elev, min(max(L("cloud") / 10, 0), 1))
    return elev, L("az"), dni, dhi, L("h"), min(max(L("cloud") / 10, 0), 1)


def rollout(sess, fr, start_tilt=45.0):
    fixed = FIXED_ANNUAL
    tilt = start_tilt
    steps = []
    for step in range(STEPS):
        t01 = step / (STEPS - 1)
        elev, az, dni, dhi, hr, C = sample(fr, t01)
        azr = np.radians(az)
        ov = np.array([[min(max(elev / 90, 0), 1), np.sin(azr), np.cos(azr),
                        min(max(dni / 1000, 0), 1), min(max(dhi / 400, 0), 1),
                        tilt / 90, RATIO, t01]], np.float32)
        a = float(np.clip(sess.run(
            ["deterministic_continuous_actions"],
            {"obs_0": ov, "action_masks": np.zeros((1, 0), np.float32),
             "recurrent_in": np.zeros((1, 1, 0), np.float32)})[0][0, 0], -1, 1))
        tilt = float(np.clip(tilt + a * MAX_DELTA, 0, 90))
        poa = np.asarray(eff_poa(TILTS, elev, az, dni, dhi, c=CHORD, p=PITCH, albedo=ALB)).ravel()
        rl_p = float(np.asarray(eff_poa(tilt, elev, az, dni, dhi, c=CHORD, p=PITCH, albedo=ALB)).ravel()[0])
        fix_p = float(np.asarray(eff_poa(fixed, elev, az, dni, dhi, c=CHORD, p=PITCH, albedo=ALB)).ravel()[0])
        steps.append(dict(
            h=round(float(hr), 2), elev=round(float(elev), 1), az=round(float(az), 1),
            rl=round(tilt, 1), orc=round(float(TILTS[poa.argmax()]), 1),
            rl_p=round(rl_p, 1), orc_p=round(float(poa.max()), 1), fix_p=round(fix_p, 1),
        ))
    return dict(fixed_angle=round(fixed, 1), steps=steps)


def rollout_multi(sess, frames_list):
    """연속 여러 날을 정책으로 끊김없이 롤아웃(루버각이 날 사이로 이어짐)."""
    steps = []
    tilt = 45.0
    for fr in frames_list:
        r = rollout(sess, fr, start_tilt=tilt)
        tilt = r["steps"][-1]["rl"]
        steps.extend(r["steps"])
    return dict(fixed_angle=FIXED_ANNUAL, steps=steps)


def pick_singles(days):
    """사계절 × (가장 맑은 날 + 가장 흐린 날, 단 고도>30 유지) = 8일."""
    by = {}
    for d in days:
        fr = d["f"]
        if not any(f["elev"] > 30 for f in fr):
            continue
        mon = int(d["date"][5:7])
        season = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
                  6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}[mon]
        cloud = float(np.mean([f["cloud"] for f in fr]))
        by.setdefault(season, []).append((cloud, d))
    chosen = []
    for season in ["봄", "여름", "가을", "겨울"]:
        cand = sorted(by[season], key=lambda x: x[0])
        chosen.append((season + " 맑음", cand[0][1], cand[0][0]))
        chosen.append((season + " 흐림", cand[-1][1], cand[-1][0]))
    return chosen


def pick_week(days, start="2014-06-15", n=7):
    """start 부터 연속 n일(달력 연속)을 사전에서 찾아 반환."""
    import datetime as dt
    idx = {d["date"]: d for d in days}
    d0 = dt.date.fromisoformat(start)
    run = []
    for k in range(n):
        ds = (d0 + dt.timedelta(days=k)).isoformat()
        if ds in idx:
            run.append(idx[ds])
    return run


def main():
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    days = json.load(open(LIB))["days"]
    out = dict(
        source="simulation",
        model="louver_track (PPO 3M steps)",
        note="RL 정책의 실제 출력. 실측 미연결(inverter-rnd 현장 연결 시 measured 트랙 추가).",
        params=dict(chord_mm=CHORD, pitch_mm=PITCH, albedo=ALB, steps=STEPS,
                    max_delta_deg=MAX_DELTA),
        days=[],
    )
    def make_entry(label, date, cloud, r, days_in_run=1):
        rl_e = sum(s["rl_p"] for s in r["steps"])
        orc_e = sum(s["orc_p"] for s in r["steps"])
        fix_e = sum(s["fix_p"] for s in r["steps"])
        e = dict(
            label=label, date=date, cloud_mean=round(cloud, 2),
            fixed_angle=r["fixed_angle"], days_in_run=days_in_run,
            track_pct=round(100 * rl_e / orc_e, 1) if orc_e > 1e-6 else 100.0,
            rl_vs_fixed_pct=round(100 * (rl_e - fix_e) / fix_e, 1) if fix_e > 1e-6 else 0.0,
            steps=r["steps"],
        )
        print(f"{label:9} {date}  cloud={cloud:4.1f}  fixed={r['fixed_angle']:4.1f}°  "
              f"추종={e['track_pct']:5.1f}%  RL>고정={e['rl_vs_fixed_pct']:+5.1f}%  ({len(r['steps'])}스텝)")
        return e

    # 8개 단일일(사계절 맑음+흐림)
    for label, d, cloud in pick_singles(days):
        out["days"].append(make_entry(label, d["date"], cloud, rollout(sess, d["f"])))
    # 연속 7일 런(루버각 이어짐)
    week = pick_week(days)
    if len(week) >= 2:
        cloud_w = float(np.mean([f["cloud"] for d in week for f in d["f"]]))
        rw = rollout_multi(sess, [d["f"] for d in week])
        out["days"].append(make_entry(f"연속 {len(week)}일", week[0]["date"], cloud_w, rw, days_in_run=len(week)))

    # 실측 슬롯(Phase 2 배선 데모): mock 인버터 측정치를 measured 트랙으로 오버레이
    import os
    mp_path = "static/mock_measured.json"
    if os.path.exists(mp_path):
        meas = json.load(open(mp_path))
        samp = meas["samples"]
        idx = {d["date"]: d for d in days}
        gday = idx.get("2014-05-12") or next(d for d in days if d["date"][5:7] == "05"
                                             and any(f["elev"] > 30 for f in d["f"]))
        cloud_m = float(np.mean([f["cloud"] for f in gday["f"]]))
        rm = rollout(sess, gday["f"])
        for st in rm["steps"]:                     # 각 스텝에 측정치(±0.25h 평균) 부착, 없으면 None
            vals = [s["w"] for s in samp if abs(s["h"] - st["h"]) <= 0.25]
            st["meas_p"] = round(sum(vals) / len(vals), 1) if vals else None
        e = make_entry("실측 슬롯 (mock)", "기하 2014-05-12 · 실측 2026-05 mock", cloud_m, rm)
        e["has_measured"] = True
        e["measured_source"] = meas["source"]
        out["days"].append(e)
        out["measured_note"] = ("measured = inverter-rnd mock 벤치 1대(검증 아님). "
                                "시뮬 기하와 다른 연도라 절대 보정 아님 — 같은 스키마로 흘러드는 배선 데모.")
    json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False, separators=(",", ":"))
    import os
    print(f"\n→ {OUT_JSON}  ({os.path.getsize(OUT_JSON)/1024:.1f} KB, {len(out['days'])}일)")

    # main.html 의 트윈 데이터 스크립트 태그에 인라인 주입(서버렌더 환경 fetch 회피)
    blob = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    import re
    html = open("main.html", encoding="utf-8").read()
    pat = re.compile(r'(<script type="application/json" id="bipv-twin-data">).*?(</script>)', re.S)
    if pat.search(html):
        html = pat.sub(lambda m: m.group(1) + blob + m.group(2), html, count=1)
        open("main.html", "w", encoding="utf-8").write(html)
        print("→ main.html #bipv-twin-data 주입 완료")
    else:
        print("⚠ main.html 에 #bipv-twin-data 태그 없음 — 주입 생략")


if __name__ == "__main__":
    main()
