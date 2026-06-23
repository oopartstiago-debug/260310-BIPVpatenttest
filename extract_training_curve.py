# ==============================================================================
# 훈련 모드용 — louver_track TensorBoard 이벤트에서 누적보상 학습곡선 추출.
#   출력: static/training_curve.json (다운샘플 ~40점)
#   실행: /opt/homebrew/Caskroom/miniconda/base/envs/mlagents/bin/python extract_training_curve.py
# ==============================================================================
import glob
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "louver_track": "unity_viz/results/louver_track/LouverTilt",
    "louver_cloud": "unity_viz/results/louver_cloud/LouverTilt",
}
TAG = "Environment/Cumulative Reward"
OUT = "static/training_curve.json"


def curve(run_dir, n=40):
    ev = sorted(glob.glob(f"{run_dir}/events.out.tfevents.*"))
    if not ev:
        return []
    ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    if TAG not in ea.Tags().get("scalars", []):
        return []
    pts = [(s.step, s.value) for s in ea.Scalars(TAG)]
    if len(pts) <= n:
        return [{"s": int(st), "r": round(v, 2)} for st, v in pts]
    step = len(pts) / n
    return [{"s": int(pts[min(int(i * step), len(pts) - 1)][0]),
             "r": round(pts[min(int(i * step), len(pts) - 1)][1], 2)} for i in range(n)]


out = {k: curve(v) for k, v in RUNS.items()}
json.dump(out, open(OUT, "w"), ensure_ascii=False, separators=(",", ":"))
for k, c in out.items():
    print(f"{k}: {len(c)}점  보상 {c[0]['r'] if c else '-'} → {c[-1]['r'] if c else '-'}")
print(f"→ {OUT}")
