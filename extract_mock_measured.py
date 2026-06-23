# ==============================================================================
# 실측 슬롯 배선(Phase 2 준비) — inverter-rnd mock 인버터 데이터에서 측정 발전
# 시계열을 추출해 static/mock_measured.json 으로 저장.
#   ★ mock 벤치 1대(Hoymiles HMS-500-1T, PANEL_1) · 2026-05-11~12 · 검증 아님.
#     디지털 트윈의 measured 트랙이 '같은 스키마로 흘러들어오는지'를 보이는 배선 데모.
#   ts 는 UTC → KST(+9) 변환 후 하루 시각(시)으로 환산. 벤치가 커버한 구간만 채워짐.
#
#   실행: /Volumes/AISSD/inverter-rnd/.venv/bin/python extract_mock_measured.py
#   (pyarrow/pandas 가 있는 inverter-rnd venv 필요. mlagents env엔 없음.)
# ==============================================================================
import json
import pyarrow.parquet as pq
import pandas as pd

FILES = [
    "/Volumes/AISSD/inverter-rnd/dataset/raw/yyyy=2026/mm=05/dd=11/opendtu.parquet",
    "/Volumes/AISSD/inverter-rnd/dataset/raw/yyyy=2026/mm=05/dd=12/opendtu.parquet",
]
OUT = "static/mock_measured.json"

df = pd.concat([pq.read_table(f).to_pandas() for f in FILES], ignore_index=True)
df["ts"] = pd.to_datetime(df["ts"])
kst = df["ts"] + pd.Timedelta(hours=9)              # UTC → KST
h = kst.dt.hour + kst.dt.minute / 60.0
df = df.assign(h=h)
df = df[(df["h"] >= 6.0) & (df["h"] <= 19.5)]        # 주간 구간만
df = df.dropna(subset=["ac_p"])

# 0.25시 버킷 평균(작게·매끄럽게)
df["hb"] = (df["h"] * 4).round() / 4
g = df.groupby("hb")["ac_p"].mean().reset_index()
samples = [{"h": round(float(r.hb), 2), "w": round(float(r.ac_p), 1)} for r in g.itertuples()]

out = {
    "source": "inverter-rnd mock (Hoymiles HMS-500-1T, PANEL_1, 2026-05-11~12 KST)",
    "note": "벤치 1대 mock 데이터. 검증 아님 — measured 트랙 배선 데모용. 커버 구간만.",
    "unit": "W (AC 출력)",
    "samples": samples,
}
json.dump(out, open(OUT, "w"), ensure_ascii=False, separators=(",", ":"))
print(f"→ {OUT}  ({len(samples)} 버킷)  h범위 {samples[0]['h']}~{samples[-1]['h']}시  "
      f"w범위 {min(s['w'] for s in samples):.0f}~{max(s['w'] for s in samples):.0f}W")
