# 기상청 10년 master_data CSV → 날짜별 하루 데이터셋(sun_days.json)
# 각 하루 = 시간별 [태양고도, 방위, DNI, DHI] 프레임. RL 에피소드마다 실제 하루를 샘플.
import csv, json, sys
from collections import OrderedDict

SRC = "/Volumes/AISSD/ai-tilt/bipv_ai_master_data_v17.csv"
OUT = "/Volumes/AISSD/ai-tilt/unity_viz/AITiltViz/AITiltViz/Assets/StreamingAssets/sun_days.json"

byday = OrderedDict()
with open(SRC, newline="") as f:
    for row in csv.DictReader(f):
        d = row["timestamp"][:10]
        try:
            fr = {
                "h":    int(row["timestamp"][11:13]),   # 실제 시각(시)
                "elev": round(float(row["solar_elevation"]), 2),
                "az":   round(float(row["solar_azimuth"]), 2),
                "dni":  round(float(row["dni"]), 1),
                "dhi":  round(float(row["dhi"]), 1),
                "cloud": round(float(row["cloud_cover"]), 2),  # 구름량(흐림/비 트리거용)
            }
        except ValueError:
            continue
        byday.setdefault(d, []).append(fr)

days = [{"date": d, "f": fr} for d, fr in byday.items() if len(fr) >= 4]
json.dump({"days": days}, open(OUT, "w"), ensure_ascii=False, separators=(",", ":"))

import os
print(f"날짜수: {len(days)}  (원본 {len(byday)})  파일: {OUT}")
print(f"크기: {os.path.getsize(OUT)/1e6:.2f} MB")
print("프레임수 예:", days[0]["date"], len(days[0]["f"]), "/", days[180]["date"], len(days[180]["f"]))
