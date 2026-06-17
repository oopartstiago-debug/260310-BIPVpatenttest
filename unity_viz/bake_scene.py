#!/usr/bin/env python3
"""
AI Tilt — Unity 베이크 브리지 (1단계)
검증된 physics_v3(oracle) → scene_data.json. Unity는 이 파일을 읽어 렌더만.
  타임라인: 태양궤적 / oracle 최적각 / 고정 baseline / 음영률 / POA(유효일사) / 누적에너지
모든 수치는 physics_v3.eff_poa 단일소스. by-construction 금지(각도는 그리드 argmax로 실측).
"""
import sys, os, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import physics_v3 as P

GRID = np.arange(0, 90.0001, 1.0)          # 각도 후보 (전수 argmax)
N_BLADES = 20                              # 도면 1043A: 20날개
FRAME_DEPTH = 100.0                        # mm

def oracle_and_poa(elev, az, dni, dhi):
    """각 시점의 oracle 최적각 + 각도별 POA(고정 baseline 비교용으로 전 그리드 반환)."""
    poa = np.array([float(P.eff_poa(np.array([t]), np.array([elev]), np.array([az]),
                                    np.array([dni]), np.array([dhi]))[0]) for t in GRID])
    k = int(np.argmax(poa))
    return GRID[k], poa  # (oracle_tilt, poa_over_grid)

def bake(date, baseline_fixed, out):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "bipv_ai_master_data_v17.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    if date is None:
        # 하지(6/21)에 가장 가까운, 낮길이 긴 날
        target = pd.Timestamp(df["timestamp"].dt.year.min(), 6, 21).date()
        date = min(df["date"].unique(), key=lambda d: abs((d - target).days))
    else:
        date = pd.to_datetime(date).date()
    day = df[(df["date"] == date) & (df["solar_elevation"] > 0)].sort_values("timestamp")
    if len(day) == 0:
        raise SystemExit(f"일조 데이터 없음: {date}")

    frames, cum_o, cum_b = [], 0.0, 0.0
    bi = int(round(baseline_fixed))
    for _, r in day.iterrows():
        elev, az, dni, dhi = float(r.solar_elevation), float(r.solar_azimuth), float(r.dni), float(r.dhi)
        otilt, poa = oracle_and_poa(elev, az, dni, dhi)
        poa_o = float(poa[int(round(otilt))]); poa_b = float(poa[bi])
        sf_o = float(P.panel_sf(np.array([otilt]), np.array([elev]), np.array([az]))[0])
        sf_b = float(P.panel_sf(np.array([float(bi)]), np.array([elev]), np.array([az]))[0])
        cum_o += poa_o; cum_b += poa_b
        frames.append({
            "time": r.timestamp.strftime("%H:%M"), "hour": round(r.timestamp.hour + r.timestamp.minute/60, 3),
            "sun_elev": round(elev, 2), "sun_az": round(az, 2),
            "dni": round(dni, 1), "dhi": round(dhi, 1),
            "oracle_tilt": round(float(otilt), 1), "baseline_tilt": bi,
            "poa_oracle": round(poa_o, 1), "poa_baseline": round(poa_b, 1),
            "shade_oracle": round(sf_o, 3), "shade_baseline": round(sf_b, 3),
            "cum_oracle": round(cum_o, 1), "cum_baseline": round(cum_b, 1),
        })
    gain = (cum_o - cum_b) / cum_b * 100 if cum_b else 0.0
    scene = {
        "meta": {
            "physics": "physics_v3 (검증완료, env_check 교차합의)", "location": "Seoul",
            "date": str(date), "baseline_fixed_deg": bi,
            "day_gain_oracle_vs_baseline_pct": round(gain, 2),
            "note": "2D 무한슬랫 모델 — 한 시점 전 블레이드 동일각·동일음영(끝단 효과 미모델). 시각: 태양이동+루버회전+그림자+발전곡선.",
        },
        "geometry": {
            "chord_mm": P.CHORD, "pitch_mm": P.PITCH, "n_blades": N_BLADES,
            "frame_depth_mm": FRAME_DEPTH, "strip_frac": round(P.STRIP_FRAC, 4),
            "strip_lo": round(P.STRIP_LO, 4), "panel_width_mm": 1043, "panel_height_mm": 2135,
        },
        "timeline": frames,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)
    print(f"baked {out}  date={date}  frames={len(frames)}  oracle {round(otilt,0)}deg range "
          f"{frames[0]['oracle_tilt']}~{max(fr['oracle_tilt'] for fr in frames)}  "
          f"day gain vs {bi}deg = +{gain:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (기본=하지 근접일)")
    ap.add_argument("--baseline", type=float, default=60.0, help="비교 고정각 (기본 60)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "scene_data.json"))
    a = ap.parse_args()
    bake(a.date, a.baseline, a.out)
