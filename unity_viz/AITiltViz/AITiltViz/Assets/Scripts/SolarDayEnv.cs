// AI Tilt — RL 환경의 '하루' 모델.
//   ★ sun_days.json(기상청 10년, 서울 2014~2023, 3652일)을 로드해
//     에피소드마다 '실제 하루'를 무작위로 뽑는다 → 날짜·계절·실제 날씨(DNI/DHI)가 진짜로 달라짐.
//   sun_days.json이 없으면 scene_data.json(검증된 단일 하루)로 폴백.
//   설계 파라미터(피치·알베도)는 별도로 도메인 랜덤화 → 비정상 환경 유지.
//   SceneData/SceneFrame 타입은 SolarLouverViz.cs 정의를 재사용(폴백용).
using System.IO;
using UnityEngine;

public class SolarDayEnv {
    public SceneData data;                  // 폴백: 단일 검증 하루
    public string currentDate = "";         // 현재 에피소드의 실제 날짜
    public float dayPeakDni;                // 그날 최대 DNI(날씨 라벨용)
    DayRec[] days;                          // 10년 날짜별 데이터셋
    DayFrame[] cur;                         // 현재 하루의 시간별 프레임
    int dayCursor = -1;                     // 순차 진행 커서

    public bool Loaded => (days != null && days.Length > 0) ||
                          (data != null && data.timeline != null && data.timeline.Length > 1);

    // 에피소드별 상태(관측/보상에 노출)
    public float cloud = 1f, diffuseBoost = 1f;     // 실측 DNI/DHI 사용 시 1 고정
    public float chord = LouverPhysics.CHORD;
    public float pitch = LouverPhysics.PITCH;
    public float albedo = LouverPhysics.ALBEDO;

    [System.Serializable] public class DayFrame { public float elev, az, dni, dhi; }
    [System.Serializable] public class DayRec   { public string date; public DayFrame[] f; }
    [System.Serializable] public class DayLibrary { public DayRec[] days; }

    public bool Load(string dir, string fileName = "scene_data.json") {
        // 1) 10년 날짜별 데이터셋 우선
        string libPath = Path.Combine(dir, "sun_days.json");
        if (File.Exists(libPath)) {
            var lib = JsonUtility.FromJson<DayLibrary>(File.ReadAllText(libPath));
            days = (lib != null) ? lib.days : null;
        }
        // 2) 폴백 단일 하루
        string path = Path.Combine(dir, fileName);
        if (File.Exists(path)) data = JsonUtility.FromJson<SceneData>(File.ReadAllText(path));

        if (days != null && days.Length > 0) { Debug.Log($"[SolarDayEnv] 10년 데이터셋 로드: {days.Length}일 (서울)"); PickDay(0); }
        else Debug.Log("[SolarDayEnv] 단일 하루(scene_data) 폴백");
        return Loaded;
    }

    void PickDay(int i) {
        if (days == null || days.Length == 0) return;
        var d = days[Mathf.Clamp(i, 0, days.Length - 1)];
        cur = d.f; currentDate = d.date;
        float pk = 0f; foreach (var fr in cur) if (fr.dni > pk) pk = fr.dni; dayPeakDni = pk;
    }

    // 에피소드 시작: 실제 하루를 '순차로' 진행 + 설계 파라미터 랜덤화
    public void Randomize(bool domainRandomize) {
        if (days != null && days.Length > 0) { dayCursor = (dayCursor + 1) % days.Length; PickDay(dayCursor); }

        if (!domainRandomize) {
            chord = LouverPhysics.CHORD; pitch = LouverPhysics.PITCH; albedo = LouverPhysics.ALBEDO;
            cloud = 1f; diffuseBoost = 1f; return;
        }
        chord = LouverPhysics.CHORD;                 // 현은 고정(하드웨어)
        pitch = Random.Range(80f, 140f);             // 피치 → 현/피치비 변화
        albedo = Random.Range(0.08f, 0.30f);         // 바닥 반사율
        cloud = 1f; diffuseBoost = 1f;               // 날씨는 실측 데이터가 제공
    }

    public struct SunState { public float elev, az, dni, dhi; }

    // t01: 0=그날 첫 시각, 1=마지막 시각
    public SunState Sample(float t01) {
        var s = new SunState();
        if (cur != null && cur.Length > 1) {
            int n = cur.Length;
            float f = Mathf.Clamp01(t01) * (n - 1);
            int i0 = Mathf.Clamp((int)f, 0, n - 1), i1 = Mathf.Min(i0 + 1, n - 1);
            float fr = f - i0;
            var a = cur[i0]; var b = cur[i1];
            s.elev = Mathf.Lerp(a.elev, b.elev, fr); s.az = Mathf.Lerp(a.az, b.az, fr);
            s.dni  = Mathf.Lerp(a.dni,  b.dni,  fr) * cloud;
            s.dhi  = Mathf.Lerp(a.dhi,  b.dhi,  fr) * diffuseBoost;
            return s;
        }
        // 폴백
        int m = data.timeline.Length;
        float ff = Mathf.Clamp01(t01) * (m - 1);
        int j0 = Mathf.Clamp((int)ff, 0, m - 1), j1 = Mathf.Min(j0 + 1, m - 1);
        float ffr = ff - j0;
        var x = data.timeline[j0]; var y = data.timeline[j1];
        s.elev = Mathf.Lerp(x.sun_elev, y.sun_elev, ffr); s.az = Mathf.Lerp(x.sun_az, y.sun_az, ffr);
        s.dni  = Mathf.Lerp(x.dni, y.dni, ffr) * cloud;
        s.dhi  = Mathf.Lerp(x.dhi, y.dhi, ffr) * diffuseBoost;
        return s;
    }
}
