// AI Tilt — Unity 뷰어 드라이버 (1~3단계)
// 검증된 physics_v3가 구운 scene_data.json(StreamingAssets)을 읽어:
//   · 태양(Directional Light)을 실제 궤적으로 이동
//   · 루버 블레이드 N장을 oracle 최적각(또는 고정 baseline)으로 회전
//   · 좌상단 HUD에 POA·하루이득% 표시 + 재생/슬라이더/모드토글
// Unity = 뷰어, oracle = 진실. 이 스크립트는 물리를 계산하지 않음(렌더만).
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable] public class SceneMeta {
    public string physics, location, date, note;
    public int baseline_fixed_deg;
    public float day_gain_oracle_vs_baseline_pct;
}
[Serializable] public class SceneGeometry {
    public float chord_mm, pitch_mm, frame_depth_mm, strip_frac, strip_lo, panel_width_mm, panel_height_mm;
    public int n_blades;
}
[Serializable] public class SceneFrame {
    public string time;
    public float hour, sun_elev, sun_az, dni, dhi, oracle_tilt, poa_oracle, poa_baseline,
                 shade_oracle, shade_baseline, cum_oracle, cum_baseline;
    public int baseline_tilt;
}
[Serializable] public class SceneData {
    public SceneMeta meta;
    public SceneGeometry geometry;
    public SceneFrame[] timeline;
}

public class SolarLouverViz : MonoBehaviour {
    [Header("인스펙터에서 연결")]
    public Light sun;                       // Directional Light
    public Transform louverRoot;            // 원점의 빈 오브젝트 — 블레이드가 이 밑에 생성됨
    public string jsonFileName = "scene_data.json";  // StreamingAssets 안

    [Header("재생")]
    public bool play = true;
    public float secondsPerDay = 20f;       // 하루 전체를 N초에 재생
    public bool showOracle = true;          // true=AI 최적각, false=고정 baseline

    SceneData data;
    readonly List<Transform> blades = new List<Transform>();
    float t01 = 0f;                          // 0..1 (일출~일몰)
    SceneFrame fa, fb; float frac, curElev, curAz, curTilt;

    void Start() {
        string path = Path.Combine(Application.streamingAssetsPath, jsonFileName);
        if (!File.Exists(path)) { Debug.LogError("scene_data.json 없음: " + path); enabled = false; return; }
        data = JsonUtility.FromJson<SceneData>(File.ReadAllText(path));
        BuildBlades();
    }

    void BuildBlades() {
        const float m = 0.001f;             // mm -> m
        float chord = data.geometry.chord_mm * m;
        float pitch = data.geometry.pitch_mm * m;
        float width = data.geometry.panel_width_mm * m;
        int n = Mathf.Max(1, data.geometry.n_blades);
        float y0 = -(n - 1) * pitch * 0.5f;
        for (int i = 0; i < n; i++) {
            var b = GameObject.CreatePrimitive(PrimitiveType.Cube);
            b.name = "Blade_" + i;
            b.transform.SetParent(louverRoot, false);
            b.transform.localScale = new Vector3(width, 0.008f, chord);   // X=길이(동서) Y=두께 Z=현
            b.transform.localPosition = new Vector3(0, y0 + i * pitch, 0);
            blades.Add(b.transform);
        }
    }

    void Update() {
        if (data == null || data.timeline.Length == 0) return;
        if (play) { t01 += Time.deltaTime / Mathf.Max(0.1f, secondsPerDay); if (t01 > 1f) t01 -= 1f; }

        int n = data.timeline.Length;
        float f = Mathf.Clamp01(t01) * (n - 1);
        int i0 = Mathf.Clamp((int)f, 0, n - 1), i1 = Mathf.Min(i0 + 1, n - 1);
        frac = f - i0; fa = data.timeline[i0]; fb = data.timeline[i1];

        curElev = Mathf.Lerp(fa.sun_elev, fb.sun_elev, frac);
        curAz   = Mathf.Lerp(fa.sun_az,   fb.sun_az,   frac);
        curTilt = showOracle ? Mathf.Lerp(fa.oracle_tilt, fb.oracle_tilt, frac) : fa.baseline_tilt;

        // 태양 방향 (az=북기준 시계방향, elev=지평선기준). 거울상이면 ar 부호만 뒤집으면 됨.
        float er = curElev * Mathf.Deg2Rad, ar = curAz * Mathf.Deg2Rad;
        Vector3 dirToSun = new Vector3(Mathf.Sin(ar) * Mathf.Cos(er), Mathf.Sin(er), Mathf.Cos(ar) * Mathf.Cos(er));
        if (sun) sun.transform.forward = (-dirToSun).normalized;

        foreach (var b in blades) b.localRotation = Quaternion.Euler(curTilt, 0, 0);  // 동서축(X) 기준 틸트
    }

    void OnGUI() {
        if (data == null || fa == null) return;
        float poaO = Mathf.Lerp(fa.poa_oracle, fb.poa_oracle, frac);
        float poaB = Mathf.Lerp(fa.poa_baseline, fb.poa_baseline, frac);
        var st = new GUIStyle(GUI.skin.label) { fontSize = 15 };
        GUILayout.BeginArea(new Rect(18, 18, 440, 250), GUI.skin.box);
        GUILayout.Label($"AI Tilt — {data.meta.location}  {data.meta.date}", st);
        GUILayout.Label($"시각 {fa.time}    태양 고도 {curElev:0.0}°  방위 {curAz:0.0}°", st);
        GUILayout.Label($"블레이드 {curTilt:0.0}°   ({(showOracle ? "AI 최적" : "고정 " + data.meta.baseline_fixed_deg + "°")})", st);
        GUILayout.Label($"유효일사 POA   AI {poaO:0.0}   vs 고정 {poaB:0.0}", st);
        GUILayout.Label($"하루 이득  AI vs {data.meta.baseline_fixed_deg}° = +{data.meta.day_gain_oracle_vs_baseline_pct:0.0}%", st);
        if (GUILayout.Button(showOracle ? "→ 고정 baseline 보기" : "→ AI 최적각 보기")) showOracle = !showOracle;
        play = GUILayout.Toggle(play, "재생");
        t01 = GUILayout.HorizontalSlider(t01, 0, 1);
        GUILayout.EndArea();
    }
}
