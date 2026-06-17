// AI Tilt — 샌드박스 탐색 패널 (5단계)
// 검증된 LouverPhysics(physics_v3 C# 포팅)로 "우리 현장이면?"을 실시간 재계산.
//   슬라이더: 알베도 / 현(chord) / 피치(pitch) / 비교 고정각
//   결과: oracle 최적각(정오) · 하루 최고고정각 · AI 이득(vs 비교각, vs 최고고정)
// 태양궤적은 scene_data.json의 베이크값을 재사용(날짜 바꾸려면 bake_scene.py 재실행).
// 3D 애니는 SolarLouverViz가 베이크된 검증일을 보여주고, 이 패널은 숫자 탐색 전용.
using System.IO;
using UnityEngine;

public class SolarLouverSandbox : MonoBehaviour {
    public string jsonFileName = "scene_data.json";
    [Range(0.05f, 0.4f)] public float albedo = 0.15f;
    [Range(50f, 130f)]   public float chord  = 97.5f;
    [Range(60f, 130f)]   public float pitch  = 97.5f;
    [Range(0f, 90f)]     public float baseline = 90f;

    SceneData data; string selftest = "";
    float lA = -1, lC, lP, lB; // 마지막 계산 파라미터(변할 때만 재계산)
    float oracleNoon, bestFixed, dayGainVsBase, dayGainVsBest;

    void Start() {
        string sa = Application.streamingAssetsPath;
        LouverPhysics.Load(sa);
        selftest = LouverPhysics.SelfTest(sa);
        Debug.Log(selftest);
        string path = Path.Combine(sa, jsonFileName);
        if (File.Exists(path)) data = JsonUtility.FromJson<SceneData>(File.ReadAllText(path));
        Recompute();
    }

    void Update() {
        if (data == null) return;
        if (albedo != lA || chord != lC || pitch != lP || baseline != lB) Recompute();
    }

    void Recompute() {
        if (data == null || data.timeline.Length == 0) return;
        lA = albedo; lC = chord; lP = pitch; lB = baseline;

        // 하루 적분: oracle, 비교각, 그리고 모든 고정각(최고고정 탐색)
        float[] fixedSum = new float[91];
        float sumO = 0f, sumB = 0f; float noonE = -1f;
        foreach (var f in data.timeline) {
            float ot = LouverPhysics.OracleTilt(f.sun_elev, f.sun_az, f.dni, f.dhi, albedo, chord, pitch);
            sumO += LouverPhysics.EffPoa(ot, f.sun_elev, f.sun_az, f.dni, f.dhi, albedo, chord, pitch);
            sumB += LouverPhysics.EffPoa(baseline, f.sun_elev, f.sun_az, f.dni, f.dhi, albedo, chord, pitch);
            for (int a = 0; a <= 90; a++)
                fixedSum[a] += LouverPhysics.EffPoa(a, f.sun_elev, f.sun_az, f.dni, f.dhi, albedo, chord, pitch);
            if (f.sun_elev > noonE) { noonE = f.sun_elev; oracleNoon = ot; } // 최고고도=정오 근사
        }
        int bf = 0; for (int a = 1; a <= 90; a++) if (fixedSum[a] > fixedSum[bf]) bf = a;
        bestFixed = bf;
        dayGainVsBase = sumB > 0 ? (sumO - sumB) / sumB * 100f : 0f;
        dayGainVsBest = fixedSum[bf] > 0 ? (sumO - fixedSum[bf]) / fixedSum[bf] * 100f : 0f;
    }

    void OnGUI() {
        var st = new GUIStyle(GUI.skin.label) { fontSize = 13 };
        float W = 340;
        GUILayout.BeginArea(new Rect(Screen.width - W - 18, 18, W, 372), GUI.skin.box);
        GUILayout.Label("우리 현장이면? — 슬라이더로 탐색", st);
        GUILayout.Label($"알베도(바닥 반사) {albedo:0.00}", st);   albedo   = GUILayout.HorizontalSlider(albedo, 0.05f, 0.4f);
        GUILayout.Label($"날개 폭 chord {chord:0.0}mm", st);  chord    = GUILayout.HorizontalSlider(chord, 50f, 130f);
        GUILayout.Label($"날개 간격 pitch {pitch:0.0}mm", st); pitch   = GUILayout.HorizontalSlider(pitch, 60f, 130f);
        GUILayout.Label($"비교 고정각 {baseline:0}°", st); baseline = GUILayout.HorizontalSlider(baseline, 0f, 90f);
        GUILayout.Space(8);
        GUILayout.Label($"폭/간격 비율 = {chord / pitch:0.00}", st);
        GUILayout.Label($"AI 최적각(정오) = {oracleNoon:0}°", st);
        GUILayout.Label($"하루 최고 고정각 = {bestFixed:0}°", st);
        GUILayout.Label($"AI 이득  vs 고정{baseline:0}° = +{dayGainVsBase:0.0}%", st);
        GUILayout.Label($"AI 이득  vs 최고 고정각 = +{dayGainVsBest:0.0}%", st);
        GUILayout.Space(6);
        GUILayout.Label(selftest, new GUIStyle(GUI.skin.label) { fontSize = 11 });
        GUILayout.EndArea();
    }
}
