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
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

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
    bool shotPending = false;                // 에디터 캡처 요청 플래그
    float t01 = 0f;                          // 0..1 (일출~일몰)
    SceneFrame fa, fb; float frac, curElev, curAz, curTilt;
    Material bladeMat, groundMat;            // 런타임 생성 PBR
    Texture2D px;                            // 1x1 흰색(막대 그리기용)

    void Start() {
        string path = Path.Combine(Application.streamingAssetsPath, jsonFileName);
        if (!File.Exists(path)) { Debug.LogError("scene_data.json 없음: " + path); enabled = false; return; }
        data = JsonUtility.FromJson<SceneData>(File.ReadAllText(path));
        SetupVisuals();
        BuildBlades();
    }

    // 2~3단계 비주얼: 그림자 + PBR 머티리얼 (전부 런타임 → 코웍 루프 = 재컴파일+Play)
    void SetupVisuals() {
        px = new Texture2D(1, 1); px.SetPixel(0, 0, Color.white); px.Apply();

        var lit = Shader.Find("Universal Render Pipeline/Lit");
        // 블레이드 = 아노다이즈드 알루미늄. 메탈릭 낮춰 직사광에서 밝게 읽히게(과한 금속=환경반사 의존→검게 죽음)
        bladeMat = new Material(lit);
        bladeMat.SetColor("_BaseColor", new Color(0.70f, 0.72f, 0.76f));
        bladeMat.SetFloat("_Metallic", 0.30f);
        bladeMat.SetFloat("_Smoothness", 0.45f);
        // 바닥 = 밝은 무광 회색 → 그림자가 어두운 띠로 또렷이 대비
        groundMat = new Material(lit);
        groundMat.SetColor("_BaseColor", new Color(0.34f, 0.35f, 0.37f));
        groundMat.SetFloat("_Metallic", 0f);
        groundMat.SetFloat("_Smoothness", 0.08f);
        var g = GameObject.Find("Ground");
        if (g) { var r = g.GetComponent<Renderer>(); if (r) r.sharedMaterial = groundMat; }

        // 조명: 따뜻한 직사광 + 낮은 평면 주변광 → 자기그림자/바닥그림자가 또렷
        if (sun) {
            sun.shadows = LightShadows.Soft; sun.shadowStrength = 0.92f;
            sun.intensity = 1.35f; sun.color = new Color(1f, 0.96f, 0.90f);
        }
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.40f, 0.43f, 0.50f);   // 하늘빛 약한 채움광(너무 밝으면 그림자 사라짐)
        var cam = Camera.main;
        if (cam) {
            cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = new Color(0.55f, 0.70f, 0.86f);  // 맑은 하늘
            // 3/4 측면·근접 앵글 + 좁은 FOV → 슬랫 엣지가 줄줄이 분리돼 보이고 루버가 화면을 채움
            Vector3 center = louverRoot ? louverRoot.position : new Vector3(0, 1, 0);
            cam.transform.position = center + new Vector3(1.8f, 0.15f, -2.6f);
            cam.transform.LookAt(center);
            cam.fieldOfView = 40f;
        }
        var urp = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
        if (urp != null) urp.shadowDistance = 15f;
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
            b.transform.localScale = new Vector3(width, 0.004f, chord);   // X=길이(동서) Y=두께 Z=현 (얇게=날개 분리감)
            b.transform.localPosition = new Vector3(0, y0 + i * pitch, 0);
            var rend = b.GetComponent<Renderer>();
            if (rend && bladeMat) rend.sharedMaterial = bladeMat;          // PBR 알루미늄 (그림자 cast/receive는 기본 on)
            blades.Add(b.transform);
        }
    }

    void Update() {
        if (data == null || data.timeline.Length == 0) return;
#if UNITY_EDITOR
        // Unity가 직접 PNG 기록 → macOS 화면녹화 권한 불필요. 절대경로라 외부에서 바로 읽힘.
        if (shotPending) {
            shotPending = false;
            const string p = "/Volumes/AISSD/ai-tilt/unity_viz/unity_capture.png";
            ScreenCapture.CaptureScreenshot(p);
            Debug.Log("[AI Tilt] 캡처 저장(다음 프레임): " + p);
        }
#endif
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
        GUILayout.BeginArea(new Rect(18, 18, 410, 280), GUI.skin.box);
        GUILayout.Label($"AI Tilt — {data.meta.location}  {data.meta.date}", st);
        GUILayout.Label($"시각 {fa.time}    태양 고도 {curElev:0.0}°  방위 {curAz:0.0}°", st);
        GUILayout.Label($"날개 각도 {curTilt:0.0}°   ({(showOracle ? "AI 최적" : "고정 " + data.meta.baseline_fixed_deg + "°")})", st);
        GUILayout.Label($"유효일사 POA   AI {poaO:0.0}   vs 고정 {poaB:0.0}", st);
        GUILayout.Label($"하루 이득  AI vs {data.meta.baseline_fixed_deg}° = +{data.meta.day_gain_oracle_vs_baseline_pct:0.0}%", st);
        if (GUILayout.Button(showOracle ? "→ 고정 baseline 보기" : "→ AI 최적각 보기")) showOracle = !showOracle;
        play = GUILayout.Toggle(play, "재생");
        t01 = GUILayout.HorizontalSlider(t01, 0, 1);
#if UNITY_EDITOR
        if (GUILayout.Button("📸 스크린샷 저장 (unity_capture.png)")) shotPending = true;
#endif
        GUILayout.EndArea();

        DrawPowerCurve();
    }

    // 하루 발전 프로파일 막대: 회색=고정 baseline, 녹색=AI 초과분, 노란선=현재 시각
    void DrawPowerCurve() {
        if (data == null || data.timeline.Length < 2 || px == null) return;
        int n = data.timeline.Length;
        float W = 520, H = 140, x0 = 18, y0 = Screen.height - H - 18;
        GUI.Box(new Rect(x0, y0, W, H), GUIContent.none);

        float maxP = 1f;
        for (int i = 0; i < n; i++) maxP = Mathf.Max(maxP, data.timeline[i].poa_oracle);
        float padL = 12, padB = 10, padT = 26;
        float gw = W - padL - 12, gh = H - padB - padT;
        float bw = gw / n, baseY = y0 + padT + gh;
        for (int i = 0; i < n; i++) {
            var fr = data.timeline[i];
            float bx = x0 + padL + i * bw;
            float ho = gh * (fr.poa_oracle / maxP);     // AI(오라클) 전체 높이
            float hb = gh * (fr.poa_baseline / maxP);   // 고정 baseline 높이 (≤ ho, 오라클이 매 순간 최대)
            GUI.color = new Color(0.50f, 0.52f, 0.55f, 0.92f);
            GUI.DrawTexture(new Rect(bx, baseY - hb, bw * 0.8f, hb), px);
            GUI.color = new Color(0.30f, 0.80f, 0.45f, 0.92f);
            GUI.DrawTexture(new Rect(bx, baseY - ho, bw * 0.8f, Mathf.Max(0, ho - hb)), px);
        }
        float cx = x0 + padL + Mathf.Clamp01(t01) * gw;
        GUI.color = new Color(1f, 0.85f, 0.2f, 0.95f);
        GUI.DrawTexture(new Rect(cx, y0 + padT, 2, gh), px);
        GUI.color = Color.white;
        GUI.Label(new Rect(x0 + padL, y0 + 5, W, 18),
                  $"하루 발전 프로파일 (POA)   ■ 고정 {data.meta.baseline_fixed_deg}°   ■ AI 초과분",
                  new GUIStyle(GUI.skin.label) { fontSize = 12 });
    }
}
