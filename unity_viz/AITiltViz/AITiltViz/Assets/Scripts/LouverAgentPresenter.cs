// AI Tilt — RL 사실적 그래픽 시연 프레젠터.
//   실제 대기 스카이박스+태양 · 루버 자기음영 · 리플렉션 프로브(금속/유리 반사) ·
//   건물 외벽/바닥 · 안전모/조끼 작업자 + 제어콘솔 · 포스트프로세싱 · 날씨 시각화 · 계절/날씨 HUD ·
//   5초마다 플레이어 스크린샷(/tmp/aitilt_frame.png)으로 자가 검증.
//   ★ 시각 전용. Agent 로직/보상과 분리 — 학습 결과 불변.
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Unity.MLAgents;

[RequireComponent(typeof(LouverAgent))]
public class LouverAgentPresenter : MonoBehaviour {
    public Transform louverRoot;
    public int nBlades = 20;
    public float chordMm = 97.5f, pitchMm = 97.5f, panelWidthMm = 1043f;

    LouverAgent agent;
    Material bladeMat, frameMat, panelMat, wallMat, groundMat, skyMat;
    Material vestMat, hatMat, skinMat, pantsMat, podiumMat, screenMat;
    Camera cam;
    Transform charArm;
    Vector3 center;
    float width;
    float nextShot = 3f;
    bool shotPathLogged;

    // 평가 점수 추적
    float lastNonZeroPct, bestPct;
    int prevEpisodes;
    readonly float[] recent = new float[20];
    int recentIdx, recentFilled;

    void Awake() {
        agent = GetComponent<LouverAgent>();
        Application.runInBackground = true;   // 창이 비활성이어도 계속 실행(학습 타임아웃 죽음 방지)
    }

    void Start() {
        if (!louverRoot) louverRoot = agent.louverRoot ? agent.louverRoot : transform;
        center = louverRoot ? louverRoot.position : new Vector3(0, 1, 0);
        width = panelWidthMm * 0.001f;
        MakeMaterials();
        SetupSkyAndLight();
        SetupCameraAndPost();
        BuildGroundAndWall();
        BuildBlades();
        BuildPanelBehind();
        BuildFrame();
        BuildReflectionProbe();
        BuildCharacter();
    }

    void MakeMaterials() {
        var lit = Shader.Find("Universal Render Pipeline/Lit");
        bladeMat  = M(lit, new Color(0.74f, 0.76f, 0.80f), 0.85f, 0.62f);
        frameMat  = M(lit, new Color(0.16f, 0.17f, 0.20f), 0.55f, 0.45f);
        panelMat  = M(lit, new Color(0.05f, 0.09f, 0.20f), 0.0f, 0.62f);
        wallMat   = M(lit, new Color(0.50f, 0.51f, 0.53f), 0.0f,  0.12f);   // 콘크리트 외벽
        groundMat = M(lit, new Color(0.50f, 0.50f, 0.52f), 0.0f,  0.16f);
        vestMat   = M(lit, new Color(0.95f, 0.45f, 0.05f), 0.0f,  0.30f);   // 형광 조끼
        hatMat    = M(lit, new Color(0.95f, 0.80f, 0.10f), 0.0f,  0.45f);   // 안전모
        skinMat   = M(lit, new Color(0.92f, 0.74f, 0.58f), 0.0f,  0.25f);
        pantsMat  = M(lit, new Color(0.16f, 0.20f, 0.32f), 0.0f,  0.25f);   // 작업바지
        podiumMat = M(lit, new Color(0.30f, 0.32f, 0.36f), 0.7f,  0.55f);   // 제어콘솔 금속
        screenMat = M(lit, new Color(0.10f, 0.20f, 0.28f), 0.0f,  0.6f);
        screenMat.EnableKeyword("_EMISSION"); screenMat.SetColor("_EmissionColor", new Color(0.2f, 0.9f, 1f) * 1.6f);
    }

    Material M(Shader s, Color c, float metallic, float smooth) {
        var m = new Material(s); m.SetColor("_BaseColor", c); m.SetFloat("_Metallic", metallic); m.SetFloat("_Smoothness", smooth); return m;
    }

    void SetupSkyAndLight() {
        var sky = Shader.Find("Skybox/Procedural");
        if (sky) {
            skyMat = new Material(sky);
            skyMat.SetFloat("_SunSize", 0.05f);
            skyMat.SetFloat("_SunSizeConvergence", 4f);
            skyMat.SetFloat("_AtmosphereThickness", 1.0f);
            skyMat.SetFloat("_Exposure", 1.0f);
            skyMat.SetColor("_SkyTint", new Color(0.42f, 0.58f, 0.85f));
            skyMat.SetColor("_GroundColor", new Color(0.36f, 0.37f, 0.39f));
            RenderSettings.skybox = skyMat;
        }
        if (agent.sun) {
            agent.sun.shadows = LightShadows.Soft;
            agent.sun.intensity = 1.4f;
            agent.sun.color = new Color(1f, 0.97f, 0.92f);
            agent.sun.shadowStrength = 0.9f;
            RenderSettings.sun = agent.sun;
        }
        QualitySettings.shadowDistance = 60f;   // 루버 슬랫 자기음영이 또렷이 보이게
        RenderSettings.ambientMode = AmbientMode.Trilight;
        RenderSettings.ambientSkyColor     = new Color(0.55f, 0.64f, 0.80f);
        RenderSettings.ambientEquatorColor = new Color(0.47f, 0.49f, 0.52f);
        RenderSettings.ambientGroundColor  = new Color(0.28f, 0.28f, 0.29f);
    }

    void SetupCameraAndPost() {
        cam = Camera.main; if (!cam) return;
        cam.clearFlags = CameraClearFlags.Skybox;
        cam.transform.position = center + new Vector3(2.2f, 0.7f, -4.2f);
        cam.transform.LookAt(center + new Vector3(-0.45f, 1.15f, 0.05f));
        cam.fieldOfView = 48f;
        cam.farClipPlane = Mathf.Max(cam.farClipPlane, 80f);

        var data = cam.GetUniversalAdditionalCameraData();
        if (data != null) { data.renderPostProcessing = true; data.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing; }
        var volGo = new GameObject("PostVolume");
        var vol = volGo.AddComponent<Volume>(); vol.isGlobal = true;
        var prof = ScriptableObject.CreateInstance<VolumeProfile>(); vol.profile = prof;
        var bloom = prof.Add<Bloom>(true); bloom.intensity.Override(0.8f); bloom.threshold.Override(1.1f); bloom.scatter.Override(0.6f);
        var tm = prof.Add<Tonemapping>(true); tm.mode.Override(TonemappingMode.ACES);
        var vig = prof.Add<Vignette>(true); vig.intensity.Override(0.25f); vig.smoothness.Override(0.4f);
        var co = prof.Add<ColorAdjustments>(true); co.postExposure.Override(0.1f); co.saturation.Override(8f); co.contrast.Override(10f);
    }

    void BuildReflectionProbe() {
        var go = new GameObject("ReflProbe");
        go.transform.position = center + new Vector3(0, 1.0f, -0.6f);
        var rp = go.AddComponent<ReflectionProbe>();
        rp.mode = ReflectionProbeMode.Realtime;
        rp.refreshMode = ReflectionProbeRefreshMode.EveryFrame;
        rp.timeSlicingMode = ReflectionProbeTimeSlicingMode.IndividualFaces;
        rp.size = new Vector3(16, 16, 16);
        rp.resolution = 128;
        rp.cullingMask = 0;          // 하늘만 반사(잡오브젝트 제외, 가볍고 깔끔)
        rp.clearFlags = ReflectionProbeClearFlags.Skybox;
    }

    void BuildGroundAndWall() {
        var ground = GameObject.Find("Ground");
        if (!ground) { ground = GameObject.CreatePrimitive(PrimitiveType.Plane); ground.name = "Ground"; ground.transform.localScale = new Vector3(3, 1, 3); }
        var gr = ground.GetComponent<Renderer>(); if (gr) gr.sharedMaterial = groundMat;

        float pitch = pitchMm * 0.001f; float h = (nBlades + 1) * pitch;
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = "Facade";
        wall.transform.position = center + new Vector3(0, h * 0.5f - center.y + 0.5f, 0.34f);
        wall.transform.localScale = new Vector3(width + 0.7f, center.y + h + 1.2f, 0.30f);
        var wr = wall.GetComponent<Renderer>(); if (wr) wr.sharedMaterial = wallMat;
    }

    void BuildBlades() {
        float chord = chordMm * 0.001f, pitch = pitchMm * 0.001f;
        float y0 = -(nBlades - 1) * pitch * 0.5f;
        for (int i = 0; i < nBlades; i++) {
            var b = GameObject.CreatePrimitive(PrimitiveType.Cube);
            b.name = "Blade_" + i;
            b.transform.SetParent(louverRoot, false);
            b.transform.localScale = new Vector3(width, 0.008f, chord * 0.9f);   // 살짝 두껍게 + 슬랫 간격
            b.transform.localPosition = new Vector3(0, y0 + i * pitch, 0);
            var r = b.GetComponent<Renderer>(); if (r) r.sharedMaterial = bladeMat;
        }
    }

    void BuildPanelBehind() {
        float pitch = pitchMm * 0.001f; float h = (nBlades + 1) * pitch;
        var panel = GameObject.CreatePrimitive(PrimitiveType.Cube);
        panel.name = "PVPanel";
        panel.transform.position = center + new Vector3(0, 0, 0.12f);
        panel.transform.localScale = new Vector3(width * 1.02f, h, 0.03f);
        var r = panel.GetComponent<Renderer>(); if (r) r.sharedMaterial = panelMat;
    }

    void BuildFrame() {
        float pitch = pitchMm * 0.001f; float h = (nBlades + 1) * pitch;
        var holder = new GameObject("LouverFrame"); holder.transform.position = center;
        for (int s = -1; s <= 1; s += 2) AddBar(holder.transform, new Vector3(0.05f, h + 0.06f, 0.18f), new Vector3(width * 0.5f * s, 0, 0));
        for (int s = -1; s <= 1; s += 2) AddBar(holder.transform, new Vector3(width + 0.10f, 0.05f, 0.18f), new Vector3(0, h * 0.5f * s, 0));
    }

    void AddBar(Transform parent, Vector3 scale, Vector3 pos) {
        var bar = GameObject.CreatePrimitive(PrimitiveType.Cube);
        bar.transform.SetParent(parent, false); bar.transform.localScale = scale; bar.transform.localPosition = pos;
        var r = bar.GetComponent<Renderer>(); if (r) r.sharedMaterial = frameMat;
    }

    // 안전모+형광조끼 작업자 + 제어콘솔. 인체 비율로 다듬음. 오른팔이 각도 따라 움직임.
    void BuildCharacter() {
        var holder = new GameObject("Operator");
        holder.transform.position = center + new Vector3(-(width * 0.5f) - 1.25f, -center.y, -0.2f);  // 바닥 좌측
        // 다리
        AddPart(holder.transform, PrimitiveType.Capsule, new Vector3(0.17f, 0.36f, 0.17f), new Vector3(-0.11f, 0.36f, 0), pantsMat);
        AddPart(holder.transform, PrimitiveType.Capsule, new Vector3(0.17f, 0.36f, 0.17f), new Vector3( 0.11f, 0.36f, 0), pantsMat);
        // 몸통(조끼)
        AddPart(holder.transform, PrimitiveType.Capsule, new Vector3(0.38f, 0.34f, 0.24f), new Vector3(0, 1.02f, 0), vestMat);
        // 머리 + 안전모
        AddPart(holder.transform, PrimitiveType.Sphere, new Vector3(0.21f, 0.23f, 0.21f), new Vector3(0, 1.52f, 0), skinMat);
        AddPart(holder.transform, PrimitiveType.Sphere, new Vector3(0.27f, 0.17f, 0.27f), new Vector3(0, 1.62f, 0), hatMat);
        AddPart(holder.transform, PrimitiveType.Cylinder, new Vector3(0.30f, 0.012f, 0.30f), new Vector3(0, 1.55f, 0), hatMat); // 챙
        // 왼팔(고정)
        AddPart(holder.transform, PrimitiveType.Capsule, new Vector3(0.11f, 0.30f, 0.11f), new Vector3(-0.27f, 1.02f, 0.02f), vestMat);
        // 오른 어깨 피벗 + 팔(각도 따라 회전)
        var shoulder = new GameObject("Shoulder"); shoulder.transform.SetParent(holder.transform, false);
        shoulder.transform.localPosition = new Vector3(0.27f, 1.22f, 0.05f);
        charArm = shoulder.transform;
        var arm = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        arm.transform.SetParent(shoulder.transform, false);
        arm.transform.localScale = new Vector3(0.10f, 0.26f, 0.10f);
        arm.transform.localPosition = new Vector3(0.04f, -0.16f, 0.10f);
        var ar = arm.GetComponent<Renderer>(); if (ar) ar.sharedMaterial = skinMat;
        var ac = arm.GetComponent<Collider>(); if (ac) Destroy(ac);
        // 제어 콘솔(작업자와 루버 사이)
        var podium = GameObject.CreatePrimitive(PrimitiveType.Cube);
        podium.transform.position = holder.transform.position + new Vector3(0.5f, 0.5f, 0.15f);
        podium.transform.localScale = new Vector3(0.4f, 1.0f, 0.3f);
        var pr = podium.GetComponent<Renderer>(); if (pr) pr.sharedMaterial = podiumMat;
        var screen = GameObject.CreatePrimitive(PrimitiveType.Cube);
        screen.transform.SetParent(podium.transform, false);
        screen.transform.localScale = new Vector3(0.8f, 0.45f, 0.12f);
        screen.transform.localPosition = new Vector3(0, 0.42f, -0.45f);
        var sc = screen.GetComponent<Collider>(); if (sc) Destroy(sc);
        screen.transform.localRotation = Quaternion.Euler(35f, 0, 0);
        var sr = screen.GetComponent<Renderer>(); if (sr) sr.sharedMaterial = screenMat;
    }

    void AddPart(Transform parent, PrimitiveType t, Vector3 scale, Vector3 pos, Material mat) {
        var g = GameObject.CreatePrimitive(t);
        var col = g.GetComponent<Collider>(); if (col) Destroy(col);
        g.transform.SetParent(parent, false); g.transform.localScale = scale; g.transform.localPosition = pos;
        var r = g.GetComponent<Renderer>(); if (r) r.sharedMaterial = mat;
    }

    void Update() {
        float clear = Mathf.Clamp01(agent.CurrentDni / 750f);
        if (agent.sun) agent.sun.intensity = Mathf.Lerp(0.30f, 1.65f, clear);
        if (skyMat) {
            skyMat.SetFloat("_AtmosphereThickness", Mathf.Lerp(1.9f, 0.9f, clear));
            skyMat.SetFloat("_Exposure", Mathf.Lerp(0.6f, 1.15f, clear));
        }
        if (charArm) charArm.localRotation = Quaternion.Euler(-30f - agent.CurrentTilt * 0.5f, 0, 0);  // 콘솔 조작하듯

        if (agent.DayTrackingPct > 0.01f) lastNonZeroPct = agent.DayTrackingPct;
        int ep = agent.CompletedEpisodes;
        if (ep > prevEpisodes) {
            prevEpisodes = ep;
            if (lastNonZeroPct > bestPct) bestPct = lastNonZeroPct;
            recent[recentIdx] = lastNonZeroPct; recentIdx = (recentIdx + 1) % recent.Length;
            if (recentFilled < recent.Length) recentFilled++;
        }
        // 자가 검증용 스크린샷 (persistentDataPath = 항상 쓰기 가능)
        if (Time.unscaledTime > nextShot) {
            nextShot = Time.unscaledTime + 5f;
            string p = System.IO.Path.Combine(Application.persistentDataPath, "frame.png");
            ScreenCapture.CaptureScreenshot(p);
            if (!shotPathLogged) { Debug.Log("[Presenter] screenshot path = " + p); shotPathLogged = true; }
        }
    }

    float RecentAvg() { if (recentFilled == 0) return 0f; float s = 0f; for (int i = 0; i < recentFilled; i++) s += recent[i]; return s / recentFilled; }

    string Season(string d) {
        if (string.IsNullOrEmpty(d) || d.Length < 7) return "";
        int mo; int.TryParse(d.Substring(5, 2), out mo);
        if (mo == 12 || mo <= 2) return "겨울"; if (mo <= 5) return "봄"; if (mo <= 8) return "여름"; return "가을";
    }
    string Weather(float peak) { return peak >= 600 ? "맑음 ☀" : peak >= 300 ? "구름 ⛅" : "흐림 ☁"; }

    void OnGUI() {
        if (agent == null) return;
        var title = new GUIStyle(GUI.skin.label) { fontSize = 18, fontStyle = FontStyle.Bold, wordWrap = true };
        var body  = new GUIStyle(GUI.skin.label) { fontSize = 15, wordWrap = true };
        var big   = new GUIStyle(GUI.skin.label) { fontSize = 16, fontStyle = FontStyle.Bold, wordWrap = true };
        GUILayout.BeginArea(new Rect(18, 18, 580, 360), GUI.skin.box);
        GUILayout.Label("AI가 태양을 따라 블라인드 각도를 맞추는 법을 스스로 배우는 중", title);
        GUILayout.Space(4);
        GUILayout.Label($"📅 {agent.Env.currentDate}   {Season(agent.Env.currentDate)} · {Weather(agent.Env.dayPeakDni)}   (서울 · 기상청 실제 10년, 날짜 순차 진행)", big);
        GUILayout.Space(6);
        int t = (int)Time.realtimeSinceStartup;
        long steps = Academy.IsInitialized ? Academy.Instance.TotalStepCount : 0;
        GUILayout.Label($"⏱ 학습 시간 {t/60:00}:{t%60:00}    학습 스텝 {steps:n0}    완료한 하루 {agent.CompletedEpisodes}일", body);
        GUILayout.Space(4);
        GUILayout.Label($"★ 평가 점수 — 최근 {recentFilled}일 평균  {RecentAvg():0.0}%      역대 최고  {bestPct:0.0}%", big);
        GUILayout.Label("   (100% = 컴퓨터가 찾은 '정답' 각도만큼 햇빛을 모음. 학습될수록 올라감)", body);
        GUILayout.Space(6);
        GUILayout.Label($"●  AI 각도 {agent.CurrentTilt:0}°    정답(이론상 최적) {agent.CurrentOracleTilt:0}°", body);
        GUILayout.Label($"●  지금 받는 햇빛 {agent.CurrentPoa:0}  (최대 {agent.CurrentOraclePoa:0})    오늘 점수 {agent.DayTrackingPct:0}%", body);
        GUILayout.EndArea();
    }
}
