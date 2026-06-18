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

    // 하루 발전 프로파일(시각화) — 시간위상 24구간에 AI/정답 POA 기록 (에피소드마다 초기화)
    const int NBINS = 24;
    readonly float[] binAi = new float[NBINS], binOra = new float[NBINS];
    readonly bool[] binSet = new bool[NBINS];

    // ── V3 현상설명(Explanation) 모드 — 시각 전용(학습 불변) ──
    // 맑은 한여름 정오를 고정하고 루버를 0°→90° 스윕하며 왜 ~80°가 최적인지(저각 자기음영/균형/빔스침) 보여줌.
    bool explainMode;
    float explainAngle; bool explainDescending;
    Vector3 savedCamPos; Quaternion savedCamRot; float savedFov;
    float[] explainCurve; float explainCurveMax, explainOptTilt, explainOptPoa;
    const float EX_ELEV = 72f, EX_AZ = 180f, EX_DNI = 880f, EX_DHI = 110f, EX_ALB = 0.15f, EX_C = 97.5f, EX_P = 97.5f;
    const float EX_SWEEP_SPEED = 11f;  // 도/초

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
        foreach (var a in System.Environment.GetCommandLineArgs()) if (a == "-explain") { EnterExplain(); break; }
    }

    // ── 현상설명 모드 진입/이탈 ──
    void EnterExplain() {
        if (explainMode) return;
        explainMode = true;
        if (explainCurve == null) {
            explainCurve = new float[91]; explainCurveMax = 1f; explainOptPoa = -1f;
            for (int t = 0; t <= 90; t++) {
                float e = LouverPhysics.EffPoa(t, EX_ELEV, EX_AZ, EX_DNI, EX_DHI, EX_ALB, EX_C, EX_P);
                explainCurve[t] = e;
                if (e > explainCurveMax) explainCurveMax = e;
                if (e > explainOptPoa) { explainOptPoa = e; explainOptTilt = t; }
            }
        }
        if (cam) {
            savedCamPos = cam.transform.position; savedCamRot = cam.transform.rotation; savedFov = cam.fieldOfView;
            cam.transform.position = center + new Vector3(2.5f, 1.05f, -3.7f);  // 살짝 높여 PV 면을 내려다봄
            cam.transform.LookAt(center + new Vector3(-0.95f, 0.35f, 0.0f));    // 루버를 화면 우측에(좌측 텍스트 안 가림)
            cam.fieldOfView = 42f;
        }
        explainAngle = 0f; explainDescending = false;
        FreezeSun();
    }

    void ExitExplain() {
        if (!explainMode) return; explainMode = false;
        if (cam) { cam.transform.position = savedCamPos; cam.transform.rotation = savedCamRot; cam.fieldOfView = savedFov; }
    }

    void FreezeSun() {
        if (!agent.sun) return;
        float er = EX_ELEV * Mathf.Deg2Rad, ar = EX_AZ * Mathf.Deg2Rad;
        Vector3 d = new Vector3(Mathf.Sin(ar) * Mathf.Cos(er), Mathf.Sin(er), Mathf.Cos(ar) * Mathf.Cos(er));
        agent.sun.transform.forward = (-d).normalized;
        agent.sun.intensity = 1.6f;
    }

    // 시각 전용: 에이전트가 매 스텝 설정한 블레이드/태양을 LateUpdate 에서 덮어써 스윕을 보여줌(학습 불변).
    void LateUpdate() {
        if (!explainMode) return;
        explainAngle += (explainDescending ? -1f : 1f) * EX_SWEEP_SPEED * Time.deltaTime;
        if (explainAngle >= 90f) { explainAngle = 90f; explainDescending = true; }
        else if (explainAngle <= 0f) { explainAngle = 0f; explainDescending = false; }
        if (louverRoot)
            for (int i = 0; i < louverRoot.childCount; i++)
                louverRoot.GetChild(i).localRotation = Quaternion.Euler(explainAngle, 0, 0);
        FreezeSun();
    }

    string ExplainCaption(float a, out string head) {
        if (a < 25f) { head = "① 저각 — 자기음영"; return "슬랫을 낮게 눕히면 위 슬랫이 아래를 가려(자기음영) 직사광이 패널에 거의 못 닿는다. 발전 급감."; }
        if (a <= 86f) { head = "② 약 80° — 균형(최적)"; return "자기음영이 사라지고 입사각도 적정. 두 손실이 모두 작은 '평평한 꼭대기' → 발전 최대."; }
        head = "③ 90° — 빔 스침"; return "완전히 세우면 직사광이 표면을 스치듯 입사(cosθ 급감) + 유리 반사(프레넬)가 커져 다시 손실.";
    }

    void MakeMaterials() {
        var lit = Shader.Find("Universal Render Pipeline/Lit");
        bladeMat  = M(lit, Color.white, 0.10f, 0.82f);                       // ★ 블레이드 = BIPV 유리 PV 모듈(5셀/날개)
        frameMat  = M(lit, new Color(0.16f, 0.17f, 0.20f), 0.55f, 0.45f);
        panelMat  = M(lit, new Color(0.09f, 0.10f, 0.12f), 0.0f, 0.12f);     // 뒤=어두운 리세스(개구부 안쪽)
        wallMat   = M(lit, new Color(0.50f, 0.51f, 0.53f), 0.0f,  0.12f);   // 콘크리트 외벽
        groundMat = M(lit, new Color(0.50f, 0.50f, 0.52f), 0.0f,  0.16f);
        vestMat   = M(lit, new Color(0.95f, 0.45f, 0.05f), 0.0f,  0.30f);   // 형광 조끼
        hatMat    = M(lit, new Color(0.95f, 0.80f, 0.10f), 0.0f,  0.45f);   // 안전모
        skinMat   = M(lit, new Color(0.92f, 0.74f, 0.58f), 0.0f,  0.25f);
        pantsMat  = M(lit, new Color(0.16f, 0.20f, 0.32f), 0.0f,  0.25f);   // 작업바지
        podiumMat = M(lit, new Color(0.30f, 0.32f, 0.36f), 0.7f,  0.55f);   // 제어콘솔 금속
        screenMat = M(lit, new Color(0.10f, 0.20f, 0.28f), 0.0f,  0.6f);
        screenMat.EnableKeyword("_EMISSION"); screenMat.SetColor("_EmissionColor", new Color(0.2f, 0.9f, 1f) * 1.6f);

        // 절차적 텍스처(런타임 생성 — 임포트 불필요). 블레이드=PV 셀 모듈, 바닥=콘크리트 타일.
        bladeMat.SetTexture("_BaseMap", MakeBladePvTex(384, 96));
        groundMat.SetTexture("_BaseMap", MakeTileTex(256));        groundMat.SetColor("_BaseColor", Color.white);
        groundMat.SetTextureScale("_BaseMap", new Vector2(8, 8));
    }

    Material M(Shader s, Color c, float metallic, float smooth) {
        var m = new Material(s); m.SetColor("_BaseColor", c); m.SetFloat("_Metallic", metallic); m.SetFloat("_Smoothness", smooth); return m;
    }

    // BIPV 날개 = 폭방향 5셀 PV 모듈(1043A: 5셀/날개). 짙은 남색 글라스 셀 + 셀 경계 + 버스바 + 가장자리 프레임.
    Texture2D MakeBladePvTex(int w, int h) {
        var t = new Texture2D(w, h, TextureFormat.RGB24, true);
        Color frame = new Color(0.20f, 0.22f, 0.25f), grout = new Color(0.06f, 0.08f, 0.13f);
        Color cell = new Color(0.10f, 0.17f, 0.40f), bus = new Color(0.72f, 0.76f, 0.82f);   // 데모 가독: 셀 약간 밝게
        int cols = 5, cw = w / cols;
        int fb = Mathf.Max(2, h / 16);  // 가장자리 알루미늄 프레임 두께
        var px = new Color[w * h];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            Color c;
            if (x < fb || x >= w - fb || y < fb || y >= h - fb) c = frame;   // 프레임 테두리
            else {
                int cx = x % cw;
                if (cx < 2) c = grout;                                       // 셀 사이 경계
                else if (cx == cw / 3 || cx == 2 * cw / 3) c = bus;          // 버스바 2줄(폭방향)
                else c = cell;
            }
            px[y * w + x] = c;
        }
        t.SetPixels(px); t.Apply(); t.wrapMode = TextureWrapMode.Clamp; return t;
    }

    // 콘크리트 타일: 밝은 회색 타일 + 어두운 줄눈
    Texture2D MakeTileTex(int n) {
        var t = new Texture2D(n, n, TextureFormat.RGB24, true);
        Color tile = new Color(0.50f, 0.50f, 0.52f), grout = new Color(0.34f, 0.34f, 0.36f);
        int g = n / 4; var px = new Color[n * n];
        for (int y = 0; y < n; y++) for (int x = 0; x < n; x++)
            px[y * n + x] = (x % g < 2 || y % g < 2) ? grout : tile;
        t.SetPixels(px); t.Apply(); return t;
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
        cam.transform.position = center + new Vector3(2.0f, 0.65f, -4.3f);
        cam.transform.LookAt(center + new Vector3(-1.0f, 1.0f, 0.0f));   // 좌측을 봐서 루버가 화면 우측에 옴(텍스트 안 가림)
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

        // 하루 발전 프로파일: 현재 시간위상 구간에 최신 POA 기록
        int bi = Mathf.Clamp(Mathf.RoundToInt(agent.DayPhase01 * (NBINS - 1)), 0, NBINS - 1);
        binAi[bi] = agent.CurrentPoa; binOra[bi] = agent.CurrentOraclePoa; binSet[bi] = true;

        if (agent.DayTrackingPct > 0.01f) lastNonZeroPct = agent.DayTrackingPct;
        int ep = agent.CompletedEpisodes;
        if (ep > prevEpisodes) {
            prevEpisodes = ep;
            if (lastNonZeroPct > bestPct) bestPct = lastNonZeroPct;
            recent[recentIdx] = lastNonZeroPct; recentIdx = (recentIdx + 1) % recent.Length;
            if (recentFilled < recent.Length) recentFilled++;
            for (int i = 0; i < NBINS; i++) binSet[i] = false;   // 새 하루 → 프로파일 초기화
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

        // 모드 토글 버튼(우상단) — 사용자가 .app 에서 클릭
        var btn = new GUIStyle(GUI.skin.button) { fontSize = 15, fontStyle = FontStyle.Bold };
        if (GUI.Button(new Rect(Screen.width - 236, 18, 218, 38),
                explainMode ? "◀ AI 시연으로 돌아가기" : "▶ 왜 80°가 최적인가 (설명)", btn)) {
            if (explainMode) ExitExplain(); else EnterExplain();
        }

        if (explainMode) { DrawExplainOverlay(); return; }

        var title = new GUIStyle(GUI.skin.label) { fontSize = 15, fontStyle = FontStyle.Bold, wordWrap = true };
        var body  = new GUIStyle(GUI.skin.label) { fontSize = 13, wordWrap = true };
        var big   = new GUIStyle(GUI.skin.label) { fontSize = 14, fontStyle = FontStyle.Bold, wordWrap = true };
        var hud = new Rect(14, 14, 446, 300); Panel(hud);
        GUILayout.BeginArea(new Rect(hud.x + 12, hud.y + 10, hud.width - 24, hud.height - 20));
        GUILayout.Label("AI가 태양을 따라 블라인드 각도를 맞추는 법을 스스로 배우는 중", title);
        GUILayout.Space(4);
        GUILayout.Label($"📅 {agent.Env.currentDate}   {Season(agent.Env.currentDate)} · {Weather(agent.Env.dayPeakDni)}   (서울 기상청 10년, 순차)", big);
        GUILayout.Space(6);
        int t = (int)Time.realtimeSinceStartup;
        long steps = Academy.IsInitialized ? Academy.Instance.TotalStepCount : 0;
        GUILayout.Label($"⏱ 학습 {t/60:00}:{t%60:00}   스텝 {steps:n0}   완료 {agent.CompletedEpisodes}일", body);
        GUILayout.Space(4);
        GUILayout.Label($"★ 평가 점수 — 최근 {recentFilled}일 평균 {RecentAvg():0.0}%   역대 최고 {bestPct:0.0}%", big);
        GUILayout.Label("   (100% = '정답' 각도만큼 햇빛을 모음. 학습될수록 ↑)", body);
        GUILayout.Space(6);
        GUILayout.Label($"●  AI 각도 {agent.CurrentTilt:0}°    정답(이론상 최적) {agent.CurrentOracleTilt:0}°", body);
        GUILayout.Label($"●  지금 햇빛 {agent.CurrentPoa:0} (최대 {agent.CurrentOraclePoa:0})   오늘 {agent.DayTrackingPct:0}%", body);
        DrawPowerBar(GUILayoutUtility.GetRect(410, 14), agent.CurrentPoa, agent.CurrentOraclePoa);
        GUILayout.EndArea();

        DrawDayChart(new Rect(14, 322, 446, 124));
    }

    static void Fill(Rect r, Color c) { var old = GUI.color; GUI.color = c; GUI.DrawTexture(r, Texture2D.whiteTexture); GUI.color = old; }
    // 반투명 패널 배경 — 루버를 가리지 않게 텍스트는 좌측 컬럼에만, 배경은 살짝 비침
    static void Panel(Rect r) { Fill(r, new Color(0.04f, 0.05f, 0.07f, 0.60f)); }

    // 즉시 발전량 막대: 정답(회색) 위에 AI(하늘색) 겹침
    void DrawPowerBar(Rect r, float ai, float ora) {
        float mx = Mathf.Max(1f, ora);
        Fill(r, new Color(0, 0, 0, 0.35f));
        Fill(new Rect(r.x, r.y, r.width * Mathf.Clamp01(ora / mx), r.height), new Color(0.6f, 0.62f, 0.66f, 0.7f));
        Fill(new Rect(r.x, r.y, r.width * Mathf.Clamp01(ai / mx), r.height), new Color(0.2f, 0.85f, 1f, 0.95f));
    }

    // 하루 발전 프로파일: 시간대별 정답(회색) vs AI(하늘색) 막대 → 학습=정답 곡선 추종을 시각화
    void DrawDayChart(Rect area) {
        Panel(area);
        var lab = new GUIStyle(GUI.skin.label) { fontSize = 13, fontStyle = FontStyle.Bold };
        GUI.Label(new Rect(area.x + 10, area.y + 5, area.width - 20, 20), "오늘 발전 프로파일 (시간대별) — ■ 정답  ■ AI", lab);
        Rect plot = new Rect(area.x + 12, area.y + 30, area.width - 24, area.height - 42);
        float mx = 1f; for (int i = 0; i < NBINS; i++) if (binOra[i] > mx) mx = binOra[i];
        float bw = plot.width / NBINS;
        for (int i = 0; i < NBINS; i++) {
            if (!binSet[i]) continue;
            float x = plot.x + i * bw;
            float ho = plot.height * Mathf.Clamp01(binOra[i] / mx);
            float ha = plot.height * Mathf.Clamp01(binAi[i] / mx);
            Fill(new Rect(x + 1, plot.yMax - ho, bw - 2, ho), new Color(0.6f, 0.62f, 0.66f, 0.85f));                  // 정답
            Fill(new Rect(x + 1 + bw * 0.18f, plot.yMax - ha, (bw - 2) * 0.64f, ha), new Color(0.2f, 0.85f, 1f, 0.95f)); // AI
        }
    }

    // ── 현상설명 오버레이 ──
    void DrawExplainOverlay() {
        float curPoa = explainCurve != null ? explainCurve[Mathf.Clamp(Mathf.RoundToInt(explainAngle), 0, 90)] : 0f;
        string head; string body = ExplainCaption(explainAngle, out head);
        var title = new GUIStyle(GUI.skin.label) { fontSize = 15, fontStyle = FontStyle.Bold, wordWrap = true };
        var big   = new GUIStyle(GUI.skin.label) { fontSize = 18, fontStyle = FontStyle.Bold, wordWrap = true };
        var headS = new GUIStyle(GUI.skin.label) { fontSize = 15, fontStyle = FontStyle.Bold, wordWrap = true };
        var bodyS = new GUIStyle(GUI.skin.label) { fontSize = 13, wordWrap = true };

        var cap = new Rect(14, 14, 452, 212); Panel(cap);
        GUILayout.BeginArea(new Rect(cap.x + 12, cap.y + 10, cap.width - 24, cap.height - 20));
        GUILayout.Label("왜 약 80°가 최적인가 — 0°→90° 훑어보기 (맑은 한여름 정오)", title);
        GUILayout.Space(6);
        GUILayout.Label($"현재 각도 {explainAngle:0}°    받는 햇빛 {curPoa:0} / 최대 {explainOptPoa:0} (최적 {explainOptTilt:0}°)", big);
        GUILayout.Space(4);
        GUILayout.Label(head, headS);
        GUILayout.Label(body, bodyS);
        GUILayout.EndArea();

        DrawExplainCurve(new Rect(14, 234, 452, 212));
    }

    void DrawExplainCurve(Rect area) {
        Panel(area);
        var lab = new GUIStyle(GUI.skin.label) { fontSize = 12 };
        GUI.Label(new Rect(area.x + 10, area.y + 5, area.width - 20, 20), "발전량(POA) vs 루버 각도", new GUIStyle(GUI.skin.label) { fontSize = 13, fontStyle = FontStyle.Bold });
        Rect plot = new Rect(area.x + 14, area.y + 28, area.width - 28, area.height - 52);
        if (explainCurve == null) return;
        // 구간 배경: 저각 자기음영 / 빔스침
        Fill(new Rect(plot.x, plot.y, plot.width * 25f / 90f, plot.height), new Color(0.95f, 0.35f, 0.25f, 0.10f));
        Fill(new Rect(plot.x + plot.width * 86f / 90f, plot.y, plot.width * 4f / 90f, plot.height), new Color(0.98f, 0.6f, 0.1f, 0.12f));
        // 곡선(도별 막대, 구간별 색)
        for (int t = 0; t <= 90; t++) {
            float h = plot.height * Mathf.Clamp01(explainCurve[t] / explainCurveMax);
            float x = plot.x + plot.width * t / 90f;
            Color c = t < 25 ? new Color(0.95f, 0.4f, 0.3f, 0.9f)
                    : t > 86 ? new Color(0.98f, 0.62f, 0.12f, 0.9f)
                             : new Color(0.35f, 0.85f, 0.5f, 0.9f);
            Fill(new Rect(x, plot.yMax - h, Mathf.Max(1f, plot.width / 91f - 0.5f), h), c);
        }
        // 최적각(흰선) + 현재각(시안선)
        float ox = plot.x + plot.width * explainOptTilt / 90f;
        Fill(new Rect(ox - 1, plot.y, 2, plot.height), new Color(1f, 1f, 1f, 0.55f));
        float cx = plot.x + plot.width * explainAngle / 90f;
        Fill(new Rect(cx - 1.5f, plot.y, 3, plot.height), new Color(0.2f, 0.85f, 1f, 1f));
        GUI.Label(new Rect(plot.x, plot.yMax + 2, 90, 18), "0° (눕힘)", lab);
        GUI.Label(new Rect(plot.xMax - 70, plot.yMax + 2, 70, 18), "90° (세움)", lab);
        GUI.Label(new Rect(ox - 16, plot.y - 1, 90, 18), $"최적 {explainOptTilt:0}°", lab);
    }
}
