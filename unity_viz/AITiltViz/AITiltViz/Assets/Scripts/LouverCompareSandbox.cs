// AI Tilt — 밀폐 루버 자기음영/간격 비교 샌드박스 (인터랙티브)
//   목적: 각도·간격(gcr)·태양을 손으로 조작 → 자기음영이 PV 띠를 삼키는 걸 눈으로.
//   좌: 우리 (gcr 1.169 고정)  ·  우: HDC (가정 gcr 1.1 — 측정값 아님, 슬라이더로 탐색)
//   물리는 전부 검증된 LouverPhysics(physics_v3 C# 포팅, Unity≡Python) 호출. 새 물리 없음.
//   진짜 Unity 그림자 = 자기음영(광원=태양). 옆 2D 단면 = 정확한 PanelSf 수치와 일치.
//   ★HDC 간격은 특허표로 원리적 미상(3모델 확정) → "가정"이라 명시. 편심/도8 "주장 vs 실제≈0" 토글.
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class LouverCompareSandbox : MonoBehaviour {
    [Header("연결 (없으면 자동 생성)")]
    public Light sun;

    // 기하 (mm)
    const float OUR_CHORD = 114f, OUR_PITCH = 97.5f;   // 우리 실기하 gcr=1.169
    const float HDC_CHORD = 114f;                        // HDC 가정: 같은 폭, 간격만 gcr로
    const int   NB = 9;                                  // 스택당 블레이드 수
    const float BLADE_W = 0.55f;                         // 날개 길이(동서, m) — 두 스택 나란히 놓게 좁힘

    // 조작 상태
    float tilt = 78f, sunElev = 45f, sunAz = 180f, gcrHDC = 1.10f;
    bool showEcc = false;

    // 씬
    Transform ourRoot, hdcRoot;
    readonly List<Transform> ourB = new List<Transform>(), hdcB = new List<Transform>();
    Material ourMat, hdcMat, groundMat, pvMat;
    Texture2D px;
    string selftest = "";

    // 일 적분 캐시 (scene_data.json 타임라인)
    SceneData day;
    int   ourBestFixed, hdcBestFixed;
    float ourTrackGain, hdcTrackGain, dayGenRatio;   // 트래킹 이득(vs 최고고정), 일발전 우리/HDC
    float lastGcr = -1f;

    void Start() {
        string sa = Application.streamingAssetsPath;
        LouverPhysics.Load(sa);
        selftest = LouverPhysics.SelfTest(sa);
        Debug.Log(selftest);
        string p = Path.Combine(sa, "scene_data.json");
        if (File.Exists(p)) day = JsonUtility.FromJson<SceneData>(File.ReadAllText(p));

        px = new Texture2D(1, 1); px.SetPixel(0, 0, Color.white); px.Apply();
        EnsureScene();
        ourRoot = MakeRoot("OurLouver", new Vector3(-0.5f, 1f, 0f));
        hdcRoot = MakeRoot("HdcLouver", new Vector3( 0.5f, 1f, 0f));
        BuildStack(ourRoot, OUR_CHORD, OUR_PITCH, ourB, ourMat);
        RebuildHdc();
        RecomputeDay();
    }

    Transform MakeRoot(string name, Vector3 pos) {
        var g = GameObject.Find(name); if (g == null) g = new GameObject(name);
        g.transform.position = pos; return g.transform;
    }

    void EnsureScene() {
        var lit = Shader.Find("Universal Render Pipeline/Lit");
        ourMat = new Material(lit); ourMat.SetColor("_BaseColor", new Color(0.62f, 0.66f, 0.72f));
        ourMat.SetFloat("_Metallic", 0.30f); ourMat.SetFloat("_Smoothness", 0.45f);
        hdcMat = new Material(lit); hdcMat.SetColor("_BaseColor", new Color(0.72f, 0.64f, 0.58f));
        hdcMat.SetFloat("_Metallic", 0.30f); hdcMat.SetFloat("_Smoothness", 0.45f);
        pvMat = new Material(lit); pvMat.SetColor("_BaseColor", new Color(0.10f, 0.16f, 0.34f));  // PV 띠 = 짙은 남색
        pvMat.SetFloat("_Metallic", 0.10f); pvMat.SetFloat("_Smoothness", 0.55f);

        if (GameObject.Find("Ground") == null) {
            var pl = GameObject.CreatePrimitive(PrimitiveType.Plane); pl.name = "Ground";
            pl.transform.localScale = new Vector3(2, 1, 2);
            groundMat = new Material(lit); groundMat.SetColor("_BaseColor", new Color(0.34f, 0.35f, 0.37f));
            groundMat.SetFloat("_Metallic", 0f); groundMat.SetFloat("_Smoothness", 0.08f);
            pl.GetComponent<Renderer>().sharedMaterial = groundMat;
        }
        // 벽(내부/실외기실 쪽) — 태양은 -Z(외부), 벽은 +Z(내부). 상부가 이 벽쪽으로 기욺.
        if (GameObject.Find("Wall") == null) {
            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube); wall.name = "Wall";
            wall.transform.position = new Vector3(0f, 1.0f, 0.45f);
            wall.transform.localScale = new Vector3(2.4f, 2.2f, 0.06f);
            var wm = new Material(lit); wm.SetColor("_BaseColor", new Color(0.46f, 0.47f, 0.50f));
            wm.SetFloat("_Metallic", 0f); wm.SetFloat("_Smoothness", 0.05f);
            wall.GetComponent<Renderer>().sharedMaterial = wm;
        }
        if (sun == null)
            foreach (var l in FindObjectsOfType<Light>()) if (l.type == LightType.Directional) { sun = l; break; }
        if (sun == null) { var go = new GameObject("Directional Light"); sun = go.AddComponent<Light>(); sun.type = LightType.Directional; }
        sun.shadows = LightShadows.Soft; sun.shadowStrength = 0.92f; sun.intensity = 1.35f; sun.color = new Color(1f, 0.96f, 0.90f);
        RenderSettings.ambientMode = AmbientMode.Flat; RenderSettings.ambientLight = new Color(0.40f, 0.43f, 0.50f);

        var cam = Camera.main;
        if (cam) {
            cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = new Color(0.55f, 0.70f, 0.86f);
            cam.transform.position = new Vector3(0f, 1.35f, -2.9f); cam.transform.LookAt(new Vector3(0, 1f, 0)); cam.fieldOfView = 44f;
        }
        var urp = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
        if (urp != null) urp.shadowDistance = 15f;
    }

    // 블레이드 스택 생성: 날개(맨 프레임 회색) + 가운데 PV 띠(남색, 현의 STRIP_FRAC 만큼)
    void BuildStack(Transform root, float chordMm, float pitchMm, List<Transform> list, Material mat) {
        foreach (Transform c in root) Destroy(c.gameObject);
        list.Clear();
        const float m = 0.001f;
        float chord = chordMm * m, pitch = pitchMm * m;
        float y0 = -(NB - 1) * pitch * 0.5f;
        float stripFrac = 83f / chordMm;         // physics_v3 STRIP_FRAC (M6 하프 83mm)
        for (int i = 0; i < NB; i++) {
            var b = GameObject.CreatePrimitive(PrimitiveType.Cube);
            b.name = "Blade_" + i; b.transform.SetParent(root, false);
            b.transform.localScale = new Vector3(BLADE_W, 0.004f, chord);
            b.transform.localPosition = new Vector3(0, y0 + i * pitch, 0);
            b.GetComponent<Renderer>().sharedMaterial = mat;
            // PV 띠 (현 방향 가운데 stripFrac): 남색 얇은 판, 그림자 받게 살짝 위로
            var pv = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pv.name = "PV_" + i; pv.transform.SetParent(b.transform, false);
            pv.transform.localScale = new Vector3(0.96f, 1.4f, stripFrac);
            pv.transform.localPosition = new Vector3(0, 0.5f, 0);
            pv.GetComponent<Renderer>().sharedMaterial = pvMat;
            list.Add(b.transform);
        }
    }

    void RebuildHdc() {
        float pitch = HDC_CHORD / Mathf.Max(0.5f, gcrHDC);   // gcr=현/피치 → 피치=현/gcr
        BuildStack(hdcRoot, HDC_CHORD, pitch, hdcB, hdcMat);
        lastGcr = gcrHDC;
    }

    void Update() {
        if (Mathf.Abs(gcrHDC - lastGcr) > 1e-3f) { RebuildHdc(); RecomputeDay(); }
        // 상부=내부(벽)·하부=외부(태양쪽) — panel_sf 관례와 일치 (2026-07-13 사용자 지적: 렌더 부호 반대였음)
        var rot = Quaternion.Euler(-tilt, 0, 0);
        foreach (var b in ourB) b.localRotation = rot;
        foreach (var b in hdcB) b.localRotation = rot;
        // 태양 방향 (SolarLouverViz와 동일 관례: az=북기준, elev=지평선기준)
        float er = sunElev * Mathf.Deg2Rad, ar = sunAz * Mathf.Deg2Rad;
        Vector3 toSun = new Vector3(Mathf.Sin(ar) * Mathf.Cos(er), Mathf.Sin(er), Mathf.Cos(ar) * Mathf.Cos(er));
        if (sun) sun.transform.forward = (-toSun).normalized;
    }

    // 두 기하의 하루 적분(scene_data.json 타임라인) → 최고고정각·트래킹이득·일발전비
    void RecomputeDay() {
        if (day == null || day.timeline == null || day.timeline.Length == 0) return;
        float ourP = OUR_PITCH, hdcP = HDC_CHORD / Mathf.Max(0.5f, gcrHDC);
        float[] fixO = new float[91], fixH = new float[91];
        float sumOraO = 0f, sumOraH = 0f;
        foreach (var f in day.timeline) {
            float otO = LouverPhysics.OracleTilt(f.sun_elev, f.sun_az, f.dni, f.dhi, LouverPhysics.ALBEDO, OUR_CHORD, ourP);
            float otH = LouverPhysics.OracleTilt(f.sun_elev, f.sun_az, f.dni, f.dhi, LouverPhysics.ALBEDO, HDC_CHORD, hdcP);
            sumOraO += LouverPhysics.EffPoa(otO, f.sun_elev, f.sun_az, f.dni, f.dhi, LouverPhysics.ALBEDO, OUR_CHORD, ourP);
            sumOraH += LouverPhysics.EffPoa(otH, f.sun_elev, f.sun_az, f.dni, f.dhi, LouverPhysics.ALBEDO, HDC_CHORD, hdcP);
            for (int a = 0; a <= 90; a++) {
                fixO[a] += LouverPhysics.EffPoa(a, f.sun_elev, f.sun_az, f.dni, f.dhi, LouverPhysics.ALBEDO, OUR_CHORD, ourP);
                fixH[a] += LouverPhysics.EffPoa(a, f.sun_elev, f.sun_az, f.dni, f.dhi, LouverPhysics.ALBEDO, HDC_CHORD, hdcP);
            }
        }
        ourBestFixed = ArgMax(fixO); hdcBestFixed = ArgMax(fixH);
        ourTrackGain = fixO[ourBestFixed] > 0 ? (sumOraO - fixO[ourBestFixed]) / fixO[ourBestFixed] * 100f : 0f;
        hdcTrackGain = fixH[hdcBestFixed] > 0 ? (sumOraH - fixH[hdcBestFixed]) / fixH[hdcBestFixed] * 100f : 0f;
        dayGenRatio = fixH[hdcBestFixed] > 0 ? fixO[ourBestFixed] / fixH[hdcBestFixed] * 100f : 100f;
    }
    static int ArgMax(float[] a) { int b = 0; for (int i = 1; i < a.Length; i++) if (a[i] > a[b]) b = i; return b; }

    void OnGUI() {
        var st = new GUIStyle(GUI.skin.label) { fontSize = 13 };
        var stB = new GUIStyle(GUI.skin.label) { fontSize = 13, fontStyle = FontStyle.Bold };

        // ── 좌: 조작 슬라이더 ──
        GUILayout.BeginArea(new Rect(16, 16, 320, 232), GUI.skin.box);
        GUILayout.Label("조작 — 자기음영을 손으로", stB);
        GUILayout.Label($"날개 각도 {tilt:0}°  (0=수평 열림 · 90=수직 닫힘)", st); tilt = GUILayout.HorizontalSlider(tilt, 0, 90);
        GUILayout.Label($"태양 고도 {sunElev:0}°", st); sunElev = GUILayout.HorizontalSlider(sunElev, 5, 85);
        GUILayout.Label($"태양 방위 {sunAz:0}° (180=정남)", st); sunAz = GUILayout.HorizontalSlider(sunAz, 120, 240);
        GUILayout.Space(6);
        GUILayout.Label($"HDC 간격 가정  gcr {gcrHDC:0.00}  ⚠측정값 아님", stB); gcrHDC = GUILayout.HorizontalSlider(gcrHDC, 0.90f, 1.30f);
        showEcc = GUILayout.Toggle(showEcc, "편심/도8 '주장 vs 실제' 보기");
        GUILayout.EndArea();

        // ── 우: 실시간 비교 ──
        float su = LouverPhysics.PanelSf(tilt, sunElev, sunAz, OUR_CHORD, OUR_PITCH);
        float sh = LouverPhysics.PanelSf(tilt, sunElev, sunAz, HDC_CHORD, HDC_CHORD / Mathf.Max(0.5f, gcrHDC));
        float ovu = LouverPhysics.StripShade(su), ovh = LouverPhysics.StripShade(sh);
        float W = 386;
        GUILayout.BeginArea(new Rect(Screen.width - W - 16, 16, W, 300), GUI.skin.box);
        GUILayout.Label("비교 모니터 (현재 각도·태양에서)", stB);
        Row(st, "", "우리", "HDC(가정)");
        Row(st, "간격 gcr", $"{OUR_CHORD / OUR_PITCH:0.000}", $"{gcrHDC:0.00}");
        Row(st, "자기음영률", $"{su * 100:0.0}%", $"{sh * 100:0.0}%");
        Row(st, "PV띠 유효음영", $"{ovu * 100:0.0}%", $"{ovh * 100:0.0}%");
        GUILayout.Space(4); GUILayout.Label("─ 하루 적분 (서울 검증일) ─", st);
        Row(st, "최고 고정각", $"{ourBestFixed}°", $"{hdcBestFixed}°");
        Row(st, "트래킹 이득", $"+{ourTrackGain:0.0}%", $"+{hdcTrackGain:0.0}%");
        Row(st, "일 발전 (우리/HDC)", $"{dayGenRatio:0.0}%", "기준100");
        GUILayout.Space(4);
        GUILayout.Label("→ 둘 다 밀폐권이라 거의 동급. 간격을 벌릴수록\n   트래킹 이득이 커지지만 그럼 '밀폐'를 포기하는 것.", new GUIStyle(GUI.skin.label){ fontSize = 11 });
        GUILayout.EndArea();

        // ── 하: 2D 단면 스키마 (정확한 PanelSf 시각화) ──
        DrawSection(new Rect(16, Screen.height - 190, 300, 174), "우리 gcr 1.169", su, OUR_CHORD / OUR_PITCH);
        DrawSection(new Rect(Screen.width - 316, Screen.height - 190, 300, 174), $"HDC 가정 gcr {gcrHDC:0.00}", sh, gcrHDC);

        // ── 편심 정직 토글 ──
        if (showEcc) {
            GUILayout.BeginArea(new Rect(Screen.width / 2 - 230, Screen.height - 150, 460, 132), GUI.skin.box);
            GUILayout.Label("HDC 편심축 / 도8 각도표 — 주장 vs 실제", stB);
            GUILayout.Label("· 편심: HDC 스냅샷 지표론 여름 +18%~+22.7% 주장", st);
            GUILayout.Label("· 실제 연간 발전(빔·확산·IAM·선형셀): ≈ +0.0~0.3% (증발)", new GUIStyle(GUI.skin.label){ fontSize = 13, normal = { textColor = new Color(1f,0.7f,0.4f) } });
            GUILayout.Label("· 도8 아침·저녁 저각(30°) = 우리 물리로 자기음영 88~100% = 발전0", st);
            GUILayout.Label("Claude+codex+Fable 3모델 교차검증. 밀집 선형셀선 편심 무효.", new GUIStyle(GUI.skin.label){ fontSize = 11 });
            GUILayout.EndArea();
        }

        GUI.Label(new Rect(16, Screen.height - 14, 900, 14), selftest, new GUIStyle(GUI.skin.label){ fontSize = 10 });
    }

    void Row(GUIStyle st, string a, string b, string c) {
        GUILayout.BeginHorizontal();
        GUILayout.Label(a, st, GUILayout.Width(150));
        GUILayout.Label(b, st, GUILayout.Width(110));
        GUILayout.Label(c, st, GUILayout.Width(110));
        GUILayout.EndHorizontal();
    }

    // 측면 단면: 반복단위 4장 + 태양광선, 각 날개 상단 sf 만큼 붉게(그림자), PV띠 표시
    void DrawSection(Rect area, string title, float sf, float gcr) {
        GUI.Box(area, GUIContent.none);
        GUI.Label(new Rect(area.x + 8, area.y + 4, area.width, 16), title, new GUIStyle(GUI.skin.label){ fontSize = 12, fontStyle = FontStyle.Bold });
        float cx = area.x + 128, cyTop = area.y + 40;
        float pitchPx = 28f, chordPx = pitchPx * gcr;   // chord=gcr·pitch → gcr>1이면 현이 피치보다 길어 겹침
        float tr = tilt * Mathf.Deg2Rad;
        // 상부=내부(화면 왼쪽) · 하부=외부(오른쪽). tilt0=수평, tilt90=수직닫힘.
        Vector2 dir = new Vector2(Mathf.Cos(tr), Mathf.Sin(tr)) * (chordPx * 0.5f);
        GUI.Label(new Rect(area.x + 6, cyTop - 6, 60, 40), "내부\n(벽)", new GUIStyle(GUI.skin.label){ fontSize = 10, alignment = TextAnchor.MiddleCenter });
        GUI.Label(new Rect(area.xMax - 58, cyTop - 6, 54, 40), "외부\n(태양)", new GUIStyle(GUI.skin.label){ fontSize = 10, alignment = TextAnchor.MiddleCenter });
        // 태양 광선 (좌상단서)
        float er = sunElev * Mathf.Deg2Rad;
        Vector2 ray = new Vector2(Mathf.Cos(er), -Mathf.Sin(er)) * 26f;
        for (int i = 0; i < 4; i++) {
            Vector2 c = new Vector2(cx, cyTop + i * pitchPx);
            Vector2 top = c - dir, bot = c + dir;
            // 그림자 부분(상단 sf) = 붉게 / 밝은 부분 = 청록
            Vector2 sfEnd = Vector2.Lerp(top, bot, Mathf.Clamp01(sf));
            DrawSeg(top, sfEnd, 4f, new Color(0.85f, 0.25f, 0.20f, 0.95f));
            DrawSeg(sfEnd, bot, 4f, new Color(0.30f, 0.72f, 0.70f, 0.95f));
            if (i == 1) { GUI.color = new Color(1f, 0.85f, 0.25f, 0.9f); DrawSeg(top + new Vector2(-26, 26) - ray, top + new Vector2(-26, 26), 1.5f, new Color(1f,0.85f,0.25f,0.9f)); GUI.color = Color.white; }
        }
        GUI.Label(new Rect(area.x + 8, area.yMax - 34, area.width, 30),
                  $"자기음영 {sf * 100:0.0}%   ■그림자 ■수광\n{(gcr >= 1f ? "닫으면 밀폐(gcr≥1)" : "닫아도 틈(gcr<1)")}",
                  new GUIStyle(GUI.skin.label){ fontSize = 11 });
    }

    void DrawSeg(Vector2 a, Vector2 b, float w, Color col) {
        Vector2 d = b - a; float len = d.magnitude; if (len < 0.01f) return;
        float ang = Mathf.Atan2(d.y, d.x) * Mathf.Rad2Deg;
        Matrix4x4 save = GUI.matrix;
        GUIUtility.RotateAroundPivot(ang, a);
        GUI.color = col; GUI.DrawTexture(new Rect(a.x, a.y - w * 0.5f, len, w), px);
        GUI.color = Color.white; GUI.matrix = save;
    }
}
