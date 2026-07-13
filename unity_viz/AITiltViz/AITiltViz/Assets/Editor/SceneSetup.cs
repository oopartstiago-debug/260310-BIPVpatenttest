// AI Tilt — 씬 자동 생성+배선 (에디터 전용, 1단계 수동 클릭 제거)
// Unity 메뉴 [AI Tilt → Setup Scene] 한 번이면:
//   · Plane(바닥) · LouverRoot(원점 빈 오브젝트) · Directional Light(없으면 생성)
//   · 카메라 위치 · SolarLouverViz / SolarLouverSandbox 부착 + sun·louverRoot 자동 배선
// 이 파일은 반드시 Assets/Editor/ 안에 둘 것(UnityEditor 네임스페이스는 빌드에 안 들어감).
#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

public static class SceneSetup {
    [MenuItem("AI Tilt/Setup Scene")]
    public static void Build() {
        // 바닥
        if (GameObject.Find("Ground") == null) {
            var plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
            plane.name = "Ground";
            plane.transform.position = Vector3.zero;
            plane.transform.localScale = new Vector3(2, 1, 2);
        }

        // 루버 루트
        var root = GameObject.Find("LouverRoot");
        if (root == null) { root = new GameObject("LouverRoot"); root.transform.position = new Vector3(0, 1, 0); }

        // 방향광(태양)
        Light sun = null;
        foreach (var l in Object.FindObjectsOfType<Light>()) if (l.type == LightType.Directional) { sun = l; break; }
        if (sun == null) {
            var go = new GameObject("Directional Light");
            sun = go.AddComponent<Light>();
            sun.type = LightType.Directional;
        }

        // 카메라
        var cam = Camera.main;
        if (cam != null) {
            cam.transform.position = new Vector3(3.0f, 1.4f, -3.0f);
            cam.transform.rotation = Quaternion.Euler(12, -38, 0);
        }

        // 드라이버 부착 + 배선
        var viz = root.GetComponent<SolarLouverViz>(); if (viz == null) viz = root.AddComponent<SolarLouverViz>();
        viz.sun = sun;
        viz.louverRoot = root.transform;
        var sb = root.GetComponent<SolarLouverSandbox>(); if (sb == null) sb = root.AddComponent<SolarLouverSandbox>();
        sb.baseline = 90f;   // 직렬화된 옛 기본값(60) 덮어쓰기 → 비교각 90° 통일

        EditorUtility.SetDirty(root);
        Debug.Log("[AI Tilt] 씬 구성 완료 — Play ▶ 누르면 됩니다. (Sun/LouverRoot 자동 배선됨)");
    }

    // 비교 샌드박스(우리 vs HDC 가정) — 별도 씬. LouverCompareSandbox가 나머지(바닥/태양/카메라/스택) 자가 생성.
    [MenuItem("AI Tilt/Setup Compare (HDC vs Ours)")]
    public static void BuildCompare() {
        var host = GameObject.Find("CompareSandbox") ?? new GameObject("CompareSandbox");
        Light sun = null;
        foreach (var l in Object.FindObjectsOfType<Light>()) if (l.type == LightType.Directional) { sun = l; break; }
        if (sun == null) { var go = new GameObject("Directional Light"); sun = go.AddComponent<Light>(); sun.type = LightType.Directional; }
        var cs = host.GetComponent<LouverCompareSandbox>() ?? host.AddComponent<LouverCompareSandbox>();
        cs.sun = sun;
        EditorUtility.SetDirty(host);
        Debug.Log("[AI Tilt] 비교 씬 구성 완료 — Play ▶. 슬라이더로 각도/간격/태양 조작, 우/HDC 자기음영 실시간 대조.");
    }
}
#endif
