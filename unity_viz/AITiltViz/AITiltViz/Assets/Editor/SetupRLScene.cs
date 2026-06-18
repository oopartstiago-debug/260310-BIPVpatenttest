// AI Tilt — 강화학습 씬 자동 구성 (에디터 전용)
// 메뉴 [AI Tilt → Setup RL Scene (Train)]  : 학습용(BehaviorType.Default → mlagents-learn에 연결)
// 메뉴 [AI Tilt → Setup RL Scene (Heuristic Test)] : 키보드(←/→) 수동 테스트용
// 현재 씬을 RL 구성으로 전환한다(뷰어 컴포넌트 SolarLouverViz/Sandbox는 충돌 방지 위해 제거).
#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

public static class SetupRLScene {
    [MenuItem("AI Tilt/Setup RL Scene (Train)")]
    public static void BuildTrain() { Build(BehaviorType.Default); }

    [MenuItem("AI Tilt/Setup RL Scene (Heuristic Test)")]
    public static void BuildHeuristic() { Build(BehaviorType.HeuristicOnly); }

    // 추론(InferenceOnly) 구성 + 학습된 onnx 모델 주입 → 스탠드얼론에서 학습 정책이 실제로 태양 추종.
    public static void BuildInferenceWith(Unity.InferenceEngine.ModelAsset model) {
        Build(BehaviorType.InferenceOnly);
        var root = GameObject.Find("LouverRoot");
        var bp = root.GetComponent<BehaviorParameters>();
        bp.Model = model;
    }

    static void Build(BehaviorType behaviorType) {
        // 바닥
        if (GameObject.Find("Ground") == null) {
            var plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
            plane.name = "Ground";
            plane.transform.localScale = new Vector3(2, 1, 2);
        }

        // 태양(없으면 생성)
        Light sun = null;
        foreach (var l in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
            if (l.type == LightType.Directional) { sun = l; break; }
        if (sun == null) {
            var go = new GameObject("Directional Light");
            sun = go.AddComponent<Light>();
            sun.type = LightType.Directional;
        }

        // 루버 루트 = Agent 호스트
        var root = GameObject.Find("LouverRoot");
        if (root == null) { root = new GameObject("LouverRoot"); root.transform.position = new Vector3(0, 1, 0); }

        // 뷰어 컴포넌트 제거(블레이드 생성·회전 충돌 방지)
        RemoveIfPresent<SolarLouverViz>(root);
        RemoveIfPresent<SolarLouverSandbox>(root);
        // 기존 블레이드(에디트 모드에 남아있을 수 있는) 정리
        for (int i = root.transform.childCount - 1; i >= 0; i--) {
            var c = root.transform.GetChild(i);
            if (c.name.StartsWith("Blade_")) Object.DestroyImmediate(c.gameObject);
        }

        // Agent
        var agent = root.GetComponent<LouverAgent>(); if (agent == null) agent = root.AddComponent<LouverAgent>();
        agent.louverRoot = root.transform;
        agent.sun = sun;
        agent.stepsPerDay = 120;
        agent.MaxStep = 120;

        // BehaviorParameters
        var bp = root.GetComponent<BehaviorParameters>(); if (bp == null) bp = root.AddComponent<BehaviorParameters>();
        bp.BehaviorName = "LouverTilt";
        bp.BrainParameters.VectorObservationSize = 8;
        bp.BrainParameters.NumStackedVectorObservations = 1;
        bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
        bp.BehaviorType = behaviorType;

        // DecisionRequester
        var dr = root.GetComponent<DecisionRequester>(); if (dr == null) dr = root.AddComponent<DecisionRequester>();
        dr.DecisionPeriod = 1;
        dr.TakeActionsBetweenDecisions = false;

        // Presenter(데모 시각)
        if (root.GetComponent<LouverAgentPresenter>() == null) root.AddComponent<LouverAgentPresenter>();

        // 카메라 근접 측면
        var cam = Camera.main;
        if (cam != null) {
            cam.transform.position = new Vector3(1.8f, 1.15f, -2.6f);
            cam.transform.LookAt(root.transform.position);
            cam.fieldOfView = 40f;
        }

        EditorUtility.SetDirty(root);
        Debug.Log($"[AI Tilt] RL 씬 구성 완료 — Behavior=LouverTilt, obs=8, act=1(연속), MaxStep=120, BehaviorType={behaviorType}. " +
                  (behaviorType == BehaviorType.HeuristicOnly
                     ? "▶ Play 후 ←/→ 키로 수동 테스트."
                     : "학습: 터미널에서 mlagents-learn rl/config/louver_ppo.yaml --run-id=louver01 실행 후 ▶ Play."));
    }

    static void RemoveIfPresent<T>(GameObject go) where T : Component {
        var c = go.GetComponent<T>();
        if (c != null) Object.DestroyImmediate(c);
    }
}
#endif
