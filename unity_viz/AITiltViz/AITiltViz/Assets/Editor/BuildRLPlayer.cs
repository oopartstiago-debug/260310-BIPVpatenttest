// AI Tilt — 배치모드 무인 빌드 (CLI: -executeMethod BuildRLPlayer.PerformBuild)
// SampleScene 을 RL(Train, BehaviorType.Default)로 배선·저장 후 macOS 스탠드얼론 플레이어로 빌드.
// 빌드된 .app 을  mlagents-learn --env=<...>/AITiltRL.app  으로 자동 실행 → Play 불필요.
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildRLPlayer {
    const string ScenePath = "Assets/Scenes/SampleScene.unity";
    const string OutPath   = "/Volumes/AISSD/ai-tilt/unity_viz/Build/AITiltRL.app";

    public static void PerformBuild() {
        // 1) 씬 열고 RL(Train) 구성 후 저장
        var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        SetupRLScene.BuildTrain();                 // LouverAgent + BehaviorParameters(Default) 배선
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);

        // 2) 스탠드얼론 빌드(창 뜨는 일반 플레이어 → 학습 과정 눈으로 볼 수 있음)
        var opts = new BuildPlayerOptions {
            scenes = new[] { ScenePath },
            locationPathName = OutPath,
            target = BuildTarget.StandaloneOSX,
            options = BuildOptions.None,
        };
        var s = BuildPipeline.BuildPlayer(opts).summary;
        Debug.Log($"[BuildRLPlayer] result={s.result} size={s.totalSize} errors={s.totalErrors} time={s.totalTime}");
        if (s.result != BuildResult.Succeeded) EditorApplication.Exit(1);
    }
}
#endif
