// AI Tilt — 추론(Inference) 스탠드얼론 빌드 (CLI: -executeMethod BuildRLInference.PerformBuild)
// 학습된 LouverTilt.onnx 를 Assets 로 임포트 → ModelAsset 으로 로드 → BehaviorType.InferenceOnly 로 배선 →
// 스탠드얼론 .app 빌드. 이 빌드는 트레이너 없이도 학습된 정책으로 태양을 추종한다.
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildRLInference {
    const string ScenePath = "Assets/Scenes/SampleScene.unity";
    const string OutPath   = "/Volumes/AISSD/ai-tilt/unity_viz/Build/AITiltRL_Infer.app";
    // mlagents 최종 export 는 외부 가중치(.onnx.data)를 분리 저장 → onnx + data 쌍을 함께 복사해야 한다.
    const string OnnxSrc   = "/Volumes/AISSD/ai-tilt/unity_viz/results/louver_demo/LouverTilt/LouverTilt-3000120.onnx";
    const string DataSrc   = "/Volumes/AISSD/ai-tilt/unity_viz/results/louver_demo/LouverTilt/LouverTilt-3000120.onnx.data";
    const string AssetPath = "Assets/Models/LouverTilt-3000120.onnx";

    public static void PerformBuild() {
        // 1) onnx + 외부가중치 → Assets 복사 후 ModelAsset 으로 임포트
        string assetDir = System.IO.Path.Combine(Application.dataPath, "Models");
        System.IO.Directory.CreateDirectory(assetDir);
        System.IO.File.Copy(OnnxSrc, System.IO.Path.Combine(assetDir, "LouverTilt-3000120.onnx"), true);
        System.IO.File.Copy(DataSrc, System.IO.Path.Combine(assetDir, "LouverTilt-3000120.onnx.data"), true);
        AssetDatabase.ImportAsset(AssetPath, ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
        var model = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(AssetPath)
                    ?? AssetDatabase.LoadMainAssetAtPath(AssetPath) as Unity.InferenceEngine.ModelAsset;
        if (model == null) { Debug.LogError("[BuildRLInference] ModelAsset 로드 실패: " + AssetPath); EditorApplication.Exit(2); return; }

        // 2) 씬을 추론 구성으로 배선(모델 주입) 후 저장
        var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        SetupRLScene.BuildInferenceWith(model);
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);

        // 3) 스탠드얼론 빌드(별도 .app — Train 빌드 보존)
        var opts = new BuildPlayerOptions {
            scenes = new[] { ScenePath },
            locationPathName = OutPath,
            target = BuildTarget.StandaloneOSX,
            options = BuildOptions.None,
        };
        string modelName = model.name;  // 빌드 후 에셋 언로드되므로 미리 캡처
        var s = BuildPipeline.BuildPlayer(opts).summary;
        Debug.Log($"[BuildRLInference] result={s.result} size={s.totalSize} errors={s.totalErrors} model={modelName}");
        if (s.result != BuildResult.Succeeded) EditorApplication.Exit(1);
    }
}
#endif
