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
        // 0) Assets/Env 의 HDRI(.hdr/.exr) → Resources/EnvSky.mat(Skybox/Panoramic) 베이크(드롭인)
        BakeEnvSky();
        // 0b) PBR 콘크리트 텍스처 → Resources/GroundMat·WallMat 베이크(드롭인)
        BakeSurfaces();
        // 0c) Assets/Characters 의 FBX(Mixamo) → Humanoid 임포트 + 루프 애니 + 컨트롤러 → Resources
        BakeCharacter();

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

    // Assets/Env 의 첫 HDRI 를 Skybox/Panoramic 머티리얼로 구워 Resources/EnvSky.mat 에 저장.
    // 파일이 없으면 기존 EnvSky.mat 제거(→ 프레젠터가 절차적 하늘로 폴백). HDRI 드롭인 = 자동 적용.
    static void BakeEnvSky() {
        const string outMat = "Assets/Resources/EnvSky.mat";
        string hdri = null;
        if (System.IO.Directory.Exists(Application.dataPath + "/Env"))
            foreach (var f in System.IO.Directory.GetFiles(Application.dataPath + "/Env"))
                if (f.EndsWith(".hdr") || f.EndsWith(".exr")) { hdri = "Assets/Env/" + System.IO.Path.GetFileName(f); break; }
        if (hdri == null) { if (AssetDatabase.LoadAssetAtPath<Material>(outMat) != null) AssetDatabase.DeleteAsset(outMat); return; }

        AssetDatabase.ImportAsset(hdri, ImportAssetOptions.ForceSynchronousImport);
        var tex = AssetDatabase.LoadAssetAtPath<Texture>(hdri);
        var sh = Shader.Find("Skybox/Panoramic");
        if (tex == null || sh == null) { Debug.LogWarning("[BakeEnvSky] HDRI/shader 로드 실패 — 절차적 하늘 폴백"); return; }
        var mat = new Material(sh);
        mat.SetTexture("_MainTex", tex);
        mat.SetFloat("_Mapping", 1f);      // Latitude-Longitude Layout
        mat.SetFloat("_ImageType", 0f);    // 360°
        mat.SetFloat("_Exposure", 1.05f);
        System.IO.Directory.CreateDirectory(Application.dataPath + "/Resources");
        if (AssetDatabase.LoadAssetAtPath<Material>(outMat) != null) AssetDatabase.DeleteAsset(outMat);
        AssetDatabase.CreateAsset(mat, outMat);
        AssetDatabase.SaveAssets();
        Debug.Log("[BakeEnvSky] HDRI 스카이박스 베이크 완료: " + hdri);
    }

    // Assets/Env/tex 의 PBR 콘크리트(diffuse+normal) → Resources/GroundMat·WallMat(URP Lit) 베이크.
    static void BakeSurfaces() {
        BakeMat("GroundMat", "Assets/Env/tex/ground_diff.png", "Assets/Env/tex/ground_nor.png", new Vector2(6, 6), 0.12f);
        BakeMat("WallMat",   "Assets/Env/tex/wall_diff.png",   "Assets/Env/tex/wall_nor.png",   new Vector2(2.5f, 3.5f), 0.10f);
    }

    static void BakeMat(string matName, string diffPath, string norPath, Vector2 tiling, float smooth) {
        string outMat = "Assets/Resources/" + matName + ".mat";
        if (System.IO.File.Exists(Application.dataPath + diffPath.Substring("Assets".Length)))
            AssetDatabase.ImportAsset(diffPath, ImportAssetOptions.ForceSynchronousImport);
        var diff = AssetDatabase.LoadAssetAtPath<Texture2D>(diffPath);
        if (diff == null) { if (AssetDatabase.LoadAssetAtPath<Material>(outMat) != null) AssetDatabase.DeleteAsset(outMat); return; }
        // 노멀맵 임포트 타입 보정
        if (AssetImporter.GetAtPath(norPath) is TextureImporter ti && ti.textureType != TextureImporterType.NormalMap) {
            ti.textureType = TextureImporterType.NormalMap; ti.SaveAndReimport();
        }
        var nor = AssetDatabase.LoadAssetAtPath<Texture2D>(norPath);
        var sh = Shader.Find("Universal Render Pipeline/Lit");
        if (sh == null) return;
        var mat = new Material(sh);
        mat.SetTexture("_BaseMap", diff); mat.SetTextureScale("_BaseMap", tiling);
        if (nor != null) { mat.SetTexture("_BumpMap", nor); mat.SetTextureScale("_BumpMap", tiling); mat.EnableKeyword("_NORMALMAP"); }
        mat.SetFloat("_Smoothness", smooth); mat.SetFloat("_Metallic", 0f);
        System.IO.Directory.CreateDirectory(Application.dataPath + "/Resources");
        if (AssetDatabase.LoadAssetAtPath<Material>(outMat) != null) AssetDatabase.DeleteAsset(outMat);
        AssetDatabase.CreateAsset(mat, outMat);
        AssetDatabase.SaveAssets();
        Debug.Log("[BakeSurfaces] " + matName + " 베이크 완료");
    }

    // Assets/Characters 의 FBX(Mixamo) → Humanoid 임포트 + 클립 루프 + AnimatorController → Resources.
    static void BakeCharacter() {
        const string outModel = "Assets/Resources/OperatorModel.fbx";
        const string outCtrl  = "Assets/Resources/OperatorAnim.controller";
        string srcDir = Application.dataPath + "/Characters";
        string fbx = null;
        if (System.IO.Directory.Exists(srcDir))
            foreach (var f in System.IO.Directory.GetFiles(srcDir))
                if (f.EndsWith(".fbx")) { fbx = f; break; }
        if (fbx == null) {
            if (AssetDatabase.LoadMainAssetAtPath(outModel) != null) AssetDatabase.DeleteAsset(outModel);
            if (AssetDatabase.LoadMainAssetAtPath(outCtrl) != null) AssetDatabase.DeleteAsset(outCtrl);
            return;
        }
        System.IO.Directory.CreateDirectory(Application.dataPath + "/Resources");
        System.IO.File.Copy(fbx, Application.dataPath + "/Resources/OperatorModel.fbx", true);
        AssetDatabase.ImportAsset(outModel, ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
        var mi = AssetImporter.GetAtPath(outModel) as ModelImporter;
        if (mi != null) {
            mi.animationType = ModelImporterAnimationType.Human;
            mi.materialImportMode = ModelImporterMaterialImportMode.ImportStandard;   // 머티리얼 생성
            var clips = mi.defaultClipAnimations;
            for (int i = 0; i < clips.Length; i++) clips[i].loopTime = true;
            if (clips.Length > 0) mi.clipAnimations = clips;
            mi.SaveAndReimport();
            // FBX 내장 텍스처 추출 → 머티리얼에 색/텍스처 적용(흰색 방지)
            string texDir = "Assets/Resources/CharTex";
            System.IO.Directory.CreateDirectory(Application.dataPath + "/Resources/CharTex");
            mi.ExtractTextures(texDir);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            mi.SaveAndReimport();
            // 추출된 diffuse/normal을 고정 이름(CharDiffuse/CharNormal)으로 복사 → 런타임에 머티리얼 직접 적용(바인딩 누락 우회)
            string ctxDir = Application.dataPath + "/Resources/CharTex";
            if (System.IO.Directory.Exists(ctxDir))
                foreach (var tf in System.IO.Directory.GetFiles(ctxDir)) {
                    if (!tf.EndsWith(".png")) continue;
                    if (tf.Contains("Diffuse")) System.IO.File.Copy(tf, Application.dataPath + "/Resources/CharDiffuse.png", true);
                    else if (tf.Contains("Normal")) System.IO.File.Copy(tf, Application.dataPath + "/Resources/CharNormal.png", true);
                }
            AssetDatabase.ImportAsset("Assets/Resources/CharDiffuse.png", ImportAssetOptions.ForceSynchronousImport);
            AssetDatabase.ImportAsset("Assets/Resources/CharNormal.png", ImportAssetOptions.ForceSynchronousImport);
            if (AssetImporter.GetAtPath("Assets/Resources/CharNormal.png") is TextureImporter cni && cni.textureType != TextureImporterType.NormalMap) {
                cni.textureType = TextureImporterType.NormalMap; cni.SaveAndReimport();
            }
        }
        AnimationClip clip = null;
        foreach (var a in AssetDatabase.LoadAllAssetsAtPath(outModel))
            if (a is AnimationClip ac && !ac.name.StartsWith("__")) { clip = ac; break; }
        if (clip != null) {
            if (AssetDatabase.LoadMainAssetAtPath(outCtrl) != null) AssetDatabase.DeleteAsset(outCtrl);
            UnityEditor.Animations.AnimatorController.CreateAnimatorControllerAtPathWithClip(outCtrl, clip);
        }
        AssetDatabase.SaveAssets();
        Debug.Log("[BakeCharacter] 캐릭터 베이크: " + System.IO.Path.GetFileName(fbx) + "  clip=" + (clip != null ? clip.name : "none"));
    }
}
#endif
