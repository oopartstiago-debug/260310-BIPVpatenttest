// AI Tilt — 추론(Inference) 스탠드얼론 빌드 (CLI: -executeMethod BuildRLInference.PerformBuild)
// 학습된 LouverTilt.onnx 를 Assets 로 임포트 → ModelAsset 으로 로드 → BehaviorType.InferenceOnly 로 배선 →
// 스탠드얼론 .app 빌드. 이 빌드는 트레이너 없이도 학습된 정책으로 태양을 추종한다.
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using System.Collections.Generic;
using System.Reflection;

public static class BuildRLInference {
    const string ScenePath = "Assets/Scenes/SampleScene.unity";
    const string OutPath   = "/Volumes/AISSD/ai-tilt/unity_viz/Build/AITiltRL_Infer.app";
    // mlagents 최종 export 는 외부 가중치(.onnx.data)를 분리 저장 → onnx + data 쌍을 함께 복사해야 한다.
    const string OnnxSrc   = "/Volumes/AISSD/ai-tilt/unity_viz/results/louver_track/LouverTilt/LouverTilt-3000120.onnx";
    const string DataSrc   = "/Volumes/AISSD/ai-tilt/unity_viz/results/louver_track/LouverTilt/LouverTilt-3000120.onnx.data";
    const string AssetPath = "Assets/Models/LouverTilt-3000120.onnx";

    public static void PerformBuild() {
        // 0-) 런타임 Shader.Find 로 쓰는 셰이더가 빌드서 스트립돼 마젠타로 뜨지 않게 Always-Included 에 등록
        EnsureGltfastShaders();                    // gltfast 런타임 머티리얼 셰이더(실외기 실모델, 빌드서 스트립 방지=마젠타 회피)
        EnsureSSAO();                              // 화면공간 앰비언트 오클루전(접지 그림자=photoreal 핵심)
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

    // URP 렌더러에 SSAO(화면공간 앰비언트 오클루전) 렌더러 기능 주입 — 물체가 바닥·접합부에 접지 그림자를 받아 photoreal.
    //   헤드리스 주입이라 내부 맵 갱신이 불안정할 수 있어 try/catch 비치명적. 실패 시 에디터에서 1클릭 토글로 대체 가능.
    static void EnsureSSAO() {
        try {
            var guids = AssetDatabase.FindAssets("t:UniversalRendererData");
            if (guids.Length == 0) { Debug.LogWarning("[SSAO] UniversalRendererData 없음 — 건너뜀"); return; }
            var t = System.Type.GetType("UnityEngine.Rendering.Universal.ScreenSpaceAmbientOcclusion, Unity.RenderPipelines.Universal.Runtime");
            if (t == null) { Debug.LogWarning("[SSAO] 타입 못 찾음 — 건너뜀"); return; }
            foreach (var g in guids) {   // ★Mobile·PC 등 모든 렌더러에 주입(어느 게 활성이든 적용되게)
                string path = AssetDatabase.GUIDToAssetPath(g);
                var rd = AssetDatabase.LoadAssetAtPath<ScriptableRendererData>(path);
                if (rd == null) continue;
                bool has = false;
                foreach (var f in rd.rendererFeatures) if (f != null && f.GetType().Name.Contains("ScreenSpaceAmbientOcclusion")) has = true;
                if (has) { Debug.Log("[SSAO] 이미 있음: " + path); continue; }
                var feature = ScriptableObject.CreateInstance(t) as ScriptableRendererFeature;
                feature.name = "SSAO";
                rd.rendererFeatures.Add(feature);
                AssetDatabase.AddObjectToAsset(feature, rd);
                var so = new SerializedObject(rd);
                var map = so.FindProperty("m_RendererFeatureMap");
                var feats = so.FindProperty("m_RendererFeatures");
                if (map != null && feats != null) { while (map.arraySize < feats.arraySize) map.InsertArrayElementAtIndex(map.arraySize); so.ApplyModifiedProperties(); }
                var onValidate = typeof(ScriptableRendererData).GetMethod("OnValidate", BindingFlags.NonPublic | BindingFlags.Instance);
                onValidate?.Invoke(rd, null);
                EditorUtility.SetDirty(rd);
                Debug.Log("[SSAO] 렌더러 기능 추가: " + path);
            }
            AssetDatabase.SaveAssets(); AssetDatabase.Refresh();
        } catch (System.Exception e) { Debug.LogWarning("[SSAO] 자동 주입 실패(무시, 에디터 토글로 대체): " + e.Message); }
    }

    // gltfast 는 머티리얼을 런타임에 생성 → shadergraph 가 어떤 빌드 에셋에도 참조되지 않아 스트립됨(마젠타).
    // 패키지 경로에서 셰이더 에셋을 직접 로드해 Always-Included 에 등록(이름 추정 불필요·robust).
    static void EnsureGltfastShaders() {
        string[] paths = {
            "Packages/com.unity.cloud.gltfast/Runtime/Shader/glTF-pbrMetallicRoughness.shadergraph",
            "Packages/com.unity.cloud.gltfast/Runtime/Shader/glTF-unlit.shadergraph",
            "Packages/com.unity.cloud.gltfast/Runtime/Shader/glTF-pbrSpecularGlossiness.shadergraph",
        };
        var assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset");
        if (assets == null || assets.Length == 0) return;
        var so = new SerializedObject(assets[0]);
        var arr = so.FindProperty("m_AlwaysIncludedShaders");
        foreach (var p in paths) {
            var sh = AssetDatabase.LoadAssetAtPath<Shader>(p);
            if (sh == null) { Debug.LogWarning("[gltfast] 셰이더 못 찾음: " + p); continue; }
            bool has = false;
            for (int i = 0; i < arr.arraySize; i++)
                if (arr.GetArrayElementAtIndex(i).objectReferenceValue == sh) { has = true; break; }
            if (has) continue;
            int idx = arr.arraySize;
            arr.InsertArrayElementAtIndex(idx);
            arr.GetArrayElementAtIndex(idx).objectReferenceValue = sh;
            Debug.Log("[gltfast] Always-Included 추가: " + sh.name);
        }
        so.ApplyModifiedProperties();
        AssetDatabase.SaveAssets();
    }

    // Assets/Env 의 첫 HDRI 를 Skybox/Panoramic 머티리얼로 구워 Resources/EnvSky.mat 에 저장.
    // 파일이 없으면 기존 EnvSky.mat 제거(→ 프레젠터가 절차적 하늘로 폴백). HDRI 드롭인 = 자동 적용.
    static void BakeEnvSky() {
        const string outMat = "Assets/Resources/EnvSky.mat";
        string hdri = null, fallback = null;
        if (System.IO.Directory.Exists(Application.dataPath + "/Env"))
            foreach (var f in System.IO.Directory.GetFiles(Application.dataPath + "/Env"))
                if (f.EndsWith(".hdr") || f.EndsWith(".exr")) {
                    string ap = "Assets/Env/" + System.IO.Path.GetFileName(f), ln = System.IO.Path.GetFileName(f).ToLower();
                    // 옥상 도시 전경 위해 도시/옥상 HDRI 우선(없으면 첫 파일)
                    if (ln.Contains("platz") || ln.Contains("city") || ln.Contains("urban") || ln.Contains("rooftop") || ln.Contains("courtyard")) { hdri = ap; break; }
                    if (fallback == null) fallback = ap;
                }
        if (hdri == null) hdri = fallback;
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
        BakeMat("UnitMat",   "Assets/Env/tex/unit_diff.png",   "Assets/Env/tex/unit_nor.png",   new Vector2(1.5f, 1.5f), 0.55f, 0.75f);  // 실외기=브러시드 메탈(CC0 ambientCG)
    }

    static void BakeMat(string matName, string diffPath, string norPath, Vector2 tiling, float smooth, float metallic = 0f) {
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
        mat.SetFloat("_Smoothness", smooth); mat.SetFloat("_Metallic", metallic);
        System.IO.Directory.CreateDirectory(Application.dataPath + "/Resources");
        if (AssetDatabase.LoadAssetAtPath<Material>(outMat) != null) AssetDatabase.DeleteAsset(outMat);
        AssetDatabase.CreateAsset(mat, outMat);
        AssetDatabase.SaveAssets();
        Debug.Log("[BakeSurfaces] " + matName + " 베이크 완료");
    }

    // fbx 를 Humanoid 로 임포트. sourceAvatar 가 있으면 Copy From Other(anim-only=Without Skin 리타게팅),
    // 없으면 CreateFromThisModel(스킨 모델은 자기 아바타 생성). 클립 loopTime=true.
    static void ImportFbxHuman(string assetPath, Avatar sourceAvatar) {
        var mi = AssetImporter.GetAtPath(assetPath) as ModelImporter;
        if (mi == null) return;
        mi.animationType = ModelImporterAnimationType.Human;
        mi.materialImportMode = ModelImporterMaterialImportMode.ImportStandard;
        if (sourceAvatar != null) { mi.avatarSetup = ModelImporterAvatarSetup.CopyFromOther; mi.sourceAvatar = sourceAvatar; }
        else mi.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
        var clips = mi.defaultClipAnimations;
        for (int i = 0; i < clips.Length; i++) clips[i].loopTime = true;
        if (clips.Length > 0) mi.clipAnimations = clips;
        mi.SaveAndReimport();
    }

    static AnimationClip FirstClip(string assetPath) {
        foreach (var a in AssetDatabase.LoadAllAssetsAtPath(assetPath))
            if (a is AnimationClip ac && !ac.name.StartsWith("__")) return ac;
        return null;
    }
    static Avatar FirstAvatar(string assetPath) {
        foreach (var a in AssetDatabase.LoadAllAssetsAtPath(assetPath))
            if (a is Avatar av) return av;
        return null;
    }

    // Assets/Characters 의 FBX(Mixamo) → Humanoid 임포트 + 순환 AnimatorController → Resources.
    //   • 스킨 모델(가장 큰 fbx = with-skin)을 OperatorModel 로, 그 아바타를 생성
    //   • 나머지 anim-only(Without Skin) fbx 는 Copy From Other 로 그 아바타를 복사 → Humanoid 리타게팅
    //   • idle(스킨 모델 클립) + 작업 클립들을 exit-time 으로 번갈아 재생('뚝딱뚝딱')
    static void BakeCharacter() {
        const string outModel = "Assets/Resources/OperatorModel.fbx";
        const string outCtrl  = "Assets/Resources/OperatorAnim.controller";
        string srcDir = Application.dataPath + "/Characters";
        var fbxFiles = new List<string>();
        if (System.IO.Directory.Exists(srcDir))
            foreach (var f in System.IO.Directory.GetFiles(srcDir))
                if (f.EndsWith(".fbx")) fbxFiles.Add(f);
        if (fbxFiles.Count == 0) {
            if (AssetDatabase.LoadMainAssetAtPath(outModel) != null) AssetDatabase.DeleteAsset(outModel);
            if (AssetDatabase.LoadMainAssetAtPath(outCtrl) != null) AssetDatabase.DeleteAsset(outCtrl);
            return;
        }
        // 스킨 모델 = 가장 큰 fbx(mesh+텍스처 포함 = with-skin). anim-only(Without Skin)는 용량이 작음.
        fbxFiles.Sort((a, b) => new System.IO.FileInfo(b).Length.CompareTo(new System.IO.FileInfo(a).Length));
        string skinFbx = fbxFiles[0];

        // 1) 스킨 모델 → Resources/OperatorModel.fbx, Humanoid(CreateFromThisModel) + 텍스처 추출(흰색 방지)
        System.IO.Directory.CreateDirectory(Application.dataPath + "/Resources");
        System.IO.File.Copy(skinFbx, Application.dataPath + "/Resources/OperatorModel.fbx", true);
        AssetDatabase.ImportAsset(outModel, ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
        ImportFbxHuman(outModel, null);
        var smi = AssetImporter.GetAtPath(outModel) as ModelImporter;
        if (smi != null) {
            string texDir = "Assets/Resources/CharTex";
            System.IO.Directory.CreateDirectory(Application.dataPath + "/Resources/CharTex");
            smi.ExtractTextures(texDir);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            smi.SaveAndReimport();
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
        Avatar srcAvatar = FirstAvatar(outModel);      // 작업 클립 리타게팅용 소스 아바타(스킨 모델)
        AnimationClip idleClip = FirstClip(outModel);   // 기본(쉬는) idle = 스킨 모델 클립

        // 2) anim-only(Without Skin) fbx → Copy From Other(srcAvatar) 로 임포트 → 작업 클립 수집
        var work = new List<AnimationClip>();
        foreach (var f in fbxFiles) {
            if (f == skinFbx) continue;
            string ap = "Assets/Characters/" + System.IO.Path.GetFileName(f);
            ImportFbxHuman(ap, srcAvatar);
            var c = FirstClip(ap);
            if (c != null) work.Add(c);
        }

        // 3) 재생 순서 = idle(기본) + 작업 클립들
        var seq = new List<AnimationClip>();
        if (idleClip != null) seq.Add(idleClip);
        seq.AddRange(work);
        if (seq.Count == 0) { Debug.LogWarning("[BakeCharacter] 클립 없음"); return; }

        // 4) AnimatorController — 1개면 루프, 여러 개면 exit-time 순환(각 클립 끝까지 재생 후 다음)
        if (AssetDatabase.LoadMainAssetAtPath(outCtrl) != null) AssetDatabase.DeleteAsset(outCtrl);
        var ctrl = UnityEditor.Animations.AnimatorController.CreateAnimatorControllerAtPathWithClip(outCtrl, seq[0]);
        if (seq.Count > 1) {
            var sm = ctrl.layers[0].stateMachine;
            var states = new List<UnityEditor.Animations.AnimatorState>();
            states.Add(sm.defaultState); states[0].motion = seq[0];
            for (int i = 1; i < seq.Count; i++) {
                var st = sm.AddState(seq[i].name + "_" + i, new Vector3(260, 60 * i, 0));
                st.motion = seq[i]; states.Add(st);
            }
            for (int i = 0; i < states.Count; i++) {   // 각 상태 끝까지 재생 후 다음으로 → 분주한 작업 루프
                var tr = states[i].AddTransition(states[(i + 1) % states.Count]);
                tr.hasExitTime = true; tr.exitTime = 0.92f; tr.hasFixedDuration = true; tr.duration = 0.3f;
            }
        }
        AssetDatabase.SaveAssets();
        Debug.Log($"[BakeCharacter] 스킨={System.IO.Path.GetFileName(skinFbx)} idle={(idleClip!=null?1:0)} 작업클립={work.Count} 총상태={seq.Count}");
    }
}
#endif
