# AI Tilt — Stage 3/4 외부 에셋 가이드 (사용자 작업분)

V1·V2·V3 + 절차적 Stage 3(블레이드=PV 유리 모듈)까지는 코드로 완료. **더 사실적인** 단계는
외부 에셋/로그인이 필요해 아래만 사용자가 준비하면 내가 배선한다.

원칙: **FBX·OBJ 는 Unity 가 기본 임포트**(패키지 불필요). glTF 는 별도 패키지 필요 → 가능하면 FBX/OBJ.

---

## Stage 3 — 정교한 루버 3D (실 CAD 메시)

목표: 절차적 박스 블레이드 → bipv-cad 의 실제 블레이드 형상(프레임 프로파일·모서리·마운트)으로 교체.

### 경로 A (권장, 패키지 0) — STL → OBJ → Unity 기본 임포트
1. bipv-cad 에서 **블레이드 1개**를 STL 로 내보내기(`/Volumes/AISSD/bipv-cad`, env `bipv-cad`).
   - 단일 블레이드 메시면 충분(20개는 Unity가 인스턴싱). 단위 mm.
2. STL→OBJ 변환(트라이메시, 추가 설치 없이):
   ```bash
   /Volumes/AISSD/bipv-cad/.venv/bin/python -c "import trimesh; m=trimesh.load('blade.stl'); m.export('blade.obj')"
   ```
3. `blade.obj` 를 `unity_viz/AITiltViz/AITiltViz/Assets/Models/Louver/` 에 두기.
4. 나에게 "블레이드 OBJ 넣었어" → `LouverAgentPresenter.BuildBlades()` 가 박스 대신 그 메시를
   인스턴싱하도록 배선(스케일 mm→m, PV 머티리얼 적용, 피치 97.5 간격 유지). 물리/학습 무관(시각만).

### 경로 B — glTF 직접
- `Packages/manifest.json` 에 `"com.unity.cloud.gltfast"` 추가(내가 가능) 후 `.gltf` 드롭.
- bipv-cad CadQuery 가 glTF 내보내기를 지원하면 경로 A 의 변환 단계 생략.

### 부가(에디터 필요)
- 블레이드 PBR 유리 머티리얼(반투명·반사)·환경 HDRI 는 에디터에서 드롭/연결이 깔끔(헤드리스는 .meta GUID 위험).

---

## Stage 4 — 사람 캐릭터 (Mixamo)

목표: 절차적 프리미티브 작업자 → 리깅된 실제 캐릭터 + 애니메이션.

1. https://www.mixamo.com **Adobe 로그인**(사용자만 가능).
2. 캐릭터 1개 + 애니메이션(예: "Idle", "Standing", 콘솔 조작이면 "Typing"/"Looking Around") 선택.
3. **Download → FBX (with skin)**, 본 = 표준 휴머노이드.
4. `Assets/Characters/` 에 FBX 드롭 → Unity 기본 임포트. 임포트 설정 **Rig=Humanoid**, **Animation=Loop**.
5. 나에게 알려주면 `BuildCharacter()` 의 프리미티브를 끄고 그 모델을 배치(루버 좌측 바닥),
   Animator 로 애니 재생. 오른팔 각도 연동은 선택(휴머노이드 IK 필요 시 별도).

---

## 지금 바로 가능한(에셋 없는) 후속 — 내가 단독 진행 가능
- V3 심화: 겨울/여름 비교(계절별 최적각 이동 토글), 캡션 페이드 애니.
- 환경 절차적 디테일: 하늘 그라디언트/구름 강화, 바닥 반사, 난간/실외기 등 소품.
- 블레이드 베벨/프레임을 절차적으로 더 정교화(메시 직접 생성).

요청 시 위 중 무엇이든 바로 착수.
