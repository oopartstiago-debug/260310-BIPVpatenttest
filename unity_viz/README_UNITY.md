# AI Tilt — Unity 시각화 (스탠드얼론 데모)

**아키텍처: Unity = 뷰어, oracle(physics_v3) = 진실.**
Python이 검증된 물리로 `scene_data.json`을 굽고, Unity는 그걸 읽어 렌더만 한다.
물리 계산이 게임엔진 근사에 오염되지 않는다.

```
[physics_v3] --bake_scene.py--> scene_data.json --(StreamingAssets)--> [Unity 씬] --build--> 데모.app
```

## 파일
- `bake_scene.py` — 베이크 브리지. `scene_data.json` 생성. (프로젝트 .venv로 실행)
- `scene_data.json` — 구워진 하루 타임라인(태양궤적·oracle각·음영·POA·누적에너지).
- `SolarLouverViz.cs` — Unity 드라이버(JSON 읽어 태양이동+루버회전+HUD).

### 베이크 다시 굽기
```bash
cd /Users/wonetiago/projects/260310-BIPVpatenttest
.venv/bin/python unity_viz/bake_scene.py                 # 하지 근접일, 60° 비교
.venv/bin/python unity_viz/bake_scene.py --date 2014-12-21 --baseline 45
```

---

## 0단계 — Unity 설치 (당신 손, 계정 로그인 필요)
1. https://unity.com/download 에서 **Unity Hub** 설치 (또는 `brew install --cask unity-hub`).
2. Hub 실행 → 무료 **Personal** 계정 로그인.
3. Installs → Install Editor → **2022.3 LTS (Silicon)** 선택.
   모듈에서 **macOS Build Support (IL2CPP)** 반드시 체크. (윈도우 데모도 필요하면 Windows Build Support 추가)

## 1단계 — 프로젝트 + 씬 만들기 (클릭 순서)
1. Hub → New Project → **3D (URP)** 템플릿 → 이름 `AITiltViz` → Create.
2. 프로젝트 창 Assets 아래에 폴더 두 개 생성:
   - `Assets/Scripts/`  ← `SolarLouverViz.cs`를 여기로 복사
   - `Assets/StreamingAssets/`  ← `scene_data.json`을 여기로 복사  (폴더명 정확히, 대소문자 주의)
3. 씬 구성 (Hierarchy 우클릭):
   - **3D Object → Plane** (바닥). Position (0,0,0).
   - **Create Empty** → 이름 `LouverRoot`. Position (0, 1, 0). (블레이드가 이 밑에 자동 생성됨)
   - 기본 **Directional Light**는 이미 있음 (없으면 Light → Directional Light).
4. 드라이버 부착:
   - `LouverRoot` 선택 → Inspector → **Add Component** → `Solar Louver Viz`.
   - 컴포넌트의 **Sun** 칸에 Hierarchy의 Directional Light를 드래그.
   - **Louver Root** 칸에 LouverRoot 자신을 드래그.
5. 카메라: Main Camera를 루버가 보이게 이동(예: Position (3, 1, -3), 살짝 루버 쪽 회전). 나중에 조정.
6. **Play ▶** — 태양이 하루를 돌고 루버가 oracle각으로 회전, 좌상단 HUD에 POA·이득% 표시.
   - HUD 버튼으로 "AI 최적 ↔ 고정 60°" 토글 → 60°에서 자기음영 그림자가 생기는 게 보임.

> 첫 실행에서 거울상(태양이 반대쪽)·블레이드 축이 어긋나면 알려줘. `SolarLouverViz.cs`에서
> 태양 방위 부호(`ar`) 또는 블레이드 회전축만 손보면 됨 — 같이 디버그.

## 2~3단계 (이후)
- 실시간 자기그림자(URP Shadow), 블레이드 PBR 재질, 발전 막대/곡선 UI, 그림자 토글.

## 6단계 — 스탠드얼론 빌드 (데모 파일)
1. File → Build Settings → 현재 씬 Add Open Scenes → Platform **macOS** → Build.
2. 나온 `.app`을 데모 노트북으로 복사 → **우클릭 → 열기**(미서명 Gatekeeper 1회 우회) → 오프라인 실행.

## 한계 (정직)
2D 무한슬랫 모델이라 **한 시점엔 전 블레이드가 동일각·동일음영**(끝단/측면 효과 미모델).
"블레이드별 색이 제각각"인 히트맵은 외부 불균일 음영 시나리오에서만 의미 — 기본 데모에선 다같이 회전.
