# AI Tilt — 강화학습(ML-Agents) 빌드업 핸드오프

검증된 `physics_v3`(40/40 PASS)를 **보상 엔진**으로 그대로 쓰고, 그 위에 ML-Agents 강화학습을 올린 구성입니다. Gemini 초안의 핵심 결함(검증 물리 폐기, 단일 God-class, 에피소드/정규화 부재)을 바로잡아 재설계했습니다.

## 0. 지금까지 자동으로 끝낸 것 ✅

- **ML-Agents ↔ Unity 6.5 호환성 검증** — `com.unity.ml-agents@4.0.3`(Unity 6000.0+ 요구) ↔ Python `mlagents==1.1.0`, Python 3.10.12.
- **패키지 설치** — `Packages/manifest.json`에 `com.unity.ml-agents: 4.0.3` 추가 → Unity가 정상 해석(`com.unity.ai.inference` 2.6.1 동반). **컴파일 에러 0**.
- **RL 아키텍처 4개 스크립트 작성 + 컴파일 통과**:
  - `Assets/Scripts/SolarDayEnv.cs` — scene_data.json 태양 타임라인 보간 + 도메인 랜덤화
  - `Assets/Scripts/LouverAgent.cs` — Agent(관측 8 / 연속행동 1 / 보상=검증 EffPoa−구동비용)
  - `Assets/Scripts/LouverAgentPresenter.cs` — 블레이드 생성 + HUD(Agent vs Oracle)
  - `Assets/Editor/SetupRLScene.cs` — 메뉴 한 번에 씬 배선
- **트레이너 config / Python 셋업 스크립트** 작성(`rl/`).

## 1. 설계 — "또 83°로 픽스되지 않게"

순간 발전량만 보상하면 최적 정책은 "항상 oracle 최적각"이라 RL이 상수로 수렴합니다(학습 곡선 평평). 그래서 **보상源은 검증 물리로 두되, 환경을 비정상·제약조건부**로 만들어 정책이 상수가 될 수 없게 했습니다:

1. **이동하는 태양** — 에피소드(=하루, 120스텝) 동안 06:00→19:00 태양이 이동. 추종 목표가 매 스텝 변함.
2. **모터 속도 한계** — 행동은 각속도(스텝당 ±3°). 순간이동 불가 → 램프·예측 필요.
3. **구동비용 패널티** — `wMotor`로 불필요한 떨림을 벌점화 → 트레이드오프 발생.
4. **도메인 랜덤화** — 에피소드마다 구름(DNI/DHI)·현/피치비·알베도를 바꿈 → 매번 최적각이 달라져 **관측→각도 함수**를 학습해야 함. 이게 학습 곡선을 실제로 움직이는 핵심.

**보상** = `wEnergy·(EffPoa/350) − wMotor·(|Δ각|/3)`, 하루 끝에 oracle 상한 대비 추종률 보너스. oracle은 **평가 지표(상한)**로만 쓰고 보상에 직접 넣지 않습니다(베끼기 방지).

**관측(8)**: 태양고도, 방위 sin/cos, DNI, DHI, 현재각, 현/피치비, 하루위상.

## 2. 남은 단계 — 사람이 해야 하는 것

자동화로 끝낼 수 없는(당신 머신·계정·터미널이 필요한) 부분입니다.

### A. RL 씬 배선 (클릭 1번)
Unity 메뉴 **AI Tilt → Setup RL Scene (Train)**.
→ LouverRoot에 LouverAgent + BehaviorParameters(LouverTilt, obs 8, act 1) + DecisionRequester + Presenter 자동 배선, 뷰어 컴포넌트는 충돌 방지로 제거.
> 키보드로 먼저 감 잡고 싶으면 **Setup RL Scene (Heuristic Test)** → ▶ Play → ←/→ 키로 블레이드 조작(파이썬 불필요).

### B. Python 학습 환경 (터미널)
```bash
bash rl/setup_mlagents.sh      # conda env + mlagents==1.1.0 + torch
```
Conda 미설치면 먼저: https://docs.conda.io (miniforge 권장, Apple Silicon).

### C. 학습 실행
```bash
cd /Volumes/AISSD/ai-tilt/unity_viz
conda activate mlagents
mlagents-learn rl/config/louver_ppo.yaml --run-id=louver01
```
"Play 버튼을 누르라"는 메시지가 뜨면 → Unity에서 ▶ Play. (BehaviorType=Default여야 연결됨)

### D. 보상곡선·평가
```bash
tensorboard --logdir results     # http://localhost:6006
```
- `LouverTilt/Environment/Cumulative Reward` 우상향 = 학습 중.
- HUD의 **하루 추종률(oracle 상한 대비 %)** 90%+면 양호.

### E. 학습된 모델 적용
`results/louver01/LouverTilt.onnx` → BehaviorParameters의 **Model**에 드래그, BehaviorType=**Inference Only** → ▶ Play로 추론 시연.

## 3. 튜닝 메모

- 곡선이 평평 → `domainRandomize` 범위 축소(쉬운 커리큘럼부터) 또는 `wMotor`↓ / `beta`↑(탐험).
- 떨림이 심함 → `wMotor`↑ 또는 `maxDeltaDegPerStep`↓.
- 보상 스케일 기준 `poaRef=350`(최대 poa_oracle≈343). 구름 랜덤화로 더 낮은 일사면 자동 정규화됨.
- `OracleTilt`를 매 스텝 호출(91회 EffPoa)하므로 대규모 병렬학습 시 평가만 코어스닝하면 빠름.

## 4. 검증 물리 API (재사용 지점)

`LouverPhysics`(static):
- `EffPoa(tilt, elev, az, dni, dhi, albedo, chord, pitch)` → 유효 POA(보상源)
- `OracleTilt(elev, az, dni, dhi, albedo, chord, pitch)` → 그리드 argmax 최적각(상한)
- `PanelSf(...)`/`StripShade(...)` → 자기음영(EffPoa 내부에서 이미 반영)
- `Load(streamingAssetsDir)` / `SelfTest(dir)` → 테이블 로드 / 40-벡터 교차검증

Agent의 `Initialize()`에서 `Load` + `SelfTest`를 1회 호출해 **보상이 진짜 physics_v3와 일치함**을 매 실행 게이트합니다.
