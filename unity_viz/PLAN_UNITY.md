# AI Tilt — Unity 시각화 마스터 진행계획 (세션 연속용)

> 재개 트리거: **"AI Tilt 유니티 진행"** → 이 파일부터 읽기.
> 원칙: **Unity = 뷰어, 검증물리(physics_v3 → LouverPhysics) = 진실.** 보상源·oracle 모두 검증물리.
> 검증 원칙: by-construction 금지 — 각도는 그리드 argmax 실측, C# 포팅은 Python 검증벡터 대조.

## 목표
BIPV 루버 AI 각도제어를 **오프라인 스탠드얼론 데모**로. 시연 = 내가 통제하는 Mac에
`.app` 복사 → 우클릭-열기(미서명 Gatekeeper 1회 우회) → 인터넷/Python 없이 실행.
(AI가 태양을 따라 스스로 각도를 학습하는 과정 + 검증물리 대비 추종률을 보여줌)

## 두 버전 (★현재 = 신버전 RL)
- **구버전 (베이크 뷰어)**: `physics_v3` → `scene_data.json` 베이크 → `SolarLouverViz`가 하루를 재생.
  oracle각이 사실상 상수(83°)라 "학습"의 의미가 약하고 비주얼도 빈약 → **보류**(코드는 유효, 참조용).
- **신버전 (RL 학습환경)**: ML-Agents(Unity 6 + `com.unity.ml-agents@4.0.3` ↔ `mlagents==1.1.0`).
  검증물리(`LouverPhysics.EffPoa`)를 **보상源**으로 두고, 이동태양·모터속도한계·구동비용·도메인랜덤화로
  "상수 83° 베끼기"를 차단 → 관측→각도 함수를 실제로 학습. **시연용 비주얼 고도화 진행 중.**
  설계 상세 = `rl/AI_TILT_RL_HANDOFF.md`.

## 단계 현황
| # | 단계 | 담당 | 상태 |
|---|---|---|---|
| 0 | Unity 설치(Hub + **Unity 6 (6000.0)** Silicon + macOS Build Support) | 사용자(계정로그인) | ✅ 완료 |
| 1 | 베이크 브리지 + scene_data.json + C# 드라이버 + 가이드 (구버전) | Claude | ✅ 완료(d34b31e) |
| 5a | physics_v3 → C# 포팅(LouverPhysics) + Python 검증벡터 대조 | Claude | ✅ 완료(40/40 PASS, POA 0.002%) |
| 5b | 샌드박스 패널(알베도/현/피치/비교각 실시간 재계산) | Claude | ✅ 완료(로직 Python 검증) |
| R1 | RL 환경 4스크립트(SolarDayEnv/LouverAgent/Presenter/SetupRLScene) + 컴파일 | Claude | ✅ 완료 |
| R2 | sun_days.json(기상청 10년·서울 3652일) 베이크 → 실제 하루 학습 | Claude | ✅ 완료 |
| R3 | ML-Agents 학습 실행 → `results/louver_demo/LouverTilt.onnx` | 사용자(터미널) | ✅ 완료(3.0M스텝, 보상 77.7) |
| R4 | 스탠드얼론 macOS .app 빌드(`Build/AITiltRL.app`) | 사용자 | ✅ 완료 |
| V1 | 비주얼 고도화: 사실적 스카이/태양·자기음영·반사프로브·외벽·작업자·포스트프로세싱 | Claude | ✅ 1차 완료(Presenter) |
| V2 | 데이터 오버레이(발전 막대·하루 POA 곡선 AI vs oracle) + PV 셀 텍스처 | Claude | 🔄 진행 중 |
| V3 | 현상설명 모드(가이드 카메라+캡션: 저각자기음영/80°균형/90°빔스침) | Claude | ⬜ |

**다음 행동(사용자)**: Unity에서 `AITiltViz` 열기 → **AI Tilt → Setup RL Scene** 메뉴로 씬 배선 →
학습된 `LouverTilt.onnx`를 BehaviorParameters의 Model에 넣고 Inference로 Play(또는 `Build/AITiltRL.app` 실행).
비주얼 코드(V2)를 갱신했으면 **Play(에디터)** 또는 **재빌드**해야 반영됨(런타임 절차적 생성).

## 파일맵 (unity_viz/)
- `bake_scene.py` — physics_v3 → scene_data.json. `--date YYYY-MM-DD --baseline 60`
- `gen_csharp_tables.py` — C#용 테이블/검증벡터 생성(아래 3개 산출)
- `scene_data.json` — 하루 타임라인(StreamingAssets로 복사)
- `vf_flat_v17.json` / `iam_diffuse_v17.json` — C# 물리 테이블(StreamingAssets)
- `physics_testvectors.json` — C# SelfTest용 40벡터(StreamingAssets)
- `SolarLouverViz.cs` — 베이크 하루 애니(태양이동+루버회전+HUD)
- `LouverPhysics.cs` — physics_v3 C# 포팅(panel_sf·VF·IAM·eff_poa·oracle) + SelfTest()
- `SolarLouverSandbox.cs` — 실시간 탐색 패널(LouverPhysics 사용)
- `SolarLouverViz.cs`/`Sandbox.cs` + json 4개 → **Assets/Scripts/** 와 **Assets/StreamingAssets/** 로 복사

## 재현 (프로젝트 .venv)
```bash
cd /Volumes/AISSD/ai-tilt
.venv/bin/python unity_viz/bake_scene.py                  # scene_data.json
.venv/bin/python unity_viz/gen_csharp_tables.py           # vf_flat/iam/testvectors
```

## 검증 상태 (Unity 없이 가능한 한도)
- ✅ 포팅 로직: C# 수식을 Python으로 1:1 재현 → 검증벡터 40/40 PASS(POA 상대 0.002%, sf 1.2e-6)
- ✅ 샌드박스 로직: scene_data 위 재계산이 물리 정합(albedo↑→최적각↑·이득↑, ratio↓→최적각↓, AIvs최고 +0.1~0.5% 천장)
- ✅ C# 컴파일/런타임: Unity 6에서 컴파일 0에러 + RL 학습 3.0M스텝 완주 + .app 빌드 성공
  (`LouverPhysics.SelfTest`가 Agent.Initialize에서 매 실행 자동 검증)
- ⚠️ 한계: 2D 무한슬랫(한 시점 전 블레이드 동일각·동일음영, 끝단효과 미모델)

## 열린 질문
- 어느 시나리오를 구울지(기본=하지근접 6/21맑음, +12.2% vs60). 겨울/대표일 추가 베이크 가능.
- 날짜를 샌드박스에서 바꾸려면 태양위치 알고리즘(SPA) C# 포팅 필요 → 지금은 베이크 재실행으로 충분.
- 완전 클린 배포(아무 Mac 경고0)는 Apple Developer 서명($99/년) — 현재 불필요.
