# AI Tilt — Unity 시각화 마스터 진행계획 (세션 연속용)

> 재개 트리거: **"AI Tilt 유니티 진행"** → 이 파일부터 읽기.
> 원칙: **Unity = 뷰어, oracle(physics_v3) = 진실.** 물리는 Python이 굽고 Unity는 렌더만.
> 검증 원칙: by-construction 금지 — 각도는 그리드 argmax 실측, C# 포팅은 Python 검증벡터 대조.

## 목표
BIPV 루버 AI 각도제어를 **오프라인 스탠드얼론 데모**로. 시연 = 내가 통제하는 Mac에
`.app` 복사 → 우클릭-열기(미서명 Gatekeeper 1회 우회) → 인터넷/Python 없이 실행.
(왜 80°가 최적인지 보여주는 현상설명 + 우리현장 탐색 샌드박스)

## 단계 현황
| # | 단계 | 담당 | 상태 |
|---|---|---|---|
| 0 | Unity 설치(Hub+2022.3 LTS Silicon+macOS Build Support) | **사용자**(계정로그인) | ⬜ 대기 |
| 1 | 베이크 브리지 + scene_data.json + C# 드라이버 + 가이드 | Claude | ✅ 완료(d34b31e) |
| 5a | physics_v3 → C# 포팅(LouverPhysics) + Python 검증벡터 대조 | Claude | ✅ 완료(40/40 PASS, POA 0.002%) |
| 5b | 샌드박스 패널(알베도/현/피치/비교각 실시간 재계산) | Claude | ✅ 완료(로직 Python 검증) |
| 1b | 씬 제작 + 배선 + Play(첫 실행 디버그: 거울상/축) | 사용자+Claude | ⬜ Unity 설치 후 |
| 2 | 실시간 자기그림자(URP) + 블레이드 PBR 재질 | Claude(코드)+사용자(에디터) | ⬜ |
| 3 | 데이터 오버레이(블레이드 음영색·발전 막대·하루 곡선) | Claude | ⬜ |
| 4 | 현상설명 모드(가이드 카메라+캡션: 저각자기음영/80°균형/90°빔스침) | Claude | ⬜ |
| 6 | 스탠드얼론 macOS .app 빌드 → 데모 노트북 복사 | 사용자 | ⬜ |

**다음 행동**: 사용자가 0단계(Unity 설치) → README_UNITY.md 1단계 클릭순서로 씬 만들고
`SolarLouverViz`(애니) + `SolarLouverSandbox`(탐색) 부착 → Play. 그다음 2~4단계.

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
cd /Users/wonetiago/projects/260310-BIPVpatenttest
.venv/bin/python unity_viz/bake_scene.py                  # scene_data.json
.venv/bin/python unity_viz/gen_csharp_tables.py           # vf_flat/iam/testvectors
```

## 검증 상태 (Unity 없이 가능한 한도)
- ✅ 포팅 로직: C# 수식을 Python으로 1:1 재현 → 검증벡터 40/40 PASS(POA 상대 0.002%, sf 1.2e-6)
- ✅ 샌드박스 로직: scene_data 위 재계산이 물리 정합(albedo↑→최적각↑·이득↑, ratio↓→최적각↓, AIvs최고 +0.1~0.5% 천장)
- ⬜ C# 컴파일/런타임: Unity 설치 후 첫 Play에서 SelfTest 로그 + 거울상/축 디버그 필요
- ⚠️ 한계: 2D 무한슬랫(한 시점 전 블레이드 동일각·동일음영, 끝단효과 미모델)

## 열린 질문
- 어느 시나리오를 구울지(기본=하지근접 6/21맑음, +12.2% vs60). 겨울/대표일 추가 베이크 가능.
- 날짜를 샌드박스에서 바꾸려면 태양위치 알고리즘(SPA) C# 포팅 필요 → 지금은 베이크 재실행으로 충분.
- 완전 클린 배포(아무 Mac 경고0)는 Apple Developer 서명($99/년) — 현재 불필요.
