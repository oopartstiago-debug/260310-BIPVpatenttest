# Mixamo 캐릭터 받아 AI Tilt 데모에 넣기 (작업 가이드 · 코웍 공유용)

## 목적
Unity 옥상 BIPV(건물일체형 태양광) 루버 데모에 등장할 **현장 작업자(기술자) 캐릭터**가 필요합니다.
요구사항: **리깅된 휴머노이드 캐릭터 + 자연스러운 대기/조작 애니메이션**, 포맷 **FBX**(Unity 기본 임포트).

## 1) 로그인
- https://www.mixamo.com 접속 → 우상단 **Log in**.
- **Adobe ID 필요(무료)**. 없으면 "Create an account"로 무료 가입(이메일만). Creative Cloud 유료 구독 불필요 — Mixamo는 무료입니다.

## 2) 캐릭터 선택 (Characters 탭)
- 상단 **Characters** → 검색창에 `worker`, `business`, `suit`, `casual` 등.
- 현장 작업자 느낌에 가장 가까운 캐릭터 선택(예: Mixamo 기본 캐릭터 중 작업복/캐주얼). 안전모·형광조끼가 정확히 맞는 기본 캐릭터는 없을 수 있으니 **가장 근접한 것**을 고르면 됩니다(색/디테일은 Unity에서 보정 가능). 자체 캐릭터 FBX 업로드도 가능.
- 캐릭터 클릭 → 우측 프리뷰에 적용.

## 3) 애니메이션 선택 (Animations 탭)
- 상단 **Animations** → 검색: `Idle`, `Looking Around`, `Standing`, `Talking On Phone`(콘솔 조작 느낌).
- 데모엔 **서서 살피는 동작**이 어울림 → **Idle** 또는 **Looking Around** 추천. 프리뷰로 자연스러움 확인.
- **In Place** 체크 = 제자리 동작(이동 안 함). 데모엔 제자리가 맞습니다.

## 4) 다운로드 설정 (★중요)
우상단 **Download** → 대화상자에서:
- **Format**: `FBX Binary (.fbx)` (또는 "FBX for Unity"가 보이면 그것)
- **Skin**: `With Skin` (메시 포함 — 필수)
- **Frames per Second**: `30`
- **Keyframe Reduction**: `none`
→ **Download**. 캐릭터+스켈레톤+스킨+애니가 든 `.fbx` 1개가 받아집니다.

## 5) 파일 둘 위치
받은 `.fbx`를 프로젝트 폴더에 넣어주세요(없으면 폴더 생성):
```
/Volumes/AISSD/ai-tilt/unity_viz/AITiltViz/AITiltViz/Assets/Characters/
```

## 6) 그다음 (내가 처리)
파일 넣고 "캐릭터 넣었어"라고 알려주시면 제가:
- Unity 임포트 설정 **Rig = Humanoid**, 애니 **Loop** 설정
- 기존 프리미티브 작업자 제거 → 이 캐릭터를 루버 좌측에 배치, Animator로 애니 재생
- 스케일(보통 cm→m)·바닥 정렬 보정 후 재빌드·캡처 검증

## 참고
- Mixamo 캐릭터는 시각화용으로 충분히 사실적입니다. 더 고품질 실사 인물이 필요하면 RenderPeople(유료) 등도 있으나, **애니메이션까지 무료로 한 번에** 되는 건 Mixamo가 최선입니다.

---

## 7) ★작업자 '뚝딱뚝딱' 동작 추가 (지금 필요한 것)
현재 캐릭터(Timmy)는 **idle(가만히 서 있기) 1개**라 분주해 보이지 않습니다.
**작업하는 동작 클립**을 추가로 받아 넣으면, 코드가 자동으로 **여러 동작을 번갈아 재생**(작업1→작업2→…→idle→반복)합니다.

### 받는 법 (같은 Timmy 캐릭터 그대로)
1. Mixamo 로그인 상태에서 **Characters에서 Timmy(또는 현재 쓰는 캐릭터)를 그대로 선택**해 둡니다(스킨은 이미 있으니 애니만 받으면 됩니다).
2. **Animations 탭**에서 아래 동작들을 검색해 자연스러운 걸 고릅니다(2~3개 추천):
   - `Using Tablet` / `Typing` — 콘솔·태블릿 점검하는 느낌(이 데모에 가장 잘 맞음)
   - `Hammering` / `Plastering` — 손으로 작업하는 동작(뚝딱뚝딱)
   - `Looking Around` / `Inspection` — 둘러보며 점검
   - `Talking On Phone` — 무전/통화
3. 각 동작마다 **In Place 체크**(제자리, 이동 안 함).
4. **Download** 설정 — ★작업 클립은 **Skin: `Without Skin`**(애니메이션만, 메시 불필요):
   - Format `FBX Binary (.fbx)` · Skin **Without Skin** · FPS `30` · Keyframe Reduction `none`
5. 받은 `.fbx`들을 같은 폴더에 넣습니다:
   ```
   /Volumes/AISSD/ai-tilt/unity_viz/AITiltViz/AITiltViz/Assets/Characters/
   ```
   - 파일명에 동작 이름이 들어가면 좋습니다(예: `Worker_UsingTablet.fbx`). 파일명에 `idle`이 들어간 것만 '대기'로 인식해 맨 뒤로 돌립니다.

### 그다음 (내가 처리)
"작업 클립 넣었어"라고 알려주시면 재빌드만 하면 됩니다 — `BakeCharacter`가 가장 큰 fbx(=Timmy, 스킨 모델)를 캐릭터로, 나머지 작업 클립들을 모아 **순환 재생 컨트롤러**를 자동 구성합니다. (코드 이미 준비 완료)

> 팁: 동작 1개만 받아도 됩니다(예: `Using Tablet` 하나). 그러면 그 동작을 계속 반복 = "작업 중"으로 보입니다. 여러 개면 더 분주해 보입니다.
