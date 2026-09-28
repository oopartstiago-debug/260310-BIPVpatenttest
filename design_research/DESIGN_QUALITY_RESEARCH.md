# 디자인 품질("AI 슬롭" 탈피) 외부조사 종합 보고서

작성일: 2026-09-28 · 대상: 웹 대시보드(`main_screen` / `ops` / `shading` / `validation`), `explainer.html`, Unity 시연(`unity_viz/AITiltViz`)
방법: 병렬 조사 에이전트 4개(웹 UI · 3D 사실감 · AI 디자인 프로세스/평가 · 도메인 시각 언어), 웹 검색 약 200회. 현 화면은 Playwright로 렌더해 직접 진단했습니다.

> **출처 신뢰도 표기.** 조사 환경의 프록시가 일부 사이트(carbondesignsystem.com, pvsyst.com, ethz.ch, distill.pub, arxiv.org 일부 등)를 막았습니다. 그래서 일부 주장은 원문 대신 검색 요약에 근거합니다. 원문을 직접 읽은 것은 Anthropic `frontend-design` SKILL.md, impeccable 소스, Vercel 가이드라인, v0 유출 프롬프트, Grafana 문서입니다. 2026년 arXiv 논문 번호(예: 2607.xxxxx, 2608.xxxxx)는 초록 요약만 확인했으니 인용 전에 원문을 다시 확인하세요.

---

## 0. 결론 요약

**반복해서 고쳐도 안 되는 이유는 "프롬프트 문구"가 아니라 "구조"에 있습니다.** 확인된 원인은 네 가지입니다.

1. **분포 수렴.** 목표가 형용사("모던, 깔끔, 프리미엄")뿐이면 모델은 학습 데이터의 중앙값을 냅니다. Anthropic은 이것을 공식적으로 *distributional convergence*라고 부릅니다. 현재 대시보드의 `#0A0A0C` 다크 배경, `#7F77DD` 보라 액센트, Inter/Pretendard 조합이 바로 그 중앙값입니다.
2. **에이전트가 결과를 보지 않습니다.** 렌더를 보고 고치는 루프가 없으면 "끝난 것처럼 보이는" 지점에서 멈춥니다. Unity는 헤드리스(`-nographics`)로는 카메라 렌더 자체가 불가능해서 이 문제가 가장 심합니다.
3. **도메인 문법이 없습니다.** 신뢰를 주는 태양광 자료(PVsyst 손실도, 카펫 플롯, 태양궤적, 1:1 검증 산점도, 특허 도면)의 문법 대신 범용 SaaS 카드 키트를 쓰고 있습니다.
4. **"보여주기"와 "정직"이 충돌합니다.** 실패한 포토리얼은 가장 싸 보입니다. 특허·기술 데모는 의도된 스타일화(선화, 클레이, 분해도)가 오히려 고급스럽고 정직합니다.

**처방 한 줄:** 레퍼런스 → `DESIGN.md` 토큰 고정 → 방향 3안 병렬 → 스크린샷/렌더를 보고 채점 → 사람이 블라인드로 선택. 웹은 "연구보고서/도면" 방향, 3D는 "Blender 헤드리스 + 선화/클레이 우선"을 추천합니다.

---

## 1. 현 상태 진단 (직접 렌더 확인)

### 1-1. 웹 대시보드 (`main_screen.html`)

Anthropic `frontend-design` SKILL.md(2026 원문)는 "AI 생성 디자인이 모이는 특징"을 번호 목록으로 적어 두었습니다. 현재 화면은 그중 여러 항목에 걸립니다.

| 현재 코드 | 해당하는 슬롭 신호 | 출처 |
|---|---|---|
| `--bg:#0A0A0C`, `--surface:#111114` | "tinted near-black standing in for black", 요청하지 않은 다크 | SKILL.md #5 |
| `--accent:#7F77DD` (보라) | 인디고/보라 액센트. Adam Wathan(Tailwind 창시자)이 "모든 AI UI가 indigo-500인 건 우리 탓"이라고 사과한 그 계열 | [X](https://x.com/adamwathan/status/1953510802159219096) |
| `.bp-title{uppercase; letter-spacing:.08em}` "BIPV CONTROL" | "tracked-out ALL-CAPS eyebrow label" | SKILL.md #5, impeccable "eyebrow는 금지" |
| `서울 · 2026.05.14 · XGBoost V15` | "meta strings joined with middle dots" | SKILL.md #5 |
| "AI 제어 VS 고정각 — 발전량 증분" | "labels built as 'WORD — fragment'" | SKILL.md #5 |
| 거대 `+27.3%` + 작은 라벨 + 보조 스탯 2개 | "hero-metric template" | SKILL.md, impeccable |
| 같은 반경·1px 테두리 카드 반복 | SaaS card kit | SKILL.md #4 |
| 상시 초록 상태 점 + `box-shadow` 헤일로 | ISA-101 위반("모두 초록이면 초록은 의미가 없다"), impeccable "zero-offset colored halo is decoration" | ISA-101, impeccable |
| 'Inter' 숫자 폴백 | "Inter for everything" | Anthropic 블로그 |
| kWh/패널, 기준 없는 +% | 도메인 정규화 지표(kWh/kWp, PR)와 출처·기간이 없음 | PVsyst/IEC 61724 관례 |

※ 렌더에서 차트가 빈 칸으로 나온 것은 조사 컨테이너가 CDN(jsdelivr)을 막았기 때문입니다. 실제 버그는 아닙니다.

### 1-2. 설명 페이지 (`explainer.html`)

크림 배경, 주황 액센트, 왼쪽 굵은 테두리 콜아웃, `01/02/03` 번호 칩, 💡🎯 이모지, 3칸 수치 카드 그리드로 되어 있습니다. 이것은 "Claude 아티팩트 룩"이라는 또 하나의 중앙값입니다. 다이어그램(루버 단면)은 사선 막대 3개 수준이라 기술 문서의 설득력이 떨어집니다.

### 1-3. Unity 시연

- `LouverAgentPresenter.cs`에서 `CreatePrimitive`를 26회 호출합니다. 블레이드, 바닥, 패널, 팔(Capsule), 포디움, 스크린이 모두 기본 도형입니다.
- UI는 IMGUI 기본 회색 스킨입니다(`unity_capture.png`). Unity 공식 문서는 IMGUI를 "에디터 확장·디버그용"이라고 하며 런타임 UI로 권장하지 않습니다.
- ACES 톤매핑, HDRI, 블룸은 이미 있습니다. 부족한 것은 **베벨, 재질 미세 변화, AO/접촉 그림자, 스케일 단서, 카메라(수직선 평행), 맥락**입니다.

---

## 2. 원인: 왜 프롬프트로는 고쳐지지 않는가

| 원인 | 근거 |
|---|---|
| 분포 수렴. 지시가 없으면 확률 중심에서 샘플링 | [Anthropic: Improving frontend design through Skills](https://claude.com/blog/improving-frontend-design-through-skills) |
| RLHF 전형성 편향. 평가자가 익숙한 것을 선호하고 KL-RLHF가 이를 증폭 | Verbalized Sampling, [arXiv 2510.01171](https://arxiv.org/abs/2510.01171) |
| AI 공동창작은 개인 점수를 올리지만 집단 다양성을 낮춤(19개 연구 메타분석) | [ACM ECCE 2026](https://dl.acm.org/doi/10.1145/3822301.3822304) |
| 스킬 약 400토큰은 "바닥을 올릴 뿐 차별성은 명시적 브리프에서 나옴" | [wmedia.es](https://wmedia.es/en/tips/claude-code-frontend-design-skill) |
| 에이전트는 "looks done"에서 멈춤. 검증 가능한 체크가 필요 | [Claude Code Best Practices](https://code.claude.com/docs/en/best-practices) |
| 생성 UI 도구가 설명한 디자인 근거의 25% 이상이 실제로는 미구현("Design Theater") | arXiv 2607.22928 (초록 기준) |
| **목표 이미지가 있으면** MLLM이 상당히 충실하게 재현. 문제는 코드력이 아니라 목표의 부재 | Design2Code, [arXiv 2403.03163](https://arxiv.org/abs/2403.03163) |
| LLM 비평은 측정형 문제(정렬, 간격)에 강하고 취향 판단에 약함. 영역 크롭·확대를 주면 사람과의 격차가 50% 줄어듦 | UICrit [2407.08850](https://arxiv.org/abs/2407.08850), [2412.16829](https://arxiv.org/abs/2412.16829) |
| 전문 디자이너끼리도 선호 일치도가 α=0.25. 최종 판정은 **사용자 본인의 A/B 선택**이 효율적 | DesignPref, [arXiv 2511.20513](https://arxiv.org/abs/2511.20513) |
| 자기개선 루프에서 한 곳을 고치면 다른 곳이 깨짐. **라운드당 한 문제만** 고치는 편이 우수 | RubSE, arXiv 2608.24138 (초록 기준) |

---

## 3. 웹 UI 처방

### 3-1. 반(反)슬롭 규칙 원천 (설치 가능)

| 이름 | 핵심 | 링크 |
|---|---|---|
| **Anthropic frontend-design** | 주제에 뿌리를 둔 디자인, 서체 1~2종, 대문자 라벨·eyebrow 금지, "boldness는 한 곳에만", 계획(4~6색 hex, 서체 역할, ASCII 와이어프레임)을 먼저 세우고 "누구나 도달할 기본값인가"를 자기검토한 뒤 코딩 | [SKILL.md](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md) |
| **impeccable** (pbakaus, Apache-2.0) | 명령 24개(`critique`, `audit`, `distill`, `quieter`…). **LLM 없이 도는 결정론적 탐지 규칙 61개** `npx impeccable detect` (CI용 `--json`). 대시보드 전용 "Operate 모드" 포함 | [GitHub](https://github.com/pbakaus/impeccable) |
| **OneRedOak design-review** | Playwright로 실제 화면을 탐색하는 리뷰 서브에이전트와 `/design-review` 명령 | [GitHub](https://github.com/OneRedOak/claude-code-workflows/tree/main/design-review) |
| taste-skill / ui-ux-pro-max | 다이얼형 취향 규칙 / 스타일·팔레트·폰트 DB | [taste](https://github.com/leonxlnx/taste-skill), [uupm](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) |
| Vercel Web Interface Guidelines | tabular 숫자, 색에만 의존하지 않는 상태 표시, 중첩 반경, APCA, `color-scheme` | [GitHub](https://github.com/vercel-labs/web-interface-guidelines) |
| v0 시스템 프롬프트(유출본) | 색은 3~5개, 그라디언트 기본 금지, 서체 최대 2종, 이모지 아이콘 금지, 블롭 금지 | [awesome-system-prompts](https://github.com/EliFuzz/awesome-system-prompts/blob/main/leaks/v0/2025-07-20_prompt.md) |
| **korean-vibe-fonts** | 상업 사용 가능한 한글 웹폰트 464종을 분위기와 상황별로 추천. Claude Code 어댑터 포함 | [GitHub](https://github.com/seulkikaang/korean-vibe-fonts) |
| awesome-design-md / Google Stitch DESIGN.md | 실제 브랜드 70여 곳에서 추출한 DESIGN.md 모음과 에이전트용 토큰 포맷(Apache-2.0) | [awesome-design-md](https://github.com/voltagent/awesome-design-md), [Google 블로그](https://blog.google/innovation-and-ai/models-and-research/google-labs/stitch-design-md/) |

> 취향 스킬은 **하나만** 설치하세요. 여러 개를 쓰면 규칙끼리 충돌합니다. 추천 조합은 impeccable(프로세스 + 탐지기) + Playwright + design-review입니다.

### 3-2. 타이포그래피 (한글)

Pretendard는 "한국 웹의 Inter"입니다. 안전하지만 가장 흔한 기본값입니다.

| 조합 | 성격 | 라이선스 / 배포 |
|---|---|---|
| **IBM Plex Sans KR + IBM Plex Mono** (모노는 측정값·단위 전용) | 엔지니어링·계측, 한 가족이라 일관적 | OFL, Google Fonts |
| **42dot Sans** 단일 패밀리 + `tabular-nums` | 모빌리티·하드웨어 톤, 가변 300–800 | OFL, Google Fonts |
| **SUIT(UI) + 마루부리 또는 Noto Serif KR(설명문)** | 특허·보고서 맥락 | OFL, jsDelivr / 네이버 CDN |
| Spoqa Han Sans Neo | 숫자 가독성 우수 | OFL, jsDelivr |
| 비추천: G마켓 산스, 페이퍼로지 | 커머스·PPT 느낌 | OFL |
| 비추천: 산돌구름 | 구독형이고 웹 임베딩 제한 | 상용 |

- 대시보드 스케일은 고정 rem, 비율 1.125–1.2로 잡습니다(impeccable Operate). 현재 58–72px 히어로 숫자는 계측기 문법에 비해 과대합니다.
- 모노 서체는 "기술적 분위기를 내는 의상"으로 쓰면 그 자체가 AI 신호입니다. **측정값과 단위에만** 씁니다.

### 3-3. 색 체계

- **ISA-101 / High-Performance HMI**: 정상 상태는 저대비 그레이스케일(화면의 약 90%)로 두고, 색은 비정상·주의·조작 필요 상태에만 씁니다. "모든 설비가 초록이면 초록은 아무 의미가 없다." [LADX](https://ladx.ai/resources/isa-101-hmi-design), [Rockwell HMI Style Guide](https://literature.rockwellautomation.com/idc/groups/literature/documents/wp/proces-wp023_-en-p.pdf)
- **Datawrapper**: 회색이 가장 중요한 색입니다. 강조색은 차트당 하나만 씁니다. [블로그](https://www.datawrapper.de/blog/emphasize-with-color-in-data-visualizations)
- **도구**
  - OKLCH([Evil Martians](https://evilmartians.com/chronicles/oklch-in-css-why-quit-rgb-hsl))
  - Radix Colors 12단계 역할 체계([docs](https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale))
  - Adobe Leonardo(목표 대비비에서 역으로 색 생성)
  - Huetone
  - **Material HCT**: 실제 BIPV 사진에서 색을 추출하면 "도메인에서 유래한 색"이 됩니다.
- **색각이상 안전**
  - Okabe-Ito(`#E69F00` 주황, `#0072B2` 파랑, `#D55E00` 주홍, `#009E73` 청록)
  - 연속값은 **cividis**
  - 차이값은 ColorBrewer **RdBu**
  - 무지개/jet, RdYlGn은 금지

### 3-4. 차용 가능한 디자인 시스템

| 시스템 | 라이선스 | 적합성 |
|---|---|---|
| **IBM Carbon** (데이터 시각화·대시보드 가이드) | Apache-2.0 | 최우선. Plex와 한 몸 |
| Palantir Blueprint | Apache-2.0 | 고밀도 데스크톱 데이터 UI |
| Radix Colors/Themes | MIT | 색 역할 체계 |
| Vercel Geist | 서체 OFL | 새로운 개발툴 기본값이 되어 가는 중이라 주의 |
| Tremor | Apache/MIT | 구조만 참고. 기본 외관은 새로운 슬롭 |
| Atlassian ADS | 제한 | 사용 금지 |

### 3-5. 레퍼런스 수집처

- **실제 제품 화면**: Mobbin(MCP 지원), Refero와 [Refero Styles](https://styles.refero.design)(실서비스를 토큰 명세로 변환), Page Flows
- **무드보드**: Cosmos.so, Are.na
- **주의**: Dribbble과 Land-book은 학습 분포의 중심이라 슬롭의 원천일 수 있습니다.
- **가장 강력한 것은 도메인 1차 자료**입니다: SMA Sunny Portal, SolarEdge, Enphase, SCADA 화면, 건축 입면도, 계측기 전면 패널.

---

## 4. 도메인 시각 언어: 이 분야에서 "진짜처럼" 보이는 문법

심사자와 투자자는 **익숙한 업계 문법**을 보고 신뢰를 판단합니다. 새 스타일을 발명하지 말고 업계 문법을 차용하세요.

### 4-1. 반드시 넣을 업계 표준 도식

| 도식 | 원천 | 이 프로젝트에서의 용도 |
|---|---|---|
| **손실 폭포도** (GHI → POA → 음영 → IAM → … → 계통) | PVsyst loss diagram. 금융 심사자가 가장 먼저 읽는 섹션 | "루버 상호음영"과 "추적 이득"을 별도 줄로 넣기 |
| **정규화 생산량 Yf/Lc/Ls + PR** | PVsyst, IEC 61724 | kWh/패널 대신 kWh/kWp·PR |
| **iso-shading 다이어그램** (태양궤적 + 등음영선) | PVsyst | 루버 간 자기음영이 언제 문제인지 |
| **카펫 플롯 3연** (일×시 히트맵: 고정 / AI / 차이) | Solargis QC, ClimateStudio | 같은 축과 같은 컬러바, 차이는 RdBu |
| **태양궤적도** | pvlib, Ladybug, [Andrew Marsh](https://andrewmarsh.com/software/sunpath3d-web/) | 각도 슬라이더와 연동 |
| **1:1 검증 산점도** + MBE/RMSE/nRMSE + 잔차 패널 | 검증 논문 표준 | N, 기간, 해상도 명시 |
| **surplus/deficit filled line** | FT Visual Vocabulary | AI − 고정 이득 |
| **참조부호가 붙은 단면·분해도** (100 루버, 110 셀, 120 축…) | 37 CFR 1.84, KIPO 【부호의 설명】 | 청구항과 도식을 직접 대조. **이 프로젝트만의 차별점** |

### 4-2. 차트 작법

- **직접 라벨**: 범례 없이 선 끝에 이름과 값을 붙입니다(Wilke, Observable Plot).
- **중복 부호화**: AI는 실선과 색, 고정 90°는 회색 점선으로 구분합니다.
- **결론형 제목**: "추적이득" 대신 "겨울 오전, AI 제어가 고정 90° 대비 +31%"처럼 씁니다(Amanda Cox: "주석 레이어가 가장 중요하다").
- **KPI 4요소**: 숫자 + 단위 + 기간 + 비교 기준.
- **모든 차트 하단에 "출처 · 방법 · CSV"를 붙입니다**(OWID, Ember 방식). "추정 / 계측" 구분 라벨도 넣습니다(KPX 방식).
- **스몰 멀티플과 스파크라인**을 게이지·도넛 대신 씁니다(Tufte).

### 4-3. 벤치마크 대상

- **ETH Adaptive Solar Facade** (Schlüter; Nature Energy 2019 → Zurich Soft Robotics): 가장 가까운 선례입니다. "실물 사진 + 절제된 도식 + 수치 2~3개" 형식을 따릅니다.
- **Electricity Maps · Ember · Fraunhofer Energy-Charts**: 단위가 붙은 큰 숫자, 데이터 품질 라벨, 에너지원별로 고정된 색.
- **Distill.pub · Bartosz Ciechanowski · Bret Victor**: 문장과 연동되는 인터랙티브 위젯 하나. 예: "각도를 드래그하면 단면 광선·음영·발전량이 함께 변함".
- **딥테크 사이트**
  - Heliogen: "AI"라는 말 대신 물리 동작을 구체적으로 설명합니다.
  - Commonwealth Fusion(Upstatement 제작): "기억에 남되 과학 사업에 걸맞게 신뢰 가능"한 톤.
  - Onyx Solar, Heliatek: 실제 설치 사진에 kW 수치를 붙입니다.
- **Tesla 앱 Power Flow**: 흐름 선의 색을 의미에 고정합니다(태양은 주황). 단, 가정용 4노드 구성을 그대로 베끼면 제품 앱처럼 보입니다.

### 4-4. 제안하는 방향 3가지 (모두 밝은 배경, 보라 없음)

**A. 설계도서 / 특허도면**
- 오프화이트 제도용지에 흑연 선, 우하단 표제란, "FIG. 3" 캡션, 참조부호, 치수선 θ, 45° 해칭 음영. 청사진(파랑 바탕)은 클리셰이므로 피합니다.
- 서체: IBM Plex Sans KR + Plex Mono
- 토큰: `paper #F6F4EE`, `ink #1A1C1E`, `graphite #6B7075`, `hairline #D9D6CC`, `sun #E69F00`, `redline #C8431E`, `ai #0072B2`, `fixed #8A8F94`

**B. 연구보고서 / 저널** (Fraunhofer · PVsyst · Nature Energy)
- 흰 지면, 번호 붙은 그림, 방법론 캡션, 각주 출처. 위 4-1 도식을 업계 형식 그대로 씁니다.
- 서체: 제목은 Noto Serif KR 600, 본문은 Pretendard 또는 SUIT, 파라미터는 Plex Mono
- 토큰: `bg #FFFFFF`, `text #222`, `muted #666`, `rule #E2E2E2`, `measured #000`, `simulated #D55E00`, `ai #0072B2`, `fixed #999`, `irradiance #E69F00`. 연속값은 cividis, 차이값은 RdBu.

**C. 에디토리얼 에너지 데이터** (Electricity Maps · Ember · OWID)
- 따뜻한 지면, 결론형 대제목, 스몰 멀티플, 파사드 입면 셀 맵(SolarEdge 모듈 레이아웃을 응용해 루버별 발전량으로 채색).
- 토큰: `bg #FAF8F3`, `text #1F2328`, `sub #6E6A62`, `grid #E8E4DA`, `solar #F2A900`, `solar-ink #9A6700`, `gain #1F7A5C`, `fixed #A8A39A`, `loss #B5452B`

**추천:** 대시보드는 **B**, 설명 페이지는 **A + C** 혼합(단면 위젯은 도면풍, 결론 수치는 에디토리얼풍).
**모든 페이지 공통 토큰:** AI는 `#0072B2`, 고정 90°는 회색 점선, 일사는 `#E69F00`.

---

## 5. 3D 시각화 처방

### 5-1. 기본 도형처럼 보이는 원인 체크리스트

| 증상 | 해결 |
|---|---|
| 칼날 모서리 | Bevel 0.5–1 mm, segments 2–3, Harden Normals |
| 균일 재질 | roughness에 3–10% 노이즈, 유리 얼룩, 모서리 마모 |
| 스케일 불명 | 사람 컷아웃, 난간, 창틀, 보도블록 |
| 평평한 조명 | GI, 하늘 차폐, 물리 하늘 |
| 떠 보임 | AO, 접촉 그림자 |
| 톤매핑 | AgX(Blender 4+ 기본) 또는 Khronos PBR Neutral(제품색 정확). ACES는 색이 틀어짐 |
| 게임 같은 카메라 | 24–28 mm, **카메라 수평 + shift로 수직선 평행**(건축 사진 문법) |
| 빈 맥락 | 유리가 반사할 도시 HDRI와 주변 건물 |
| IMGUI 회색 패널 | UI Toolkit(USS), 또는 3D는 3D로 두고 UI는 웹으로 분리 |

### 5-2. 파이프라인 비교 (헤드리스 에이전트 기준, ★5 만점)

| 경로 | 노력(★ 많을수록 쉬움) | 사실감 | 자동화 | 특허데모 적합 | 평 |
|---|---|---|---|---|---|
| **A. Blender Cycles 헤드리스 스틸** (`blender -b -P build.py`, CPU+OIDN) | ★★★ | ★★★★ | ★★★★★ | ★★★★★ | **최우선.** 렌더를 PNG로 보고 고치는 루프 가능, 기하 정확 |
| **B. 같은 씬의 Freestyle 선화 / 클레이 AO / 분해도** | ★★★★ | 스타일 | ★★★★★ | ★★★★★ | 특허 도면과 데모를 한 소스에서 |
| C. fSpy 실사 배경판 + CG 루버 합성 | ★★ | ★★★★★ | ★★★ (사진은 사람이 제공) | ★★★★ | 히어로컷 1–2장 |
| D. Unity URP 보강 (SSAO, APV, UI Toolkit, 베벨 FBX) | ★★★ | ★★ | ★★ | ★★★ | 인터랙티브를 유지할 때. 룩은 클레이+선 권장 |
| E. Unity HDRP | ★ | ★★★★ | ★★ | ★★★ | 사람이 에디터에서 룩데브할 때만 가치 |
| F. three.js / R3F + drei + three-gpu-pathtracer | ★★★ | ★★★☆ | ★★★★ (Playwright) | ★★★★ | 웹 공유, explorable 위젯에 최적 |
| G. UE5 Python + Movie Render Queue | ★ | ★★★★★ | ★★★ | ★★★★ | GPU 워크스테이션 필요 |
| H. Twinmotion / D5 / Enscape / Lumion | — | ★★★★ | ☆ (API 없음) | ★★★ | 사람이 GUI로 작업할 때만 |
| I. 생성형 렌더 강화 (FLUX Depth/Canny, Kontext, Krea, Gemini, Veras) | ★★★★★ | 겉보기 ★★★★★ | ★★★★ | ★ | **무드컷 전용.** 블레이드 수·피치·각도가 몰래 바뀜 |

**기술 포인트**
- EEVEE는 디스플레이 없이 실패하는 사례가 많아 헤드리스에서는 Cycles가 안전합니다.
- **Unity `-batchmode -nographics`에서는 카메라 렌더가 불가능**합니다. 에이전트가 Unity 결과를 "보는" 루프를 만들기가 가장 어렵습니다.
- Blender MCP(ahujasid/blender-mcp)는 Poly Haven, Sketchfab, Hyper3D를 연동하지만 **Blender UI에서 서버를 켜야** 합니다. 헤드리스 컨테이너에서는 "bpy 스크립트 → 렌더 → PNG 확인" 파일 브리지가 더 견고합니다. 사용자 PC에 GUI Blender가 있다면 MCP를 보조로 쓰세요.
- drei: `Environment`, `AccumulativeShadows` + `RandomizedLight`, `ContactShadows`, `MeshTransmissionMaterial`. 후처리는 pmndrs `ToneMappingEffect`(AgX/Neutral) + N8AO.

### 5-3. 생성형 강화 운영 규칙 (쓴다면)

1. 원본 CG와 강화본을 항상 나란히 제시합니다.
2. 강화본에 "AI-enhanced illustrative; geometry per CG render" 표기를 붙입니다.
3. Depth와 Canny를 동시에 걸고 강도는 높게, creativity는 낮게 둡니다. 루버 영역은 마스크로 원본을 유지합니다.
4. 블레이드 수와 각도를 오버레이로 자동 대조해 불일치하면 폐기합니다.
5. 수치·기술 설명 화면에는 절대 쓰지 않습니다.

### 5-4. 스타일화가 포토리얼보다 나은 경우

- 청중이 심사관, 기술검토자, 투자자일 때
- 사람의 룩데브 반복이 불가능할 때(헤드리스 에이전트)
- 청구항 수치(피치 97.5 mm, 각도)와 관련되어 과장이 리스크일 때
- 실시간 인터랙티브일 때. "거의 된" 실시간 포토리얼이 가장 싸 보입니다.

추천 룩: **흰 클레이 + PV 블레이드만 짙은 남색 유리(단일 강조) + 태양 방향 그림자 + 윤곽선.** Gooch 쿨-웜 셰이딩(SIGGRAPH '98)도 기술 일러스트의 표준 선택지입니다.

### 5-5. 레퍼런스와 에셋

- **실물 레퍼런스**: ETH ASF/HiLo(Roman Keller 사진), Colt Shadovoltaic One River Terrace(NY), EnergyX DY-Building(고양, BIPV 105.6 kWp), 도심 루버형 BIPV 논문(DBpia, KIEAE).
- **촬영 문법**: 오전·오후 측광으로 그림자 리듬을 보여 줍니다. **같은 카메라로 09·12·16시 트립틱**을 만들면 추적 제어 메시지와 미감이 동시에 해결됩니다.
- **에셋**
  - Poly Haven, ambientCG: CC0, API 제공
  - Fab/Megascans: 2025년부터 유료화, 무료 스타터는 유지
  - Sketchfab: CC 라이선스 기록 필수
  - Kenney: CC0, 로우폴리 맥락 건물
  - BIM: NBS Source, BIMobject의 브리즈솔레유
  - 인물: 2D 컷아웃이 가장 저렴하고 효과적
- **주의**: Ready Player Me는 2026-01-31 서비스 종료(Netflix 인수)라 사용하지 마세요.

---

## 6. 프로세스: 반복 가능한 디자인 루프

### 6-1. 핵심 원칙 4가지

1. **형용사 대신 레퍼런스와 토큰을 줍니다.** 레퍼런스 3~5개를 모으고, 각각에서 "가져올 속성 1개"를 정해 `DESIGN.md`에 수치로 고정합니다.
2. **보지 않으면 커밋하지 않습니다.** Playwright 스크린샷이나 Blender PNG를 **새 컨텍스트의 평가 서브에이전트**가 채점하게 합니다(작성자와 평가자 분리). 전체 이미지와 함께 영역 크롭도 줍니다. 라운드당 한 문제만 고치고 최대 3라운드까지 돕니다.
3. **한 안이 아니라 세 안을 만듭니다.** 각 안에 서로 다른 레퍼런스와 제약을 강제로 배정합니다(위 A/B/C). "전형성이 낮은 방향부터 나열하라"고 지시합니다.
4. **사람이 블라인드로 고릅니다.** 선택 이유 한 줄을 `DESIGN.md`의 Do/Don't에 누적합니다. 이것이 사용자 취향 데이터가 됩니다.

### 6-2. 채점 루브릭 (각 0–2점, 20점 만점. 16점 이상 + 블라인드 승리면 통과)

1. 금지 신호 0개(Inter, 보라, eyebrow 대문자, 가운뎃점 메타, 히어로 메트릭, 이모지 아이콘, 3카드 그리드)
2. 서체가 `DESIGN.md`와 일치하고 스케일 비율이 일정함
3. 색이 역할 기반이고, 강조색은 1개이며 면적 10% 이하, 정상 상태는 무채색
4. 간격이 4/8 배수이고 일관됨
5. 흐리게 봐도(squint test) 1순위 초점이 명확함
6. 밀도가 도메인에 맞음
7. 레퍼런스에서 가져온 속성이 식별됨
8. 도메인 고유 도식이 1개 이상 있음(손실도, 카펫, 태양궤적, 부호 단면)
9. 단위, 기간, 출처, 추정·계측 구분이 있음
10. WCAG AA 또는 APCA 대비 충족, 반응형이 깨지지 않음

자동 보조 도구:
- `npx impeccable detect` 건수
- 금지 hex·폰트 grep
- (선택) UIClip 상대점수: 변형끼리 비교하는 용도로만. [uiclip](https://uimodeling.github.io/uiclip/)

### 6-3. 이 저장소에서 실행할 단계

| 단계 | 내용 | 주체 · 시간 |
|---|---|---|
| 0 | Playwright MCP, impeccable, design-review 설치. CLAUDE.md에 "UI는 DESIGN.md를 따르고 완료 전 스크린샷 리뷰 통과" 한 줄만 추가 | 에이전트 · 30분 |
| 1 | 기준선 측정: 4개 화면 × 1440/390px 스크린샷, detect 건수, 루브릭 점수 | 에이전트 · 15분 |
| 2 | 레퍼런스 3~5개 선정(PVsyst 보고서, Carbon 대시보드, Electricity Maps, ETH ASF 사진, 인버터 포털) | **사람** · 30분 |
| 3 | DESIGN.md 3종(A/B/C) 작성. 필요하면 FLUX로 방향별 무드 이미지 생성 | 에이전트 |
| 4 | `main_screen` **한 화면만** 3안으로 병렬 구현(worktree), 각각 스크린샷→비평→수정 최대 3회 | 에이전트 |
| 5 | 블라인드 선택. 이유를 DESIGN.md에 기록. 막히면 이 시점에 디자이너에게 1화면 시안 의뢰 | **사람** · 10분 |
| 6 | 선택안을 나머지 화면과 explainer로 확장. 화면마다 새 컨텍스트 리뷰, detect 0건 | 에이전트 |
| 7 | 3D: Blender 헤드리스로 베이스 씬(베벨 블레이드, 태양궤적 정합, AgX, 수평 카메라) → 트립틱, 선화, 클레이, 분해도 스틸 | 에이전트 · 1–2일 |
| 8 | Unity는 결정 필요: (a) UI Toolkit + 클레이/선 스타일로 정직화, 또는 (b) 인터랙티브를 three.js explorable 위젯으로 이관 | **사람 결정** |

### 6-4. 사람 자원 옵션 (비용 대비 효과 최대)

- **디자이너에게 "메인 대시보드 1화면 Figma 시안 + 색·타이포 토큰"만 의뢰**하고, 나머지는 에이전트가 Figma MCP(`get_design_context`, `get_variable_defs`)로 확장합니다.
  - 크몽: 1장 7~12만 원 수준
  - 라우드소싱: 콘테스트로 여러 방향 시안을 받을 수 있음(30~100만 원)
  - Fiverr: 시간당 $30~70
- **템플릿**(Tailwind Plus $249, Untitled UI)은 구조와 밀도만 가져오고 토큰은 자체 DESIGN.md로 덮어씁니다. 그대로 쓰면 또 다른 중앙값이 됩니다.

---

## 7. 바로 적용할 변경 Top 12 (우선순위 순)

1. **보라 `#7F77DD`를 폐기**하고 공통 의미 토큰을 씁니다: AI `#0072B2`, 일사 `#E69F00`, 고정 90°는 회색 점선.
2. **ISA-101식 색 계약**: 정상 상태는 무채색으로 두고, 상시 초록 점과 헤일로를 삭제하고, 색은 경보와 핵심 데이터 1개에만 씁니다.
3. **다크 기본값을 해제**합니다. 회의실 프로젝터가 사용 장면이므로 밝은 배경(B안)을 쓰고, 다크는 보조로 둡니다.
4. **히어로 메트릭을 해체**합니다. 첫 화면 주인공은 "오늘 일사·각도·발전 시계열(AI vs 고정, 차이 음영)"로 하고, +%는 주석으로 내립니다.
5. **템플릿 크롬을 제거**합니다: 대문자·자간 eyebrow, 가운뎃점 메타, "A — B" 라벨, `01/02` 번호 칩, 이모지.
6. **서체를 교체**합니다: IBM Plex Sans KR + Plex Mono(측정값 전용) 또는 42dot Sans. Inter 폴백은 제거합니다.
7. **정규화 지표로 바꿉니다**: kWh/kWp, PR. KPI는 숫자, 단위, 기간, 기준 4요소를 갖춥니다.
8. **업계 도식을 추가**합니다: PVsyst식 손실 폭포도, 카펫 플롯 3연, iso-shading, 1:1 검증 산점도(nRMSE/MBE).
9. **참조부호 단면도**: explainer의 막대 도식을 특허 명세와 같은 부호(100, 110, …)를 쓴 도면풍 SVG로 교체하고, 각도 드래그 위젯을 붙입니다.
10. **출처 표기**: 모든 차트 하단에 "출처 · 방법 · CSV", 그리고 추정·계측 라벨을 붙입니다.
11. **Unity IMGUI를 중단**합니다: UI Toolkit으로 바꾸거나 UI를 웹으로 분리하고, 블레이드는 Blender에서 베벨 처리한 FBX로 교체하고, 룩은 클레이 + 강조 1색 + 윤곽선으로 합니다.
12. **검증 루프를 고정**합니다: DESIGN.md, Playwright 스크린샷, 별도 평가 에이전트, impeccable detect, 블라인드 선택.

---

## 부록: 주요 출처 모음

- Anthropic: [frontend-design SKILL.md](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md) · [블로그](https://claude.com/blog/improving-frontend-design-through-skills) · [Harness design(생성기/평가기 분리)](https://www.anthropic.com/engineering/harness-design-long-running-apps) · [Claude Code Best Practices](https://code.claude.com/docs/en/best-practices)
- 반슬롭 도구: [impeccable](https://github.com/pbakaus/impeccable) · [design-review](https://github.com/OneRedOak/claude-code-workflows/tree/main/design-review) · [taste-skill](https://github.com/leonxlnx/taste-skill) · [Vercel guidelines](https://github.com/vercel-labs/web-interface-guidelines) · [awesome-design-md](https://github.com/voltagent/awesome-design-md) · [korean-vibe-fonts](https://github.com/seulkikaang/korean-vibe-fonts)
- 연구: Design2Code [2403.03163](https://arxiv.org/abs/2403.03163) · UICrit [2407.08850](https://arxiv.org/abs/2407.08850) · [2412.16829](https://arxiv.org/abs/2412.16829) · DesignRepair [2411.01606](https://arxiv.org/abs/2411.01606) · UIClip [2404.12500](https://arxiv.org/abs/2404.12500) · DesignPref [2511.20513](https://arxiv.org/abs/2511.20513) · Verbalized Sampling [2510.01171](https://arxiv.org/abs/2510.01171)
- 데이터 시각화: [FT Visual Vocabulary](https://github.com/Financial-Times/chart-doctor/tree/main/visual-vocabulary) · [Wilke](https://clauswilke.com/dataviz/) · [Datawrapper 블로그](https://www.datawrapper.de/blog/) · [Carbon 데이터 시각화](https://carbondesignsystem.com/data-visualization/color-palettes/) · [cividis 논문](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0199239) · [Reuters style](https://github.com/reuters-graphics/style)
- 태양광 도식: [PVsyst 손실도](https://www.pvsyst.com/help/project-design/results/loss-diagram.html) · [iso-shading](https://www.pvsyst.com/help/project-design/shadings/calculation-and-model/iso-shading-diagram.html) · [pvlib sunpath](https://pvlib-python.readthedocs.io/en/stable/gallery/solar-position/plot_sunpath_diagrams.html) · [Ladybug](https://docs.ladybug.tools/ladybug-primer/) · [Andrew Marsh](https://andrewmarsh.com/software/)
- 레퍼런스: [ETH ASF](https://systems.arch.ethz.ch/research/adaptive-solar-facade) · [Electricity Maps](https://app.electricitymaps.com/) · [Ember](https://ember-energy.org/data/electricity-data-explorer/) · [OWID Grapher](https://github.com/owid/owid-grapher) · [Distill guide](https://distill.pub/guide/) · [ciechanow.ski](https://ciechanow.ski/)
- HMI: [ISA-101 요약](https://ladx.ai/resources/isa-101-hmi-design) · [Rockwell HMI 가이드 PDF](https://literature.rockwellautomation.com/idc/groups/literature/documents/wp/proces-wp023_-en-p.pdf) · [Grafana best practices](https://github.com/grafana/grafana/blob/main/docs/sources/visualizations/dashboards/build-dashboards/best-practices/index.md)
- 3D: [Blender CLI 렌더](https://docs.blender.org/manual/en/latest/advanced/command_line/render.html) · [Freestyle](https://docs.blender.org/manual/en/latest/render/freestyle/introduction.html) · [blender-mcp](https://github.com/ahujasid/blender-mcp) · [Poly Haven API](https://polyhaven.com/our-api) · [fSpy](https://fspy.io/) · [three-gpu-pathtracer](https://github.com/gkjohnson/three-gpu-pathtracer) · [drei](https://github.com/pmndrs/drei) · [Unity UI 시스템 비교](https://docs.unity3d.com/6000.3/Documentation/Manual/UI-system-compare.html) · [headless Unity 한계](https://partiallydisassembled.net/posts/unity-headless.html) · [37 CFR 1.84](https://www.ecfr.gov/current/title-37/chapter-I/subchapter-A/part-1/subpart-B/subject-group-ECFRc7605aa2d3f3782/section-1.84)
