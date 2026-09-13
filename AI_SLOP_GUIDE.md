# AI 슬롭 회피 가이드

<!-- slop-lint: off (이 문서는 금지 패턴을 인용한다) -->

작성 2026-09-13. 대상: 이 레포의 대시보드(main.html, app.py), 설명 페이지(explainer.html), Unity 시연, 마크다운 보고서, 레포 구조.
근거 자료는 각 절 끝에 URL로 붙였다. 현재 상태 수치는 `python slop_lint.py` 출력(2026-09-13 기준)이다.

## 0. 결론

이 레포는 세 층에서 AI 생성 흔적이 뚜렷하다. 린트 집계는 HIGH 57, MED 80, LOW 29이다.

시각. 라이브 대시보드(main.html v2.1)는 2025년식 "근흑 배경, 보라 강조, Inter" 룩을 이미 벗어났다. 문제는 도착지다. 현재 팔레트(배경 #FAFAF7, 제목 Newsreader 세리프, 강조 #C8580F)는 Anthropic이 2026년 9월 frontend-design 스킬에 hex 값까지 적어 둔 2세대 AI 클리셰(크림 #F4F1EA, 세리프 디스플레이, 테라코타 #D97757)와 같은 조합이다. 그 위에 템플릿 크롬이 얹혀 있다. 번호 아이브로우 7개("01 · 이번 달 성과"), 가운뎃점 구분자 48개, 화살표 5개, 로고 흔들림과 태양 맥동 등 무한 애니메이션 5개, 스크롤 페이드업, 동일 카드 3열 STEP 1/2/3. 레거시 다크 화면 4개(main_screen, ops_screen, shading_screen, validation_screen)는 1세대 클리셰 그 자체이고 app.py에서 참조되지 않는 고아 파일이다.

텍스트. 11개 마크다운 문서에 em-dash 206개, 볼드 라벨 불릿 110개, "X가 아니라 Y" 대구법 21회, 자기평가 수식어(정직, 진짜, 완전, 핵심, 유일) 126회, 체크와 경고 이모지 143개가 있다. TRACKING_GAIN_INVESTIGATION.md 한 파일에 볼드 217쌍, 장식 기호 203개다.

내용. 가장 무거운 슬롭은 스타일이 아니라 숫자다. 대시보드 헤드라인 "+5.7%"는 고정 60° 대비 수치다. POSITIONING.md 5절은 "나쁜 베이스라인 대비 수치는 외부 검증에서 반드시 깨진다"고 쓰고, 정직한 헤드라인은 최고 고정각(80°) 대비 +1.67%라고 결론냈다. 문서와 화면이 다른 숫자를 말한다.

레포. 루트에 verify_*.py 42개, 결과 JSON 11개, _vN 접미사 파일 12개, md와 중복인 docx 3개가 있다.

## 1. 사람들이 쓰는 방법

### 1.1 시각과 UI

원인은 분포다. Tailwind 저자 Adam Wathan은 2025년 8월 "Tailwind UI의 모든 버튼을 indigo-500으로 만든 것을 사과한다. 그래서 지구상 모든 AI 생성 UI가 인디고가 됐다"고 썼다. Anthropic 쿡북은 모델이 "분포의 중심으로 수렴하며, 프론트엔드에서는 이것이 사용자들이 AI 슬롭이라 부르는 미학을 만든다"고 설명한다. Adrian Krebs가 Show HN 랜딩 페이지 1,590개를 16개 DOM/CSS 패턴으로 채점한 결과 22%가 심한 슬롭, 32%가 경미한 슬롭이었다.

1세대 신호(2024~2025). 인디고/보라 강조, 보라에서 분홍으로 가는 그라디언트, Inter/Roboto/Space Grotesk, 가운데 정렬 히어로와 버튼 2개, 아이콘+제목+두 줄짜리 동일 카드 3열, rounded-2xl과 shadow-lg를 모든 블록에, 글래스모피즘, 그라디언트 텍스트, 네온 글로우, 이모지 아이콘, 왼쪽 강조 바 카드, "+12.5%" 초록 화살표 KPI 타일. Anti-AI-UI 저장소는 이 중 3개 이상이 겹치면 "2초 안에 AI로 읽힌다"고 정리한다.

2세대 신호(2026). "보라와 Inter를 피하라"는 프롬프트가 퍼지자 모델은 새 기본값으로 수렴했다. Anthropic frontend-design 스킬(anthropics/skills, 2026-09 판)은 세 가지를 hex와 함께 명시한다. (a) 크림 배경 #F4F1EA에 세리프 디스플레이와 테라코타 #D97757 강조, (b) 근흑 배경에 산성 초록 또는 주홍 강조 하나, (c) 헤어라인 괘선과 radius 0, 촘촘한 컬럼의 신문 레이아웃. 같은 문서는 "템플릿 크롬"도 이름 붙였다. ALL-CAPS 아이브로우, "A · B · C" 가운뎃점 구분자, 모노스페이스 라벨, 링크마다 붙는 화살표.

수정 방법으로 합의된 것.
- 토큰을 먼저 정한다. 색 4~6개 hex, 서체 역할 2개, 간격 단위, radius, 그림자 깊이를 DESIGN.md 같은 파일에 쓰고 생성기에 넘긴다. anti-slop 규칙 37은 이 파일이 없으면 결과물을 "방향 없는 초안"으로 표시한다.
- 주제에서 꺼낸다. 그 분야의 재료, 계기, 문서 관습에서 색과 형태를 가져온다. 왜 그 선택인지 한 줄로 쓸 수 없으면 무효(anti-slop 규칙 31).
- 서체는 register별로 고른다. 기술 문서는 IBM Plex 계열, 편집은 Fraunces나 Crimson Pro, 코드는 JetBrains Mono. Inter, Roboto, Arial, 시스템 폰트, Space Grotesk는 스킬이 금지 목록에 올렸다. 한국어에서 Pretendard는 Inter를 기반으로 만든 서체라 같은 인상을 준다.
- 테두리, 채움, radius, 그림자는 역할별로 쓴다. 들어 올려야 할 것 하나만 들어 올린다.
- 애니메이션은 한 곳에 몰아 넣는다. "산발적인 효과보다 잘 짜인 한 순간이 낫다"(Anthropic 스킬). 무한 반복 장식 모션은 빼고 prefers-reduced-motion을 존중한다.
- 구조 장치는 정보를 실어야 한다. 순서가 정보인 곳에만 번호를 붙인다.
- 실제 콘텐츠로 만든다. 자리표시 문구, 가짜 통계, 눌러도 아무 일 없는 버튼은 하드 게이트다.

도구. Design Slop Cop(URL 입력, 16패턴), no-slop-ui(에이전트 규칙과 리뷰 체크리스트), anti-slop(38규칙), Anti-AI-UI(P0~LOW 심각도), Vercel Web Interface Guidelines(에이전트 스킬), Anthropic frontend-design 스킬(계획, 브리프 대조, 빌드, 자기비평 루프).

출처. https://x.com/adamwathan/status/1953510802159219096 , https://platform.claude.com/cookbook/coding-prompting-for-frontend-aesthetics , https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md , https://www.adriankrebs.ch/blog/design-slop/ , https://github.com/Vanszs/Anti-AI-UI , https://github.com/miqdadbadjuber/anti-slop , https://github.com/LeoStehlik/no-slop-ui , https://github.com/vercel-labs/web-interface-guidelines

### 1.2 차트와 대시보드

- 회색 바탕에 강조색 하나. Datawrapper의 Lisa Charlotte Muth는 "회색이 데이터 시각화에서 가장 중요한 색"이라 쓴다. 시리즈마다 색을 주는 기본 팔레트는 색을 정체성에 쓰고 메시지에는 못 쓴다.
- 제목은 발견이다. "분기별 매출"이 아니고 "3분기 매출 23% 상승"이다. 결론이 있으면 글로 써서 차트 위에 둔다(Storytelling with Data).
- 범례 대신 직접 라벨. Eugene Wei: "범례는 독자 눈을 차트와 범례 사이에서 왕복시킨다."
- 이중 y축은 없는 상관을 암시한다. 위아래 두 차트(small multiples)로 나눈다.
- 파이와 도넛은 3~5조각이 아니면 표로 바꾼다(Tufte, FT Visual Vocabulary).
- KPI 타일은 3~7개, 각 숫자에 비교 기준(목표, 임계값, 전기)이 있어야 한다. 화살표만 있고 기준이 없는 타일은 장식이다.
- 눈금선은 기본 0, 필요하면 가로만 얇게. 테두리와 그림자는 뺀다.
- 불확실성은 유효숫자 2자리까지(JCGM 100 GUM 7절). "6.210 kWh"는 방법이 못 받치는 정밀도다.

출처. https://www.datawrapper.de/blog/colors , https://www.eugenewei.com/blog/2017/11/13/remove-the-legend , https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md , https://github.com/caylent/tufte-data-viz/blob/main/SKILL.md , https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e_FILES/MAIN_C045315e/07_e.html

### 1.3 텍스트

기준 목록은 Wikipedia의 "Signs of AI writing"(WikiProject AI Cleanup)이다. 이 문서 스스로 "특정 단어나 문장부호 금지 목록이 아니다. 하나의 신호는 약하고, 겹칠 때 강하다"고 밝힌다. 내용 층에서는 근거 없는 의의 부여("~의 증거다", "중요한 역할을 한다"), 분사 꼬리 분석(", highlighting", ", ensuring"), 홍보 어휘, 출처 없는 권위("전문가들은"), 교훈성 면책("~에 유의해야 한다"), 요약 신호("결론적으로")를 든다. 언어 층에서는 delve, robust, seamless, leverage 같은 어휘, 계사 회피("serves as", "boasts"), 부정 대구("단순한 X가 아니라 Y"), 3항 나열 강박, 유의어 순환을 든다. 서식 층에서는 Title Case 제목, 핵심어마다 볼드, 인라인 헤더 불릿("- **라벨**: 설명"), 제목과 불릿의 이모지, em-dash 남용, 문장이면 될 것을 표로 만드는 습관을 든다.

Anthropic 자체 규칙(Claude 시스템 프롬프트 2026-09-01판)은 "명확성에 필요한 최소한의 서식", "목록은 요청받았거나 내용이 다면적일 때만", "'genuinely', 'honestly'는 오히려 불성실하게 들리므로 피한다", "모든 단어가 서로 다른 것을 더해야 한다"고 적는다. Claude Code의 사용자 소통 프롬프트는 "결과부터", "읽기 쉬움이 짧음보다 우선", "A → B → 실패 같은 화살표 사슬로 압축하지 말 것", "표는 짧은 열거 가능한 사실에만, 설명은 셀이 아니라 본문에"라고 적는다.

한국어 신호. woonjangahn의 gist와 slop-gate의 korean.json, humanizer-kr이 정리한 것. 예고 문장("지금부터 ~을 살펴보겠습니다"), 완곡 계사("~라고 할 수 있습니다", "~것으로 보입니다"), 빈 수식어("다양한", "핵심적인", "중요한 것은"), 접속사 쌓기("또한/아울러/나아가/이를 통해"), "첫째, 둘째, 셋째" 기계적 열거, "단순한 X가 아닙니다. Y입니다", 번역투("~의 경우", "~에 있어서", "~함으로써", "~을 통해", 이중 피동 "되어지다"), 같은 어미 3연속. KatFishNet(arXiv 2503.00032)은 한국어 AI 문장의 61%에 쉼표가 있고 사람 문장은 26%라고 측정했다. 연결어미 "-고/-어서/-지만" 뒤의 영어식 쉼표가 특히 그렇다.

구조 슬롭. 모든 절이 도입 한 문장, 불릿 셋, 마무리 한 줄로 같은 모양인 것(jooray/humanizer 41번 "구조 균일성"), 서론과 각 절과 결론에서 같은 논지를 세 번 말하는 것("프랙탈 요약"), 500단어 미만 절에 제목 3단계, 본문을 다시 적는 표, "Overview / Key Features / Conclusion" 뼈대.

한 줄 논지. MikkoParkkola/anti-ai-tell: "밋밋함은 분산 결핍이다. 치료는 더 긴 금지어 목록이 아니라 선택된 관점이다." 금지어만 걸러내면 깨끗하지만 텅 빈 글이 나온다(humanizer-kr).

린터. vale-ai-tells(Vale 규칙 136개, 커밋 메시지 팩 포함), ai-slop-linter(npx, PR 게이트, 베이스라인), slop-gate(한국어 팩), deslop, slopscore. 이 레포에는 같은 취지의 `slop_lint.py`를 넣었다(6절).

출처. https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing , https://platform.claude.com/docs/en/release-notes/system-prompts/claude-fable-5-1 , https://gist.github.com/woonjangahn/3ad4d8fe1804aed2e7cafc9493ec566f , https://github.com/hwajongpark/slop-gate , https://github.com/monologg/humanizer-kr , https://github.com/MikkoParkkola/anti-ai-tell , https://github.com/tbhb/vale-ai-tells , https://arxiv.org/abs/2503.00032

### 1.4 코드, 레포, 기술 보고서

코드. Karpathy 방식 CLAUDE.md(GitHub 스타 17만 이상)의 네 규칙. 요청한 것 이상의 기능 없음, 한 번 쓰는 코드에 추상화 없음, 외과적 수정(옆 코드 손대지 않음), 불가능한 시나리오의 에러 처리 없음. 주석은 "무엇"이 아니라 "왜"만, 변경 이력("fixed X")은 커밋 메시지로. 임시 스크립트와 덤프는 temp/ 아래에 두고 git에서 제외하며, 회귀 테스트가 될 만한 것만 tests/로 승격한다. 커밋 본문은 "무엇과 왜"를 쓰고 "어떻게"는 diff에 맡긴다(cbea.ms). 이 레포의 커밋 메시지는 구체적이고 좋다. 유지할 것.

기술 보고서. 가짜 정밀도("97.3%")는 유효숫자 2자리와 포함 인자로 고친다. "significantly improves", "comprehensive", "robust" 같은 무정량 평가어는 검정과 구간, 열거된 범위, 성립 구간으로 바꾼다. "검증됨"은 ASME VVUQ 용어로는 활동이다. 무엇을 무엇과 어떻게 비교했는지가 없으면 단어를 쓰면 안 된다. 실패 사례 절("어디서 깨지는가")과 "무엇이 나오면 결론이 바뀌는가" 절을 둔다. 모든 숫자는 스크립트 이름과 커밋으로 추적 가능해야 한다. arXiv CS는 2026년부터 검토 안 된 LLM 출력(가짜 인용, "이 표는 예시입니다" 잔재)이 있는 논문의 저자를 1년 차단한다. Kobak 등(Science Advances 2025)은 2024년 초록의 최소 13.5%가 LLM 처리됐다고 추정한다.

출처. https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md , https://code.claude.com/docs/en/best-practices , https://cbea.ms/git-commit/ , https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty , https://arxiv.org/abs/2406.07016 , https://thenextweb.com/news/arxiv-ai-slop-ban-researchers-preprint

## 2. 현재 시스템 진단

### 2.1 라이브 대시보드 main.html

app.py가 실제로 임베드하는 유일한 화면이다. 렌더 결과에서 확인한 것.

| 위치 | 관찰 | 해당 신호 |
|---|---|---|
| 전체 | 배경 #FAFAF7, 제목 Newsreader, 강조 #C8580F | 2세대 클리셰(크림, 세리프, 테라코타) |
| 헤더 | 로고 마크가 5초 주기로 1° 흔들림 | 장식용 무한 애니메이션 |
| 01 히어로 | "+5.7%" 60px 숫자, 고정 60° 대비 | 나쁜 베이스라인. 문서 결론(80° 대비 +1.67%)과 불일치 |
| 01 히어로 | 태양 궤적 일러스트, 맥동하는 태양 | 장식 일러스트, 무한 애니메이션 |
| 01 하단 | 동일 타일 3열(오늘 발전, 수익, 차이) | 비교 기준 없는 KPI 타일 |
| 모든 절 | "01 · 이번 달 성과" 번호 아이브로우, 가운뎃점 | 템플릿 크롬. 절 순서는 정보가 아니다 |
| 02 차트 | GHI 막대와 각도 선을 이중 y축에, 범례 상단 | 이중축, 범례 |
| 03 비교 | "6.210 kWh", "5.870 kWh" | 소수 3자리 가짜 정밀도 |
| 04 시뮬 | "AI 의 딜레마" 불릿 2개에 화살표 4개 | 화살표 사슬 |
| RL 트윈 | "정직한 결과:" 볼드 라벨로 시작하는 8줄 문단, 10px | UI 카피에 들어온 보고서 슬롭 |
| 05 학습 | STEP 1 / STEP 2 / STEP 3 동일 카드 3열 | 3열 카드 그리드 |
| 05 지표 | R² 0.9665, 추종률 99.1% | 유효숫자 과다 |
| 모든 절 | opacity 0에서 IntersectionObserver로 페이드업 | 스크롤 리빌. 썸네일과 첫 프레임이 비어 보인다 |

잘한 것도 있다. 월별 차트는 회색 막대에 최대 월 하나만 강조했다. 피처 중요도는 막대 옆에 값을 직접 적었다. 각 절에 용어 툴팁이 있다. 이 부분은 그대로 둔다.

### 2.2 레거시 화면 4개와 preview

main_screen.html, ops_screen.html, shading_screen.html, validation_screen.html, main_screen.preview.html. 배경 #0A0A0C, 강조 #7F77DD(보라), Inter와 Pretendard, ALL-CAPS 아이브로우 "BIPV CONTROL", 필 상태칩과 글로우 점, 아이콘 박스가 붙은 동일 카드 4열, "R² 0.9966" 64px 히어로. 1세대 클리셰의 모든 항목이 있다. app.py는 이 파일들을 읽지 않는다. 삭제한다.

### 2.3 explainer.html

편집형 페이지로 방향은 좋다. 남은 신호. 배경 #f4f2ed에 강조 #c2600a(2세대 팔레트), ALL-CAPS 키커, 왼쪽 강조 바 카드, 01~04 번호 배지, 동일 칩 3열이 두 번, 전구와 과녁 이모지 콜아웃, em-dash 14개, "X가 아니라 Y" 4회, "엔지니어용 한 줄:", "한 문장으로:" 볼드 라벨 마무리, "정직성 원장" 같은 자기평가 제목.

### 2.4 Unity 시연

unity_capture.png. 반투명 회색 패널 위에 같은 크기 글자 8줄이 왼쪽 위에, 슬라이더 5개가 오른쪽 위에 쌓여 있다. 시연 화면에 디버그 문자열("[LouverPhysics 셀프테스트] 40/40 PASS maxRel(POA)=0.002%")이 노출된다. 3D 장면은 좋다. HUD가 장면을 가린다.

### 2.5 마크다운 문서 11개

| 파일 | 단어 | em-dash | 볼드 라벨 불릿 | 이모지 | 수식어 | 대구법 |
|---|---:|---:|---:|---:|---:|---:|
| TRACKING_GAIN_INVESTIGATION.md | 4113 | 65 | 58 | 73 | 51 | 4 |
| POSITIONING.md | 1497 | 21 | 5 | 14 | 11 | 5 |
| ADVERSARIAL_REVIEW_RESULTS.md | 917 | 26 | 6 | 19 | 12 | 1 |
| FABLE_REVIEW.md | 1128 | 18 | 2 | 20 | 9 | 2 |
| CHANGES_REAL_GEOMETRY.md | 1510 | 17 | 10 | 6 | 12 | 2 |

나머지 6개 파일도 같은 패턴이다. 내용은 좋다. 반증 시도, 스스로 수치를 낮춘 이력, 재현 스크립트 명시는 1.4절이 요구하는 것과 일치한다. 문제는 그것을 "정직", "진짜", "완전", "유일"이라는 단어로 스스로 평가하는 것이다. 근거를 보여 주면 독자가 판단한다.

### 2.6 app.py와 레포

app.py에는 라이트 테마로 바꾼 뒤 남은 다크 상수(PT, C_AI 등 Tailwind violet-500 #8b5cf6, DARK_BG)가 정의만 되고 쓰이지 않는다. 상태 문자열에 체크와 경고 이모지가 있다. 루트에 verify_*.py 42개, 결과 JSON 11개, docx 3개(md의 변환본), _vN 파일 12개.

## 3. 적용 방법

우선순위 순이다. P0은 숫자, P1은 시각 방향, 나머지는 청소다.

### P0. 헤드라인 숫자를 문서 결론과 맞춘다

- 히어로에서 "고정 60° 대비 +5.7%" 하나를 띄우는 대신, 비교 상대를 명시한 두 숫자를 같은 크기로 나란히 둔다. "최고 고정각 80° 대비 +1.7%"와 "방치(15° 고착) 대비 +33%". POSITIONING.md 1절의 표가 그대로 근거다.
- 60°는 관행 45°도 교과서 37.5°도 아니어서 상대로서 설명이 안 된다. 쓰려면 "회사 관행 45° 대비 +10%"로 바꾼다.
- 소수점. kWh는 1자리, %는 1자리, R²는 2자리(0.97), 추종률은 정수. 모델이 못 받치는 자릿수를 화면에 내지 않는다.
- "정직한 결과:", "AI 의 딜레마" 같은 UI 카피는 라벨을 떼고 두 문장으로 줄인다.

### P1. 시각 방향을 주제에서 다시 꺼낸다

현재 팔레트를 유지하면 2026년 기준으로 "AI가 만든 예쁜 페이지"로 읽힌다. 다른 클리셰(근흑에 산성 초록, 신문형 헤어라인)로 옮기는 것도 답이 아니다. 이 제품의 재료에서 꺼낸다.

재료. 도면 1043A, 알루미늄 압출 블레이드, 결정질 실리콘 셀(짙은 남색), 기상청 관측 시계열, 특허 도면(검은 선, 도8), 그리고 설계 검토에서 쓰는 레드라인 마크업.

제안 토큰.
- 바탕: 도면 트레이싱지 톤의 차가운 회백 #F3F4F2. 크림 계열(황색 기운)을 피한다.
- 잉크: #17191C.
- 괘선: #C9CDC8. 카드 테두리 대신 가로 괘선 하나로 절을 나눈다.
- PV 남색: #1E3A5F. 측정값, 물리 모델, 실제 셀 면을 나타낸다.
- 레드라인: #C0272D. AI가 내린 결정(각도, 스케줄), 도면 위 마크업, 주석. 설계 검토에서 붉은 선이 "변경 지시"인 관습과 맞는다.
- 알루미늄: #8A9096. 보조 텍스트, 비교 대상(고정각).
- 다크 테마는 같은 역할로 #16181B, #E4E6E2, #33373B, #8FB3DD, #F0605F, #9AA0A6.

서체. 본문과 제목 IBM Plex Sans KR(300/400/600), 숫자와 파일명 IBM Plex Mono. Google Fonts에서 제공된다. Plex는 기술 문서 register로 쓰이는 서체이고 한국어 판이 있어 Pretendard(Inter 파생)를 대체할 수 있다. 큰 숫자는 굵기를 올리는 대신 300 굵기로 크기를 키운다.

레이아웃. 카드는 상호작용이 있는 곳(시뮬레이터, 트윈)에만. 통계 타일 3열은 한 줄 텍스트 표로 바꾼다. STEP 1~3은 세로 타임라인 한 줄로 바꾸거나 본문 문단 셋으로 푼다. 절 번호와 가운뎃점 구분자는 뺀다. 아이브로우가 필요하면 소문자 라벨 하나만.

아이콘. Tabler 아이콘 폰트 대신 도면 기호식 라인 SVG 3~4개만(블레이드 단면, 태양 고도, 기상). 이모지는 UI 어디에도 쓰지 않는다.

모션. 3D 시뮬레이터 재생 한 곳만 남긴다. 로고 흔들림, 태양 맥동, 블레이드 기울임 루프, 스크롤 페이드업은 제거하고 prefers-reduced-motion을 존중한다.

차트. 02 차트는 GHI와 각도를 위아래 두 패널로 나누고 선 끝에 직접 라벨을 붙인다. 범례 제거. 눈금선은 가로만, 색은 괘선색. 03 비교는 지금처럼 가로 막대에 값 직접 표기를 유지하되 AI 막대만 레드라인, 나머지는 알루미늄. 월별 차트의 "회색 + 하나 강조"는 이미 맞다.

Unity HUD. 왼쪽 패널을 시각, 각도, 이득 3줄로 줄이고 나머지는 토글로 숨긴다. 셀프테스트 문자열은 -debug 플래그 뒤로 옮긴다. 슬라이더 라벨 크기를 값보다 한 단계 작게.

### P2. 템플릿 크롬 제거(main.html, explainer.html)

- 번호 아이브로우 7개와 배지 01~04 삭제.
- 가운뎃점 구분자 48개는 문장 또는 줄바꿈으로.
- 화살표 글리프는 "이면", "으로" 같은 조사로.
- 무한 애니메이션 5개 삭제(로고, 태양 2개, 맥동, 블레이드).
- 페이드업 삭제. 페이지가 로드된 순간 모든 절이 보여야 한다.
- 999px 필 7개는 radius 2px 사각 라벨로.
- backdrop-filter 2개 제거.
- explainer의 왼쪽 강조 바, 이모지 콜아웃, 칩 3열 두 벌 제거.

### P3. 레거시 삭제

`git rm main_screen.html main_screen.preview.html ops_screen.html shading_screen.html validation_screen.html`. app.py에서 PT, C_AI, C_F60, C_V90, C_GHI, DARK_BG 정의를 지운다(사용처 없음). st.set_page_config의 page_icon 이모지와 상태 문자열의 이모지를 텍스트로 바꾼다.

### P4. 텍스트 규칙

아래 블록을 CLAUDE.md로 두면 이 레포에서 생성되는 문서에 적용된다. 짧게 유지한다. Anthropic 문서는 규칙 파일이 길어지면 통째로 무시된다고 경고한다.

```
# 문서 규칙
- em-dash(—)를 쓰지 않는다. 쉼표, 마침표, 괄호를 쓴다.
- 불릿을 "**라벨**: 설명"으로 시작하지 않는다. 문장으로 쓴다.
- "X가 아니라 Y" 대구법을 문서당 1회 이하로 쓴다.
- 정직, 진짜, 완전, 핵심, 유일, 근본, 견고, 엄격을 자기 결과에 붙이지 않는다. 근거를 보여준다.
- 이모지, ★, ①②③, → 를 본문에 쓰지 않는다.
- "정리하면", "한 문장으로", "한 줄로" 마무리 공식을 쓰지 않는다. 결론은 첫 문단에 쓴다.
- 숫자는 방법이 받치는 자릿수까지만 쓴다. %는 소수 1자리, R²는 2자리.
- 볼드는 문서당 5회 이하. 제목은 3단계 이하.
- 표는 열거 가능한 사실에만. 본문을 다시 적는 표를 만들지 않는다.
- 같은 어미를 3문장 연속 쓰지 않는다.
- 결과에는 실패 사례와 "무엇이 나오면 결론이 바뀌는가"를 한 단락 넣는다.
- 커밋 전 python slop_lint.py --strict 를 통과한다.
```

기존 문서 11개를 한 번에 고칠 필요는 없다. POSITIONING.md와 AITILT_OVERVIEW.md는 외부에 나가는 문서이므로 먼저 고친다. TRACKING_GAIN_INVESTIGATION.md는 작업 일지 성격이므로 docs/worklog/로 옮기고 린트 대상에서 뺀다.

### P5. 레포 위생

- verify_*.py 42개는 verify/ 디렉터리로. 문서나 app.py가 이름으로 참조하는 31개는 그대로 옮기고, 참조가 없는 11개는 삭제하거나 verify/archive/로.
- *_results.json은 results/로. 결정론 재현이 목적이면 스크립트가 다시 만들면 되므로 커밋 대상에서 뺄 수 있다.
- docx 3개는 삭제하고 필요할 때 md에서 변환한다.
- _v15, _v16, _v17 접미사는 현재 버전만 남기고 이전 버전 파일은 삭제한다. 이력은 git에 있다.
- _debate_zerobase_result.json(78KB), _debate_zerobase.wf.js는 docs/worklog/로.

### P6. 린트 운영

`python slop_lint.py`는 마크다운 텍스트, HTML 텍스트와 CSS, 파이썬 문자열, 레포 구조를 검사한다. `--strict`를 붙이면 HIGH가 하나라도 있으면 종료 코드 1이다. 마크다운의 코드 블록은 건너뛰고, 파일 앞 400자 안에 `slop-lint: off`가 있으면 그 파일은 통째로 건너뛴다(이 가이드처럼 금지 패턴을 인용하는 문서용). 규칙은 전부 휴리스틱이라 히트를 읽고 판단한다. 새 문서와 화면은 HIGH 0을 목표로 하고, 기존 파일은 P4 순서대로 줄인다. CSS 검사는 보라 강조(색상 250~290°), 크림+세리프+테라코타 조합, 근흑+보라 조합, Inter/Pretendard, 글래스, ALL-CAPS 아이브로우, 무한 애니메이션, 스크롤 리빌, 번호 아이브로우, 가운뎃점, 화살표를 본다.

## 4. 발행 전 체크리스트

시각.
- 강조색 조합이 1세대(근흑+보라), 2세대(크림+세리프+테라코타) 어느 쪽도 아닌가.
- 서체가 Inter, Pretendard, Roboto, Space Grotesk가 아닌가.
- 번호 아이브로우, 가운뎃점 구분자, 화살표 글리프, ALL-CAPS 라벨이 없는가.
- 동일 카드 3열 또는 4열 그리드가 없는가.
- 무한 애니메이션이 0개이고 스크롤 리빌이 없는가.
- 이모지가 UI에 0개인가.
- 차트에 범례 대신 직접 라벨이 있고 이중 y축이 없는가.
- 큰 숫자마다 비교 상대가 같은 화면에 있는가.
- 소수점 자릿수가 방법의 정밀도를 넘지 않는가.

텍스트.
- 첫 문단에 결론이 있는가.
- em-dash 0, 볼드 라벨 불릿 0, 이모지 0인가.
- 자기평가 수식어를 근거로 바꿨는가.
- 실패 사례와 반증 조건이 있는가.
- 모든 숫자에 스크립트 이름이 붙어 있는가.

## 5. 이 문서의 한계

린트 규칙은 표면 신호만 본다. 관점 없는 글, 근거 없는 확신, 같은 논지의 반복은 사람이 읽어야 잡힌다. 위 진단은 렌더 스크린샷(합성 데이터 주입, 2026-09-13)과 정적 분석에 기댔고, 실제 배포 화면의 데이터와 다를 수 있다. P1의 토큰 제안은 하나의 방향이고, 도면과 셀 재료에서 다른 팔레트를 꺼낼 수도 있다. 어느 쪽이든 왜 그 색인지 한 줄로 쓸 수 있어야 한다.
