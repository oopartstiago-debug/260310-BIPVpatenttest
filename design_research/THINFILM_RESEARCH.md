# 다층박막(컬러 BIPV 간섭 코팅) 웹 보고서: 외부조사와 구현 방법 점검

작성 2026-09-28. 조사 에이전트 3개(도메인 시각 언어, 구현 방법, 다른 모델 결과물 비교), 검색 약 100회.
sciencedirect, optica, fraunhofer, arxiv 등은 프록시에 막혀 원문 대신 검색 요약으로 확인한 항목이 있다. 그런 항목은 [미확인]으로 표시했다.
구현 결과는 `thinfilm/` (README 참고).

## 1. "Astra"와 "Fable" 결과물이 그럴듯해 보인 이유

- Fable은 Claude Fable 5.1(2026-09-01 출시)이다. Astra는 OpenAI GPT-6 Astra(2026-09-03 출시)일 가능성이 가장 높다. OpenAI가 프론트엔드 시각 판단력 강화를 내세웠고, Codex 모델 ID는 `gpt-6-astra`이다. Google Project Astra는 실시간 비서라서 해당하지 않는다.
- 공개 리더보드(스냅샷 날짜마다 다르며 검색 요약 기준)에서 Image-to-WebDev 부문은 Astra 1위, Fable 5.1 2위였다. Design Arena 웹 부문은 Kimi K3, Muse Spark 1.3, Fable 5.1, Opus 5.5가 비슷한 점수로 경합한다.
- 실무 비교에서 반복해서 나온 원인은 모델 자체보다 과정이었다.
  - Astra는 이미지를 직접 생성해 레이아웃에 넣는다. Claude 결과물은 CSS 도형과 자리표시자에 기댄다.
  - Kimi K3는 렌더를 보고 고치는 루프(vision in the loop)를 기본으로 돈다.
  - Fable은 절제와 마감에서 강하다는 평이 많다.
- 시사점: Claude Code에 스크린샷 루프와 실제 콘텐츠(실측 데이터, 실제 이미지)를 넣으면 격차 대부분이 줄어든다.
  - 모델을 바꾸려면 디자인 판정 서브에이전트만 `model: fable`로 두고, 반복 구현은 Opus로 한다([sub-agents 문서](https://code.claude.com/docs/en/sub-agents), [model-config](https://code.claude.com/docs/en/model-config)).
  - Astra는 `openai/codex-plugin-cc` 또는 Claude Code 밖에서 시안을 만든다. 그 결과 HTML과 스크린샷을 저장소에 넣어 DESIGN.md 추출용 레퍼런스로 쓴다.
- 이긴 시안을 규칙으로 만드는 비교표 항목: 타입 스케일 비, 간격 단위, 고유 색 수, 뷰포트당 정보량, 자리표시자 여부, 카피, 시그니처 장치 1개.

## 2. 이 분야에서 신뢰를 주는 시각 문법

선행 사례
- Kromatix(SwissINSO와 EPFL LESO, Schüler): 스퍼터 다층 간섭 필터에 각도별 CIELAB를 측정했다.
- Fraunhofer ISE MorphoColor(Bläsi et al., IEEE JPV 11, 1305, 2021): 텍스처 유리 뒷면 Bragg 스택으로 기준 모듈의 94% 이상 출력을 낸다. Megasol이 사업화했다.
- 모델링: Wessels et al., Opt. Express 30, 14586 (2022). microfacet BSDF와 TMM을 결합했다.

핵심 논문
- Halme & Mäkinen, EES 12, 1274 (2019): 효율 손실을 정하는 것은 명도라는 점을 보였다. 어두운 세트의 근거다.
- Røyset et al., Energy Build. 298, 113517 (2023).
- Ortiz Lizcano et al., Solar RRL (2023): 텍스처로 색상과 채도의 각도 안정성을 높였다.
- Gewohn et al., AIP Adv. 11, 095104 (2021): 예측색과 측정색의 ΔE00 1.34 [미확인].
- IEA PVPS T15-07 (2019).

논문의 표준 그림 세트
- R(λ)와 색 패치
- 각도별 R(λ, θ)
- a*b* 궤적
- ΔE00 대 각도(임계선 1, 3, 5)
- 상대 광전류 대 L*
- 두께 비례 단면도
- 예측 대 측정

그리는 규칙
- 가시역 380–780 nm를 주 그래프로 둔다. AM1.5G는 옅은 면, V(λ)는 점선으로 뒤에 깐다.
- 각도 계열은 한 색상의 명도 단계로 칠하고 직접 라벨을 단다.
- 데이터 계열 색과 물리색(스와치)을 섞지 않는다.
- 스와치는 N5 회색(#777) 주변에 평면 사각형으로 둔다. 그라데이션, 그림자, 광택은 넣지 않는다. Lab과 ΔE00을 병기한다.
- 모든 그림에 조건 태그(TMM 방식, 광원, 관찰자, Δλ)와 n,k 출처를 붙인다.

"19°" 가설
- 코팅이 유리 뒷면에 있으면 스택이 받는 각도는 유리 내부 굴절각이다. 공기 30°는 n = 1.52 유리 안에서 19.2°가 된다. 각도는 항상 공기와 유리 두 기준으로 병기한다. [원문 미확인. 기준판에서 수치로 확인함]

흔한 함정
- max 정규화로 어두운 색이 비비드해진다.
- 유리 앞면 반사(약 4%)를 빠뜨린다.
- 두꺼운 유리를 coherent로 처리해 가짜 프린지가 생긴다.
- Kischkat n,k(1.54–14.3 µm)를 가시광에 쓴다.
- 채널별 클리핑으로 색상이 이동한다.
- 흰색이나 검은색 배경에 스와치를 둔다.

## 3. 구현 방법 비교

| 선택지 | 품질 상한 | 에이전트 친화 | 공수 | 장기 위험 |
|---|---|---|---|---|
| Astro+MDX + Svelte + Plot/D3 + TS TMM + Python 사전계산 | 최고 | 높음 | 중 | 낮음 |
| Quarto + Closeread + OJS | 중상 | 최고 | 낮음 | 낮음 |
| Observable Framework | 상 | 높음 | 낮음 | 중상 (Cloud 폐지, 릴리스 정체) |
| Streamlit/Panel | 낮음 (모두 같은 모양) | 최고 | 최저 | 낮음 |
| Plotly 기본 템플릿 | 대시보드 느낌 | | | |

- 물리 계산: Python(`tmm`, `colour-science`)을 기준 구현이자 정답지로 두고, 웹은 TS/JS 포팅으로 실시간 계산한다. 두 구현을 golden 벡터로 대조한다. Pyodide는 첫 로드 10 MB 이상이라 보조 모드로만 쓴다.
- three.js `MeshPhysicalMaterial.iridescence`는 Belcour–Barla 단일 박막 근사다. 다층, 흡수, 두꺼운 유리를 표현하지 못하므로 정량 색에 쓰면 안 된다. 3D 미리보기에는 TMM으로 만든 각도별 RGB LUT 텍스처를 쓴다.
- 화면 색: sRGB와 P3를 분기하고, 색역 밖은 OKLCh에서 채도만 줄인다(Color.js 방식). ΔE00과 Lab을 항상 병기한다.
- 검증 루프: pytest와 Vitest(golden), Playwright 스크린샷과 axe, `npx impeccable detect`, Lighthouse CI.

## 4. 이번 기준판에서 택한 것과 이유

기준판은 의존성 없는 정적 페이지(ES 모듈과 손으로 그린 SVG)로 만들었다.
- 이유 1: 클라우드 컨테이너가 CDN을 막아 외부 라이브러리 렌더를 검증할 수 없었다.
- 이유 2: 맥 세션이 Astro나 Quarto로 옮길 때도 `physics/`, `tmm.js`, `color.js`, `model.js`는 그대로 재사용할 수 있다.
- 글꼴: IBM Plex Sans KR과 Plex Mono를 저장소에 넣었다(OFL, 약 1 MB).
- 색: 흰 지면에 무채색 데이터를 쓰고, 강조는 #1F4E79 하나만 쓴다. 크림, 세리프, 테라코타, 보라는 배제했다.

## 출처

- https://ieeexplore.ieee.org/document/9475075/
- https://opg.optica.org/abstract.cfm?uri=oe-30-9-14586
- https://pubs.rsc.org/en/content/articlelanding/2019/ee/c8ee03161d
- https://www.sciencedirect.com/science/article/pii/S0378778823007478
- https://onlinelibrary.wiley.com/doi/full/10.1002/solr.202300256
- https://pubs.aip.org/aip/adv/article/11/9/095104/661448
- https://iea-pvps.org/wp-content/uploads/2020/01/IEA-PVPS_15_R07_Coloured_BIPV_report.pdf
- https://github.com/sbyrnes321/tmm
- https://colour.readthedocs.io/
- https://www.nature.com/articles/s41597-023-02898-2
- https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_iridescence/README.md
- https://github.com/pbakaus/impeccable
- https://quarto.org/docs/interactive/ojs/
- https://observablehq.com/release-notes/2025-04-15-deprecating-observable-cloud
- https://github.com/openai/codex-plugin-cc
- https://www.kla.com/products/instruments/reflectance-calculator
