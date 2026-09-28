# thinfilm: 간섭 코팅 컬러 모듈 계산기

커버유리 뒷면 TiO₂/SiO₂ 다층 코팅의 반사색, 관찰 각도별 색 변화, 상대 광전류를 계산하는 웹 리포트다.
2026-09-28 클라우드 세션에서 맥 로컬의 다층박막(sputtering) 프로젝트를 새로 구현하는 기준판으로 만들었다.
기존 로컬 코드와의 통합은 `HANDOFF_MAC.md` 참고.

## 실행

```bash
pip install tmm colour-science pvlib scipy pyyaml
cd thinfilm
python3 physics/design_presets.py   # 선택: 프리셋 재설계 (약 4분) → physics/stacks.json
python3 physics/make_data.py        # web/data/optics.json, tests/golden.json 생성 (약 15초)
node tests/tmm.test.mjs             # 웹 계산 ↔ Python 대조 (196 검사)
cd web && python3 -m http.server 8765   # http://localhost:8765 (file:// 로는 모듈이 로드되지 않는다)
```

## 구조

| 경로 | 역할 |
|---|---|
| `physics/optics.py` | 기준 구현. Byrnes `tmm.inc_tmm` + colour-science(D65, CIE 1931 2°) + AM1.5G |
| `physics/fastspec.py` | 최적화용 벡터화 TMM. `optics.spectrum` 과 2.6e-15 이내 일치, 약 200배 빠름 |
| `physics/design_presets.py` | 목표 Lab 에 맞춘 5층 프리셋 탐색 (차등진화) |
| `physics/make_data.py` | 웹 데이터와 golden 테스트 벡터 생성 |
| `physics/nk/` | refractiveindex.info 원본 YAML (CC0). TiO₂ Siefke, SiO₂ Malitson, Si₃N₄ Luke, Si Green 2008 |
| `web/src/tmm.js` | TMM JS 포팅 (coh/inc) |
| `web/src/color.js` | XYZ → Lab → CIEDE2000, sRGB 표시와 OKLCh 색역 매핑 |
| `web/src/model.js` | 적층 구성과 평가 (optics.py 와 같은 규약) |
| `web/src/app.js`, `charts.js` | 화면과 의존성 없는 SVG 그림 |
| `tests/tmm.test.mjs` | golden 대조, Sharma 2005 CIEDE2000 표준쌍, R+T ≤ 1, 30° → 19.2° |

## 검증 상태 (2026-09-28)

- JS ↔ Python: 196 검사 통과. 최대 오차 R 1.1e-8, T 2.0e-8, Lab 4.3e-7, ΔE00 4.0e-5.
- `npx impeccable detect web/index.html`: 0건. 이전 세션의 `slop_lint.py`: HIGH/MED 0건.
- Playwright 1440px, 390px 렌더: 가로 넘침 없음, 콘솔 오류 없음(favicon 404 제외).

## 모델 가정 (페이지 4절과 같음)

평면 유리만 다룬다. 유리 n = 1.52, EVA n = 1.48 은 분산 없는 상수다. 굴절률은 문헌 박막 값이라 스퍼터 조건에 따라 다르다.
광전류는 EQE = 1 인 상대값이며 셀 전기 손실은 없다. 두께 공차, 텍스처, 확산 반사, 측정 기하는 아직 모델에 없다.
