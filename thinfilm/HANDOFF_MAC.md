# 맥 로컬 다층박막 세션용 인계

클라우드 세션(2026-09-28)은 맥의 sputtering 프로젝트 파일에 접근할 수 없었다. 그래서 외부조사 결과를 적용한
새 구현을 이 저장소 `thinfilm/` 에 먼저 만들고 검증했다. 맥 세션은 아래 프롬프트로 이어 가면 된다.

## 맥에서 붙여 넣을 프롬프트

```
sputtering 프로젝트를 점검하고 웹 보고서/시뮬레이터를 새 기준판으로 재구현해줘.

참고 기준판: GitHub oopartstiago-debug/260310-BIPVpatenttest 브랜치 claude/ai-design-quality-improvement-bv0bp1 의 thinfilm/
(git clone 또는 git fetch 후 thinfilm/README.md, thinfilm/HANDOFF_MAC.md, design_research/*.md 를 먼저 읽어라)

순서:
1. 기존 sputtering 코드와 HANDOFF.md 의 최신 "인계" 절, P2_최종판정_어두운세트.md, P6_텍스처_각도해소_조건.md 를 읽고
   기준판 thinfilm/physics/optics.py 와 계산 규약을 표로 비교해라: 입사 매질(공기/유리), 각도 기준, 유리·EVA 처리(coherent/incoherent),
   편광, 광원/관찰자, 색 적분법, 광전류 정의, n,k 출처. 다른 항목마다 어느 쪽이 맞는지 근거를 적어라.
2. "Fraunhofer 19° 불일치"를 먼저 확인해라. 공기 30° 가 유리(n≈1.52) 안에서 19.2° 가 된다.
   원문 각도가 공기 기준인지 유리 기준인지, 우리 TMM 의 입사 매질이 무엇이었는지 확인하고 결론을 BACKLOG 에 기록해라.
3. 기존 프로젝트의 측정 n,k(엘립소메트리)가 있으면 thinfilm/physics/nk 에 같은 YAML 형식으로 넣고 문헌 값을 대체해라.
   refractiveindex.info 의 Kischkat TiO₂/Si₃N₄ 는 1.54 µm 이상 적외선 데이터라 가시광에 쓰면 안 된다.
4. 기존 "어두운 세트"와 텍스처 결과를 기준판 형식(그림 번호, 캡션, 계산 조건 태그, N5 회색 스와치, 공기/유리 각도 병기)으로 옮겨라.
5. 매 변경 후: python3 physics/make_data.py && node tests/tmm.test.mjs, Playwright 스크린샷(1440, 390)을 직접 보고,
   npx impeccable detect web/index.html 가 0건인지 확인해라. 크림 배경+세리프+테라코타, 보라, 카드 그리드,
   대문자 라벨, 가운뎃점 메타, 이모지, 히어로 숫자는 쓰지 마라(2세대 AI 기본값 포함).
6. 디자인 판단이 막히면 같은 브리프로 Fable(/model fable)과 Opus 시안을 따로 만들고 스크린샷을 A/B 로 이름 바꿔 나에게 보여줘라.
```

## 기준판에서 확정한 것

- 적층: 공기 | 유리 3.2 mm (incoherent) | 코팅 (coherent) | EVA 0.45 mm (incoherent) | SiNₓ 75 nm | c-Si.
- 색: D65, CIE 1931 2°, colour-science 적분 가중치를 JSON 으로 내보내 웹이 그대로 쓴다. ΔE 는 CIEDE2000.
- 광전류: AM1.5G 광자속 × Si 유입 투과율, EQE = 1, 코팅 없는 모듈 대비.
- 각도: 입력은 공기 기준, 화면에는 유리 내부 각도를 함께 표기.
- 프리셋(5층 TiO₂/SiO₂, 5 nm 격자 재평가):
  - 청회색: L* 45.0, a* −4.1, b* −18.0, 상대 광전류 95.5 %, ΔE00(60°) 5.9.
  - 녹색: L* 53.5, a* −18.8, b* 4.5, 91.1 %.
  - 브론즈: L* 59.3, a* 3.9, b* 12.2, 82.8 %. 목표(52, 6, 20)보다 밝고 채도가 낮다. 평면 5층으로는 광전류 손실과 함께 맞추기 어려웠다.

## 아직 안 한 것

- 기존 로컬 결과와의 수치 대조(1번 단계).
- 두께 공차 몬테카를로 밴드, 텍스처 유리(microfacet + TMM), 측정 스펙트럼 대비 그림.
- 공유용 단일 HTML 번들(지금은 http 서버가 필요).
