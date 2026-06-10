# AI Tilt — Fable 5 적대검토 프롬프트

> 사용법: 같은 Claude Code 세션에서 `/clear` → `/model`로 Fable 5 선택 → 아래 블록을 붙여넣기.
> (파일·Bash·RAG 접근이 필요하므로 웹 claude.ai chat 말고 반드시 세션 안에서.)

```
역할: 너는 Opus 4.8이 만든 BIPV 루버 각도제어 시스템(이름: AI Tilt) 진단(SHADING_POWER_AUDIT.md)을
적대적으로 검증하고, 핵심 물리모델을 더 옳게 재설계하는 SOTA 리뷰어다.
처음부터 재진단하지 마라(수확체감). 기존 audit를 의심하고, 어려운 물리에 집중해라.

대상 코드/데이터 (전부 실제로 읽고 근거로 인용):
- 프로젝트: /Users/wonetiago/projects/260310-BIPVpatenttest/
- 핵심 물리: app.py:119 eff_poa(), :104 panel_sf(), :116 svf()
  → 결정적 한 줄 app.py:125  np.maximum(pd2*(1-sf*0.7)+dd*s, 0)
- 오라클: argmax(eff_poa) == target_angle_v15 (CSV에서 100% 일치라고 audit가 주장)
- 데이터: bipv_ai_master_data_v15.csv (9.4MB) / 모델 bipv_xgboost_model_v15.pkl
- 진단서: SHADING_POWER_AUDIT.md (이게 검증 대상)
- 사내 근거(RAG): curl GET http://localhost:8200/api/query 로 인용 검증
- 하드웨어 전제: 마이크로인버터 HMS-500-1T(모듈별 MPPT), 낮은 시스템전압, KS C 8577 인증

반드시 의심하고 검증할 것 (rubber-stamp 금지, 못 미더우면 "검증불가"로 명시):
1. 음영 항의 형태 자체가 틀렸나? `beam*(1-sf*k) + diffuse*svf`라는 가법 분리가 옳은가.
   `0.7` derate가 진짜 주범인지, 아니면 모델 형태가 주범인지 구분하라.
2. audit의 "블레이드=3-substring 직렬+바이패스" 추상화가 HMS-500-1T 모듈별 MPPT 구조에서
   맞는 전기적 단위인가? 실제로 직렬인 건 블레이드 내부 셀인가, 모듈 내 블레이드인가,
   마이크로인버터당 모듈 몇 개인가. 추상화 레벨이 틀리면 retain 함수 전체가 무의미.
3. 계단형 retain(=substring 통째 바이패스)이 물리적으로 옳은가? 부분음영 셀은 바이패스
   다이오드가 순방향 도통하기 전까지 전류를 줄여서 통과시키는 regime이 있다. 계단 대신
   single-diode I-V 모델 기반 retain이 더 옳지 않은지 설계·비교하라.
4. panel_sf의 기하학적 음영률(선분교차)이 전기적 손실로 옳게 매핑되나? 그늘진 면적 ≠ 전력손실.
5. 오라클 정의가 옳은가: eff_poa의 "순간값 argmax"로 라벨을 만든다. 연간 에너지 수율 최적화가
   아니라 순간 최적이라면, 그 자체가 라벨 편향 아닌가.
6. albedo 누락(audit는 +5.8° 편향 주장)·온도 12% 가짜중요도 주장을 CSV로 재현/반박하라.
7. 동~서향 설치인데 az=180(남향) 가정 — 영향 정량화.

산출물 (정확히 이 구조):
A. audit 주장 판정표: [주장 | 확정/반박/검증불가 | 증거(file:line 또는 CSV 통계 또는 RAG 인용)]
B. audit가 놓친 신규 결함 (각각 증거 첨부)
C. 재설계한 retain/eff_poa 함수: 실제 Python 코드 + 물리적 근거 + 기존 대비 최적각/우위 변화
D. 수정된 P0 실행계획 (라벨 재생성→재학습 파이프라인 포함), 위험·되돌리기 포함
E. 자기 한계: 데이터/근거 부족으로 단정 못 한 부분 명시
```
