export const meta = {
  name: 'angle-control-zerobase-debate',
  description: '각도제어 부활 가능성: 제약 최대 해제(회전축·분절·밀폐·gcr 자유) 제로베이스 적대토론 — 5방향 제안→적대반박→재제안→심판, 물리계산 강제',
  phases: [
    { title: '제안', detail: '5개 서로 다른 메커니즘 방향, 각자 physics로 이득 숫자화' },
    { title: '반박', detail: '각 제안을 적대적으로 재계산해 죽이기 시도' },
    { title: '재제안', detail: '반박을 받고 살리거나 인정' },
    { title: '심판', detail: '전체 종합·부활 가능성 최종 판정' },
  ],
}

const FACTS = `
확정 사실(재발견 말고 여기서 출발):
- 제품: 에어컨 실외기실 벽면 BIPV 루버. 블레이드 현114mm·피치97.5mm·gcr=현/피치=1.169(겹침16.5mm 밀폐). 셀=M6 하프컷 83mm, 현 위에서 상단마진24/셀83/하7mm(하단 치우침).
- 지금 블레이드는 "위아래(고도)축"으로만 회전(tilt). 방위(동서)로는 못 돎.
- 자기음영: 촘촘히 겹쳐 위 블레이드가 아래를 25~40% 가림. tilt=수직(~90°)이 최적. 반대로 눕히면 음영0%지만 PV가 태양 등져 발전0(음수).
- 순수 태양추종(고도축) 이득 = 최고 고정각(~78°) 대비 연 +1.67~2.34%. 사실상 의미없음.
- 이미 죽은 후보(같은 이유로): 에어컨수요매칭·시간대요금·동서향벽·계절재고정·여름피크·차양겸용·눈비바람·개별블레이드각(0%)·가변피치(0%)·편심축(선형셀서 ≈0).
- 근본 사망원인 둘: (1)고도축만 돌아 동서 해를 못 쫓음 (2)값어치 큰 상황선 최적각이 거의 수직(90°)에 붙어 평평→각도 바꿔도 발전 거의 불변.
`

const PHYSICS = `
물리계산 필수(말로만 하지 말고 직접 python 실행해 숫자 대라). 실행 패턴(모듈 import는 프로젝트 디렉터리서만 됨):
  cd /Volumes/AISSD/ai-tilt && .venv/bin/python - <<'PY'
  import numpy as np, pandas as pd
  from physics_v3 import eff_poa      # eff_poa(tilt,elev,az,dni,dhi, sa=180.0) → 시간당 발전 대용치(빔·cos입사각·IAM·자기음영+확산+지면반사)
  from physics_v2 import panel_sf     # panel_sf(tilt,elev,az, hd=114, p=97.5, sa=180.0) → 현 전체 자기음영률(0~1)
  df = pd.read_csv('bipv_ai_master_data_v17.csv')      # 서울 10년: solar_elevation, solar_azimuth, dni, dhi (+timestamp)
  day = df[df.solar_elevation>0]
  # tilt=블레이드 경사(0=수평/90=수직), sa=표면방위(180=남향). 방위축 제어를 보려면 sa를 시간에 따라 바꿔라.
  PY
핵심 후크: sa(표면방위)가 파라미터다 → 방위축/2축 회전은 sa를 태양방위 az로 추종시켜 테스트할 수 있다. 다만 밀폐 루버는 물리적으로 tilt만 도는 구조임을 유념(sa를 돌리려면 제품이 달라짐 — 대가 명시).
분절/서브패널 독립각은 physics_v3의 균일무한루버 가정을 벗어나므로, 필요하면 N블레이드 광선추적을 직접 짜서 자기음영 재분배가 선형셀서 총량보존인지 깨보라.
`

const PROPOSAL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['mechanism', 'value_metric', 'claim', 'computed_number', 'baseline_stated', 'product_cost', 'self_verdict'],
  properties: {
    mechanism: { type: 'string', description: '물리적으로 무엇을 바꾸는가(회전축/분절/밀폐완화/구동 등)' },
    value_metric: { type: 'string', description: '어떤 값어치 축인가: 연kWh / 피크kW / 부하매칭 / 수명 / 쾌적 / 화재안전 등' },
    claim: { type: 'string', description: '주장 한 줄' },
    computed_number: { type: 'string', description: 'python으로 실제 계산한 이득 크기(반드시 실행한 숫자)' },
    baseline_stated: { type: 'string', description: '무엇 대비 이득인가(공정한 baseline 명시)' },
    product_cost: { type: 'string', description: '제품변경·비용·특허리스크·밀폐훼손 대가' },
    self_verdict: { type: 'string', description: '스스로 보기에 이게 팔 만한가(정직하게)' },
  },
}

const REBUT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'killer', 'recomputed', 'residual'],
  properties: {
    verdict: { type: 'string', enum: ['DIES', 'WOUNDED', 'SURVIVES'], description: '제안의 운명' },
    killer: { type: 'string', description: '죽이는 물리적/제품적 급소(baseline 반칙·자기음영·밀폐훼손·구동비용·특허 등)' },
    recomputed: { type: 'string', description: '내가 직접 재계산한 숫자(제안자 숫자를 검증/반증)' },
    residual: { type: 'string', description: '그래도 남는 진짜 값어치가 있다면(숫자로), 없으면 "없음"' },
  },
}

const REVISE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['concede', 'salvaged_claim', 'final_number', 'honest_status'],
  properties: {
    concede: { type: 'boolean', description: '반박을 받아들여 접는가' },
    salvaged_claim: { type: 'string', description: '살아남은 주장(있으면), 없으면 "없음"' },
    final_number: { type: 'string', description: '최종적으로 방어 가능한 숫자' },
    honest_status: { type: 'string', description: 'DEAD / MARGINAL / ALIVE 중 하나 + 한 줄 근거' },
  },
}

const JUDGE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['revival_verdict', 'best_mechanism', 'best_number', 'ranking', 'what_to_tell_user'],
  properties: {
    revival_verdict: { type: 'string', enum: ['NO', 'MARGINAL', 'YES'], description: '제약을 풀어도 각도제어가 부활 가능한가' },
    best_mechanism: { type: 'string', description: '그나마 가장 살아남은 메커니즘' },
    best_number: { type: 'string', description: '그 메커니즘의 방어 가능한 최고 숫자와 baseline' },
    ranking: {
      type: 'array',
      description: '5방향 순위',
      items: {
        type: 'object', additionalProperties: false,
        required: ['mechanism', 'status', 'number'],
        properties: {
          mechanism: { type: 'string' }, status: { type: 'string' }, number: { type: 'string' },
        },
      },
    },
    what_to_tell_user: { type: 'string', description: '사용자에게 보고할 결론(정직·구체·과대주장 금지, 한국어)' },
  },
}

const MANDATES = [
  { key: 'azimuth', dir: '회전축에 방위(동서) 자유도 추가 — 방위단독축 또는 2축. sa(표면방위)를 태양방위 az로 추종시켜 physics로 이득을 재라. 밀폐 루버가 방위로 돌려면 제품이 어떻게 달라지고(구동부·방수·특허) 그 대가가 이득을 잡아먹는지까지.' },
  { key: 'segment', dir: '블레이드를 상하 분절하거나 서브패널로 나눠 각 구획 독립각. 위 구획은 그림자 안 지는 각, 아래는 다른 각으로. N블레이드 광선추적을 직접 짜서, 선형셀(현방향 단일스트립)서 자기음영 재분배가 총량보존(Jensen)을 깨는지 숫자로 확인하라.' },
  { key: 'relax-closure', dir: '"닫으면 완전밀폐" 제약을 부분완화 허용(가변 gcr, 접이/텔레스코픽으로 낮엔 벌리고 밤엔 닫음). 벌렸을 때 발전이득을 physics로, 밀폐 본업 훼손·프라이버시·비 들이침 대가를 명시. gcr<1로 도망갈 수 있으나 그 대가를 정직하게.' },
  { key: 'nonpower', dir: '발전 총kWh 말고 다른 값어치로 각도제어를 정당화 — 피크시간 발전집중(요금)·부하매칭·PV수명(온도)·차양쾌적·화재안전·소음. 반드시 그 값어치를 숫자로. 단 "연최적고정 vs 여름각도제어" 같은 baseline 반칙 금지.' },
  { key: 'freeform', dir: '위 어디에도 안 묶임. 완전 제로베이스. 움직이는 블레이드(각도제어)가 고정보다 나은 시나리오를 무엇이든 발명하라 — 단 반드시 physics로 숫자를 대고 공정 baseline을 명시.' },
]

phase('제안')
log(`제로베이스 적대토론 시작: ${MANDATES.length}개 메커니즘 방향, 제약 최대 해제`)

const results = await pipeline(
  MANDATES,
  // stage 1: 제안
  (m) => agent(
    `너는 낙관적 발명가다. 이번엔 제약을 최대한 풀었다. 아래 방향으로 "각도제어(움직이는 블레이드)가 값어치를 낸다"를 성립시켜라.\n\n${FACTS}\n\n너의 방향: ${m.dir}\n\n${PHYSICS}\n\n반드시 python을 직접 실행해 이득 크기를 숫자로 대라. 공정한 baseline(무엇 대비인가)을 명시하라. 제품을 바꾸면 그 대가(구동·비용·특허·밀폐훼손)를 숨기지 마라.`,
    { label: `제안:${m.key}`, phase: '제안', effort: 'high', schema: PROPOSAL_SCHEMA }
  ).then(proposal => ({ mandate: m, proposal })),

  // stage 2: 적대 반박
  (prev, m) => agent(
    `너는 냉정한 반박자다. 아래 제안을 물리로 죽여라. 제안자 숫자를 그대로 믿지 말고 직접 python으로 재계산하라. baseline 반칙(연최적 vs 계절제어 등), 자기음영, 밀폐훼손, 구동/방수 비용, 특허(HDC KR102683082B1 냉각연동·KR102460661B1 편심축) 저촉을 급소로 삼아라.\n\n${FACTS}\n\n제안 방향(${m.key}): ${m.dir}\n제안 내용: ${JSON.stringify(prev.proposal)}\n\n${PHYSICS}\n\n직접 계산해서 DIES/WOUNDED/SURVIVES 판정하라. 그래도 남는 진짜 값어치가 있으면 숫자로 인정하라.`,
    { label: `반박:${m.key}`, phase: '반박', effort: 'high', schema: REBUT_SCHEMA }
  ).then(rebuttal => ({ ...prev, rebuttal })),

  // stage 3: 재제안
  (prev, m) => agent(
    `너는 다시 제안자다. 반박을 받았다. 정직하게 — 살릴 수 있으면 물리로 살리고(재계산), 죽었으면 인정하라. 과대주장 금지.\n\n제안 방향(${m.key}): ${m.dir}\n너의 제안: ${JSON.stringify(prev.proposal)}\n반박: ${JSON.stringify(prev.rebuttal)}\n\n${PHYSICS}\n\nDEAD/MARGINAL/ALIVE로 최종 상태를 정하라.`,
    { label: `재제안:${m.key}`, phase: '재제안', effort: 'medium', schema: REVISE_SCHEMA }
  ).then(revision => ({ ...prev, revision }))
)

phase('심판')
const clean = results.filter(Boolean)
const dossier = clean.map(r => ({
  mechanism: r.mandate.key,
  proposal: r.proposal,
  rebuttal: r.rebuttal,
  revision: r.revision,
}))

const judgment = await agent(
  `너는 심판이다. 5개 메커니즘 방향의 [제안→반박→재제안]을 모두 읽고, "제약을 최대한 풀어도 각도제어가 부활 가능한가"를 최종 판정하라. 정직·구체·과대주장 금지. 살아남은 게 있으면 숫자와 공정 baseline으로, 없으면 왜 근본적으로 죽는지 한 줄로.\n\n${FACTS}\n\n토론 전문:\n${JSON.stringify(dossier, null, 1)}\n\n사용자에게 보고할 결론을 한국어로 명확히.`,
  { label: '심판', phase: '심판', effort: 'xhigh', schema: JUDGE_SCHEMA }
)

return { judgment, dossier }
