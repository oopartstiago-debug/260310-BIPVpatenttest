// AI Tilt — 강화학습 Agent
//   보상源 = 검증된 LouverPhysics.EffPoa (40/40 PASS physics_v3 C# 포팅)
//   oracle = LouverPhysics.OracleTilt (그리드 argmax 상한) → 평가/표시용
//
// 설계 의도 — "또 83°로 픽스되지 않게" 환경을 비정상·제약조건부로 만든다:
//   1) 태양이 에피소드 안에서 하루를 이동(추종 목표가 매 스텝 변함)
//   2) 모터 속도 한계(스텝당 각 변화 제한) → 순간이동 불가, 램프/예측 필요
//   3) 구동비용 패널티 → 무의미한 떨림 억제, 트레이드오프 발생
//   4) 에피소드마다 구름·현/피치·알베도 랜덤화 → 관측→각도 '함수'를 학습해야 함(상수론 못 이김)
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class LouverAgent : Agent {
    [Header("시각 연동(선택)")]
    public Transform louverRoot;   // 블레이드들의 부모(프레젠터가 이 밑에 생성)
    public Light sun;              // 가상 태양(Directional Light)

    [Header("에피소드/시간")]
    public int stepsPerDay = 120;     // 에이전트 MaxStep과 일치시킬 것
    public bool domainRandomize = true;

    [Header("제어 동역학")]
    [Range(0f, 90f)] public float minAngle = 0f;
    [Range(0f, 90f)] public float maxAngle = 90f;
    public float maxDeltaDegPerStep = 3f;   // 모터 속도 한계(스텝당 최대 각 변화)

    [Header("보상 가중치")]
    public float poaRef = 350f;   // 정규화 기준(최대 poa_oracle≈343)
    public float wEnergy = 1.0f;  // 발전 보상
    public float wMotor = 0.05f;  // 구동비용 패널티

    // 평가/표시용 공개 상태
    public float CurrentTilt { get; private set; }
    public float CurrentOracleTilt { get; private set; }
    public float CurrentPoa { get; private set; }
    public float CurrentOraclePoa { get; private set; }
    public float DayTrackingPct { get; private set; }  // 누적 POA / 누적 oracle POA (상한 대비 추종률)
    public float CurrentDni { get; private set; }       // 현재 직달일사(날씨 시각화용)
    public SolarDayEnv Env => env;

    readonly SolarDayEnv env = new SolarDayEnv();
    int step;
    float cumPoa, cumOraclePoa;

    public override void Initialize() {
        if (!env.Loaded) env.Load(Application.streamingAssetsPath);
        if (!LouverPhysics.Loaded) LouverPhysics.Load(Application.streamingAssetsPath);
        // 물리 무결성 1회 확인(검증 게이트) — 보상이 진짜 physics_v3와 일치하는지
        Debug.Log(LouverPhysics.SelfTest(Application.streamingAssetsPath));
    }

    public override void OnEpisodeBegin() {
        env.Randomize(domainRandomize);
        step = 0; cumPoa = 0f; cumOraclePoa = 0f;
        CurrentTilt = Random.Range(minAngle, maxAngle);  // 시작각 랜덤(상수 회피)
        DayTrackingPct = 0f;
        ApplySunAndBlades(0f);
    }

    public override void CollectObservations(VectorSensor sensor) {
        float t01 = stepsPerDay <= 1 ? 0f : (float)step / (stepsPerDay - 1);
        var s = env.Sample(t01);
        float azr = s.az * Mathf.Deg2Rad;
        sensor.AddObservation(Mathf.Clamp01(s.elev / 90f));                  // 1 태양 고도
        sensor.AddObservation(Mathf.Sin(azr));                              // 2 방위 sin
        sensor.AddObservation(Mathf.Cos(azr));                              // 3 방위 cos
        sensor.AddObservation(Mathf.Clamp01(s.dni / 1000f));                // 4 DNI
        sensor.AddObservation(Mathf.Clamp01(s.dhi / 400f));                 // 5 DHI
        sensor.AddObservation(CurrentTilt / 90f);                          // 6 현재 블레이드 각
        sensor.AddObservation(Mathf.Clamp01((env.chord / env.pitch) / 2f));// 7 현/피치비(설계 변수)
        sensor.AddObservation(t01);                                        // 8 하루 위상
    }

    public override void OnActionReceived(ActionBuffers actions) {
        float a = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float prev = CurrentTilt;
        CurrentTilt = Mathf.Clamp(CurrentTilt + a * maxDeltaDegPerStep, minAngle, maxAngle);
        float moved = Mathf.Abs(CurrentTilt - prev);

        float t01 = stepsPerDay <= 1 ? 0f : (float)step / (stepsPerDay - 1);
        var s = env.Sample(t01);

        // === 검증 물리로 발전량(보상源) ===
        float poa = LouverPhysics.EffPoa(CurrentTilt, s.elev, s.az, s.dni, s.dhi, env.albedo, env.chord, env.pitch);
        float oraTilt = LouverPhysics.OracleTilt(s.elev, s.az, s.dni, s.dhi, env.albedo, env.chord, env.pitch);
        float oraPoa = LouverPhysics.EffPoa(oraTilt, s.elev, s.az, s.dni, s.dhi, env.albedo, env.chord, env.pitch);

        float rEnergy = wEnergy * (poa / Mathf.Max(1f, poaRef));
        float rMotor  = wMotor * (moved / Mathf.Max(0.01f, maxDeltaDegPerStep));
        AddReward(rEnergy - rMotor);

        cumPoa += poa; cumOraclePoa += oraPoa;
        CurrentPoa = poa; CurrentOraclePoa = oraPoa; CurrentOracleTilt = oraTilt; CurrentDni = s.dni;
        DayTrackingPct = cumOraclePoa > 1e-3f ? 100f * cumPoa / cumOraclePoa : 0f;

        ApplySunAndBlades(t01);

        step++;
        if (step >= stepsPerDay) {
            // 종료 보너스: 하루 추종률(상한 대비) — 발전 보상과 같은 스케일로 소량
            AddReward(0.5f * (cumOraclePoa > 1e-3f ? cumPoa / cumOraclePoa : 0f));
            EndEpisode();
        }
    }

    // 개발자 수동 테스트(Behavior Type=Heuristic Only): ←/→ 로 눕히기/세우기
    // 이 프로젝트는 새 Input System을 쓰므로 레거시 Input.GetAxisRaw는 예외를 던진다 → 키보드 직접 조회.
    public override void Heuristic(in ActionBuffers actionsOut) {
        var ca = actionsOut.ContinuousActions;
        float v = 0f;
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null) {
            if (kb.rightArrowKey.isPressed || kb.dKey.isPressed) v += 1f;
            if (kb.leftArrowKey.isPressed  || kb.aKey.isPressed) v -= 1f;
        }
#else
        v = Input.GetAxisRaw("Horizontal");
#endif
        ca[0] = v;
    }

    void ApplySunAndBlades(float t01) {
        var s = env.Sample(t01);
        if (sun) {
            float er = s.elev * Mathf.Deg2Rad, ar = s.az * Mathf.Deg2Rad;
            Vector3 dirToSun = new Vector3(Mathf.Sin(ar) * Mathf.Cos(er), Mathf.Sin(er), Mathf.Cos(ar) * Mathf.Cos(er));
            sun.transform.forward = (-dirToSun).normalized;
        }
        if (louverRoot) {
            for (int i = 0; i < louverRoot.childCount; i++)
                louverRoot.GetChild(i).localRotation = Quaternion.Euler(CurrentTilt, 0, 0);
        }
    }
}
