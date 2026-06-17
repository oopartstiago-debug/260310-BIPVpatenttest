#!/usr/bin/env bash
# AI Tilt — ML-Agents 학습 환경(Python) 셋업.  사용자 Mac 터미널에서 실행.
# Unity 패키지 com.unity.ml-agents@4.0.3  ↔  Python mlagents==1.1.0  (Python 3.10.12)
# 공식 설치 문서 기준(2026): conda 가상환경 권장.
set -e

ENV_NAME="mlagents"
PYVER="3.10.12"

echo "[1/4] conda 환경 생성 ($ENV_NAME, python $PYVER)"
# Conda(또는 miniforge/Anaconda) 선행 설치 필요: https://docs.conda.io
conda create -y -n "$ENV_NAME" python="$PYVER"

# conda activate를 스크립트에서 쓰려면 hook 필요
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[2/4] grpcio 사전 설치(Apple Silicon 빌드 이슈 예방)"
conda install -y "grpcio=1.48.2" -c conda-forge || true

echo "[3/4] mlagents(==1.1.0) 설치 — PyTorch 등 의존성 자동"
python -m pip install --upgrade pip
python -m pip install mlagents==1.1.0

echo "[4/4] 설치 확인"
mlagents-learn --help >/dev/null && echo "  [OK] mlagents-learn 준비 완료"

cat <<'EOF'

=== 다음 단계 ===
1) Unity 에디터에서 메뉴  AI Tilt → Setup RL Scene (Train)  (BehaviorType=Default)
2) 터미널(이 conda 환경)에서:
     cd /Volumes/AISSD/ai-tilt/unity_viz
     mlagents-learn rl/config/louver_ppo.yaml --run-id=louver01
   "Start training by pressing the Play button" 메시지가 뜨면
3) Unity로 가서 ▶ Play.  학습 시작.
4) 보상곡선 보기(다른 터미널):
     tensorboard --logdir results
   브라우저에서 http://localhost:6006  →  LouverTilt/Environment/Cumulative Reward 우상향 확인.
5) 학습 종료 후 results/louver01/LouverTilt.onnx 를 Unity의
   BehaviorParameters > Model 에 넣고 BehaviorType=Inference Only 로 추론 실행.
EOF
