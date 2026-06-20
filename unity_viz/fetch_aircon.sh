#!/usr/bin/env bash
# Tier2 실외기 실모델(CC0) 재다운로드 — Poly Haven "Exterior Aircon Unit".
#   gltfast 가 런타임 로드하는 드롭인. gitignore 대상이라 클론 후 1회 실행하면 복구됨.
#   배선: LouverAgentPresenter.LoadAircon (실패 시 절차적 실외기 폴백).
set -e
cd "$(dirname "$0")/AITiltViz/AITiltViz/Assets/StreamingAssets/models/aircon" 2>/dev/null || {
  mkdir -p "$(dirname "$0")/AITiltViz/AITiltViz/Assets/StreamingAssets/models/aircon"
  cd "$(dirname "$0")/AITiltViz/AITiltViz/Assets/StreamingAssets/models/aircon"
}
API="https://api.polyhaven.com/files/exterior_aircon_unit"
echo "Poly Haven 파일 목록 조회..."
curl -s "$API" | python3 -c "
import sys,json,os
d=json.load(sys.stdin); g=d['gltf']['1k']['gltf']
print(os.path.basename(g['url'])+'\t'+g['url'])
for rel,info in g['include'].items(): print(rel+'\t'+info['url'])
" | while IFS=$'\t' read -r rel url; do
  mkdir -p "$(dirname "$rel")"
  curl -sL -A "Mozilla/5.0" -o "$rel" "$url"
  echo "  $rel"
done
echo "완료 → $(pwd)"
