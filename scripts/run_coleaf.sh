#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CATEGORY="${CATEGORY:?Set CATEGORY}"
PYTHON="${PYTHON:-python}"
CANDIDATES="${CANDIDATES:-results/$CATEGORY/candidates}"
OUT="${OUT:-results/$CATEGORY/coleaf}"
ALPHA_GRID="${ALPHA_GRID:-0,0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0}"
mkdir -p "$OUT"
common=(--category="$CATEGORY" --vq_method=rqkmeans --window=5 --neighbors_per_item=200
  --history_lengths=10 --memory_ks=50 --candidate_mode=base_only --fusion_mode=score)
"$PYTHON" -m colagr.eval.evaluate_coleaf "${common[@]}" \
  --base_candidates="$CANDIDATES/val.pt" --split=val --alphas="$ALPHA_GRID" > "$OUT/validation.log"
ALPHA="$("$PYTHON" - "$OUT/validation.log" <<'PY'
import json, sys
for line in reversed(open(sys.argv[1], encoding='utf-8').read().splitlines()):
    if line.startswith('COLAGR_RESULT='):
        print(json.loads(line.split('=', 1)[1])['top_configs'][0]['alpha'])
        break
else:
    raise SystemExit('No validation result')
PY
)"
"$PYTHON" -m colagr.eval.evaluate_coleaf "${common[@]}" \
  --base_candidates="$CANDIDATES/test.pt" --split=test --alphas="$ALPHA" > "$OUT/test.log"
printf 'Selected CoLeaf alpha: %s\nValidation: %s\nTest: %s\n' "$ALPHA" "$OUT/validation.log" "$OUT/test.log"
