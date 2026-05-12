#!/bin/bash
# tools/eval_clip_main.sh — CLIP main 분류 결과 평가 (v3 baseline과 동일 포맷)
# Usage: bash tools/eval_clip_main.sh <exp_name> <jsonl_path> [splits_file]
# Example: bash tools/eval_clip_main.sh clip_canonical outputs/clip_main_canonical.jsonl

set -e

EXP_NAME=${1:?"Usage: $0 <exp_name> <jsonl_path> [splits_file]"}
OUT_FILE=${2:?"Usage: $0 <exp_name> <jsonl_path> [splits_file]"}
SPLITS=${3:-splits/splits.json}
SPLIT="test"

LOG_DIR="experiments"
mkdir -p "$LOG_DIR"
EVAL_FILE="${LOG_DIR}/${EXP_NAME}_eval.txt"

python3 -c "
import json, collections

splits = json.load(open('${SPLITS}'))
gt_map = {r['file_name']: r['main_category'] for r in splits['${SPLIT}']}

correct = total = main_fail = 0
class_correct = collections.Counter()
class_total = collections.Counter()
confusions = collections.Counter()

for line in open('${OUT_FILE}'):
    r = json.loads(line)
    fn = r['file_name']
    gt = gt_map.get(fn)
    if gt is None:
        continue
    pred = r.get('pred_main')
    class_total[gt] += 1
    total += 1
    if pred is None:
        main_fail += 1
        confusions[(gt, 'NO_DETECTION')] += 1
    elif pred == gt:
        correct += 1
        class_correct[gt] += 1
    else:
        confusions[(gt, pred)] += 1

detected = total - main_fail
print(f'total: {total}')
print(f'main_fail: {main_fail} ({main_fail/total*100:.1f}%)')
print(f'main_ok: {detected} ({detected/total*100:.1f}%)')
print(f'correct: {correct} ({correct/total*100:.1f}%)')
if detected > 0:
    print(f'accuracy (among detected): {correct/detected*100:.1f}%')

print()
print('=== per-class accuracy ===')
for cls in sorted(class_total, key=lambda c: class_correct[c]/max(class_total[c],1)):
    t = class_total[cls]
    c = class_correct[cls]
    print(f'  {cls}: {c}/{t} ({c/t*100:.1f}%)')

print()
print('=== top 20 confusions ===')
for (gt_cat, pred_cat), cnt in confusions.most_common(20):
    print(f'  {gt_cat} -> {pred_cat}: {cnt}')
" 2>&1 | tee "$EVAL_FILE"

echo ""
echo "Saved: $EVAL_FILE"
