#!/bin/bash
# tools/eval_e2e_v2.sh — Pipeline v2 E2E 평가 (main + sub 정확도)
# Usage: bash tools/eval_e2e_v2.sh <exp_name> <jsonl_path> [splits_file]

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
gt_main_map = {r['file_name']: r['main_category'] for r in splits['${SPLIT}']}
gt_sub_map  = {r['file_name']: r['sub_category']  for r in splits['${SPLIT}']}

main_correct = sub_correct = sub_correct_given_main = total = main_fail = 0
class_main_correct = collections.Counter()
class_main_total = collections.Counter()
class_sub_correct = collections.Counter()  # keyed by main
class_sub_total = collections.Counter()    # keyed by main (only main-correct)
main_confusions = collections.Counter()
sub_confusions = collections.Counter()  # within-main sub errors

for line in open('${OUT_FILE}'):
    r = json.loads(line)
    fn = r['file_name']
    gt_main = gt_main_map.get(fn)
    gt_sub  = gt_sub_map.get(fn)
    if gt_main is None: continue
    total += 1

    a = r.get('stage_a', {})
    c = r.get('stage_c', {})
    pred_main = a.get('pred_main')
    pred_sub  = c.get('pred_sub') if c else None

    class_main_total[gt_main] += 1
    if pred_main is None:
        main_fail += 1
        main_confusions[(gt_main, 'NO_DETECTION')] += 1
    elif pred_main == gt_main:
        main_correct += 1
        class_main_correct[gt_main] += 1
        class_sub_total[gt_main] += 1
        if pred_sub == gt_sub:
            sub_correct_given_main += 1
            class_sub_correct[gt_main] += 1
        else:
            sub_confusions[(gt_main, gt_sub, pred_sub)] += 1
    else:
        main_confusions[(gt_main, pred_main)] += 1

    # Overall sub accuracy: pred_sub == gt_sub regardless of main correctness
    if pred_sub == gt_sub:
        sub_correct += 1

print('===== Pipeline v2 E2E 평가 =====')
print(f'total: {total}')
print(f'main_fail: {main_fail} ({main_fail/total*100:.1f}%)')
print(f'main_correct: {main_correct} ({main_correct/total*100:.1f}%)')
print(f'sub_correct (E2E): {sub_correct} ({sub_correct/total*100:.1f}%)')
print(f'sub_correct (given main correct): {sub_correct_given_main}/{main_correct} '
      f'({sub_correct_given_main/max(main_correct,1)*100:.1f}%)')

print()
print('===== per-class main accuracy =====')
for cls in sorted(class_main_total, key=lambda c: class_main_correct[c]/max(class_main_total[c],1)):
    t = class_main_total[cls]; c = class_main_correct[cls]
    sub_t = class_sub_total[cls]; sub_c = class_sub_correct[cls]
    sub_str = f' | sub: {sub_c}/{sub_t} ({sub_c/max(sub_t,1)*100:.1f}%)' if sub_t else ''
    print(f'  {cls}: main {c}/{t} ({c/t*100:.1f}%){sub_str}')

print()
print('===== top 15 main confusions =====')
for (gt_cat, pred_cat), cnt in main_confusions.most_common(15):
    print(f'  {gt_cat} -> {pred_cat}: {cnt}')

print()
print('===== top 10 sub confusions (within correct main) =====')
for (m, gs, ps), cnt in sub_confusions.most_common(10):
    print(f'  [{m}] {gs} -> {ps}: {cnt}')
" 2>&1 | tee "$EVAL_FILE"

echo ""
echo "Saved: $EVAL_FILE"
