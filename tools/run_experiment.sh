#!/bin/bash
# tools/run_experiment.sh — 실험 실행 + 평가 + 기록 자동화
# Usage: bash tools/run_experiment.sh <exp_name> [box_thr] [text_thr] [splits_file] [extra_args]
# Example:
#   bash tools/run_experiment.sh v3_baseline 0.25 0.15 splits/splits.json
#   bash tools/run_experiment.sh v7_alias 0.25 0.15 splits/splits.json

set -e

EXP_NAME=${1:?"Usage: $0 <exp_name> [box_thr] [text_thr] [splits_file] [extra_args]"}
BOX_THR=${2:-0.25}
TEXT_THR=${3:-0.15}
SPLITS=${4:-splits/splits.json}
EXTRA_ARGS=${5:-""}
SPLIT="test"

OUT_DIR="outputs"
LOG_DIR="experiments"
mkdir -p "$OUT_DIR" "$LOG_DIR"

OUT_FILE="${OUT_DIR}/e2e_${EXP_NAME}.jsonl"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
EVAL_FILE="${LOG_DIR}/${EXP_NAME}_eval.txt"

echo "=============================================" | tee "$LOG_FILE"
echo "Experiment: $EXP_NAME" | tee -a "$LOG_FILE"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Splits: $SPLITS" | tee -a "$LOG_FILE"
echo "Split: $SPLIT" | tee -a "$LOG_FILE"
echo "Box threshold: $BOX_THR" | tee -a "$LOG_FILE"
echo "Text threshold: $TEXT_THR" | tee -a "$LOG_FILE"
echo "Extra args: $EXTRA_ARGS" | tee -a "$LOG_FILE"
echo "Output: $OUT_FILE" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# ── 1) 프롬프트 정보 기록 ──
echo "" >> "$LOG_FILE"
echo "=== Prompt Info ===" >> "$LOG_FILE"
python3 -c "
from src.label_mapping import build_gdino_text_prompt, PROMPT_BOOST
p = build_gdino_text_prompt()
print(f'Segments: {p.count(\".\")}'  )
print(f'Chars: {len(p)}')
print(f'PROMPT_BOOST: {sorted(PROMPT_BOOST)}')
print(f'Prompt: {p[:500]}...')
" >> "$LOG_FILE" 2>&1

# ── 2) 토큰 예산 확인 ──
echo "" >> "$LOG_FILE"
echo "=== Token Budget ===" >> "$LOG_FILE"
python3 -c "
from src.dino import DINODetector
from src.label_mapping import build_gdino_text_prompt
detector = DINODetector()
prompt = build_gdino_text_prompt()
n = detector.verify_prompt_budget(prompt)
print(f'Single prompt tokens: {n} / 256')
" >> "$LOG_FILE" 2>&1

# ── 3) 파이프라인 실행 ──
echo "" | tee -a "$LOG_FILE"
echo "=== Pipeline Run ===" | tee -a "$LOG_FILE"
START=$(date +%s)

python3 10_e2e_pipeline.py \
  --splits "$SPLITS" \
  --split "$SPLIT" \
  --box-threshold "$BOX_THR" \
  --text-threshold "$TEXT_THR" \
  --out "$OUT_FILE" \
  $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"

END=$(date +%s)
ELAPSED=$((END - START))
echo "Elapsed: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)" | tee -a "$LOG_FILE"

# ── 4) 평가 ──
echo "" | tee -a "$LOG_FILE"
echo "=== Evaluation ===" | tee -a "$LOG_FILE"

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
    pred = r['stage_a']['pred_main']
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
" 2>&1 | tee "$EVAL_FILE" | tee -a "$LOG_FILE"

# ── 5) 요약 기록 (CSV 한 줄 추가) ──
SUMMARY_CSV="${LOG_DIR}/summary.csv"
if [ ! -f "$SUMMARY_CSV" ]; then
  echo "exp_name,date,splits,box_thr,text_thr,extra_args,total,main_fail,main_fail_pct,correct,correct_pct,det_acc,elapsed_s" > "$SUMMARY_CSV"
fi

python3 -c "
import json
splits = json.load(open('${SPLITS}'))
gt_map = {r['file_name']: r['main_category'] for r in splits['${SPLIT}']}
correct = total = main_fail = 0
for line in open('${OUT_FILE}'):
    r = json.loads(line)
    gt = gt_map.get(r['file_name'])
    if gt is None: continue
    pred = r['stage_a']['pred_main']
    total += 1
    if pred is None: main_fail += 1
    elif pred == gt: correct += 1
detected = total - main_fail
mf_pct = main_fail/total*100 if total else 0
c_pct = correct/total*100 if total else 0
d_acc = correct/detected*100 if detected else 0
print(f'${EXP_NAME},$(date +%Y-%m-%d),${SPLITS},${BOX_THR},${TEXT_THR},${EXTRA_ARGS},{total},{main_fail},{mf_pct:.1f},{correct},{c_pct:.1f},{d_acc:.1f},${ELAPSED}')
" >> "$SUMMARY_CSV"

echo ""
echo "============================================="
echo "Done! Results saved:"
echo "  Log:     $LOG_FILE"
echo "  Eval:    $EVAL_FILE"
echo "  Summary: $SUMMARY_CSV"
echo "  Output:  $OUT_FILE"
echo "============================================="
