"""
Direct sub-category classification with CLIP — no Stage A/B split.

Instead of the 3-stage pipeline (main → detect/crop → sub), classify images
directly into one of the 54 active sub_categories in a single CLIP forward pass.

This tests whether CLIP can handle the full 54-way taxonomy at once and whether
skipping the main-first decomposition hurts or helps accuracy.

Model: SigLIP2-L-512 (google/siglip2-large-patch16-512) — best Stage A baseline.

Metrics:
  - sub_acc      : pred_sub == gt_sub  (E2E equivalent)
  - main_acc     : implied main(pred_sub) == gt_main  (upper bound view)
  - per-class breakdown for both levels

Usage:
    python exp_clip_direct_sub.py
    python exp_clip_direct_sub.py --model-id google/siglip-large-patch16-384
    python exp_clip_direct_sub.py --splits splits/splits.json --split test
    python exp_clip_direct_sub.py --limit 200  # quick smoke-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from config import CATEGORY_CONFIG
from src.clip_zeroshot import DEFAULT_MODEL_ID, CLIPZeroShot
from src.dataset import load_splits

SIGLIP2_L_512 = "google/siglip2-large-patch16-512"


def build_sub_candidates() -> dict[str, str]:
    """Flatten CATEGORY_CONFIG → {sub_label: clip_prompt} for all 54 active subs."""
    candidates: dict[str, str] = {}
    for entry in CATEGORY_CONFIG.values():
        candidates.update(entry["sub_categories"])
    return candidates


def build_sub_to_main() -> dict[str, str]:
    """Map each sub_label → main_kor using CATEGORY_CONFIG keys."""
    mapping: dict[str, str] = {}
    for main_kor, entry in CATEGORY_CONFIG.items():
        for sub_label in entry["sub_categories"]:
            mapping[sub_label] = main_kor
    return mapping


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, default=Path("splits/splits.json"))
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model-id", type=str, default=SIGLIP2_L_512)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=Path, default=None,
                   help="default: outputs/clip_direct_sub_{model_short}.jsonl")
    return p.parse_args()


def model_short_name(model_id: str) -> str:
    import re
    base = model_id.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def main():
    args = parse_args()

    short = model_short_name(args.model_id)
    if args.out is None:
        args.out = Path(f"outputs/clip_direct_sub_{short}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.splits)
    if args.split not in splits:
        raise SystemExit(f"[error] split '{args.split}' not in {args.splits}")
    records = splits[args.split]
    if args.limit:
        records = records[: args.limit]

    candidates = build_sub_candidates()
    sub_to_main = build_sub_to_main()
    n_subs = len(candidates)

    print(f"[info] model={args.model_id}", file=sys.stderr)
    print(f"[info] direct 54-way sub classification  ({n_subs} candidates)", file=sys.stderr)
    print(f"[info] images={len(records)}", file=sys.stderr)
    for sub_label, prompt in sorted(candidates.items())[:5]:
        print(f"  {sub_label!r}: {prompt!r}", file=sys.stderr)
    print("  ...", file=sys.stderr)

    clf = CLIPZeroShot(model_id=args.model_id)

    n_sub_correct = 0
    n_main_correct = 0
    n_total = 0

    # {sub_label: [total, correct]}
    per_sub: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # {main_kor: [total, correct_main, correct_sub]}
    per_main: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    fout = args.out.open("w", encoding="utf-8")
    t0 = time.time()

    for i, rec in enumerate(records, 1):
        try:
            image = Image.open(rec.image_path).convert("RGB")
        except Exception as e:
            fout.write(json.dumps(
                {"file_name": rec.file_name, "error": f"open_failed:{e}"},
                ensure_ascii=False) + "\n")
            continue

        result = clf.classify(image, candidates)
        pred_sub = result["pred"]
        pred_main = sub_to_main.get(pred_sub, "")
        score = result["score"]
        all_scores = result["all_scores"]

        gt_sub = rec.sub_category
        gt_main = rec.main_category

        sub_ok = pred_sub == gt_sub
        main_ok = pred_main == gt_main

        n_total += 1
        if sub_ok:
            n_sub_correct += 1
        if main_ok:
            n_main_correct += 1

        per_sub[gt_sub][0] += 1
        if sub_ok:
            per_sub[gt_sub][1] += 1

        per_main[gt_main][0] += 1
        if main_ok:
            per_main[gt_main][1] += 1
        if sub_ok:
            per_main[gt_main][2] += 1

        topk = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
        record = {
            "file_name": rec.file_name,
            "gt_main": gt_main,
            "gt_sub": gt_sub,
            "pred_sub": pred_sub,
            "pred_main": pred_main,
            "score": round(score, 6),
            "sub_correct": sub_ok,
            "main_correct": main_ok,
            "topk": [{"label": k, "score": round(s, 6)} for k, s in topk],
            "model_id": args.model_id,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        if i % 200 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(records) - i) / rate
            print(
                f"  [{i}/{len(records)}] "
                f"sub={n_sub_correct/n_total*100:.1f}% "
                f"main={n_main_correct/n_total*100:.1f}% "
                f"rate={rate:.2f} img/s eta={eta/60:.1f}m",
                file=sys.stderr,
            )

    fout.close()
    dt = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[done] {n_total} images in {dt:.1f}s ({dt/max(1,n_total):.3f}s/img)",
          file=sys.stderr)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"DIRECT 54-WAY SUB CLASSIFICATION — {args.model_id}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  implied main_acc : {n_main_correct}/{n_total}  "
          f"({n_main_correct/n_total*100:.2f}%)", file=sys.stderr)
    print(f"  sub_acc (=E2E)   : {n_sub_correct}/{n_total}  "
          f"({n_sub_correct/n_total*100:.2f}%)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    print(f"\n--- Per main_category (sorted by sub_acc) ---", file=sys.stderr)
    rows = []
    for main_kor, (tot, main_ok, sub_ok) in sorted(per_main.items()):
        rows.append((main_kor, tot, main_ok, sub_ok,
                     main_ok / tot * 100, sub_ok / tot * 100))
    rows.sort(key=lambda r: r[5])
    for main_kor, tot, main_ok, sub_ok, main_pct, sub_pct in rows:
        print(f"  {main_kor:<30}  n={tot:>4}  "
              f"main={main_pct:>5.1f}%  sub={sub_pct:>5.1f}%", file=sys.stderr)

    print(f"\n--- Per sub_category (sorted by acc) ---", file=sys.stderr)
    sub_rows = []
    for sub_label, (tot, ok) in sorted(per_sub.items()):
        sub_rows.append((sub_label, tot, ok, ok / tot * 100))
    sub_rows.sort(key=lambda r: r[3])
    for sub_label, tot, ok, pct in sub_rows:
        print(f"  {sub_label:<50}  n={tot:>4}  acc={pct:>5.1f}%", file=sys.stderr)

    print(f"\n[saved] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
