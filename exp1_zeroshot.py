"""
Experiment 1 — CLIP only vs DINO + CLIP (Zero-Shot).

Both methods classify sub_category using CLIP zero-shot similarity.
  - CLIP only   : full original image  (from splits/splits.json)
  - DINO + CLIP : pre-extracted crop   (from splits/crop_splits.json)

Results are saved to results/exp1/ and a comparison report is printed.

Usage:
    python exp1_zeroshot.py --split test
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import CATEGORY_CONFIG
from src.clip_zeroshot import CLIPZeroShot
from src.dataset import load_splits
from src.metrics import compare_reports, compute_report


def parse_args():
    p = argparse.ArgumentParser(description="Exp 1: CLIP only vs DINO+CLIP zero-shot")
    p.add_argument("--splits", type=Path, default=Path("splits/splits.json"))
    p.add_argument("--crop-splits", type=Path, default=Path("splits/crop_splits.json"))
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out-dir", type=Path, default=Path("results/exp1"))
    return p.parse_args()


def run_zeroshot(records, classifier: CLIPZeroShot, class_names: list[str]) -> dict:
    """Run CLIP zero-shot on records and return a metrics report."""
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    y_true, y_pred, probs_list = [], [], []

    for i, rec in enumerate(records):
        candidates = CATEGORY_CONFIG[rec.main_category]["sub_categories"]

        try:
            image = Image.open(rec.image_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] {rec.image_path}: {e}")
            continue

        result = classifier.classify(image, candidates)
        gt_idx = label_to_idx.get(rec.sub_category, -1)
        pred_idx = label_to_idx.get(result["pred"], -1)

        y_true.append(gt_idx)
        y_pred.append(pred_idx)

        prob_row = np.array([result["all_scores"].get(c, 0.0) for c in class_names])
        probs_list.append(prob_row)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(records)}] processed")

    probs_np = np.stack(probs_list) if probs_list else np.zeros((0, len(class_names)))
    return compute_report(y_true, y_pred, probs_np, class_names)


def print_comparison(name_a: str, name_b: str, rep_a: dict, rep_b: dict, class_names: list[str]):
    delta = compare_reports(rep_a, rep_b, class_names)
    print(f"\n{'='*65}")
    print(f"{'Metric':<35} {name_a:>10} {name_b:>10} {'Delta':>8}")
    print(f"{'-'*65}")
    print(f"{'Top1':<35} {rep_a['top1']:>10.4f} {rep_b['top1']:>10.4f} {delta['top1']:>+8.4f}")
    if rep_a.get('top3') and rep_b.get('top3'):
        print(f"{'Top3':<35} {rep_a['top3']:>10.4f} {rep_b['top3']:>10.4f} {delta['top3']:>+8.4f}")
    print(f"{'Macro F1':<35} {rep_a['macro_f1']:>10.4f} {rep_b['macro_f1']:>10.4f} {delta['macro_f1']:>+8.4f}")
    print(f"{'-'*65}")
    for c in class_names:
        va = rep_a["class_wise_top1"].get(c)
        vb = rep_b["class_wise_top1"].get(c)
        d = delta["class_wise_top1"].get(c)
        sa = f"{va:.4f}" if va is not None else "  N/A"
        sb = f"{vb:.4f}" if vb is not None else "  N/A"
        sd = f"{d:>+8.4f}" if d is not None else "    N/A"
        print(f"  {c:<33} {sa:>10} {sb:>10} {sd}")
    print(f"{'='*65}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits_orig = load_splits(args.splits)
    splits_crop = load_splits(args.crop_splits)

    orig_records = splits_orig[args.split]
    crop_records = splits_crop[args.split]

    # Build class list from config
    class_names = sorted({
        sub_label
        for cat in CATEGORY_CONFIG.values()
        for sub_label in cat["sub_categories"]
    })
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    print(f"[INFO] Evaluating split='{args.split}'  n={len(orig_records)}")

    classifier = CLIPZeroShot()

    print(f"\n--- Running CLIP only (full image, n={len(orig_records)}) ---")
    clip_only_report = run_zeroshot(orig_records, classifier, class_names)

    print(f"\n--- Running DINO + CLIP (pre-extracted crop, n={len(crop_records)}) ---")
    dino_clip_report = run_zeroshot(crop_records, classifier, class_names)

    clip_only_report["class_names"] = class_names
    clip_only_report["split"] = args.split
    dino_clip_report["class_names"] = class_names
    dino_clip_report["split"] = args.split

    out_a = args.out_dir / "clip_only.json"
    out_b = args.out_dir / "dino_clip.json"
    with open(out_a, "w", encoding="utf-8") as f:
        json.dump(clip_only_report, f, ensure_ascii=False, indent=2)
    with open(out_b, "w", encoding="utf-8") as f:
        json.dump(dino_clip_report, f, ensure_ascii=False, indent=2)

    print_comparison("CLIP only", "DINO+CLIP", clip_only_report, dino_clip_report, class_names)

    comparison = {
        "split": args.split,
        "class_names": class_names,
        "clip_only": clip_only_report,
        "dino_clip": dino_clip_report,
        "delta_dino_minus_clip_only": compare_reports(clip_only_report, dino_clip_report, class_names),
    }
    out_cmp = args.out_dir / "comparison.json"
    with open(out_cmp, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {out_a}")
    print(f"[SAVED] {out_b}")
    print(f"[SAVED] {out_cmp}")


if __name__ == "__main__":
    main()
