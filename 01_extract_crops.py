"""
Step 1 — Extract DINO crops for all records (run once, never again).

Reads splits/splits.json, runs Grounding DINO on every image, saves the
best-confidence crop to data/crops/{split}/{file_name}, and writes
splits/crop_splits.json with updated image_path fields.

Both Exp 1 (DINO+CLIP zero-shot) and Exp 2 (Linear Probe) use the same
pre-extracted crops to ensure a fair comparison.

Usage:
    python 01_extract_crops.py
"""
import argparse
import json
from pathlib import Path

from config import CATEGORY_CONFIG
from src.dataset import SampleRecord, load_splits, save_splits
from src.dino import DINODetector


def parse_args():
    p = argparse.ArgumentParser(description="Extract DINO crops for all split records")
    p.add_argument("--splits", type=Path, default=Path("splits/splits.json"))
    p.add_argument("--crop-dir", type=Path, default=Path("data/crops"))
    p.add_argument("--out", type=Path, default=Path("splits/crop_splits.json"))
    return p.parse_args()


def main():
    args = parse_args()

    if args.out.exists():
        print(f"[WARN] {args.out} already exists. Delete it first to re-extract.")
        return

    splits = load_splits(args.splits)
    detector = DINODetector()
    print("[DINO] Model loaded.")

    crop_splits: dict[str, list[SampleRecord]] = {}
    total = sum(len(v) for v in splits.values())
    done = 0
    fallback_counts: dict[str, int] = {}

    for split_name, records in splits.items():
        out_dir = args.crop_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        updated: list[SampleRecord] = []
        fallbacks = 0

        for rec in records:
            done += 1
            crop_path = out_dir / rec.file_name

            if crop_path.exists():
                updated.append(SampleRecord(
                    image_path=str(crop_path),
                    file_name=rec.file_name,
                    main_category=rec.main_category,
                    sub_category=rec.sub_category,
                    group_id=rec.group_id,
                ))
                print(f"[{done}/{total}] skip (exists) {split_name}/{rec.file_name}")
                continue

            from PIL import Image
            try:
                image = Image.open(rec.image_path).convert("RGB")
            except Exception as e:
                print(f"[{done}/{total}] ERROR open {rec.image_path}: {e}")
                updated.append(rec)
                continue

            dino_prompt = CATEGORY_CONFIG.get(rec.main_category, {}).get("dino_prompt", "object")
            crop, is_fallback = detector.best_crop(image, dino_prompt)

            if is_fallback:
                fallbacks += 1
                print(f"[{done}/{total}] {split_name}/{rec.file_name} → fallback (no detection)")
            else:
                print(f"[{done}/{total}] {split_name}/{rec.file_name} → crop saved")

            crop.save(crop_path)
            updated.append(SampleRecord(
                image_path=str(crop_path),
                file_name=rec.file_name,
                main_category=rec.main_category,
                sub_category=rec.sub_category,
                group_id=rec.group_id,
            ))

        crop_splits[split_name] = updated
        fallback_counts[split_name] = fallbacks

    save_splits(args.out, crop_splits)
    print(f"\n[SAVED] {args.out}")
    for split_name, recs in crop_splits.items():
        fb = fallback_counts.get(split_name, 0)
        print(f"  {split_name}: {len(recs)} records, {fb} fallbacks (full image used)")
    print("[NOTE] crop_splits.json is fixed. Do not regenerate unless splits.json changes.")


if __name__ == "__main__":
    main()
