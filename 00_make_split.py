"""
Step 0 — Create a fixed train/val/test split (run once, never again).

Reads all CSVs from data/trash-data/csv/, filters by CATEGORY_CONFIG,
performs a grouped stratified split, and saves splits/splits.json.

Usage:
    python 00_make_split.py
    python 00_make_split.py --train 0.7 --val 0.15 --seed 42
"""
import argparse
import json
from pathlib import Path

from config import CATEGORY_CONFIG
from src.dataset import distribution, grouped_stratified_split, load_records, sample_per_category, save_splits


def parse_args():
    p = argparse.ArgumentParser(description="Create fixed train/val/test split")
    p.add_argument("--data-root", type=Path, default=Path("/data/trash-data"))
    p.add_argument("--out", type=Path, default=Path("splits/splits.json"))
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-per-category", type=int, default=100, metavar="N",
        help="Keep at most N records per sub_category (randomly sampled). "
             "Set to 0 to disable (use all records).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.out.exists():
        print(f"[WARN] {args.out} already exists. Delete it first to regenerate.")
        return

    csv_dir = args.data_root / "csv"
    image_dir = args.data_root / "image"

    include = list(CATEGORY_CONFIG.keys())
    print(f"[INFO] Loading records for: {include}")
    records = load_records(csv_dir=csv_dir, image_dir=image_dir, include_categories=include)

    if not records:
        raise RuntimeError(
            f"No records found. Check that {csv_dir} and {image_dir} exist and are populated."
        )

    print(f"[INFO] Total records: {len(records)}")
    sub_dist = distribution(records)
    for label, cnt in sorted(sub_dist.items()):
        print(f"  {label}: {cnt}")

    if args.max_per_category > 0:
        print(f"\n[INFO] Sampling at most {args.max_per_category} records per sub_category (seed={args.seed})")
        records = sample_per_category(records, max_per_category=args.max_per_category, seed=args.seed)
        print(f"[INFO] Records after sampling: {len(records)}")

    splits = grouped_stratified_split(
        records, seed=args.seed, train_ratio=args.train, val_ratio=args.val
    )

    print(f"\n[SPLIT] seed={args.seed}  train={args.train}  val={args.val}")
    for split_name, recs in splits.items():
        print(f"  {split_name}: {len(recs)} records")
        for label, cnt in sorted(distribution(recs).items()):
            print(f"    {label}: {cnt}")

    save_splits(args.out, splits)
    print(f"\n[SAVED] {args.out}")
    print("[NOTE] This file is fixed. Do not regenerate unless starting a new experiment.")


if __name__ == "__main__":
    main()
