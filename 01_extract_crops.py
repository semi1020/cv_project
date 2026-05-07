"""
Step 1 — Extract DINO crops for all records (run once per splits.json).

Reads splits/splits.json, runs Grounding DINO on every image with the
main_category-specific dino_prompt, saves the best-confidence crop to
data/crops/{split}/{file_name}, and writes splits/crop_splits.json with:
  - image_path: path to the saved crop (or full image on fallback)
  - dino_meta : detection metadata (success, score, box, label_en, image_size)

dino_meta is the contract for downstream evaluation (02_dino_eval.py) and
is forwarded transparently by the CLIP pipeline (it ignores the field).

Usage:
    python 01_extract_crops.py
    python 01_extract_crops.py --crop-dir data/crops --out splits/crop_splits.json
"""
import argparse
from pathlib import Path

from PIL import Image

from config import CATEGORY_CONFIG
from src.dataset import SampleRecord, load_splits, save_splits
from src.dino import DINODetector


def parse_args():
    p = argparse.ArgumentParser(description="Extract DINO crops for all split records")
    p.add_argument("--splits", type=Path, default=Path("splits/splits.json"))
    p.add_argument("--crop-dir", type=Path, default=Path("/data/trash-data/crops"))
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
    summary: dict[str, dict[str, int]] = {}

    for split_name, records in splits.items():
        out_dir = args.crop_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        updated: list[SampleRecord] = []
        n_success = 0
        n_fallback = 0
        n_cached = 0
        n_error = 0

        for rec in records:
            done += 1
            crop_path = out_dir / rec.file_name

            if crop_path.exists():
                # Crop already on disk from a prior run. Skip detection to save
                # time but mark dino_meta as None so 02_dino_eval can flag the
                # gap. Delete the crop dir for a clean stat run.
                updated.append(SampleRecord(
                    image_path=str(crop_path),
                    file_name=rec.file_name,
                    main_category=rec.main_category,
                    sub_category=rec.sub_category,
                    group_id=rec.group_id,
                    dino_meta=None,
                ))
                n_cached += 1
                print(f"[{done}/{total}] skip (cached) {split_name}/{rec.file_name}")
                continue

            try:
                image = Image.open(rec.image_path).convert("RGB")
            except Exception as e:
                print(f"[{done}/{total}] ERROR open {rec.image_path}: {e}")
                updated.append(SampleRecord(
                    image_path=rec.image_path,
                    file_name=rec.file_name,
                    main_category=rec.main_category,
                    sub_category=rec.sub_category,
                    group_id=rec.group_id,
                    dino_meta={
                        "detection_success": False,
                        "fallback": True,
                        "score": None,
                        "box": None,
                        "label_en": None,
                        "image_size": None,
                        "error": f"open_failed:{e}",
                    },
                ))
                n_error += 1
                continue

            dino_prompt = CATEGORY_CONFIG.get(rec.main_category, {}).get(
                "dino_prompt", "object"
            )
            crop, meta = detector.best_crop(image, dino_prompt)

            if meta["fallback"]:
                n_fallback += 1
                print(f"[{done}/{total}] {split_name}/{rec.file_name} → fallback")
            else:
                n_success += 1
                print(
                    f"[{done}/{total}] {split_name}/{rec.file_name} "
                    f"→ crop (score={meta['score']:.3f})"
                )

            crop.save(crop_path)
            updated.append(SampleRecord(
                image_path=str(crop_path),
                file_name=rec.file_name,
                main_category=rec.main_category,
                sub_category=rec.sub_category,
                group_id=rec.group_id,
                dino_meta=meta,
            ))

        crop_splits[split_name] = updated
        summary[split_name] = {
            "total": len(updated),
            "success": n_success,
            "fallback": n_fallback,
            "cached": n_cached,
            "error": n_error,
        }

    save_splits(args.out, crop_splits)
    print(f"\n[SAVED] {args.out}")
    for split_name, stats in summary.items():
        det_rate = (
            stats["success"] / max(1, stats["success"] + stats["fallback"])
        )
        print(
            f"  {split_name}: total={stats['total']}, "
            f"success={stats['success']}, fallback={stats['fallback']}, "
            f"cached={stats['cached']}, error={stats['error']}, "
            f"detection_rate={det_rate:.1%} (excl. cached)"
        )
    print(
        "[NOTE] crop_splits.json is fixed. To regenerate, delete the crop "
        "directory AND splits/crop_splits.json."
    )


if __name__ == "__main__":
    main()
