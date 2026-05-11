"""
E2E inference pipeline (no GT main_category).

Two stages:
  Stage A — Multi-class single-prompt GDINO.
  Stage B — Per-class GDINO re-detect + crop for CLIP.

Usage:
    python 10_e2e_pipeline.py --splits splits/splits.json --split test
    python 10_e2e_pipeline.py --splits splits/splits.json --split test --box-threshold 0.30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

from config import CATEGORY_CONFIG
from src.dataset import load_splits
from src.dino import DINODetector, BOX_THRESHOLD, TEXT_THRESHOLD
from src.label_mapping import (
    EN_ALIAS_TO_KOR,
    KOR_TO_EN,
    KEEP_MAINS,
    build_gdino_text_prompt,
    map_en_to_kor,
)


def _build_stage_a_prompt() -> tuple[str, list[str]]:
    prompt = build_gdino_text_prompt(canonical_only=True)
    active_mains = [m for m in KOR_TO_EN if m in KEEP_MAINS]
    return prompt, active_mains


def _resolve_main(label_en: str | None) -> str | None:
    if not label_en:
        return None
    s = label_en.lower().strip()
    if s in EN_ALIAS_TO_KOR:
        kor = EN_ALIAS_TO_KOR[s]
        return kor if kor in KEEP_MAINS else None
    kor = map_en_to_kor(label_en)
    return kor if kor in KEEP_MAINS else None


def _stage_a(detector: DINODetector, image: Image.Image, prompt: str,
             top_k: int,
             box_threshold: float = BOX_THRESHOLD,
             text_threshold: float = TEXT_THRESHOLD) -> dict:
    dets = detector.detect(image, prompt, with_labels=True,
                           box_threshold=box_threshold,
                           text_threshold=text_threshold)[:top_k]
    if not dets:
        return {
            "pred_main": None, "label_en": None, "score": None, "box": None,
            "topk": [],
        }

    topk = []
    pred_main = None
    pred_idx = -1
    for i, d in enumerate(dets):
        kor = _resolve_main(d.get("label_en"))
        topk.append({
            "label_en": d.get("label_en"),
            "label_kor": kor,
            "score": round(d["score"], 4),
            "box": list(d["box"]),
        })
        if pred_main is None and kor is not None:
            pred_main = kor
            pred_idx = i

    if pred_main is None:
        return {
            "pred_main": None,
            "label_en": dets[0].get("label_en"),
            "score": round(dets[0]["score"], 4),
            "box": list(dets[0]["box"]),
            "topk": topk,
        }

    best = dets[pred_idx]
    return {
        "pred_main": pred_main,
        "label_en": best.get("label_en"),
        "score": round(best["score"], 4),
        "box": list(best["box"]),
        "topk": topk,
    }


def _stage_b(detector: DINODetector, image: Image.Image, main_kor: str,
             crop_path: Path) -> dict:
    entry = CATEGORY_CONFIG.get(main_kor)
    if entry is None:
        return {
            "crop_path": None, "dino_prompt": None,
            "score": None, "box": None, "fallback": None,
            "error": f"main {main_kor!r} not in CATEGORY_CONFIG",
        }

    dino_prompt = entry["dino_prompt"]
    crop, meta = detector.best_crop(image, dino_prompt)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path)

    return {
        "crop_path": str(crop_path),
        "dino_prompt": dino_prompt,
        "score": meta["score"],
        "box": meta["box"],
        "label_en": meta.get("label_en"),
        "fallback": meta["fallback"],
    }


def _iter_image_paths(args) -> list[tuple[str, str]]:
    if args.splits is not None:
        splits = load_splits(args.splits)
        if args.split not in splits:
            raise SystemExit(f"[error] split '{args.split}' not in {args.splits}")
        recs = splits[args.split]
        return [(r.file_name, r.image_path) for r in recs]

    img_dir = args.images
    if not img_dir or not img_dir.is_dir():
        raise SystemExit("[error] pass --splits or a valid --images directory")
    paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return [(p.name, str(p)) for p in paths]


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--images", type=Path,
                     help="directory of images (no GT)")
    src.add_argument("--splits", type=Path,
                     help="splits.json (uses --split records as image source)")
    p.add_argument("--split", default="test",
                   help="which split to run (default: test)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of images (smoke test)")
    p.add_argument("--top-k", type=int, default=5,
                   help="top-K detections in stage_a.topk")
    p.add_argument("--box-threshold", type=float, default=None,
                   help="Override Stage A box_threshold (default: dino.py BOX_THRESHOLD)")
    p.add_argument("--text-threshold", type=float, default=None,
                   help="Override Stage A text_threshold (default: dino.py TEXT_THRESHOLD)")
    p.add_argument("--out", type=Path, default=Path("outputs/e2e_predictions.jsonl"))
    p.add_argument("--crop-dir", type=Path, default=Path("outputs/crops_e2e"))
    return p.parse_args()


def main():
    args = parse_args()

    items = _iter_image_paths(args)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("[error] no images to process")

    detector = DINODetector()

    # ── threshold 결정 ──
    bt = args.box_threshold if args.box_threshold is not None else BOX_THRESHOLD
    tt = args.text_threshold if args.text_threshold is not None else TEXT_THRESHOLD

    # ── 프롬프트 준비 ──
    prompt, active_mains = _build_stage_a_prompt()
    n_tokens = detector.verify_prompt_budget(prompt)
    print(
        f"[info] Stage A prompt: {len(active_mains)} active mains, "
        f"{n_tokens} tokens (limit 256)",
        file=sys.stderr,
    )

    print(f"[info] Stage A thresholds: box={bt}, text={tt}", file=sys.stderr)
    print(f"[info] images: {len(items)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out.open("w", encoding="utf-8")

    n_main_ok, n_main_fail, n_b_fallback = 0, 0, 0
    t0 = time.time()
    for i, (file_name, abs_path) in enumerate(items, 1):
        try:
            image = Image.open(abs_path).convert("RGB")
        except Exception as e:
            fout.write(json.dumps(
                {"file_name": file_name, "error": f"open_failed:{e}"},
                ensure_ascii=False) + "\n")
            continue

        # ── Stage A ──
        a = _stage_a(detector, image, prompt, args.top_k,
                     box_threshold=bt, text_threshold=tt)

        # ── Stage B ──
        if a["pred_main"]:
            n_main_ok += 1
            crop_path = args.crop_dir / file_name
            b = _stage_b(detector, image, a["pred_main"], crop_path)
            if b.get("fallback"):
                n_b_fallback += 1
        else:
            n_main_fail += 1
            b = {
                "crop_path": None, "dino_prompt": None,
                "score": None, "box": None, "fallback": None,
            }

        record = {
            "file_name": file_name,
            "image_size": [image.size[0], image.size[1]],
            "stage_a": a,
            "stage_b": b,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        if i % 50 == 0:
            print(
                f"  [{i}/{len(items)}] main_ok={n_main_ok} "
                f"main_fail={n_main_fail} b_fallback={n_b_fallback}",
                file=sys.stderr,
            )

    fout.close()
    dt = time.time() - t0
    print(
        f"\n[done] {len(items)} images in {dt:.1f}s "
        f"({dt/max(1,len(items)):.3f}s/img)",
        file=sys.stderr,
    )
    print(
        f"[summary] main_ok={n_main_ok}/{len(items)}  "
        f"main_fail={n_main_fail}  stage_b_fallback={n_b_fallback}",
        file=sys.stderr,
    )
    print(f"[saved] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
