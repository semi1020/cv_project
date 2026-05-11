"""
E2E inference pipeline v2 — CLIP-first architecture.

Three stages (each tool used per its training task):
  Stage A — CLIP zero-shot 31-way main_category classification (full image).
  Stage B — GDINO single-class prompt detection → crop (학습된 phrase grounding 용도).
  Stage C — CLIP zero-shot N-way sub_category classification on crop (or full image).

Pipeline v1 ([10_e2e_pipeline.py](10_e2e_pipeline.py)) used GDINO as Stage A classifier
which is a tool-task mismatch — see [docs/PIPELINE_HISTORY.md](docs/PIPELINE_HISTORY.md) § 2.3.

Stage A baseline: CLIP canonical 79.5% (vs GDINO v3 57.0%).
Stage B: optional. exp1 결과에서 DINO crop이 sub-acc에 +0.1pp만 기여 → --skip-stage-b 옵션.
Stage C: existing CLIP zero-shot infrastructure ([src/clip_zeroshot.py](src/clip_zeroshot.py)).

Usage:
    # Default — full pipeline (A+B+C)
    python 11_e2e_pipeline_v2.py --splits splits/splits.json --split test

    # Skip Stage B (use full image for Stage C — much faster, equivalent acc per exp1)
    python 11_e2e_pipeline_v2.py --splits splits/splits.json --split test --skip-stage-b

    # Multi-alias mode for Stage A (note: total acc identical to canonical, distribution differs)
    python 11_e2e_pipeline_v2.py --splits splits/splits.json --split test --stage-a-mode multi
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

from config import CATEGORY_CONFIG
from exp_clip_main import (
    MultiAliasClassifier,
    build_candidates_canonical,
    model_short_name,
)
from src.clip_zeroshot import DEFAULT_MODEL_ID, CLIPZeroShot
from src.dataset import load_splits
from src.dino import DINODetector
from src.label_mapping import KEEP_MAINS, KOR_TO_EN


def _stage_a_canonical(clf: CLIPZeroShot, image: Image.Image,
                       candidates: dict[str, str], top_k: int) -> dict:
    r = clf.classify(image, candidates)
    topk = sorted(r["all_scores"].items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return {
        "pred_main": r["pred"],
        "score": round(r["score"], 6),
        "topk": [{"label_kor": k, "score": round(s, 6)} for k, s in topk],
    }


def _stage_a_multi(clf: CLIPMultiAliasClassifier, image: Image.Image, top_k: int) -> dict:
    r = clf.classify(image)
    topk = sorted(r["all_scores"].items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return {
        "pred_main": r["pred"],
        "score": round(r["score"], 6),
        "topk": [{"label_kor": k, "score": round(s, 6)} for k, s in topk],
    }


def _stage_b(detector: DINODetector, image: Image.Image, main_kor: str,
             crop_path: Path) -> dict:
    """Per-class single-prompt GDINO detection → crop. Pipeline v1과 동일."""
    entry = CATEGORY_CONFIG.get(main_kor)
    if entry is None:
        return {"crop_path": None, "dino_prompt": None,
                "score": None, "box": None, "fallback": None,
                "error": f"main {main_kor!r} not in CATEGORY_CONFIG"}
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


def _stage_c(clf: CLIPZeroShot, image: Image.Image, main_kor: str) -> dict:
    """CLIP zero-shot sub classification — given main, choose sub."""
    entry = CATEGORY_CONFIG.get(main_kor)
    if entry is None or not entry.get("sub_categories"):
        return {"pred_sub": None, "score": None, "all_scores": None,
                "error": f"main {main_kor!r} has no sub candidates"}
    candidates = entry["sub_categories"]
    r = clf.classify(image, candidates)
    return {
        "pred_sub": r["pred"],
        "score": round(r["score"], 6),
        "all_scores": {k: round(v, 6) for k, v in r["all_scores"].items()},
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
    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    return [(p.name, str(p)) for p in paths]


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--images", type=Path)
    src.add_argument("--splits", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--stage-a-mode", choices=["canonical", "multi"], default="canonical",
                   help="Stage A CLIP mode (default: canonical — chosen baseline per § 2.4)")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                   help=f"HF model id for Stage A & C (default: {DEFAULT_MODEL_ID}). "
                        "Examples: openai/clip-vit-large-patch14, "
                        "facebook/metaclip-l14-fullcc2.5b, "
                        "google/siglip-large-patch16-384")
    p.add_argument("--stage-c-model-id", type=str, default=None,
                   help="Override Stage C model (default: same as --model-id)")
    p.add_argument("--skip-stage-b", action="store_true", default=False,
                   help="Skip GDINO crop. Use full image for Stage C "
                        "(per exp1, DINO crop = +0.1pp F1 only)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: outputs/e2e_v2_{model_short}_{a-mode}{_no_b}.jsonl")
    p.add_argument("--crop-dir", type=Path, default=Path("outputs/crops_e2e_v2"))
    return p.parse_args()


def main():
    args = parse_args()
    items = _iter_image_paths(args)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("[error] no images to process")

    # ── output path defaulting ──
    if args.out is None:
        short = model_short_name(args.model_id)
        suffix = "_no_b" if args.skip_stage_b else ""
        args.out = Path(f"outputs/e2e_v2_{short}_{args.stage_a_mode}{suffix}.jsonl")

    print(f"[info] model_id={args.model_id}", file=sys.stderr)
    if args.stage_c_model_id and args.stage_c_model_id != args.model_id:
        print(f"[info] stage_c_model_id={args.stage_c_model_id}", file=sys.stderr)

    # ── Stage A classifier ──
    if args.stage_a_mode == "canonical":
        stage_a_clf = CLIPZeroShot(model_id=args.model_id)
        candidates = build_candidates_canonical()
        print(f"[info] Stage A: {args.model_id} canonical, {len(candidates)} classes",
              file=sys.stderr)
        def stage_a(img):
            return _stage_a_canonical(stage_a_clf, img, candidates, args.top_k)
    else:
        stage_a_clf = MultiAliasClassifier(model_id=args.model_id)
        stage_a_clf.fit_classes({k: KOR_TO_EN[k] for k in sorted(KEEP_MAINS)})
        n_aliases = sum(len(KOR_TO_EN[k]) for k in KEEP_MAINS)
        print(f"[info] Stage A: {args.model_id} multi-alias, {len(KEEP_MAINS)} classes, "
              f"{n_aliases} aliases", file=sys.stderr)
        def stage_a(img):
            return _stage_a_multi(stage_a_clf, img, args.top_k)

    # ── Stage B detector (only if needed) ──
    detector = None
    if not args.skip_stage_b:
        detector = DINODetector()
        print(f"[info] Stage B: GDINO per-class detector enabled", file=sys.stderr)
    else:
        print(f"[info] Stage B: SKIPPED (using full image for Stage C)", file=sys.stderr)

    # ── Stage C: reuse Stage A model if same id and canonical mode, else load fresh ──
    stage_c_model_id = args.stage_c_model_id or args.model_id
    if args.stage_a_mode == "canonical" and stage_c_model_id == args.model_id:
        stage_c_clf = stage_a_clf  # 동일 instance 재사용 (모델 1번만 로드)
    else:
        stage_c_clf = CLIPZeroShot(model_id=stage_c_model_id)
    print(f"[info] Stage C: {stage_c_model_id} zero-shot sub", file=sys.stderr)

    print(f"[info] images: {len(items)}  out={args.out}", file=sys.stderr)
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

        # Stage A
        a = stage_a(image)
        pred_main = a["pred_main"]

        # Stage B (optional)
        if pred_main and detector is not None:
            n_main_ok += 1
            crop_path = args.crop_dir / file_name
            b = _stage_b(detector, image, pred_main, crop_path)
            if b.get("fallback"):
                n_b_fallback += 1
            crop_image = image
            try:
                crop_image = Image.open(b["crop_path"]).convert("RGB") if b.get("crop_path") else image
            except Exception:
                crop_image = image
        elif pred_main:
            n_main_ok += 1
            b = {"crop_path": None, "dino_prompt": None,
                 "score": None, "box": None, "fallback": None,
                 "skipped": True}
            crop_image = image
        else:
            n_main_fail += 1
            b = {"crop_path": None, "dino_prompt": None,
                 "score": None, "box": None, "fallback": None}
            crop_image = image

        # Stage C
        if pred_main:
            c = _stage_c(stage_c_clf, crop_image, pred_main)
        else:
            c = {"pred_sub": None, "score": None, "all_scores": None}

        record = {
            "file_name": file_name,
            "image_size": [image.size[0], image.size[1]],
            "stage_a": a,
            "stage_b": b,
            "stage_c": c,
            "model_id": args.model_id,
            "stage_c_model_id": stage_c_model_id,
            "stage_a_mode": args.stage_a_mode,
            "skip_stage_b": args.skip_stage_b,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(items) - i) / rate
            print(f"  [{i}/{len(items)}] main_ok={n_main_ok} "
                  f"main_fail={n_main_fail} b_fb={n_b_fallback} "
                  f"rate={rate:.2f} img/s eta={eta/60:.1f}m", file=sys.stderr)

    fout.close()
    dt = time.time() - t0
    print(f"\n[done] {len(items)} images in {dt:.1f}s ({dt/max(1,len(items)):.3f}s/img)",
          file=sys.stderr)
    print(f"[summary] main_ok={n_main_ok}/{len(items)}  main_fail={n_main_fail}  "
          f"stage_b_fallback={n_b_fallback}", file=sys.stderr)
    print(f"[saved] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
