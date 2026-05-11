"""
Open-vocab zero-shot 31-way main_category classification.

Goal: validate hypothesis that CLIP-family (image-text matching, classifier-natured)
outperforms GDINO (phrase grounding, detector-natured) on Stage A main classification.

Apples-to-apples vs v3_baseline:
  - Same test split (splits/splits.json)
  - Same 31 KEEP_MAINS classes
  - Same eval script (per-class acc + top-20 confusions)

Prompt strategies:
  --mode canonical : "a photo of a {first_alias}" (1 prompt per class, default)
  --mode multi     : average embedding over all aliases (zero-shot best-practice)

Model selection (open-vocab, no retraining):
  --model-id openai/clip-vit-large-patch14         (default, 79.5% main baseline)
  --model-id facebook/metaclip-l14-fullcc2.5b      (MetaCLIP large, CLIP-compat API)
  --model-id google/siglip-large-patch16-384       (SigLIP large, sigmoid loss)
  --model-id google/siglip-so400m-patch14-384      (SigLIP SO400M, larger)

Usage:
    python exp_clip_main.py --splits splits/splits.json --split test
    python exp_clip_main.py --mode multi --out outputs/clip_main_multi.jsonl
    python exp_clip_main.py --model-id google/siglip-large-patch16-384
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from src.clip_zeroshot import DEFAULT_MODEL_ID, CLIPZeroShot, _is_siglip_family
from src.dataset import load_splits
from src.label_mapping import KEEP_MAINS, KOR_TO_EN


CLIP_TEMPLATE = "a photo of a {}"


def build_candidates_canonical() -> dict[str, str]:
    """One prompt per class using the first (canonical) alias."""
    out = {}
    for kor in sorted(KEEP_MAINS):
        alias = KOR_TO_EN[kor][0]
        out[kor] = CLIP_TEMPLATE.format(alias)
    return out


def model_short_name(model_id: str) -> str:
    """Filesystem-safe short tag for output naming. e.g.
    openai/clip-vit-large-patch14 -> clip-vit-large-patch14
    google/siglip-large-patch16-384 -> siglip-large-patch16-384
    """
    base = model_id.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


class MultiAliasClassifier:
    """Average text embedding over all aliases per class — works for CLIP/MetaCLIP/SigLIP.

    Uses model.get_text_features / get_image_features (present in both CLIPModel and
    SiglipModel via AutoModel). Final scoring:
      - SigLIP family: sigmoid of scaled cosine
      - Others       : softmax of scaled cosine
    Argmax is invariant; differences only affect score interpretation.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        from transformers import AutoModel, AutoProcessor
        # use_safetensors=True: torch<2.6 CVE-2025-32434 회피 (clip_zeroshot.py 참고)
        model_kwargs = {"use_safetensors": True}
        try:
            self._proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            self._model = AutoModel.from_pretrained(model_id, local_files_only=True, **model_kwargs)
        except Exception:
            self._proc = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModel.from_pretrained(model_id, **model_kwargs)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).eval()
        self.model_id = model_id
        self._is_siglip = _is_siglip_family(model_id)
        self._labels: list[str] = []
        self._text_emb: torch.Tensor | None = None

    def fit_classes(self, alias_map: dict[str, list[str]]) -> None:
        """Pre-compute averaged & normalized text embeddings per class."""
        self._labels = sorted(alias_map.keys())
        all_emb_per_class = []
        for kor in self._labels:
            prompts = [CLIP_TEMPLATE.format(a) for a in alias_map[kor]]
            inputs = self._proc(
                text=prompts, return_tensors="pt",
                padding="max_length", truncation=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self._model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            class_emb = emb.mean(dim=0)
            class_emb = class_emb / class_emb.norm()
            all_emb_per_class.append(class_emb)
        self._text_emb = torch.stack(all_emb_per_class)  # [C, D]

    def classify(self, image: Image.Image) -> dict:
        inputs = self._proc(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            img_emb = self._model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        logit_scale = self._model.logit_scale.exp()
        logits = (img_emb @ self._text_emb.T).squeeze(0) * logit_scale
        if self._is_siglip:
            # Apply logit_bias if available (SigLIP only)
            if hasattr(self._model, "logit_bias"):
                logits = logits + self._model.logit_bias
            probs = logits.sigmoid().cpu().tolist()
        else:
            probs = logits.softmax(dim=-1).cpu().tolist()
        all_scores = {label: float(p) for label, p in zip(self._labels, probs)}
        pred = max(all_scores, key=lambda k: all_scores[k])
        return {"pred": pred, "score": all_scores[pred], "all_scores": all_scores}


# Backward-compat alias (older imports may use this name)
CLIPMultiAliasClassifier = MultiAliasClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, default=Path("splits/splits.json"))
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mode", choices=["canonical", "multi"], default="canonical")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                   help=f"HF model id (default: {DEFAULT_MODEL_ID}). Examples: "
                        "openai/clip-vit-large-patch14, "
                        "facebook/metaclip-l14-fullcc2.5b, "
                        "google/siglip-large-patch16-384")
    p.add_argument("--out", type=Path, default=None,
                   help="default: outputs/clip_main_{mode}_{model_short}.jsonl")
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    short = model_short_name(args.model_id)
    if args.out is None:
        args.out = Path(f"outputs/clip_main_{args.mode}_{short}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.splits)
    if args.split not in splits:
        raise SystemExit(f"[error] split '{args.split}' not in {args.splits}")
    records = splits[args.split]
    if args.limit:
        records = records[: args.limit]

    print(f"[info] model={args.model_id}", file=sys.stderr)
    print(f"[info] mode={args.mode}, classes={len(KEEP_MAINS)}, images={len(records)}",
          file=sys.stderr)
    print(f"[info] siglip_family={_is_siglip_family(args.model_id)}", file=sys.stderr)

    if args.mode == "canonical":
        clf = CLIPZeroShot(model_id=args.model_id)
        candidates = build_candidates_canonical()
        print(f"[info] canonical prompts ({len(candidates)} classes):", file=sys.stderr)
        for k in sorted(candidates)[:5]:
            print(f"  {k}: {candidates[k]!r}", file=sys.stderr)
        print("  ...", file=sys.stderr)

        def predict(img):
            r = clf.classify(img, candidates)
            return r["pred"], r["score"], r["all_scores"]
    else:
        clf = MultiAliasClassifier(model_id=args.model_id)
        alias_map = {k: KOR_TO_EN[k] for k in sorted(KEEP_MAINS)}
        clf.fit_classes(alias_map)
        n_aliases = sum(len(v) for v in alias_map.values())
        print(f"[info] multi-alias mode: {n_aliases} aliases averaged into {len(alias_map)} classes",
              file=sys.stderr)

        def predict(img):
            r = clf.classify(img)
            return r["pred"], r["score"], r["all_scores"]

    fout = args.out.open("w", encoding="utf-8")
    n_correct = 0
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        try:
            image = Image.open(rec.image_path).convert("RGB")
        except Exception as e:
            fout.write(json.dumps({"file_name": rec.file_name,
                                    "error": f"open_failed:{e}"},
                                   ensure_ascii=False) + "\n")
            continue
        pred, score, all_scores = predict(image)
        topk = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
        out_record = {
            "file_name": rec.file_name,
            "gt_main": rec.main_category,
            "pred_main": pred,
            "score": round(score, 6),
            "topk": [{"label_kor": k, "score": round(s, 6)} for k, s in topk],
            "model_id": args.model_id,
            "mode": args.mode,
        }
        fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        if pred == rec.main_category:
            n_correct += 1
        if i % 200 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(records) - i) / rate
            print(f"  [{i}/{len(records)}] correct={n_correct} "
                  f"acc={n_correct/i*100:.1f}% rate={rate:.1f} img/s eta={eta/60:.1f}m",
                  file=sys.stderr)
    fout.close()
    dt = time.time() - t0
    print(f"\n[done] {len(records)} images in {dt:.1f}s ({dt/max(1,len(records)):.3f}s/img)",
          file=sys.stderr)
    print(f"[summary] correct={n_correct}/{len(records)} ({n_correct/len(records)*100:.2f}%)",
          file=sys.stderr)
    print(f"[saved] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
