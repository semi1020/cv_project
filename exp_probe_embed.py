"""
Extract SigLIP2-L-512 image embeddings for all records in all splits.

Saves:
  runs/probes/embeds_train.pt
  runs/probes/embeds_val.pt
  runs/probes/embeds_test.pt

Each file: {"file_names": [str, ...], "embeddings": FloatTensor(N, D)}
D = 1152 for SigLIP2-large-patch16-512.

Run once before exp_probe_train.py (~30min for all splits with batch_size=16).

Usage:
    python exp_probe_embed.py
    python exp_probe_embed.py --batch-size 8 --out-dir runs/probes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.dataset import load_splits

SIGLIP2_L_512 = "google/siglip2-large-patch16-512"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits",     type=Path, default=Path("splits/splits.json"))
    p.add_argument("--model-id",   type=str,  default=SIGLIP2_L_512)
    p.add_argument("--batch-size", type=int,  default=16)
    p.add_argument("--out-dir",    type=Path, default=Path("runs/probes"))
    p.add_argument("--force",      action="store_true",
                   help="Re-extract even if output file already exists")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[info] model={args.model_id}  device={device}  batch_size={args.batch_size}",
          file=sys.stderr)
    try:
        proc  = AutoProcessor.from_pretrained(args.model_id, local_files_only=True)
        model = AutoModel.from_pretrained(args.model_id, use_safetensors=True,
                                          local_files_only=True)
    except Exception:
        proc  = AutoProcessor.from_pretrained(args.model_id)
        model = AutoModel.from_pretrained(args.model_id, use_safetensors=True)
    model = model.to(device).eval()

    splits = load_splits(args.splits)

    for split_name, records in splits.items():
        out_path = args.out_dir / f"embeds_{split_name}.pt"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} already exists (use --force to re-extract)",
                  file=sys.stderr)
            continue

        print(f"\n[info] {split_name}: {len(records)} images → {out_path}", file=sys.stderr)
        file_names: list[str] = []
        all_embs: list[torch.Tensor] = []
        t0 = time.time()

        for start in range(0, len(records), args.batch_size):
            batch = records[start: start + args.batch_size]
            images: list[Image.Image] = []
            for r in batch:
                try:
                    images.append(Image.open(r.image_path).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (512, 512), color=0))
            file_names.extend(r.file_name for r in batch)

            inputs = proc(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out   = model.get_image_features(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out  # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalize
            all_embs.append(feats.cpu())

            done = start + len(batch)
            if done % (args.batch_size * 25) == 0 or done >= len(records):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (len(records) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(records)}]  {rate:.1f} img/s  eta={eta/60:.1f}m",
                      file=sys.stderr)

        embeddings = torch.cat(all_embs, dim=0)  # (N, D)
        torch.save({"file_names": file_names, "embeddings": embeddings}, out_path)
        elapsed = time.time() - t0
        print(f"  saved shape={list(embeddings.shape)}  ({elapsed/60:.1f}m total) → {out_path}",
              file=sys.stderr)

    print("\n[done]", file=sys.stderr)


if __name__ == "__main__":
    main()
