"""
Experiment 2 — Evaluate Linear Probe and compare with DINO+CLIP zero-shot.

Loads a trained checkpoint, evaluates on the test split (using the same
pre-extracted DINO crops as zero-shot), and generates a comparison report
against Exp 1's DINO+CLIP results.

Usage:
    python exp2_evaluate.py --checkpoint runs/exp2/<run_id>/best.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor

from src.dataset import SampleRecord, load_splits
from src.linear_probe import CLIPLinearProbe
from src.metrics import compare_reports, compute_report


class CropDataset(Dataset):
    def __init__(self, records, label_to_idx, processor):
        self.records = records
        self.label_to_idx = label_to_idx
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, self.label_to_idx[rec.sub_category]


def parse_args():
    p = argparse.ArgumentParser(description="Exp 2: Evaluate linear probe + compare with zero-shot")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--zeroshot-results", type=Path, default=Path("results/exp1/dino_clip.json"),
                   help="DINO+CLIP zero-shot results from exp1 (for comparison)")
    p.add_argument("--out-dir", type=Path, default=Path("results/exp2"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def resolve_device(name):
    if name == "cpu": return torch.device("cpu")
    if name == "cuda": return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_comparison(name_a, name_b, rep_a, rep_b, class_names):
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
    device = resolve_device(args.device)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    class_names = ckpt["class_names"]
    label_to_idx = ckpt["label_to_idx"]

    # Load test records from the splits saved alongside the checkpoint
    run_dir = args.checkpoint.parent
    splits = load_splits(run_dir / "splits.json")
    test_records = splits[args.split]

    model = CLIPLinearProbe(num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    loader = DataLoader(
        CropDataset(test_records, label_to_idx, processor),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for pv, y in loader:
            logits = model(pv.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(y)
            all_preds.append(probs.argmax(1))

    probs_np = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy().tolist()
    y_pred = torch.cat(all_preds).numpy().tolist()

    probe_report = compute_report(y_true, y_pred, probs_np, class_names)
    probe_report["class_names"] = class_names
    probe_report["split"] = args.split

    out_probe = args.out_dir / "linear_probe.json"
    with open(out_probe, "w", encoding="utf-8") as f:
        json.dump(probe_report, f, ensure_ascii=False, indent=2)

    print(f"\n[Linear Probe] top1={probe_report['top1']:.4f}  "
          f"top3={probe_report.get('top3', 'N/A')}  macro_f1={probe_report['macro_f1']:.4f}")

    if args.zeroshot_results.exists():
        with open(args.zeroshot_results, encoding="utf-8") as f:
            zs_report = json.load(f)

        print_comparison("DINO+CLIP ZS", "Linear Probe", zs_report, probe_report, class_names)

        comparison = {
            "split": args.split,
            "class_names": class_names,
            "dino_clip_zeroshot": zs_report,
            "linear_probe": probe_report,
            "delta_probe_minus_zeroshot": compare_reports(zs_report, probe_report, class_names),
        }
        out_cmp = args.out_dir / "comparison.json"
        with open(out_cmp, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"\n[SAVED] {out_cmp}")
    else:
        print(f"[WARN] Zero-shot results not found at {args.zeroshot_results}. "
              f"Run exp1_zeroshot.py first.")

    print(f"[SAVED] {out_probe}")


if __name__ == "__main__":
    main()
