"""
Experiment 2 — Train CLIP Linear Probe on DINO-cropped images.

Input:  splits/crop_splits.json  (pre-extracted DINO crops)
Output: runs/exp2/{timestamp}/best.pt

Usage:
    python exp2_train.py
    python exp2_train.py --epochs 20 --lr 1e-3
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor

from config import CATEGORY_CONFIG
from src.dataset import distribution, load_splits
from src.linear_probe import CLIPLinearProbe


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
    p = argparse.ArgumentParser(description="Exp 2: Train CLIP Linear Probe on DINO crops")
    p.add_argument("--crop-splits", type=Path, default=Path("splits/crop_splits.json"))
    p.add_argument("--output-root", type=Path, default=Path("runs/exp2"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def resolve_device(name):
    if name == "cpu": return torch.device("cpu")
    if name == "cuda": return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for pv, y in loader:
            pv, y = pv.to(device), y.to(device)
            logits = model(pv)
            loss_sum += criterion(logits, y).item() * pv.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += pv.size(0)
    return (loss_sum / total, correct / total) if total else (0.0, 0.0)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    splits = load_splits(args.crop_splits)
    train_recs = splits["train"]
    val_recs = splits["val"]
    test_recs = splits["test"]

    class_names = sorted({
        sub for cat in CATEGORY_CONFIG.values() for sub in cat["sub_categories"]
    })
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    train_loader = DataLoader(
        CropDataset(train_recs, label_to_idx, processor),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        CropDataset(val_recs, label_to_idx, processor),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = CLIPLinearProbe(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    run_dir = args.output_root / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"

    params = model.count_parameters()
    print(f"[INFO] run_dir={run_dir}  device={device}")
    print(f"[INFO] train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")
    print(f"[INFO] classes={class_names}")
    print(f"[INFO] learnable params: {params['learnable']:,} ({params['ratio']} of total)")

    best_val_acc, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_total, run_correct = 0.0, 0, 0
        for pv, y in train_loader:
            pv, y = pv.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pv)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * pv.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_total += pv.size(0)

        train_loss = run_loss / max(1, run_total)
        train_acc = run_correct / max(1, run_total)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "label_to_idx": label_to_idx,
                "args": vars(args),
            }, ckpt_path)

    # Save metadata
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(run_dir / "splits.json", "w") as f:
        json.dump({k: [r.to_dict() for r in v] for k, v in splits.items()}, f,
                  ensure_ascii=False, indent=2)
    with open(run_dir / "data_summary.json", "w") as f:
        json.dump({
            "train": distribution(train_recs),
            "val": distribution(val_recs),
            "test": distribution(test_recs),
            "crop_splits_source": str(args.crop_splits),
            "learnable_params": params["learnable"],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] best checkpoint: {ckpt_path}")
    print(f"[NEXT] python exp2_evaluate.py --checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
