"""
Train per-main PyTorch linear probes on pre-extracted SigLIP2-L-512 embeddings.

Requires: runs/probes/embeds_{train,val,test}.pt  (from exp_probe_embed.py)

Each probe = nn.Linear(D, n_subs), trained with full-batch Adam + val-acc checkpoint.
Only trained for main classes with >1 sub (17 classes).

Output:
  runs/probes/sub_probes.pt    — {main_kor: {w, b, classes, D}}
  runs/probes/probe_eval.json  — per-class val/test accuracy

Usage:
    python exp_probe_train.py
    python exp_probe_train.py --epochs 300 --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from config import CATEGORY_CONFIG
from src.dataset import load_splits

# sub-given-main baseline from Pipeline v2 SigLIP2-L-512 (no_b) E2E eval
CLIP_BASELINE: dict[str, float] = {
    "TV장식장(거실장)":        0.511,
    "가방":                  1.000,
    "거울(액자형)":            0.793,
    "공기청정기및가습기":        0.803,
    "냉장고":                0.630,
    "상":                   0.614,
    "소파":                 0.564,
    "소화기":                0.719,
    "실내조명등기구":           0.603,
    "에어컨및온풍기":           0.432,
    "의자":                 0.764,
    "자전거":                0.981,
    "진열장(장식장,책장,찬장)":  0.348,
    "청소기":                0.870,
    "컴퓨터":                0.857,
    "텔레비전":               0.835,
    "형광등기구":              0.278,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits",    type=Path,  default=Path("splits/splits.json"))
    p.add_argument("--embed-dir", type=Path,  default=Path("runs/probes"))
    p.add_argument("--out-dir",   type=Path,  default=Path("runs/probes"))
    p.add_argument("--epochs",    type=int,   default=300)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--wd",        type=float, default=1e-4)
    return p.parse_args()


def load_emb_map(path: Path, device: str) -> tuple[dict[str, torch.Tensor], int]:
    data = torch.load(path, weights_only=True, map_location=device)
    fns  = data["file_names"]
    embs = data["embeddings"].to(device)
    return {fn: embs[i] for i, fn in enumerate(fns)}, int(embs.shape[1])


def gather(records, emb_map: dict, label_to_idx: dict, device: str):
    """Collect (X, y) tensors for records whose file_name is in emb_map."""
    X_list, y_list = [], []
    for r in records:
        if r.file_name in emb_map and r.sub_category in label_to_idx:
            X_list.append(emb_map[r.file_name])
            y_list.append(label_to_idx[r.sub_category])
    if not X_list:
        return None, None
    return torch.stack(X_list), torch.tensor(y_list, device=device)


def train_probe(X_tr, y_tr, X_va, y_va, D: int, n_cls: int,
                epochs: int, lr: float, wd: float, device: str):
    fc = torch.nn.Linear(D, n_cls).to(device)
    optim = torch.optim.Adam(fc.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = -1.0
    best_w = fc.weight.data.clone()
    best_b = fc.bias.data.clone()

    for epoch in range(1, epochs + 1):
        fc.train()
        optim.zero_grad(set_to_none=True)
        F.cross_entropy(fc(X_tr), y_tr).backward()
        optim.step()

        if X_va is not None and epoch % 10 == 0:
            fc.eval()
            with torch.no_grad():
                va_acc = (fc(X_va).argmax(1) == y_va).float().mean().item()
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_w = fc.weight.data.clone()
                best_b = fc.bias.data.clone()

    # Final val check if no periodic check hit
    if X_va is not None and best_val_acc < 0:
        fc.eval()
        with torch.no_grad():
            best_val_acc = (fc(X_va).argmax(1) == y_va).float().mean().item()
        best_w = fc.weight.data.clone()
        best_b = fc.bias.data.clone()

    fc.weight.data = best_w
    fc.bias.data   = best_b
    return fc, best_val_acc if X_va is not None else math.nan


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[info] loading splits...", file=sys.stderr)
    splits = load_splits(args.splits)

    print("[info] loading embeddings...", file=sys.stderr)
    emb_train, D = load_emb_map(args.embed_dir / "embeds_train.pt", device)
    emb_val,   _ = load_emb_map(args.embed_dir / "embeds_val.pt",   device)
    emb_test,  _ = load_emb_map(args.embed_dir / "embeds_test.pt",  device)
    print(f"[info] embedding dim={D}  device={device}", file=sys.stderr)

    multi_sub = {
        m: sorted(e["sub_categories"].keys())
        for m, e in CATEGORY_CONFIG.items()
        if len(e["sub_categories"]) > 1
    }
    print(f"[info] probe targets: {len(multi_sub)} main classes\n", file=sys.stderr)

    by_main: dict[str, dict[str, list]] = {
        s: defaultdict(list) for s in ("train", "val", "test")
    }
    for split_name, recs in splits.items():
        for r in recs:
            by_main[split_name][r.main_category].append(r)

    probe_bundle: dict[str, dict] = {}
    eval_results: dict[str, dict] = {}

    for main_kor, classes in sorted(multi_sub.items()):
        label_to_idx = {c: i for i, c in enumerate(classes)}
        n_cls = len(classes)

        X_tr, y_tr = gather(by_main["train"][main_kor], emb_train, label_to_idx, device)
        X_va, y_va = gather(by_main["val"][main_kor],   emb_val,   label_to_idx, device)
        X_te, y_te = gather(by_main["test"][main_kor],  emb_test,  label_to_idx, device)

        if X_tr is None:
            print(f"  [skip] {main_kor}: no train data", file=sys.stderr)
            continue

        fc, val_acc = train_probe(X_tr, y_tr, X_va, y_va, D, n_cls,
                                  args.epochs, args.lr, args.wd, device)

        test_acc = math.nan
        if X_te is not None:
            fc.eval()
            with torch.no_grad():
                test_acc = (fc(X_te).argmax(1) == y_te).float().mean().item()

        print(f"  {main_kor:<32} n={len(X_tr):>5} ({n_cls}-way)  "
              f"val={val_acc:.3f}  test={test_acc:.3f}", file=sys.stderr)

        probe_bundle[main_kor] = {
            "w":       fc.weight.data.cpu(),
            "b":       fc.bias.data.cpu(),
            "classes": classes,
            "D":       D,
        }
        eval_results[main_kor] = {
            "n_train":  int(len(X_tr)),
            "n_val":    int(len(X_va)) if X_va is not None else 0,
            "n_test":   int(len(X_te)) if X_te is not None else 0,
            "n_classes": n_cls,
            "val_acc":  val_acc,
            "test_acc": test_acc,
            "classes":  classes,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    probe_path = args.out_dir / "sub_probes.pt"
    torch.save(probe_bundle, probe_path)
    print(f"\n[saved] {probe_path}  ({len(probe_bundle)} probes)", file=sys.stderr)

    eval_path = args.out_dir / "probe_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {eval_path}", file=sys.stderr)

    # ── Comparison vs CLIP zero-shot ──────────────────────────────────────────
    print("\n=== Probe vs CLIP zero-shot (sub-given-main, test split) ===", file=sys.stderr)
    print(f"  {'클래스':<30}  {'CLIP':>6}  {'Probe':>6}  {'Δ':>7}", file=sys.stderr)
    print("  " + "-" * 54, file=sys.stderr)
    tw_clip = tw_probe = tw_n = 0.0
    for main_kor, res in sorted(eval_results.items(), key=lambda x: x[1]["test_acc"]):
        clip_base = CLIP_BASELINE.get(main_kor)
        probe_acc = res["test_acc"]
        if clip_base is None or math.isnan(probe_acc):
            continue
        delta = probe_acc - clip_base
        flag  = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else " ")
        print(f"  {main_kor:<30}  {clip_base:.3f}   {probe_acc:.3f}  {delta:>+7.3f} {flag}",
              file=sys.stderr)
        n = res["n_test"]
        tw_clip  += clip_base * n
        tw_probe += probe_acc * n
        tw_n     += n

    if tw_n:
        print(f"\n  {'weighted avg (test)':<30}  {tw_clip/tw_n:.3f}   {tw_probe/tw_n:.3f}  "
              f"{(tw_probe - tw_clip)/tw_n:>+7.3f}", file=sys.stderr)

    # E2E impact estimate: main_acc=89.2%, probe replaces sub for target classes
    print("\n=== E2E impact estimate ===", file=sys.stderr)
    # total test = 3927, main_correct = 3503
    TOTAL, MAIN_CORRECT = 3927, 3503
    clip_sub_correct = sum(
        CLIP_BASELINE.get(m, 0) * res["n_test"]
        for m, res in eval_results.items() if not math.isnan(res["test_acc"])
    )
    probe_sub_correct = sum(
        res["test_acc"] * res["n_test"]
        for res in eval_results.values() if not math.isnan(res["test_acc"])
    )
    target_n = sum(
        res["n_test"] for res in eval_results.values() if not math.isnan(res["test_acc"])
    )
    # Classes not in probe target: keep CLIP sub accuracy estimate
    non_target_sub_correct = (MAIN_CORRECT - target_n) * 1.0  # assume 100% for single-sub
    # Actually, single-sub classes are trivially correct once main is right
    # E2E = (probe_sub_correct + non_target_sub_correct) / TOTAL
    est_e2e_probe = (probe_sub_correct + (MAIN_CORRECT - target_n)) / TOTAL
    est_e2e_clip  = (clip_sub_correct  + (MAIN_CORRECT - target_n)) / TOTAL
    print(f"  current E2E (CLIP sub): {est_e2e_clip*100:.1f}%  (sanity check vs 68.9%)",
          file=sys.stderr)
    print(f"  estimated E2E (probe):  {est_e2e_probe*100:.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
