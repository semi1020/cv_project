"""
Step 2 — Evaluate Grounding DINO BBOX extraction performance.

Reads splits/crop_splits.json (produced by 01_extract_crops.py) and computes
per-split / per-class detection statistics from each record's dino_meta.

Stage B perspective: GT main_category is given; we measure how reliably DINO
finds a matching object box. Records with dino_meta=None (cached crops from
prior runs) are excluded from rate calculations and reported separately.

Outputs results/dino_eval/{split}_report.json plus a console summary.

Usage:
    python 02_dino_eval.py
    python 02_dino_eval.py --split test
    python 02_dino_eval.py --crops splits/crop_splits.json --out results/dino_eval
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.dataset import SampleRecord, load_splits


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _score_stats(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "mean": None, "p50": None, "p90": None,
                "min": None, "max": None}
    return {
        "count": len(scores),
        "mean": round(statistics.fmean(scores), 4),
        "p50": round(_quantile(scores, 0.5), 4),
        "p90": round(_quantile(scores, 0.9), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }


def _eval_split(records: list[SampleRecord]) -> dict:
    total = len(records)
    n_success = 0
    n_fallback = 0
    n_missing = 0  # dino_meta absent (cached or error)
    n_error = 0
    success_scores: list[float] = []

    per_class_total: dict[str, int] = defaultdict(int)
    per_class_success: dict[str, int] = defaultdict(int)
    per_class_fallback: dict[str, int] = defaultdict(int)
    per_class_scores: dict[str, list[float]] = defaultdict(list)

    for r in records:
        per_class_total[r.main_category] += 1
        meta = r.dino_meta
        if meta is None:
            n_missing += 1
            continue
        if "error" in meta:
            n_error += 1
            n_fallback += 1
            per_class_fallback[r.main_category] += 1
            continue
        if meta.get("fallback"):
            n_fallback += 1
            per_class_fallback[r.main_category] += 1
        elif meta.get("detection_success") and meta.get("score") is not None:
            n_success += 1
            per_class_success[r.main_category] += 1
            success_scores.append(float(meta["score"]))
            per_class_scores[r.main_category].append(float(meta["score"]))

    measured = n_success + n_fallback
    detection_rate = (n_success / measured) if measured else 0.0

    per_class: dict[str, dict] = {}
    for cls in sorted(per_class_total.keys()):
        s, f = per_class_success[cls], per_class_fallback[cls]
        m = s + f
        per_class[cls] = {
            "total": per_class_total[cls],
            "measured": m,
            "success": s,
            "fallback": f,
            "detection_rate": round(s / m, 4) if m else None,
            "score_stats": _score_stats(per_class_scores[cls]),
        }

    return {
        "total": total,
        "summary": {
            "success": n_success,
            "fallback": n_fallback,
            "missing_meta": n_missing,
            "open_error": n_error,
            "measured": measured,
            "detection_rate": round(detection_rate, 4),
        },
        "score_stats": _score_stats(success_scores),
        "per_class": per_class,
    }


def _print_split(split_name: str, report: dict) -> None:
    s = report["summary"]
    ss = report["score_stats"]
    print(f"\n=== {split_name} (n={report['total']}) ===")
    print(
        f"  success={s['success']}  fallback={s['fallback']}  "
        f"missing_meta={s['missing_meta']}  open_error={s['open_error']}"
    )
    print(
        f"  detection_rate={s['detection_rate']:.1%}  "
        f"(measured {s['measured']}/{report['total']})"
    )
    if ss["count"]:
        print(
            f"  success score: mean={ss['mean']}  p50={ss['p50']}  "
            f"p90={ss['p90']}  min={ss['min']}  max={ss['max']}"
        )
    pc = report["per_class"]
    if pc:
        print("  per-class detection rate (sorted asc):")
        rows = sorted(
            pc.items(),
            key=lambda kv: (kv[1]["detection_rate"] if kv[1]["detection_rate"] is not None else -1),
        )
        for cls, st in rows:
            rate = st["detection_rate"]
            rate_s = f"{rate:.1%}" if rate is not None else "  n/a"
            score_mean = st["score_stats"]["mean"]
            score_s = f"{score_mean:.3f}" if score_mean is not None else "  n/a"
            print(
                f"    {cls:<28} rate={rate_s}  "
                f"({st['success']}/{st['measured']})  score_mean={score_s}"
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops", type=Path, default=Path("splits/crop_splits.json"))
    p.add_argument("--out", type=Path, default=Path("results/dino_eval"))
    p.add_argument(
        "--split", default="all",
        help="split to evaluate: train | val | test | all",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.crops.exists():
        raise SystemExit(f"[error] {args.crops} not found. Run 01_extract_crops.py first.")

    splits = load_splits(args.crops)
    args.out.mkdir(parents=True, exist_ok=True)

    targets = sorted(splits.keys()) if args.split == "all" else [args.split]
    overall: dict[str, dict] = {}

    for split_name in targets:
        if split_name not in splits:
            print(f"[warn] split '{split_name}' not in crop_splits.json; skipping")
            continue
        report = _eval_split(splits[split_name])
        out_path = args.out / f"{split_name}_report.json"
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _print_split(split_name, report)
        print(f"\n  [saved] {out_path}")
        overall[split_name] = report["summary"]

    if len(overall) > 1:
        print("\n=== OVERALL ===")
        for sn, s in overall.items():
            print(
                f"  {sn:<6} detection_rate={s['detection_rate']:.1%}  "
                f"({s['success']}/{s['measured']})"
            )


if __name__ == "__main__":
    main()
