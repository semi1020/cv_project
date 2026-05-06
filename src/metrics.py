"""Evaluation metrics for classification experiments."""
import numpy as np


def compute_report(
    y_true: list[int],
    y_pred: list[int],
    probs: np.ndarray | None,
    class_names: list[str],
) -> dict:
    """Compute full evaluation report.

    Returns dict with: top1, top3, macro_f1, class_wise_top1,
    confusion_matrix, abstain_threshold.
    """
    n_classes = len(class_names)
    yt = np.array(y_true, dtype=np.int64)
    yp = np.array(y_pred, dtype=np.int64)

    top1 = float((yt == yp).mean()) if len(yt) else 0.0

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= p < n_classes:
            cm[t, p] += 1

    cls_top1 = {}
    f1_list = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        total = tp + fn
        cls_top1[name] = float(tp / total) if total else None
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1_list.append((2 * tp) / denom)

    macro_f1 = float(sum(f1_list) / len(f1_list)) if f1_list else 0.0

    top3 = None
    if probs is not None and len(probs):
        k = min(3, probs.shape[1])
        topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
        hits = sum(1 for i in range(len(yt)) if yt[i] in topk[i])
        top3 = float(hits / len(yt))

    conf = probs.max(axis=1) if probs is not None else np.zeros(len(yt))
    abstain = []
    for thresh in [0.30, 0.50, 0.70, 0.80, 0.90]:
        mask = conf >= thresh
        kept = int(mask.sum())
        acc = float((yt[mask] == yp[mask]).mean()) if kept else None
        abstain.append({
            "threshold": thresh,
            "coverage": float(kept / len(yt)) if len(yt) else 0.0,
            "accepted": kept,
            "accuracy_on_accepted": acc,
        })

    return {
        "top1": top1,
        "top3": top3,
        "macro_f1": macro_f1,
        "class_wise_top1": cls_top1,
        "confusion_matrix": cm.tolist(),
        "abstain_threshold": abstain,
    }


def compare_reports(report_a: dict, report_b: dict, class_names: list[str]) -> dict:
    """Compute delta = report_b - report_a for key metrics."""
    def _delta(a, b):
        return None if a is None or b is None else float(b - a)

    cls_delta = {
        c: _delta(
            report_a["class_wise_top1"].get(c),
            report_b["class_wise_top1"].get(c),
        )
        for c in class_names
    }
    return {
        "top1": _delta(report_a["top1"], report_b["top1"]),
        "top3": _delta(report_a.get("top3"), report_b.get("top3")),
        "macro_f1": _delta(report_a["macro_f1"], report_b["macro_f1"]),
        "class_wise_top1": cls_delta,
    }
