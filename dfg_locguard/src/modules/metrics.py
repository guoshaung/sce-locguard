from __future__ import annotations

import numpy as np


def pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    union = float((pred | gt).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    iou = tp / max(1.0, union)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def object_metrics(pred_object_masks: list[np.ndarray], gt_object_masks: list[np.ndarray], iou_threshold: float = 0.5) -> dict[str, float]:
    matched_gt: set[int] = set()
    tp = 0
    for pred in pred_object_masks:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_object_masks):
            if idx in matched_gt:
                continue
            inter = float((pred & gt).sum())
            union = float((pred | gt).sum())
            iou = inter / max(1.0, union)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
    fp = len(pred_object_masks) - tp
    fn = len(gt_object_masks) - len(matched_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"object_precision": precision, "object_recall": recall, "object_f1": f1}


def classification_accuracy(pred: list[str], gt: list[str]) -> float | None:
    if not gt:
        return None
    pairs = [(p, g) for p, g in zip(pred, gt) if g]
    if not pairs:
        return None
    return sum(1 for p, g in pairs if p == g) / len(pairs)

