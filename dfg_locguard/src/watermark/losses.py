from __future__ import annotations

import numpy as np


def localization_loss(pred_mask, gt_mask, eps: float = 1e-6) -> float:
    pred = np.clip(np.asarray(pred_mask, dtype=np.float32), eps, 1.0 - eps)
    gt = np.asarray(gt_mask, dtype=np.float32)
    bce = -np.mean(gt * np.log(pred) + (1.0 - gt) * np.log(1.0 - pred))
    dice = _dice_loss(pred, gt, eps)
    return float(bce + dice)


def semantic_guided_localization_loss(pred_mask, gt_mask, semantic_guidance_map, beta: float = 2.0, eps: float = 1e-6) -> float:
    pred = np.clip(np.asarray(pred_mask, dtype=np.float32), eps, 1.0 - eps)
    gt = np.asarray(gt_mask, dtype=np.float32)
    guidance = _normalize(np.asarray(semantic_guidance_map, dtype=np.float32))
    weights = 1.0 + beta * guidance
    bce = -np.mean(weights * (gt * np.log(pred) + (1.0 - gt) * np.log(1.0 - pred)))
    dice = _dice_loss(pred, gt, eps)
    return float(bce + dice)


def diffusion_distillation_loss(pred_mask, semantic_guidance_map, loss_type: str = "l1") -> float:
    pred = _normalize(np.asarray(pred_mask, dtype=np.float32))
    guidance = _normalize(np.asarray(semantic_guidance_map, dtype=np.float32))
    if loss_type == "mse":
        return float(np.mean((pred - guidance) ** 2))
    return float(np.mean(np.abs(pred - guidance)))


def boundary_loss(pred_mask, gt_mask) -> float:
    pred_boundary = _boundary(np.asarray(pred_mask, dtype=np.float32) > 0.5)
    gt_boundary = _boundary(np.asarray(gt_mask, dtype=np.float32) > 0.5)
    return localization_loss(pred_boundary.astype(np.float32), gt_boundary.astype(np.float32))


def _dice_loss(pred, gt, eps: float) -> float:
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def _normalize(arr: np.ndarray) -> np.ndarray:
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - min_v) / (max_v - min_v)).astype(np.float32)


def _boundary(mask: np.ndarray) -> np.ndarray:
    import cv2

    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask_u8 ^ eroded

