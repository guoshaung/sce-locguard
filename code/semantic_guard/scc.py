from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from .schema import ObjectRegion, SCCResult, pad_box


@dataclass
class SCCConfig:
    backend: str = "auto"
    device: str = "auto"
    clip_model: str = "openai/clip-vit-base-patch32"
    dino_model: str = "facebook/dinov2-small"
    crop_padding: int = 12
    allow_fallback: bool = True
    hybrid_clip_weight: float = 0.40
    hybrid_dino_weight: float = 0.45
    hybrid_pixel_weight: float = 0.15


class SemanticChangeChecker:
    """Compare object crops before/after tampering."""

    def __init__(self, config: SCCConfig):
        self.config = config
        self._clip = None
        self._clip_processor = None
        self._dino = None
        self._dino_processor = None
        self._device = None

    def compare(
        self,
        original: Image.Image,
        tampered: Image.Image,
        region: ObjectRegion,
        category_labels: Sequence[str] | None = None,
    ) -> SCCResult:
        backend = self.config.backend.lower()
        if backend not in {"auto", "clip", "dino", "hybrid", "histogram"}:
            raise ValueError(f"Unsupported SCC backend: {self.config.backend}")

        crop_original, crop_tampered = self._masked_crops(original, tampered, region)
        errors: list[str] = []

        if backend == "hybrid":
            try:
                return self._compare_hybrid(crop_original, crop_tampered, category_labels)
            except Exception as exc:
                errors.append(f"Hybrid CLIP/DINO backend unavailable: {exc}")
                if not self.config.allow_fallback:
                    raise
                result = self._compare_histogram(crop_original, crop_tampered)
                result.backend = "hybrid_fallback_histogram"
                result.notes.extend(errors)
                return result

        if backend in {"auto", "clip"}:
            try:
                return self._compare_clip(crop_original, crop_tampered, category_labels)
            except Exception as exc:
                errors.append(f"CLIP backend unavailable: {exc}")
                if backend == "clip" and not self.config.allow_fallback:
                    raise

        if backend in {"auto", "dino"}:
            try:
                result = self._compare_dino(crop_original, crop_tampered)
                result.notes.extend(errors)
                return result
            except Exception as exc:
                errors.append(f"DINO backend unavailable: {exc}")
                if backend == "dino" and not self.config.allow_fallback:
                    raise

        result = self._compare_histogram(crop_original, crop_tampered)
        result.notes.extend(errors)
        return result

    def _device_name(self) -> str:
        if self._device is not None:
            return self._device
        import torch

        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device
        return self._device

    def _compare_clip(
        self,
        crop_original: Image.Image,
        crop_tampered: Image.Image,
        category_labels: Sequence[str] | None,
    ) -> SCCResult:
        import torch
        import torch.nn.functional as F
        from transformers import CLIPModel, CLIPProcessor

        if self._clip is None:
            self._clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)
            self._clip = CLIPModel.from_pretrained(self.config.clip_model).to(self._device_name()).eval()

        inputs = self._clip_processor(images=[crop_original, crop_tampered], return_tensors="pt")
        inputs = {k: v.to(self._device_name()) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._clip.get_image_features(**inputs)
            feats = F.normalize(feats, dim=-1)
            cosine = float((feats[0] * feats[1]).sum().detach().cpu())
        similarity = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
        category, confidence = self._classify_clip(crop_tampered, category_labels)
        return SCCResult(
            backend="clip",
            similarity=similarity,
            change_score=float(1.0 - similarity),
            category=category,
            category_confidence=confidence,
            pixel_delta=self._pixel_delta(crop_original, crop_tampered),
        )

    def _compare_hybrid(
        self,
        crop_original: Image.Image,
        crop_tampered: Image.Image,
        category_labels: Sequence[str] | None,
    ) -> SCCResult:
        clip_result = self._compare_clip(crop_original, crop_tampered, category_labels)
        dino_result = self._compare_dino(crop_original, crop_tampered)
        pixel_change = float(np.clip(self._pixel_delta(crop_original, crop_tampered) * 3.0, 0.0, 1.0))
        weights = np.asarray(
            [
                self.config.hybrid_clip_weight,
                self.config.hybrid_dino_weight,
                self.config.hybrid_pixel_weight,
            ],
            dtype=np.float32,
        )
        weights = weights / max(1e-8, float(weights.sum()))
        change_score = float(
            weights[0] * clip_result.change_score
            + weights[1] * dino_result.change_score
            + weights[2] * pixel_change
        )
        change_score = float(np.clip(change_score, 0.0, 1.0))
        return SCCResult(
            backend="hybrid",
            similarity=float(1.0 - change_score),
            change_score=change_score,
            category=clip_result.category,
            category_confidence=clip_result.category_confidence,
            pixel_delta=clip_result.pixel_delta,
            components={
                "clip_similarity": clip_result.similarity,
                "clip_change": clip_result.change_score,
                "dino_similarity": dino_result.similarity,
                "dino_change": dino_result.change_score,
                "pixel_change": pixel_change,
                "clip_weight": float(weights[0]),
                "dino_weight": float(weights[1]),
                "pixel_weight": float(weights[2]),
            },
            notes=[
                f"Hybrid SCC = {weights[0]:.2f}*CLIP + {weights[1]:.2f}*DINO + {weights[2]:.2f}*pixel.",
            ],
        )

    def _classify_clip(self, image: Image.Image, category_labels: Sequence[str] | None) -> tuple[str, float]:
        if not category_labels:
            return "unknown", 0.0
        import torch

        prompts = [f"a photo of {label}" for label in category_labels]
        inputs = self._clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device_name()) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._clip(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().numpy()
        idx = int(np.argmax(probs))
        return str(category_labels[idx]), float(probs[idx])

    def _compare_dino(self, crop_original: Image.Image, crop_tampered: Image.Image) -> SCCResult:
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, AutoModel

        if self._dino is None:
            self._dino_processor = AutoImageProcessor.from_pretrained(self.config.dino_model)
            self._dino = AutoModel.from_pretrained(self.config.dino_model).to(self._device_name()).eval()

        inputs = self._dino_processor(images=[crop_original, crop_tampered], return_tensors="pt")
        inputs = {k: v.to(self._device_name()) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._dino(**inputs)
            if getattr(outputs, "pooler_output", None) is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            feats = F.normalize(feats, dim=-1)
            cosine = float((feats[0] * feats[1]).sum().detach().cpu())
        similarity = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
        return SCCResult(
            backend="dino",
            similarity=similarity,
            change_score=float(1.0 - similarity),
            pixel_delta=self._pixel_delta(crop_original, crop_tampered),
        )

    def _compare_histogram(self, crop_original: Image.Image, crop_tampered: Image.Image) -> SCCResult:
        arr_a = np.asarray(crop_original.resize((224, 224)), dtype=np.float32) / 255.0
        arr_b = np.asarray(crop_tampered.resize((224, 224)), dtype=np.float32) / 255.0
        hist_a = self._hist_features(arr_a)
        hist_b = self._hist_features(arr_b)
        similarity = float(np.dot(hist_a, hist_b) / max(1e-8, np.linalg.norm(hist_a) * np.linalg.norm(hist_b)))
        similarity = float(np.clip(similarity, 0.0, 1.0))
        pixel_delta = float(np.mean(np.abs(arr_a - arr_b)))
        change_score = float(np.clip(0.65 * (1.0 - similarity) + 0.35 * min(1.0, pixel_delta * 3.0), 0.0, 1.0))
        return SCCResult(
            backend="histogram",
            similarity=similarity,
            change_score=change_score,
            pixel_delta=pixel_delta,
            notes=["Traditional color/edge fallback; use CLIP or DINO for semantic SCC."],
        )

    @staticmethod
    def _hist_features(arr: np.ndarray) -> np.ndarray:
        feats: list[np.ndarray] = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=32, range=(0.0, 1.0), density=True)
            feats.append(hist.astype(np.float32))
        gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
        edge_hist, _ = np.histogram(edges, bins=8, range=(0.0, 1.0), density=True)
        feats.append(edge_hist.astype(np.float32))
        out = np.concatenate(feats)
        return out / max(1e-8, np.linalg.norm(out))

    def _masked_crops(
        self,
        original: Image.Image,
        tampered: Image.Image,
        region: ObjectRegion,
    ) -> tuple[Image.Image, Image.Image]:
        width, height = original.size
        box = pad_box(region.bbox, width, height, self.config.crop_padding)
        x0, y0, x1, y1 = box
        mask = region.object_mask[y0:y1, x0:x1]
        orig = np.asarray(original.crop(box), dtype=np.uint8)
        tamp = np.asarray(tampered.crop(box), dtype=np.uint8)
        if mask.size == 0 or not mask.any():
            return Image.fromarray(orig), Image.fromarray(tamp)
        fill_orig = np.full_like(orig, 127)
        fill_tamp = np.full_like(tamp, 127)
        mask3 = mask[:, :, None]
        orig = np.where(mask3, orig, fill_orig)
        tamp = np.where(mask3, tamp, fill_tamp)
        return Image.fromarray(orig), Image.fromarray(tamp)

    @staticmethod
    def _pixel_delta(crop_original: Image.Image, crop_tampered: Image.Image) -> float:
        arr_a = np.asarray(crop_original.resize((224, 224)), dtype=np.float32) / 255.0
        arr_b = np.asarray(crop_tampered.resize((224, 224)), dtype=np.float32) / 255.0
        return float(np.mean(np.abs(arr_a - arr_b)))
