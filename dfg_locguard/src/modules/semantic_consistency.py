from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .common import BICUBIC, SemanticResult, TamperRegion, pad_box


@dataclass
class SCCConfig:
    feature_method: str = "auto"
    clip_model: str = "openai/clip-vit-base-patch32"
    crop_padding: int = 8
    device: str = "cpu"


class SemanticConsistencyChecker:
    """SCC v0: compare original and tampered region features."""

    def __init__(self, config: SCCConfig):
        self.config = config
        self._clip = None
        self._clip_processor = None
        self._device = None

    def compare(self, original: Image.Image, tampered: Image.Image, tamper_region: TamperRegion) -> SemanticResult:
        crop_a, crop_b = self._crop_pair(original, tampered, tamper_region)
        method = self.config.feature_method.lower()
        if method in {"auto", "clip"}:
            try:
                return self._clip_similarity(crop_a, crop_b)
            except Exception:
                if method == "clip":
                    raise
        return self._fallback_similarity(crop_a, crop_b)

    def _crop_pair(
        self,
        original: Image.Image,
        tampered: Image.Image,
        tamper_region: TamperRegion,
    ) -> tuple[Image.Image, Image.Image]:
        width, height = original.size
        box = pad_box(tamper_region.region.bbox, width, height, self.config.crop_padding)
        x1, y1, x2, y2 = box
        region_mask = tamper_region.region.mask[y1:y2, x1:x2]
        crop_a = np.asarray(original.crop(box), dtype=np.uint8)
        crop_b = np.asarray(tampered.crop(box), dtype=np.uint8)
        if region_mask.size and region_mask.any():
            fill = np.full_like(crop_a, 127)
            crop_a = np.where(region_mask[:, :, None], crop_a, fill)
            crop_b = np.where(region_mask[:, :, None], crop_b, fill)
        return Image.fromarray(crop_a), Image.fromarray(crop_b)

    def _clip_similarity(self, crop_a: Image.Image, crop_b: Image.Image) -> SemanticResult:
        import torch
        import torch.nn.functional as F
        from transformers import CLIPModel, CLIPProcessor

        if self._device is None:
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
        if self._clip is None:
            self._clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)
            self._clip = CLIPModel.from_pretrained(self.config.clip_model).to(self._device).eval()
        inputs = self._clip_processor(images=[crop_a, crop_b], return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            feats = self._clip.get_image_features(**inputs)
            feats = F.normalize(feats, dim=-1)
            cosine = float((feats[0] * feats[1]).sum().detach().cpu())
        similarity = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
        return SemanticResult(similarity, 1.0 - similarity, "clip")

    def _fallback_similarity(self, crop_a: Image.Image, crop_b: Image.Image) -> SemanticResult:
        arr_a = np.asarray(crop_a.resize((128, 128), BICUBIC), dtype=np.float32) / 255.0
        arr_b = np.asarray(crop_b.resize((128, 128), BICUBIC), dtype=np.float32) / 255.0
        feat_a = self._fallback_feature(arr_a)
        feat_b = self._fallback_feature(arr_b)
        similarity = float(np.dot(feat_a, feat_b) / max(1e-8, np.linalg.norm(feat_a) * np.linalg.norm(feat_b)))
        similarity = float(np.clip(similarity, 0.0, 1.0))
        return SemanticResult(similarity, 1.0 - similarity, "fallback_histogram_pixel")

    @staticmethod
    def _fallback_feature(arr: np.ndarray) -> np.ndarray:
        feats = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=24, range=(0.0, 1.0), density=True)
            feats.append(hist.astype(np.float32))
        gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
        feats.append(small)
        feature = np.concatenate(feats)
        return feature / max(1e-8, np.linalg.norm(feature))

