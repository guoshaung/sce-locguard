from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import ensure_size, load_mask, load_rgb, parse_label_list, save_json, save_mask
from .scc import SCCConfig, SemanticChangeChecker
from .sse import SSEConfig, SemanticSeverityEstimator, canonical_category
from .tsa import TSAConfig, TamperedSemanticAggregator
from .visualize import save_object_index_mask, save_overlay


DEFAULT_CATEGORIES = [
    "person",
    "face",
    "text",
    "logo",
    "document",
    "license plate",
    "screen",
    "vehicle",
    "product",
    "background",
    "object",
]


@dataclass
class SemanticGuardConfig:
    tsa_backend: str = "auto"
    scc_backend: str = "auto"
    device: str = "auto"
    mask_threshold: float = 0.5
    sam_checkpoint: str | None = None
    sam_model_type: str = "vit_b"
    min_area: int = 32
    context_dilation: int = 5
    clip_model: str = "openai/clip-vit-base-patch32"
    dino_model: str = "facebook/dinov2-small"
    hybrid_clip_weight: float = 0.40
    hybrid_dino_weight: float = 0.45
    hybrid_pixel_weight: float = 0.15
    fusion_model: str | None = None
    categories: list[str] = field(default_factory=lambda: DEFAULT_CATEGORIES[:])


class SemanticGuardPipeline:
    def __init__(self, config: SemanticGuardConfig):
        self.config = config
        self.tsa = TamperedSemanticAggregator(
            TSAConfig(
                backend=config.tsa_backend,
                sam_checkpoint=config.sam_checkpoint,
                sam_model_type=config.sam_model_type,
                device=config.device,
                min_area=config.min_area,
                context_dilation=config.context_dilation,
            )
        )
        self.scc = SemanticChangeChecker(
            SCCConfig(
                backend=config.scc_backend,
                device=config.device,
                clip_model=config.clip_model,
                dino_model=config.dino_model,
                hybrid_clip_weight=config.hybrid_clip_weight,
                hybrid_dino_weight=config.hybrid_dino_weight,
                hybrid_pixel_weight=config.hybrid_pixel_weight,
            )
        )
        self.sse = SemanticSeverityEstimator(SSEConfig(fusion_model=config.fusion_model))

    def run(
        self,
        original_path: str | Path,
        tampered_path: str | Path,
        edit_mask_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        original = load_rgb(original_path)
        tampered = ensure_size(load_rgb(tampered_path), original.size)
        edit_mask = load_mask(edit_mask_path, target_size=original.size, threshold=self.config.mask_threshold)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        regions = self.tsa.extract(tampered, edit_mask)
        image_area = original.size[0] * original.size[1]
        object_payloads: list[dict[str, Any]] = []
        severities = []

        for region in regions:
            scc = self.scc.compare(original, tampered, region, self.config.categories)
            region.category = canonical_category(scc.category)
            region.category_confidence = scc.category_confidence
            severity = self.sse.score(region, scc, image_area)
            severities.append(severity)
            save_mask(region.object_mask, output_dir / f"object_{region.region_id:03d}_mask.png")
            object_payloads.append(
                {
                    "region": region.to_dict(image_area=image_area),
                    "scc": scc.to_dict(),
                    "sse": severity.to_dict(),
                }
            )

        save_overlay(tampered, edit_mask, regions, severities, output_dir / "semantic_guard_overlay.png")
        save_object_index_mask(regions, edit_mask.shape, output_dir / "semantic_objects.png")
        save_mask(edit_mask, output_dir / "editguard_mask_binary.png")

        payload: dict[str, Any] = {
            "inputs": {
                "original": str(Path(original_path)),
                "tampered": str(Path(tampered_path)),
                "edit_mask": str(Path(edit_mask_path)),
            },
            "config": {
                "tsa_backend": self.config.tsa_backend,
                "scc_backend": self.config.scc_backend,
                "device": self.config.device,
                "sam_checkpoint": self.config.sam_checkpoint,
                "sam_model_type": self.config.sam_model_type,
                "clip_model": self.config.clip_model,
                "dino_model": self.config.dino_model,
                "hybrid_clip_weight": self.config.hybrid_clip_weight,
                "hybrid_dino_weight": self.config.hybrid_dino_weight,
                "hybrid_pixel_weight": self.config.hybrid_pixel_weight,
                "fusion_model": self.config.fusion_model,
                "categories": parse_label_list(self.config.categories),
            },
            "image_size": {"width": original.size[0], "height": original.size[1]},
            "mask_pixels": int(edit_mask.sum()),
            "objects": object_payloads,
        }
        save_json(payload, output_dir / "semantic_guard_report.json")
        return payload
