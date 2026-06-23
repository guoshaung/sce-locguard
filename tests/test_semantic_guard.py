from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from semantic_guard.pipeline import SemanticGuardConfig, SemanticGuardPipeline
from semantic_guard.scc import SCCConfig, SemanticChangeChecker
from semantic_guard.schema import ObjectRegion, SCCResult


def test_semantic_guard_components_histogram(tmp_path: Path) -> None:
    original = np.full((96, 96, 3), 240, dtype=np.uint8)
    original[24:72, 24:72] = [220, 30, 30]
    tampered = original.copy()
    tampered[36:66, 36:66] = [30, 80, 230]
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[36:66, 36:66] = 255

    original_path = tmp_path / "original.png"
    tampered_path = tmp_path / "tampered.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(original).save(original_path)
    Image.fromarray(tampered).save(tampered_path)
    Image.fromarray(mask).save(mask_path)

    config = SemanticGuardConfig(
        tsa_backend="components",
        scc_backend="histogram",
        context_dilation=3,
        min_area=8,
        categories=[],
    )
    report = SemanticGuardPipeline(config).run(original_path, tampered_path, mask_path, tmp_path / "out")

    assert len(report["objects"]) == 1
    assert report["objects"][0]["scc"]["change_score"] > 0.1
    assert report["objects"][0]["sse"]["score"] > 0.0
    assert (tmp_path / "out" / "semantic_guard_report.json").exists()
    with (tmp_path / "out" / "semantic_guard_report.json").open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["mask_pixels"] == 900


def test_hybrid_scc_fuses_clip_dino_and_pixel(monkeypatch) -> None:
    checker = SemanticChangeChecker(
        SCCConfig(
            backend="hybrid",
            hybrid_clip_weight=0.4,
            hybrid_dino_weight=0.45,
            hybrid_pixel_weight=0.15,
        )
    )

    def fake_clip(*_args, **_kwargs):
        return SCCResult(
            backend="clip",
            similarity=0.80,
            change_score=0.20,
            category="person",
            category_confidence=0.70,
            pixel_delta=0.10,
        )

    def fake_dino(*_args, **_kwargs):
        return SCCResult(backend="dino", similarity=0.60, change_score=0.40, pixel_delta=0.10)

    monkeypatch.setattr(checker, "_compare_clip", fake_clip)
    monkeypatch.setattr(checker, "_compare_dino", fake_dino)
    monkeypatch.setattr(checker, "_pixel_delta", lambda *_args, **_kwargs: 0.50)

    mask = np.ones((16, 16), dtype=bool)
    region = ObjectRegion(
        region_id=1,
        object_mask=mask,
        edit_mask=mask,
        bbox=(0, 0, 16, 16),
        source="test",
        confidence=1.0,
    )
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    result = checker.compare(image, image, region)

    assert result.backend == "hybrid"
    assert result.category == "person"
    assert result.components["clip_change"] == 0.20
    assert result.components["dino_change"] == 0.40
    assert result.components["pixel_change"] == 1.0
    assert abs(result.change_score - 0.41) < 1e-6
