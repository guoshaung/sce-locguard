"""MVP integration tests for the SCE-LocGuard API wrapper.

These tests do not require checkpoints, CUDA, or Stable Diffusion cache. If the
real runtime is absent, they assert that the wrapper fails explicitly instead
of fabricating successful outputs.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from dfg_locguard.api.schemas import AIGCAttackRequest, EmbedRequest, VerifyRequest
from dfg_locguard.api.server import app
from dfg_locguard.api.service import SCELocGuardService


def dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class TestSCELocGuardApiMvp(unittest.TestCase):
    def test_health_and_capabilities_work(self):
        client = TestClient(app)
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        caps = client.get("/api/v1/capabilities")
        self.assertEqual(caps.status_code, 200)
        payload = caps.json()
        self.assertIn("embed_real_pipeline_available", payload)
        self.assertFalse(payload["supports_vlm"])
        self.assertTrue(payload["supports_64bit_capsule"])

    def test_schema_serialization(self):
        embed = EmbedRequest(
            image_path="input.png",
            owner_id="owner",
            semantic_metadata={"risk": "medium"},
            output_dir=None,
        )
        attack = AIGCAttackRequest(
            image_path="watermarked.png",
            attack_type="inpainting",
            mode="localized_composite",
        )
        verify = VerifyRequest(image_path="attacked.png")
        self.assertEqual(dump(embed)["owner_id"], "owner")
        self.assertEqual(dump(attack)["attack_type"], "inpainting")
        self.assertTrue(dump(verify)["return_overlay"])

    def test_invalid_image_path_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = SCELocGuardService(project_root=tmpdir)
            response = service.embed(
                EmbedRequest(
                    image_path=str(Path(tmpdir) / "missing.png"),
                    owner_id="owner",
                    output_dir=str(Path(tmpdir) / "jobs"),
                )
            )
            self.assertEqual(response.status, "error")
            self.assertIn("not found", response.message.lower())

    def test_service_creates_job_dirs_without_fake_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "input.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            service = SCELocGuardService(project_root=tmpdir)
            response = service.embed(
                EmbedRequest(
                    image_path=str(image_path),
                    owner_id="owner",
                    output_dir=str(tmp / "jobs"),
                )
            )
            self.assertIn(response.status, {"ok", "not_implemented"})
            self.assertTrue(response.job_id)
            report = service.export_report(response.job_id or "")
            self.assertEqual(report["job_type"], "embed")
            if response.status != "ok":
                self.assertEqual(response.implementation_level, "skeleton")
                self.assertFalse(response.watermarked_image_path)

    def test_attack_missing_model_returns_model_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "watermarked.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            service = SCELocGuardService(project_root=tmpdir)
            response = service.attack_aigc(
                AIGCAttackRequest(
                    image_path=str(image_path),
                    attack_type="inpainting",
                    mode="localized_composite",
                    output_dir=str(tmp / "jobs"),
                )
            )
            self.assertIn(response.status, {"ok", "model_unavailable"})
            self.assertTrue(response.job_id)
            if response.status != "ok":
                self.assertEqual(response.implementation_level, "skeleton")
                self.assertFalse(response.attacked_image_path)

    def test_demo_fails_explicitly_if_runtime_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "input.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            service = SCELocGuardService(project_root=tmpdir)
            summary = service.demo(
                image_path=str(image_path),
                owner_id="owner",
                attack_type="inpainting",
                mode="localized_composite",
                output_dir=str(tmp / "demo"),
            )
            self.assertIn(summary["status"], {"ok", "not_implemented"})
            if summary["status"] != "ok":
                self.assertIn(summary["failed_stage"], {"embed", "attack", "verify"})
                self.assertTrue((tmp / "demo" / "demo_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
