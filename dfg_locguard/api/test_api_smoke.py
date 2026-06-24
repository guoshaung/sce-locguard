"""Smoke tests for the SCE-LocGuard API wrapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from dfg_locguard.api.schemas import EmbedRequest, VerifyRequest
from dfg_locguard.api.server import app
from dfg_locguard.api.service import SCELocGuardService


def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class TestSCELocGuardApiSmoke(unittest.TestCase):
    def test_import_service(self):
        service = SCELocGuardService(project_root=".")
        self.assertIsNotNone(service)

    def test_schema_serialization(self):
        request = EmbedRequest(
            image_path="example.png",
            owner_id="owner",
            semantic_metadata={"risk": "medium"},
            output_dir="out",
        )
        payload = model_to_dict(request)
        self.assertEqual(payload["owner_id"], "owner")
        self.assertEqual(payload["semantic_metadata"]["risk"], "medium")

    def test_health_works(self):
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "sce-locguard")

    def test_invalid_image_path_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = SCELocGuardService(project_root=tmpdir)
            response = service.verify(
                VerifyRequest(
                    image_path=str(Path(tmpdir) / "missing.png"),
                    mode="blind",
                    output_dir=str(Path(tmpdir) / "jobs"),
                )
            )
            self.assertEqual(response.status, "error")
            self.assertIn("not found", response.message.lower())

            client = TestClient(app)
            api_response = client.post(
                "/api/v1/watermark/verify",
                json={
                    "image_path": str(Path(tmpdir) / "missing.png"),
                    "mode": "blind",
                    "output_dir": str(Path(tmpdir) / "jobs"),
                },
            )
            self.assertEqual(api_response.status_code, 400)
            self.assertIn("not found", api_response.text.lower())

    def test_verify_job_can_export_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "input.png"
            image_path.write_bytes(b"not-a-real-image-but-path-valid")
            service = SCELocGuardService(project_root=tmpdir)
            response = service.verify(
                VerifyRequest(
                    image_path=str(image_path),
                    mode="blind",
                    output_dir=str(tmp / "jobs"),
                )
            )
            self.assertEqual(response.status, "not_implemented_single_image_pipeline")
            self.assertTrue(response.job_id)
            report = service.export_report(response.job_id or "")
            self.assertEqual(report["job_type"], "verify")
            self.assertFalse(report["original_image_accessed"])


if __name__ == "__main__":
    unittest.main()
