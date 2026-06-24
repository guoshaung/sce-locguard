"""Service wrapper for SCE-LocGuard research pipelines.

The current project is script-first research code. This module provides a
stable SDK-style surface for larger systems while keeping the existing
algorithms untouched. Single-image production embedding/verification is marked
explicitly as not implemented until the research scripts are adapted into a
true one-image pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from .schemas import EmbedRequest, EmbedResponse, VerifyRequest, VerifyResponse


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class SCELocGuardService:
    """Thin wrapper around SCE-LocGuard / EditGuard-style pipelines."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).expanduser().resolve()
        self.default_job_root = self.project_root / "dfg_locguard" / "outputs" / "api_jobs"
        self.job_index_dir = self.project_root / "dfg_locguard" / "outputs" / "api_job_index"

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Register/embed an image.

        The wrapper currently validates inputs, creates a durable job record,
        and derives a deterministic 64-bit capsule preview. It does not claim
        that a watermarked image was produced unless a future single-image
        pipeline is connected.
        """

        image_path, error = self._validate_input_image(request.image_path)
        if error:
            return EmbedResponse(status="error", message=error)

        job_id, job_dir = self._create_job_dir(request.output_dir, "embed")
        capsule_bits, payload_id, auth_hash = self._build_capsule_preview(
            request.owner_id, request.semantic_metadata
        )

        response = EmbedResponse(
            status="not_implemented_single_image_pipeline",
            job_id=job_id,
            watermarked_image_path=None,
            payload_id=payload_id,
            capsule_bits=capsule_bits,
            auth_hash=auth_hash,
            message=(
                "Input image registered and API job created, but the current "
                "research code does not yet expose a real single-image embed "
                "pipeline. Existing batch scripts are left unchanged."
            ),
        )
        report = {
            "job_id": job_id,
            "job_type": "embed",
            "created_at": self._now(),
            "request": _model_to_dict(request),
            "input_image_path": str(image_path),
            "response": _model_to_dict(response),
            "algorithm_status": "single_image_embed_not_connected",
            "constraints": self._constraints(),
        }
        report_path = job_dir / "embed_report.json"
        self._write_json(report_path, report)
        self._save_index(job_id, "embed", job_dir, report_path)
        return response

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        """Blindly verify a possibly tampered image.

        The current safe wrapper does not access an original image or GT mask.
        It returns an explicit skeleton response until a single-image blind
        verification script is wired in.
        """

        if request.mode != "blind":
            return VerifyResponse(
                status="error",
                job_id=None,
                auth_status="not_evaluated",
                payload_recovered=False,
                capsule_recovered=False,
                attack_regime="unknown",
                reports=[],
                message="Only blind verification mode is supported by this API.",
            )

        image_path, error = self._validate_input_image(request.image_path)
        if error:
            return VerifyResponse(
                status="error",
                job_id=None,
                auth_status="not_evaluated",
                payload_recovered=False,
                capsule_recovered=False,
                attack_regime="unknown",
                reports=[],
                message=error,
            )

        job_id, job_dir = self._create_job_dir(request.output_dir, "verify")
        response = VerifyResponse(
            status="not_implemented_single_image_pipeline",
            job_id=job_id,
            auth_status="not_evaluated",
            payload_recovered=False,
            capsule_recovered=False,
            predicted_mask_path=None,
            overlay_path=None,
            report_json_path=str(job_dir / "verify_report.json"),
            attack_regime="unknown",
            reports=[],
            message=(
                "Input image validated and blind verification job created, but "
                "the current research code does not yet expose a real "
                "single-image verification pipeline. No original image or GT "
                "mask was accessed."
            ),
        )
        report = {
            "job_id": job_id,
            "job_type": "verify",
            "created_at": self._now(),
            "request": _model_to_dict(request),
            "input_image_path": str(image_path),
            "response": _model_to_dict(response),
            "algorithm_status": "single_image_verify_not_connected",
            "verification_mode": "blind",
            "original_image_accessed": False,
            "gt_mask_accessed": False,
            "constraints": self._constraints(),
        }
        report_path = job_dir / "verify_report.json"
        self._write_json(report_path, report)
        self._save_index(job_id, "verify", job_dir, report_path)
        return response

    def export_report(self, job_id: str) -> Dict[str, Any]:
        """Return a stored API job report."""

        if not self._is_safe_job_id(job_id):
            return {"status": "error", "message": "Invalid job_id format."}

        index_path = self.job_index_dir / f"{job_id}.json"
        if not index_path.exists():
            return {"status": "error", "message": "job_id not found."}
        index = self._read_json(index_path)
        report_path = Path(index.get("report_json_path", ""))
        if not report_path.exists():
            return {
                "status": "error",
                "message": "job report record exists, but report file is missing.",
                "job_id": job_id,
            }
        return self._read_json(report_path)

    def run_subprocess(self, command: list[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        """Execute an existing research script for future integration points."""

        return subprocess.run(
            command,
            cwd=cwd or str(self.project_root),
            check=False,
            text=True,
            capture_output=True,
        )

    def _validate_input_image(self, image_path: str) -> Tuple[Optional[Path], Optional[str]]:
        if not image_path:
            return None, "image_path is required."
        path = Path(image_path).expanduser()
        if not path.exists():
            return None, "Input image path not found."
        if not path.is_file():
            return None, "Input image path is not a file."
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
            return None, "Input file extension is not a supported image type."
        return path.resolve(), None

    def _create_job_dir(self, output_dir: str, prefix: str) -> Tuple[str, Path]:
        root = Path(output_dir).expanduser() if output_dir else self.default_job_root
        job_id = f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        job_dir = root.resolve() / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_id, job_dir

    def _build_capsule_preview(self, owner_id: str, metadata: Dict[str, Any]) -> Tuple[str, str, str]:
        payload = {
            "owner_id": owner_id,
            "semantic_metadata": metadata or {},
            "payload_bits": 64,
            "schema": "sce_locguard_semantic_capsule_v0",
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        capsule_bits = "".join(f"{byte:08b}" for byte in digest[:8])
        payload_id = hashlib.sha256((owner_id + canonical).encode("utf-8")).hexdigest()[:16]
        auth_hash = hashlib.sha256((capsule_bits + payload_id).encode("utf-8")).hexdigest()[:16]
        return capsule_bits, payload_id, auth_hash

    def _save_index(self, job_id: str, job_type: str, job_dir: Path, report_path: Path) -> None:
        self.job_index_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.job_index_dir / f"{job_id}.json",
            {
                "job_id": job_id,
                "job_type": job_type,
                "job_dir": str(job_dir),
                "report_json_path": str(report_path),
                "created_at": self._now(),
            },
        )

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _constraints(self) -> Dict[str, Any]:
        return {
            "no_training": True,
            "no_vlm": True,
            "no_128bit_payload": True,
            "watermark_embedding_modified": False,
            "verification_is_blind": True,
            "gt_mask_api_input": False,
            "semantic_report_type": "coarse_rule_based",
            "robust_payload_bits": 64,
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _is_safe_job_id(self, job_id: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", job_id or ""))

    def copy_artifact(self, source: Path, destination: Path) -> None:
        """Small helper kept for future script integration."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
