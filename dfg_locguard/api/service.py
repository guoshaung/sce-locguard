"""Service wrapper for SCE-LocGuard research pipelines.

This module upgrades the first API skeleton into a real MVP wrapper where the
existing callable research pipeline can be reused safely. It keeps the
watermarking method unchanged:

- no training
- no VLM
- no 128-bit payload
- no watermark embedding modification
- blind verification does not read the original image or GT mask

When a required model/checkpoint/cache is unavailable, the service returns a
clear `not_implemented` or `model_unavailable` response instead of fabricating
masks, reports, or payload status.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from .schemas import (
    AIGCAttackRequest,
    AIGCAttackResponse,
    CapabilityResponse,
    EmbedRequest,
    EmbedResponse,
    ReportRegion,
    VerifyRequest,
    VerifyResponse,
)


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
NEAREST = getattr(Image, "Resampling", Image).NEAREST
ATTACK_TYPES = {"object_removal", "inpainting", "local_replacement", "local_style_edit"}


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class SCELocGuardService:
    """Thin service layer around SCE-LocGuard / EditGuard-style code."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).expanduser().resolve()
        self.default_job_root = self.project_root / "dfg_locguard" / "api_jobs"
        self.job_index_dir = self.default_job_root / "_index"
        self.ckpt_path = self.project_root / "checkpoints" / "clean.pth"
        self.opt_path = self.project_root / "code" / "options" / "test_editguard.yml"
        self.recovery_threshold = 0.2

    def capabilities(self) -> CapabilityResponse:
        editguard_ok, editguard_notes = self._editguard_runtime_available()
        attack_ok, attack_notes = self._attack_runtime_available()
        notes = editguard_notes + attack_notes
        notes.extend(
            [
                "Single-image embed uses the existing EditGuard image_hiding() path when checkpoint/CUDA are available.",
                "Single-image verify uses image_recovery() with the fixed localization template; it does not access original image or GT mask.",
                "Stage8E/Stage8F full batch report calibration is not run inside the API MVP; verification report is a lightweight rule-based readout.",
            ]
        )
        return CapabilityResponse(
            embed_real_pipeline_available=editguard_ok,
            attack_real_pipeline_available=attack_ok,
            verify_real_pipeline_available=editguard_ok,
            supports_64bit_capsule=True,
            supports_vlm=False,
            supports_single_image=True,
            notes=notes,
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        image_path, error = self._validate_input_image(request.image_path)
        if error:
            return EmbedResponse(status="error", implementation_level="skeleton", message=error)

        job_id, job_dir = self._create_phase_dir(request.output_dir, "embed")
        editguard_ok, notes = self._editguard_runtime_available()
        capsule_bits, payload_id, auth_hash = self._build_capsule_preview(
            request.owner_id, request.semantic_metadata
        )
        if not editguard_ok:
            response = EmbedResponse(
                status="not_implemented",
                job_id=job_id,
                implementation_level="skeleton",
                payload_id=payload_id,
                capsule_bits=capsule_bits,
                auth_hash=auth_hash,
                message="Real single-image embed is unavailable: " + "; ".join(notes),
            )
            self._write_job_report(job_id, "embed", job_dir, request, response, {"notes": notes})
            return response

        try:
            stage1, s8c, _attack_mod = self._load_research_modules()
            model = stage1.load_editguard_model(self.project_root, self.opt_path, self.ckpt_path)
            original = self._load_rgb_512(image_path)
            variant = s8c.PayloadVariant("P2_id_plus_compact_capsule_64", 64, "direct", "")
            payload = s8c.build_payload_bits(variant, original, image_path.stem, 0, 20260624)
            message = s8c.bits_to_message(payload["bits"])
            data = stage1.image_to_editguard_batch(original)
            data["MES"] = message
            model.feed_data(data)
            model.mes = message
            watermarked = model.image_hiding()

            source_copy = job_dir / "input.png"
            watermarked_path = job_dir / "watermarked.png"
            payload_path = job_dir / "payload.json"
            Image.fromarray(original).save(source_copy)
            Image.fromarray(watermarked).save(watermarked_path)
            self._write_json(
                payload_path,
                {
                    "payload_variant": "P2_id_plus_compact_capsule_64",
                    "owner_id": request.owner_id,
                    "requested_semantic_metadata": request.semantic_metadata,
                    "payload": payload,
                    "capsule_bits": self._bits_to_string(payload["bits"]),
                    "auth_hash": self._payload_auth_hash(payload["bits"]),
                    "note": "Payload packing reuses Stage8C P2 64-bit compact capsule logic.",
                },
            )
            response = EmbedResponse(
                status="ok",
                job_id=job_id,
                implementation_level="real_pipeline",
                watermarked_image_path=str(watermarked_path),
                payload_id=payload_id,
                capsule_bits=self._bits_to_string(payload["bits"]),
                auth_hash=self._payload_auth_hash(payload["bits"]),
                psnr=self._psnr(original, watermarked),
                message="Watermarked image generated with existing EditGuard image_hiding() pipeline.",
            )
            self._write_job_report(
                job_id,
                "embed",
                job_dir,
                request,
                response,
                {
                    "input_image_path": str(image_path),
                    "payload_path": str(payload_path),
                    "watermarked_image_path": str(watermarked_path),
                },
            )
            return response
        except Exception as exc:
            response = EmbedResponse(
                status="not_implemented",
                job_id=job_id,
                implementation_level="skeleton",
                payload_id=payload_id,
                capsule_bits=capsule_bits,
                auth_hash=auth_hash,
                message=f"Real embed pipeline failed to connect: {type(exc).__name__}: {exc}",
            )
            self._write_job_report(job_id, "embed", job_dir, request, response, {"exception": repr(exc)})
            return response

    def attack_aigc(self, request: AIGCAttackRequest) -> AIGCAttackResponse:
        image_path, error = self._validate_input_image(request.image_path)
        if error:
            return AIGCAttackResponse(
                status="error",
                implementation_level="skeleton",
                attack_type=request.attack_type,
                mode=request.mode,
                message=error,
            )
        if request.attack_type not in ATTACK_TYPES:
            return AIGCAttackResponse(
                status="error",
                implementation_level="skeleton",
                attack_type=request.attack_type,
                mode=request.mode,
                message=f"Unsupported attack_type: {request.attack_type}",
            )

        job_id, job_dir = self._create_phase_dir(request.output_dir, "attack")
        attack_ok, notes = self._attack_runtime_available()
        if not attack_ok:
            response = AIGCAttackResponse(
                status="model_unavailable",
                job_id=job_id,
                implementation_level="skeleton",
                attack_type=request.attack_type,
                mode=request.mode,
                source_image_path=str(image_path),
                message="Real AIGC attack model/cache is unavailable: " + "; ".join(notes),
            )
            self._write_job_report(job_id, "attack", job_dir, request, response, {"notes": notes})
            return response

        try:
            _stage1, _s8c, attack_mod = self._load_research_modules()
            pipe_args = self._attack_args(request)
            pipe = attack_mod.load_diffusers_pipe(pipe_args, self.project_root)
            source = self._load_pil_512(image_path)
            mask, mask_source = self._load_or_create_attack_mask(request.mask_path, source.size, request.attack_type)
            if request.prompt:
                attack_mod.PROMPTS[request.attack_type] = request.prompt

            start = time.time()
            diffusion = attack_mod.generate_with_diffusers(
                pipe,
                source,
                mask,
                request.attack_type,
                self._job_seed(job_id),
                pipe_args,
            )
            runtime = time.time() - start
            mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
            src_arr = np.asarray(source.convert("RGB"), dtype=np.float32)
            diff_arr = np.asarray(diffusion.convert("RGB"), dtype=np.float32)
            if request.mode == "localized_composite":
                alpha = mask_arr[:, :, None]
                attacked_arr = (alpha * diff_arr + (1.0 - alpha) * src_arr).clip(0, 255).astype(np.uint8)
                outside_preserved = bool(np.max(np.abs(attacked_arr[mask_arr < 0.5].astype(np.float32) - src_arr[mask_arr < 0.5])) <= 1.0) if np.any(mask_arr < 0.5) else True
            else:
                attacked_arr = diff_arr.clip(0, 255).astype(np.uint8)
                outside_preserved = False

            mask_path = job_dir / "attack_mask.png"
            diffusion_path = job_dir / "diffusion_output.png"
            attacked_path = job_dir / "attacked.png"
            source_path = job_dir / "source.png"
            source.save(source_path)
            mask.save(mask_path)
            diffusion.save(diffusion_path)
            Image.fromarray(attacked_arr).save(attacked_path)
            response = AIGCAttackResponse(
                status="ok",
                job_id=job_id,
                implementation_level="real_pipeline",
                attack_type=request.attack_type,
                mode=request.mode,
                source_image_path=str(source_path),
                attacked_image_path=str(attacked_path),
                diffusion_output_path=str(diffusion_path),
                mask_path=str(mask_path),
                outside_mask_preserved=outside_preserved,
                message=f"AIGC attack generated with cached diffusers inpainting model in {runtime:.2f}s; mask_source={mask_source}.",
            )
            self._write_job_report(
                job_id,
                "attack",
                job_dir,
                request,
                response,
                {
                    "mask_source": mask_source,
                    "runtime_seconds": runtime,
                    "model_id": pipe_args.model_id,
                    "hf_cache_dir": pipe_args.model_cache_dir,
                },
            )
            return response
        except Exception as exc:
            response = AIGCAttackResponse(
                status="model_unavailable",
                job_id=job_id,
                implementation_level="skeleton",
                attack_type=request.attack_type,
                mode=request.mode,
                source_image_path=str(image_path),
                message=f"Real AIGC attack pipeline failed to connect: {type(exc).__name__}: {exc}",
            )
            self._write_job_report(job_id, "attack", job_dir, request, response, {"exception": repr(exc)})
            return response

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        image_path, error = self._validate_input_image(request.image_path)
        if error:
            return VerifyResponse(
                status="error",
                implementation_level="skeleton",
                auth_status="not_evaluated",
                payload_recovered=False,
                capsule_recovered=False,
                attack_regime="unknown",
                reports=[],
                message=error,
            )

        job_id, job_dir = self._create_phase_dir(request.output_dir, "verify")
        editguard_ok, notes = self._editguard_runtime_available()
        if not editguard_ok:
            response = VerifyResponse(
                status="not_implemented",
                job_id=job_id,
                implementation_level="skeleton",
                auth_status="not_evaluated",
                payload_recovered=False,
                capsule_recovered=False,
                attack_regime="unknown",
                reports=[],
                report_json_path=str(job_dir / "verify_report.json"),
                message="Real single-image blind verify is unavailable: " + "; ".join(notes),
            )
            self._write_job_report(job_id, "verify", job_dir, request, response, {"notes": notes})
            return response

        try:
            stage1, _s8c, _attack_mod = self._load_research_modules()
            model = stage1.load_editguard_model(self.project_root, self.opt_path, self.ckpt_path)
            suspect = self._load_rgb_512(image_path)
            data = stage1.image_to_editguard_batch(suspect)
            data["MES"] = None
            model.feed_data(data)
            pred_mask_raw, recmessage = model.image_recovery(self.recovery_threshold)
            pred_mask = (np.asarray(pred_mask_raw) > 0).astype(np.uint8)
            recovered_bits = self._recovered_bits(recmessage)
            decoded = self._decode_p2_payload(recovered_bits)
            regions = self._regions_from_mask(pred_mask, suspect, decoded.get("auth_status", "invalid"))
            pred_mask_path = job_dir / "predicted_mask.png"
            overlay_path = job_dir / "overlay.png"
            report_path = job_dir / "verify_report.json"
            Image.fromarray(pred_mask * 255).save(pred_mask_path)
            overlay_result = None
            if request.return_overlay:
                overlay_result = self._save_overlay(suspect, pred_mask, regions, overlay_path)
            attack_regime = self._attack_regime(float(pred_mask.mean()))
            response = VerifyResponse(
                status="ok",
                job_id=job_id,
                implementation_level="real_pipeline",
                auth_status=decoded["auth_status"],
                payload_recovered=bool(decoded["auth_success"]),
                capsule_recovered=bool(decoded["capsule_recovered"]),
                bit_accuracy=None,
                predicted_mask_path=str(pred_mask_path),
                overlay_path=str(overlay_result) if overlay_result else None,
                report_json_path=str(report_path),
                attack_regime=attack_regime,
                reports=regions,
                message=(
                    "Blind verification completed with existing EditGuard image_recovery(). "
                    "No original image or GT mask was accessed. bit_accuracy is unavailable in blind mode."
                ),
            )
            self._write_job_report(
                job_id,
                "verify",
                job_dir,
                request,
                response,
                {
                    "input_image_path": str(image_path),
                    "predicted_mask_area_ratio": float(pred_mask.mean()),
                    "recovered_bits": self._bits_to_string(recovered_bits),
                    "decoded_payload": decoded,
                    "verification_mode": "blind",
                    "original_image_accessed": False,
                    "gt_mask_accessed": False,
                },
            )
            return response
        except Exception as exc:
            response = VerifyResponse(
                status="not_implemented",
                job_id=job_id,
                implementation_level="skeleton",
                auth_status="not_evaluated",
                payload_recovered=False,
                capsule_recovered=False,
                attack_regime="unknown",
                reports=[],
                report_json_path=str(job_dir / "verify_report.json"),
                message=f"Real blind verify pipeline failed to connect: {type(exc).__name__}: {exc}",
            )
            self._write_job_report(job_id, "verify", job_dir, request, response, {"exception": repr(exc)})
            return response

    def export_report(self, job_id: str) -> Dict[str, Any]:
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

    def demo(
        self,
        image_path: str,
        owner_id: str,
        attack_type: str,
        mode: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        embed = self.embed(
            EmbedRequest(
                image_path=image_path,
                owner_id=owner_id,
                semantic_metadata={},
                output_dir=str(out),
            )
        )
        self._write_json(out / "demo_embed_response.json", _model_to_dict(embed))
        if embed.status != "ok" or not embed.watermarked_image_path:
            summary = {
                "status": "not_implemented",
                "failed_stage": "embed",
                "embed_response": _model_to_dict(embed),
                "message": "Demo stopped because embed did not produce a real watermarked image.",
            }
            self._write_json(out / "demo_summary.json", summary)
            return summary
        attack = self.attack_aigc(
            AIGCAttackRequest(
                image_path=embed.watermarked_image_path,
                attack_type=attack_type,  # type: ignore[arg-type]
                output_dir=str(out),
                mode=mode,  # type: ignore[arg-type]
            )
        )
        self._write_json(out / "demo_attack_response.json", _model_to_dict(attack))
        if attack.status != "ok" or not attack.attacked_image_path:
            summary = {
                "status": "not_implemented",
                "failed_stage": "attack",
                "embed_response": _model_to_dict(embed),
                "attack_response": _model_to_dict(attack),
                "message": "Demo stopped because AIGC attack did not produce a real attacked image.",
            }
            self._write_json(out / "demo_summary.json", summary)
            return summary
        verify = self.verify(VerifyRequest(image_path=attack.attacked_image_path, output_dir=str(out)))
        self._write_json(out / "demo_verify_response.json", _model_to_dict(verify))
        summary = {
            "status": "ok" if verify.status == "ok" else "not_implemented",
            "failed_stage": None if verify.status == "ok" else "verify",
            "embed_response": _model_to_dict(embed),
            "attack_response": _model_to_dict(attack),
            "verify_response": _model_to_dict(verify),
        }
        self._write_json(out / "demo_summary.json", summary)
        return summary

    def run_subprocess(self, command: list[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
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
        path = self._resolve_path(image_path)
        if not path.exists():
            return None, "Input image path not found."
        if not path.is_file():
            return None, "Input image path is not a file."
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
            return None, "Input file extension is not a supported image type."
        return path.resolve(), None

    def _create_phase_dir(self, output_dir: Optional[str], phase: str) -> Tuple[str, Path]:
        root = self._resolve_path(output_dir) if output_dir else self.default_job_root
        job_id = f"{phase}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        phase_dir = root.resolve() / job_id / phase
        phase_dir.mkdir(parents=True, exist_ok=False)
        return job_id, phase_dir

    def _write_job_report(
        self,
        job_id: str,
        job_type: str,
        job_dir: Path,
        request: Any,
        response: Any,
        extra: Dict[str, Any],
    ) -> None:
        report_path = job_dir / f"{job_type}_report.json"
        payload = {
            "job_id": job_id,
            "job_type": job_type,
            "created_at": self._now(),
            "request": _model_to_dict(request),
            "response": _model_to_dict(response),
            "constraints": self._constraints(),
        }
        payload.update(extra)
        self._write_json(report_path, payload)
        self._save_index(job_id, job_type, job_dir, report_path)

    def _load_research_modules(self) -> Tuple[Any, Any, Any]:
        script_dir = self.project_root / "dfg_locguard" / "scripts"
        code_dir = self.project_root / "code"
        for path in (script_dir, code_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import generate_editguard_outputs_stage1 as stage1
        import run_stage8c_robust_payload_semantic_capsule_audit as s8c
        import run_stage10a_real_aigc_attack_subset as attack_mod

        return stage1, s8c, attack_mod

    def _editguard_runtime_available(self) -> Tuple[bool, List[str]]:
        notes: List[str] = []
        if not self.ckpt_path.exists():
            notes.append(f"missing checkpoint: {self.ckpt_path}")
        if not self.opt_path.exists():
            notes.append(f"missing EditGuard option file: {self.opt_path}")
        try:
            import torch

            if not torch.cuda.is_available():
                notes.append("CUDA is unavailable; current EditGuard test path expects CUDA.")
        except Exception as exc:
            notes.append(f"torch import failed: {type(exc).__name__}: {exc}")
        return len(notes) == 0, notes or ["EditGuard checkpoint/config/CUDA are available."]

    def _attack_runtime_available(self) -> Tuple[bool, List[str]]:
        notes: List[str] = []
        cache_dir = self._hf_cache_dir()
        marker = cache_dir / "models--stable-diffusion-v1-5--stable-diffusion-inpainting"
        if not marker.exists():
            notes.append(f"missing cached inpainting model: {marker}")
        try:
            import torch

            if not torch.cuda.is_available():
                notes.append("CUDA is unavailable for diffusers inpainting.")
        except Exception as exc:
            notes.append(f"torch import failed: {type(exc).__name__}: {exc}")
        try:
            import diffusers  # noqa: F401
        except Exception as exc:
            notes.append(f"diffusers import failed: {type(exc).__name__}: {exc}")
        return len(notes) == 0, notes or ["Cached Stable Diffusion inpainting model is available."]

    def _attack_args(self, request: AIGCAttackRequest) -> Any:
        import argparse
        import torch

        return argparse.Namespace(
            model_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
            model_cache_dir=str(self._hf_cache_dir()),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype="fp16" if torch.cuda.is_available() else "fp32",
            allow_model_download=False,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=0.95,
        )

    def _hf_cache_dir(self) -> Path:
        return Path(os.environ.get("HF_HOME") or self.project_root / "models" / "hf_cache").resolve()

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        project_candidate = self.project_root / path
        if project_candidate.exists() or not path.exists():
            return project_candidate
        return path.resolve()

    def _load_rgb_512(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)

    def _load_pil_512(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB").resize((512, 512), BICUBIC)

    def _load_or_create_attack_mask(self, mask_path: Optional[str], size: Tuple[int, int], attack_type: str) -> Tuple[Image.Image, str]:
        if mask_path:
            path = self._resolve_path(mask_path)
            if not path.exists():
                raise FileNotFoundError(f"mask_path not found: {path}")
            mask = Image.open(path).convert("L").resize(size, NEAREST)
            if np.asarray(mask, dtype=np.uint8).max() == 0:
                raise ValueError(f"mask_path is empty: {path}")
            return mask, "provided_mask"
        width, height = size
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        if attack_type in {"object_removal", "local_replacement"}:
            box = [int(width * 0.36), int(height * 0.34), int(width * 0.66), int(height * 0.66)]
            draw.ellipse(box, fill=255)
        else:
            box = [int(width * 0.30), int(height * 0.32), int(width * 0.70), int(height * 0.68)]
            draw.rounded_rectangle(box, radius=24, fill=255)
        return mask, "auto_demo_mask"

    def _decode_p2_payload(self, bits: List[int]) -> Dict[str, Any]:
        if len(bits) < 64:
            return {"auth_status": "invalid", "auth_success": False, "capsule_recovered": False, "reason": "recovered_bit_length_lt_64"}
        bits = [int(v) for v in bits[:64]]
        observed = bits[56:64]
        expected = self._checksum_bits(bits[:56], 8)
        auth_success = observed == expected
        capsule = {
            "copyright_id": self._bits_to_int(bits[0:16]),
            "scene_id": self._bits_to_int(bits[16:24]),
            "object_id": self._bits_to_int(bits[24:32]),
            "attribute_bits": self._bits_to_int(bits[32:44]),
            "risk_bits": self._bits_to_int(bits[44:52]),
            "layout_id": self._bits_to_int(bits[52:56]),
            "checksum_bits": self._bits_to_string(observed),
        }
        return {
            "auth_status": "valid" if auth_success else "invalid",
            "auth_success": auth_success,
            "capsule_recovered": auth_success,
            "semantic_capsule": capsule,
        }

    def _regions_from_mask(self, pred_mask: np.ndarray, image: np.ndarray, auth_status: str) -> List[ReportRegion]:
        try:
            import cv2

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), 8)
        except Exception:
            return []
        height, width = pred_mask.shape
        image_area = float(height * width)
        reports: List[ReportRegion] = []
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            region_mask = labels == idx
            area_ratio = float(area / image_area)
            cx, cy = centroids[idx]
            center_score = self._center_score(float(cx), float(cy), width, height)
            crop_values = image[region_mask]
            gray = crop_values.mean(axis=1) if crop_values.size else np.asarray([0.0])
            contrast = float(np.std(gray) / 255.0)
            brightness = float(np.mean(gray) / 255.0)
            severity = self._severity_from_region(area_ratio, center_score)
            change_type = self._change_type_from_region(area_ratio, contrast, brightness, center_score)
            confidence = float(min(1.0, 0.25 + 3.0 * area_ratio + 0.35 * contrast + 0.20 * center_score))
            reports.append(
                ReportRegion(
                    region_id=len(reports) + 1,
                    bbox=[x, y, x + w, y + h],
                    area_ratio=area_ratio,
                    change_type=change_type,
                    severity=severity,
                    auth_status=auth_status,
                    confidence=confidence,
                )
            )
        return reports

    def _save_overlay(self, image: np.ndarray, pred_mask: np.ndarray, regions: List[ReportRegion], path: Path) -> Path:
        base = Image.fromarray(image).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
        mask = Image.fromarray((pred_mask.astype(np.uint8) * 110), mode="L")
        red = Image.new("RGBA", base.size, (255, 0, 0, 110))
        overlay.paste(red, (0, 0), mask)
        out = Image.alpha_composite(base, overlay).convert("RGB")
        draw = ImageDraw.Draw(out)
        for region in regions:
            if len(region.bbox) == 4:
                draw.rectangle(region.bbox, outline=(255, 255, 0), width=2)
                draw.text((region.bbox[0] + 2, region.bbox[1] + 2), f"{region.region_id}:{region.severity}", fill=(255, 255, 0))
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path)
        return path

    def _recovered_bits(self, remesg: Any) -> List[int]:
        arr = remesg.detach().float().cpu().numpy().reshape(-1)
        return [int(v > 0) for v in arr]

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

    def _payload_auth_hash(self, bits: List[int]) -> str:
        return hashlib.sha256(self._bits_to_string(bits).encode("utf-8")).hexdigest()[:16]

    def _checksum_bits(self, payload_bits: List[int], length: int = 8) -> List[int]:
        digest = hashlib.sha256(self._bits_to_string(payload_bits).encode("utf-8")).digest()
        out: List[int] = []
        for byte in digest:
            out.extend([(byte >> shift) & 1 for shift in range(7, -1, -1)])
            if len(out) >= length:
                return out[:length]
        return out[:length]

    def _bits_to_string(self, bits: List[int]) -> str:
        return "".join(str(int(bit)) for bit in bits)

    def _bits_to_int(self, bits: List[int]) -> int:
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value

    def _psnr(self, a: np.ndarray, b: np.ndarray) -> Optional[float]:
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        if mse <= 0:
            return None
        return float(20.0 * math.log10(255.0 / math.sqrt(mse)))

    def _center_score(self, cx: float, cy: float, width: int, height: int) -> float:
        max_dist = math.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)
        dist = math.sqrt((cx - width / 2.0) ** 2 + (cy - height / 2.0) ** 2)
        return float(max(0.0, 1.0 - dist / max_dist))

    def _severity_from_region(self, area_ratio: float, center_score: float) -> str:
        score = min(1.0, 8.0 * area_ratio + 0.35 * center_score)
        if score >= 0.75:
            return "critical"
        if score >= 0.50:
            return "high"
        if score >= 0.25:
            return "medium"
        return "low"

    def _change_type_from_region(self, area_ratio: float, contrast: float, brightness: float, center_score: float) -> str:
        if brightness < 0.18 and area_ratio > 0.002:
            return "removal_or_occlusion"
        if contrast > 0.22:
            return "texture_change"
        if area_ratio > 0.03 or center_score > 0.75:
            return "risk_region_change"
        if brightness > 0.70:
            return "color_change"
        return "unknown"

    def _attack_regime(self, pred_area_ratio: float) -> str:
        if pred_area_ratio > 0.80:
            return "global_rerendering_or_watermark_laundering"
        if pred_area_ratio > 0.0:
            return "localized_tamper_candidate"
        return "no_tamper_detected"

    def _job_seed(self, job_id: str) -> int:
        return int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16)

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
            "original_image_api_input": False,
            "gt_mask_api_input": False,
            "semantic_report_type": "coarse_rule_based",
            "robust_payload_bits": 64,
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _is_safe_job_id(self, job_id: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", job_id or ""))

    def copy_artifact(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
