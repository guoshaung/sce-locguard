"""Shared request/response schemas for the SCE-LocGuard service layer."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Path-based request for registering or embedding a host image."""

    image_path: str = Field(..., description="Path to the host image.")
    owner_id: str = Field(..., description="Owner or copyright identifier.")
    semantic_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Compact semantic metadata to be encoded into the 64-bit capsule.",
    )
    output_dir: Optional[str] = Field(default=None, description="Directory for API job artifacts.")
    strict: bool = Field(default=True, description="Return explicit failure instead of fallback behavior.")


class EmbedResponse(BaseModel):
    """Response returned by the embed/register endpoint."""

    status: str
    job_id: Optional[str] = None
    implementation_level: Literal["real_pipeline", "one_image_batch_wrapper", "skeleton"] = "skeleton"
    watermarked_image_path: Optional[str] = None
    payload_id: Optional[str] = None
    capsule_bits: Optional[str] = None
    auth_hash: Optional[str] = None
    psnr: Optional[float] = None
    message: str


class AIGCAttackRequest(BaseModel):
    """Request for a local AIGC editing attack."""

    image_path: str = Field(..., description="Path to a watermarked image.")
    attack_type: Literal["object_removal", "inpainting", "local_replacement", "local_style_edit"]
    mask_path: Optional[str] = Field(default=None, description="Optional local edit mask.")
    prompt: Optional[str] = Field(default=None, description="Optional inpainting prompt.")
    output_dir: Optional[str] = Field(default=None, description="Directory for API job artifacts.")
    mode: Literal["unconstrained", "localized_composite"] = "localized_composite"
    strict: bool = Field(default=True, description="Return explicit failure if the model/cache is unavailable.")


class AIGCAttackResponse(BaseModel):
    """Response returned by the AIGC attack endpoint."""

    status: str
    job_id: Optional[str] = None
    implementation_level: Literal["real_pipeline", "one_image_batch_wrapper", "skeleton"] = "skeleton"
    attack_type: Optional[str] = None
    mode: Optional[str] = None
    source_image_path: Optional[str] = None
    attacked_image_path: Optional[str] = None
    diffusion_output_path: Optional[str] = None
    mask_path: Optional[str] = None
    outside_mask_preserved: Optional[bool] = None
    message: str


class VerifyRequest(BaseModel):
    """Path-based blind verification request."""

    image_path: str = Field(..., description="Path to the possibly tampered image.")
    output_dir: Optional[str] = Field(default=None, description="Directory for API job artifacts.")
    strict: bool = Field(default=True, description="Return explicit failure instead of fallback behavior.")
    return_overlay: bool = Field(default=True, description="Save a mask overlay visualization when possible.")


class ReportRegion(BaseModel):
    """One localized region in a verification report."""

    region_id: int
    bbox: List[int] = Field(
        default_factory=list,
        description="Bounding box as [x1, y1, x2, y2].",
    )
    area_ratio: float = 0.0
    change_type: str = "unknown"
    severity: str = "unknown"
    auth_status: str = "not_evaluated"
    confidence: float = 0.0


class VerifyResponse(BaseModel):
    """Response returned by the blind verification endpoint."""

    status: str
    job_id: Optional[str] = None
    implementation_level: Literal["real_pipeline", "one_image_batch_wrapper", "skeleton"] = "skeleton"
    auth_status: str = "not_evaluated"
    payload_recovered: bool = False
    capsule_recovered: bool = False
    bit_accuracy: Optional[float] = None
    predicted_mask_path: Optional[str] = None
    overlay_path: Optional[str] = None
    report_json_path: Optional[str] = None
    attack_regime: str = "unknown"
    reports: List[ReportRegion] = Field(default_factory=list)
    message: str


class CapabilityResponse(BaseModel):
    """Runtime capability probe for callers."""

    embed_real_pipeline_available: bool
    attack_real_pipeline_available: bool
    verify_real_pipeline_available: bool
    supports_64bit_capsule: bool = True
    supports_vlm: bool = False
    supports_single_image: bool
    notes: List[str] = Field(default_factory=list)
