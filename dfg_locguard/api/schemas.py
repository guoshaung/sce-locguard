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
    output_dir: str = Field(..., description="Directory for API job artifacts.")


class EmbedResponse(BaseModel):
    """Response returned by the embed/register endpoint."""

    status: str
    job_id: Optional[str] = None
    watermarked_image_path: Optional[str] = None
    payload_id: Optional[str] = None
    capsule_bits: Optional[str] = None
    auth_hash: Optional[str] = None
    message: str


class VerifyRequest(BaseModel):
    """Path-based blind verification request."""

    image_path: str = Field(..., description="Path to the possibly tampered image.")
    mode: Literal["blind"] = Field(
        default="blind",
        description="Current API supports blind verification only.",
    )
    output_dir: str = Field(..., description="Directory for API job artifacts.")


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
    auth_status: str = "not_evaluated"
    payload_recovered: bool = False
    capsule_recovered: bool = False
    predicted_mask_path: Optional[str] = None
    overlay_path: Optional[str] = None
    report_json_path: Optional[str] = None
    attack_regime: str = "unknown"
    reports: List[ReportRegion] = Field(default_factory=list)
    message: str
