"""FastAPI server for the SCE-LocGuard service wrapper."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .schemas import EmbedRequest, EmbedResponse, VerifyRequest, VerifyResponse
from .service import SCELocGuardService


def _project_root() -> str:
    return os.environ.get("SCE_LOCGUARD_PROJECT_ROOT") or str(Path.cwd())


def get_service() -> SCELocGuardService:
    return SCELocGuardService(project_root=_project_root())


app = FastAPI(
    title="SCE-LocGuard API",
    version="0.1.0",
    description=(
        "Lightweight wrapper around the SCE-LocGuard / EditGuard-style "
        "watermark forensic pipeline."
    ),
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "sce-locguard",
        "algorithm_mode": "wrapper",
        "project_root_configured": bool(_project_root()),
        "single_image_pipeline": "not_implemented_single_image_pipeline",
        "constraints": {
            "no_training": True,
            "no_vlm": True,
            "robust_payload_bits": 64,
            "verification_mode": "blind",
        },
    }


@app.post("/api/v1/watermark/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    response = get_service().embed(request)
    if response.status == "error":
        raise HTTPException(status_code=400, detail={"status": response.status, "message": response.message})
    return response


@app.post("/api/v1/watermark/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest) -> VerifyResponse:
    response = get_service().verify(request)
    if response.status == "error":
        raise HTTPException(status_code=400, detail={"status": response.status, "message": response.message})
    return response


@app.get("/api/v1/watermark/report/{job_id}")
def export_report(job_id: str) -> dict:
    response = get_service().export_report(job_id)
    if response.get("status") == "error":
        raise HTTPException(status_code=404, detail=response)
    return response
