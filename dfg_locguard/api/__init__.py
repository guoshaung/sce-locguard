"""SCE-LocGuard API wrapper package."""

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
from .service import SCELocGuardService

__all__ = [
    "EmbedRequest",
    "EmbedResponse",
    "AIGCAttackRequest",
    "AIGCAttackResponse",
    "CapabilityResponse",
    "ReportRegion",
    "VerifyRequest",
    "VerifyResponse",
    "SCELocGuardService",
]
