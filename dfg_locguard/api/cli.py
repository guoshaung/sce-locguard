"""Command line interface for the SCE-LocGuard API wrapper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from .schemas import EmbedRequest, VerifyRequest
from .service import SCELocGuardService


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _load_metadata(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    possible_path = Path(raw)
    if possible_path.exists() and possible_path.is_file():
        return json.loads(possible_path.read_text(encoding="utf-8"))
    return json.loads(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SCE-LocGuard API wrapper CLI")
    parser.add_argument(
        "--project_root",
        default=os.environ.get("SCE_LOCGUARD_PROJECT_ROOT") or str(Path.cwd()),
        help="Project root containing dfg_locguard/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed", help="Register/embed an image through the wrapper.")
    embed.add_argument("--image", required=True, help="Host image path.")
    embed.add_argument("--owner_id", required=True, help="Owner/copyright identifier.")
    embed.add_argument("--metadata_json", default=None, help="Inline JSON or path to JSON metadata.")
    embed.add_argument("--out", required=True, help="Output directory for job artifacts.")

    verify = sub.add_parser("verify", help="Blindly verify a possibly tampered image.")
    verify.add_argument("--image", required=True, help="Tampered image path.")
    verify.add_argument("--out", required=True, help="Output directory for job artifacts.")
    verify.add_argument("--mode", default="blind", choices=["blind"], help="Verification mode.")

    report = sub.add_parser("report", help="Export a stored API job report.")
    report.add_argument("--job_id", required=True, help="Job id returned by a previous job record.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service = SCELocGuardService(project_root=args.project_root)

    if args.command == "embed":
        response = service.embed(
            EmbedRequest(
                image_path=args.image,
                owner_id=args.owner_id,
                semantic_metadata=_load_metadata(args.metadata_json),
                output_dir=args.out,
            )
        )
        print(json.dumps(_model_to_dict(response), indent=2, ensure_ascii=False))
        return 1 if response.status == "error" else 0

    if args.command == "verify":
        response = service.verify(
            VerifyRequest(image_path=args.image, mode=args.mode, output_dir=args.out)
        )
        print(json.dumps(_model_to_dict(response), indent=2, ensure_ascii=False))
        return 1 if response.status == "error" else 0

    if args.command == "report":
        response = service.export_report(args.job_id)
        print(json.dumps(response, indent=2, ensure_ascii=False))
        return 1 if response.get("status") == "error" else 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
