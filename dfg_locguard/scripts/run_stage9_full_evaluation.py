from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 9 full evaluation wrapper. This script runs the existing "
            "EditGuard/SCE-LocGuard Stage 1 -> 8C -> 8D -> 8E -> 8F pipeline "
            "inside a dedicated Stage 9 output directory."
        )
    )
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage9_full_evaluation")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--merge_distance", type=float, default=48.0)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_large_run",
        action="store_true",
        help="Required for max_samples > 50 so 200/1000-sample runs cannot start accidentally.",
    )
    return parser.parse_args()


def resolve_under_root(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def run_command(
    name: str,
    command: list[str],
    project_root: Path,
    log_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    started = datetime.now().isoformat(timespec="seconds")
    print(f"[stage9] starting {name}")
    print("[stage9] command:", " ".join(command))
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            check=False,
        )
    ended = datetime.now().isoformat(timespec="seconds")
    record = {
        "stage": name,
        "command": command,
        "returncode": proc.returncode,
        "started_at": started,
        "ended_at": ended,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    print(f"[stage9] finished {name} returncode={proc.returncode}")
    if proc.returncode != 0:
        print(f"[stage9] {name} failed; see {stderr_path}")
    return record


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def write_overview(output_dir: Path, payload: dict[str, Any]) -> None:
    stage8f = payload.get("stage8f_summary", {})
    best = stage8f.get("best_strategy") or stage8f.get("best_strategy_name") or "unknown"
    lines = [
        "# Stage 9 Full Evaluation",
        "",
        "This run reuses the existing SCE-LocGuard / DFG-LocGuard pipeline:",
        "",
        "1. Stage 1 EditGuard-only watermarked/tampered/pred-mask generation",
        "2. Stage 8C robust 64-bit semantic capsule payload audit",
        "3. Stage 8D end-to-end dual-branch semantic change report",
        "4. Stage 8E report quality audit and unknown-type reduction",
        "5. Stage 8F report-level region aggregation and fragment filtering",
        "",
        "No training, Stable Diffusion, SAM, CLIP, DINO, OmniGuard, third watermark channel, "
        "128-bit payload, or checkpoint overwrite is performed by this wrapper.",
        "",
        "## Run",
        "",
        f"- project_root: `{payload.get('project_root')}`",
        f"- output_dir: `{payload.get('output_dir')}`",
        f"- split: `{payload.get('split')}`",
        f"- start: `{payload.get('start')}`",
        f"- max_samples: `{payload.get('max_samples')}`",
        f"- completed: `{payload.get('completed')}`",
        f"- failed_stage: `{payload.get('failed_stage')}`",
        "",
        "## Stage 8F",
        "",
        f"- recommended/best strategy: `{best}`",
        f"- summary file: `{output_dir / 'stage8f' / 'stage8f_summary.json'}`",
        "",
        "For 200/1000-sample runs, launch this wrapper only after explicit confirmation and pass "
        "`--allow_large_run`.",
    ]
    (output_dir / "stage9_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_samples > 50 and not args.allow_large_run:
        raise SystemExit(
            "Refusing max_samples > 50 without --allow_large_run. "
            "This prevents accidental 200/1000-sample Stage 9 runs."
        )

    project_root = Path(args.project_root).resolve()
    output_dir = resolve_under_root(project_root, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script_dir = project_root / "dfg_locguard" / "scripts"
    python = args.python_executable
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    stage1_dir = output_dir / "stage1_editguard_generated"
    stage8c_dir = output_dir / "stage8c"
    stage8d_dir = output_dir / "stage8d"
    stage8e_dir = output_dir / "stage8e"
    stage8f_dir = output_dir / "stage8f"

    commands: list[tuple[str, list[str]]] = [
        (
            "stage1_editguard_generation",
            [
                python,
                str(script_dir / "generate_editguard_outputs_stage1.py"),
                "--project_root",
                str(project_root),
                "--split",
                args.split,
                "--start",
                str(args.start),
                "--max_samples",
                str(args.max_samples),
                "--out_dir",
                str(stage1_dir),
                "--ckpt",
                args.ckpt,
                "--opt",
                args.opt,
            ],
        ),
        (
            "stage8c_payload_capsule_audit",
            [
                python,
                str(script_dir / "run_stage8c_robust_payload_semantic_capsule_audit.py"),
                "--project_root",
                str(project_root),
                "--input_dir",
                str(stage1_dir),
                "--output_dir",
                str(stage8c_dir),
                "--max_samples",
                str(args.max_samples),
                "--ckpt",
                args.ckpt,
                "--opt",
                args.opt,
                "--mask_threshold",
                str(args.mask_threshold),
                "--contact_sheet_samples",
                str(args.contact_sheet_samples),
            ],
        ),
        (
            "stage8d_dual_branch_report",
            [
                python,
                str(script_dir / "run_stage8d_end_to_end_dual_branch_report.py"),
                "--project_root",
                str(project_root),
                "--stage1_dir",
                str(stage1_dir),
                "--stage8c_dir",
                str(stage8c_dir),
                "--output_dir",
                str(stage8d_dir),
                "--max_samples",
                str(args.max_samples),
                "--mask_threshold",
                str(args.mask_threshold),
                "--contact_sheet_samples",
                str(args.contact_sheet_samples),
            ],
        ),
        (
            "stage8e_report_quality_audit",
            [
                python,
                str(script_dir / "run_stage8e_report_quality_audit.py"),
                "--project_root",
                str(project_root),
                "--stage8d_dir",
                str(stage8d_dir),
                "--stage1_dir",
                str(stage1_dir),
                "--output_dir",
                str(stage8e_dir),
                "--max_samples",
                str(args.max_samples),
                "--contact_sheet_samples",
                str(args.contact_sheet_samples),
            ],
        ),
        (
            "stage8f_report_region_aggregation",
            [
                python,
                str(script_dir / "run_stage8f_report_region_aggregation.py"),
                "--project_root",
                str(project_root),
                "--stage8e_dir",
                str(stage8e_dir),
                "--stage1_dir",
                str(stage1_dir),
                "--output_dir",
                str(stage8f_dir),
                "--max_samples",
                str(args.max_samples),
                "--mask_threshold",
                str(args.mask_threshold),
                "--merge_distance",
                str(args.merge_distance),
                "--contact_sheet_samples",
                str(args.contact_sheet_samples),
            ],
        ),
    ]

    if args.overwrite:
        commands = [(name, command + ["--overwrite"]) for name, command in commands]

    run_records: list[dict[str, Any]] = []
    failed_stage = None
    for name, command in commands:
        record = run_command(name, command, project_root, log_dir, env)
        run_records.append(record)
        if record["returncode"] != 0:
            failed_stage = name
            break

    payload: dict[str, Any] = {
        "stage": "stage9_full_evaluation",
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "split": args.split,
        "start": args.start,
        "max_samples": args.max_samples,
        "completed": failed_stage is None,
        "failed_stage": failed_stage,
        "run_records": run_records,
        "stage1_summary": load_json(stage1_dir / "stage1_generation_summary.json"),
        "stage8c_summary": load_json(stage8c_dir / "stage8c_summary.json"),
        "stage8d_summary": load_json(stage8d_dir / "stage8d_summary.json"),
        "stage8e_summary": load_json(stage8e_dir / "stage8e_quality_summary.json"),
        "stage8f_summary": load_json(stage8f_dir / "stage8f_summary.json"),
        "constraints": {
            "training": False,
            "stable_diffusion": False,
            "sam_clip_dino_omniguard": False,
            "third_watermark_channel": False,
            "robust_payload_bits": 64,
            "requires_allow_large_run_above_50_samples": True,
        },
    }
    (output_dir / "stage9_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if failed_stage is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
