# SCE-LocGuard / DFG-LocGuard Stage 9 Preparation

## Project

SCE-LocGuard is a localized semantic tamper reporting pipeline built on an EditGuard-style dual-watermark framework.

It uses:

- a fragile localization branch for tamper mask recovery
- a robust copyright branch for copyright ID plus a compact semantic capsule/authentication payload
- verification-side semantic report generation without accessing the original image

## Current Status

Frozen local version: Stage 8F.

Completed:

- Stage 8E: semantic report quality audit and unknown-type reduction
- Stage 8F: report-level region aggregation and fragment filtering

## Stage 8F Key Result

Recommended main strategy: `F3_merge_nearby_fragments`.

- raw reports: 983
- aggregated reports with F3: 290
- reports/image with F3: 5.8
- unknown rate: 0.23398 -> 0.14483
- predicted area retained: 1.0
- GT coverage retained: 0.96478
- no-report images: 0

Compact audit-level ablations:

- `F4_main_region_top3`
- `F4_main_region_top5`

These are useful when a stricter 3-5 reports/image target is preferred.

## Repository Exclusions

This GitHub repository does not include:

- datasets
- generated full outputs
- checkpoints or model weights
- `.pth`, `.pt`, `.ckpt`, `.safetensors`
- virtual environments
- cache files
- API keys or `.env`

Only lightweight summaries and dataset manifests are tracked.

## Stage 9 Server Migration Plan

Server work is intentionally not performed yet.

Planned later steps:

1. Buy or provision a 4090 server.
2. Clone this repository on the server.
3. Upload datasets separately with `scp`, `rsync`, or a private archive.
4. Upload checkpoints separately, outside git.
5. Create a `stage9-server` branch on the server.
6. Run 200/1000-sample full evaluation from parameterized paths.

Example future server commands:

```bash
git clone https://github.com/guoshaung/sce-locguard.git
cd sce-locguard
git checkout -b stage9-server
```

Dataset and checkpoint paths should be passed through command-line arguments or config files. Do not hard-code local Windows paths such as `D:/pycharm/watermark_exps`.

