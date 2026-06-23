# Stage 10B Baseline Feasibility Audit

## Scope

This document prepares Stage 10B baseline comparison without running full baselines, downloading large checkpoints, training models, enabling VLMs, expanding the 64-bit payload, or modifying watermark embedding.

## Current Repository Baseline Availability

### EditGuard Components

The current SCE-LocGuard / DFG-LocGuard repository is already built on an EditGuard-style pipeline:

- `code/models/modules/Inv_arch.py`
- `code/models/IBSN.py`
- `code/train.py`
- `code/options/test_editguard.yml`
- `dfg_locguard/scripts/generate_editguard_outputs_stage1.py`
- `dfg_locguard/scripts/evaluate_editguard_pred_masks_stage2.py`
- `dfg_locguard/scripts/run_stage8c_robust_payload_semantic_capsule_audit.py`
- `dfg_locguard/scripts/run_stage8d_end_to_end_dual_branch_report.py`

The local/server checkpoint path used by experiments is:

- `checkpoints/clean.pth`

The checkpoint is excluded from GitHub and should remain external.

## Pure EditGuard-Style Baseline Feasibility

A pure EditGuard-style baseline is feasible from the current repo without new training:

| Output | Feasible | Notes |
|---|---|---|
| localization IoU/Dice | yes | Use predicted fragile localization mask against GT/edit mask. |
| precision/recall | yes | Already computed by Stage 2 / Stage 8D style metrics. |
| bit accuracy / copyright recovery | yes | Use robust copyright recovery from the original EditGuard branch. |
| semantic capsule | no | Pure EditGuard does not carry the SCE-LocGuard semantic capsule. |
| Stage8E/8F semantic report | no | Pure EditGuard should report masks/copyright only, not semantic audit reports. |
| audit-level aggregation | no | This is a SCE-LocGuard report-layer addition. |

Recommended implementation:

- Reuse Stage 10 localized-composite and unconstrained attack inputs.
- Evaluate only EditGuard mask and copyright recovery.
- Disable semantic capsule claims and Stage8E/8F report readout for the pure baseline.

## Metrics Extractable Without Rerunning Large Models

Several comparison rows can already be assembled from existing outputs:

| Source Output | Extractable Fields | Notes |
|---|---|---|
| Stage9C synthetic 1000 | IoU, Dice, precision, recall, bit accuracy, payload recovery, auth success, F3 unknown, reports/image, GT coverage | This is the controlled synthetic local tamper setting. |
| Stage10A unconstrained diffusion 200 | IoU, Dice, precision, recall, bit accuracy, payload recovery, auth success, predicted mask area ratio, failure mode | This should be reported as global diffusion re-rendering / watermark laundering stress test. |
| Stage10A localized-composite diagnostic 40 | IoU, Dice, payload/auth recovery, F3 unknown, reports/image | Diagnostic proof that preserving the unmasked watermarked region restores the pipeline. |
| Stage10A localized-composite full 200 | IoU, Dice, payload/auth recovery, F3 metrics, by-attack-type breakdown | Fair local AIGC edit evaluation at medium scale. |
| Stage10A localized-composite scale1000 | IoU, Dice, payload/auth recovery, F3 metrics, by-attack-type breakdown, unknown diagnosis | Fair local AIGC edit evaluation at full scale. |

These rows should be used to prepare the SCE-LocGuard side of Stage 10B before running any external baseline.

## OmniGuard Feasibility

OmniGuard is a strong related method / possible baseline because it explicitly targets robust copyright protection and tamper localization under AIGC editing and degradation.

Public resources currently identified:

- Paper: `OmniGuard: Hybrid Manipulation Localization via Augmented Versatile Deep Image Watermarking`, CVPR 2025 / arXiv.
- Repository: `https://github.com/xuanyuzhang21/OmniGuard`
- README status: code is being open sourced step by step; complete training and testing process will be made public later.
- Checkpoint download entries are listed in the README:
  - PKU Disk checkpoint zip
  - Google Drive checkpoint zip
- Light repo inspection status:
  - `git ls-remote https://github.com/xuanyuzhang21/OmniGuard.git HEAD` succeeds.
  - Shallow clone exposes `README.md`, `app.py`, `demo.py`, `test.py`, model files, `environment.yml`, and `omniguard_cop/requirements.txt`.
  - `environment.yml` is an older Python/PyTorch-style environment with CUDA 10 era dependencies.
  - `omniguard_cop/requirements.txt` lists `torch`, `torchvision`, `timm`, `kornia`, `tqdm`, `focal_frequency_loss`, `accelerate`, and `diffusers`.
  - No explicit `LICENSE` file was visible in the shallow inspection, so paper/baseline use requires manual license clarification.

Feasibility risks before running:

| Item | Status |
|---|---|
| repo URL | available |
| checkpoint availability | appears available, but size should be checked before download |
| checkpoint size | unknown until confirmed; do not download without user confirmation |
| dependencies | visible via `environment.yml` and `omniguard_cop/requirements.txt`; likely needs compatibility adjustment on the RTX 4090 server |
| expected disk | unknown; reserve several GB for checkpoint and generated outputs |
| expected runtime | unknown; likely manageable for inference on RTX 4090 after setup |
| license | no explicit license observed in shallow inspection; must clarify before paper use |
| reproducibility | partial risk because README says complete training/testing process is still being released |
| training required | not required if provided checkpoints are used for inference reproduction |
| inference-only reproduction | plausible, but should be tested only after user confirms checkpoint download |

Do not download OmniGuard checkpoint until the user confirms. First Stage 10B action should be a lightweight repository inspection and environment estimate.

## Dual-Guard Feasibility

Dual-Guard is best treated as threat-model inspiration unless runnable code/checkpoints become available. Its relevance is conceptual:

- global provenance anchor
- local content anchor
- separate evaluation for regeneration, reprompting, diffusion editing, and local tampering
- current status: no official runnable repository/checkpoint has been confirmed in this project state
- recommended use: cite as conceptual related method and threat-model/evaluation taxonomy reference, not as a numeric baseline unless official resources become available

If no official code/checkpoint is available, Stage 10B should include a conceptual comparison row rather than a numeric baseline.

## Attack Settings for Baseline Comparison

Stage 10B should compare methods under three attack settings:

| Attack Setting | Purpose |
|---|---|
| synthetic local tamper | Original Stage 9C setting; controlled local tampering. |
| localized-composited AIGC local edit | Fair local AIGC edit; unmasked watermarked region is preserved. |
| unconstrained diffusion re-rendering stress test | Global reconstruction / watermark laundering stress test. |

## Baseline Table Fields

| Field | Meaning |
|---|---|
| method | SCE-LocGuard, pure EditGuard, OmniGuard, Dual-Guard conceptual, etc. |
| attack_setting | synthetic local tamper / localized-composited AIGC / unconstrained diffusion. |
| localization IoU | pixel-level IoU. |
| Dice | pixel-level Dice/F1. |
| precision | predicted tamper precision. |
| recall | GT tamper recall. |
| bit accuracy | recovered bit accuracy. |
| payload/copyright recovery | exact payload/copyright success. |
| auth success | authentication success if supported. |
| semantic capsule available | whether compact semantic capsule is supported. |
| change type report available | whether method outputs change type reports. |
| severity report available | whether severity is reported. |
| audit-level aggregation available | whether dense region outputs are aggregated into compact audit reports. |
| failure mode under unconstrained diffusion | qualitative robustness failure mode. |

## Recommended Stage 10B Order

1. Freeze Stage 10A localized-composite full 200 results.
2. Implement pure EditGuard-style baseline extraction from existing pipeline outputs.
3. Inspect OmniGuard repository and license; estimate checkpoint size, disk, dependencies, and runtime.
4. Ask for confirmation before downloading OmniGuard checkpoint.
5. Add Dual-Guard as conceptual comparison unless official runnable resources are available.
6. Run numeric baselines only after the feasibility table is approved.

## Current Recommendation

Proceed with pure EditGuard-style baseline extraction first. Do not download OmniGuard checkpoint until the user confirms. Treat unconstrained diffusion full 200 as a watermark-laundering stress test, not as the fair local AIGC edit benchmark.
