# Stage 10 Related Work Alignment

## Purpose

This note aligns SCE-LocGuard / DFG-LocGuard with three closely related watermarking lines: EditGuard, OmniGuard, and Dual-Guard. The goal is not to copy their training recipes, but to borrow a clearer threat model and evaluation taxonomy for Stage 10 and later paper experiments.

## Related Work Positions

### EditGuard

EditGuard is the direct technical foundation for the current prototype. It frames proactive image forensics as a dual-watermark problem for copyright protection and tamper localization. In our implementation, the robust copyright branch carries identity / payload information, while the fragile localization branch produces the predicted tamper mask. This makes EditGuard the required baseline and also the base pipeline from which SCE-LocGuard inherits its dual-branch structure.

Reference:
- EditGuard: Versatile Image Watermarking for Tamper Localization and Copyright Protection, CVPR 2024 / arXiv 2023: https://arxiv.org/abs/2312.08883
- Official repository: https://github.com/xuanyuzhang21/EditGuard

### OmniGuard

OmniGuard can be understood as an enhanced successor direction to EditGuard. It keeps the proactive watermarking motivation, but strengthens the system with a hybrid manipulation localization framework. The key ideas relevant to SCE-LocGuard are:

- flexible localization watermark selection
- degradation-aware tamper extraction
- a lightweight AIGC-editing simulation layer for improved robustness across global and local editing conditions

For SCE-LocGuard, OmniGuard is important less as a module to copy and more as evidence that evaluation should distinguish strict local edits from more destructive AIGC editing / degradation settings.

Reference:
- OmniGuard: Hybrid Manipulation Localization via Augmented Versatile Deep Image Watermarking, CVPR 2025 / arXiv: https://arxiv.org/html/2412.01615v3

### Dual-Guard

Dual-Guard is a 2026 dual-channel latent watermarking framework for diffusion images. Its useful conceptual distinction is between:

- a global provenance anchor, intended to survive broader regeneration / reprompting style transformations
- a local content anchor, intended to provide spatial evidence for local tampering or diffusion editing

This taxonomy is highly relevant to SCE-LocGuard even if the implementation space is different. Dual-Guard explicitly evaluates regeneration, reprompting, diffusion editing, and local tampering. That threat taxonomy helps us interpret Stage 10A: unconstrained diffusion inpainting is closer to watermark laundering / global generative reconstruction, while localized compositing is closer to fair local AIGC tamper evaluation.

Reference:
- Dual-Guard: Dual-Channel Latent Watermarking for Provenance and Tamper Localization in Diffusion Images, arXiv 2026: https://arxiv.org/abs/2604.19090

## Design Implication for SCE-LocGuard

SCE-LocGuard should not directly reproduce the training methods from EditGuard, OmniGuard, or Dual-Guard. Instead, it should borrow their threat models and evaluation taxonomy:

- From EditGuard: keep the dual-branch view of copyright recovery plus tamper localization.
- From OmniGuard: separate clean/local editing from degradation-heavy AIGC editing and report robustness under both.
- From Dual-Guard: distinguish global provenance robustness from local content integrity, and evaluate regeneration-style attacks separately from strict local tampering.

This matters because Stage 10A showed that an unconstrained diffusion inpainting pipeline can alter the entire image outside the edit mask, destroying robust payload recovery and triggering full-image fragile alarms. That is a valid stress test, but it is not the same as a strict local edit.

## Stage 10A Taxonomy

Stage 10A should be reported as two separate settings:

| Setting | Meaning | Use in Paper |
|---|---|---|
| unconstrained diffusion re-rendering / watermark laundering stress test | Diffusion inpainting output may alter both masked and unmasked regions. This tests whether the watermark survives global generative reconstruction. | Report as a hard robustness stress test. Do not treat it as fair local tamper localization. |
| localized-composited AIGC local editing evaluation | Use diffusion content only inside the edit mask, then preserve the unmasked region from the watermarked image exactly. | Use as the fair local AIGC tamper evaluation setting. |

The Stage 10A locality audit supports this split:

- unconstrained diffusion 200 used watermarked sources, not original sources
- outside-mask distortion was high
- predicted masks became near-full-image alarms
- localized-composite diagnostic recovered strong localization and payload/auth performance

## Stage 10B Baseline Plan

Stage 10B baseline comparison should include at least:

| Baseline | Role |
|---|---|
| EditGuard | Required base baseline because SCE-LocGuard extends its dual-watermark pipeline. |
| OmniGuard | Include if code/checkpoint/reproduction resources are available. If not fully reproducible, report as partial or qualitative comparison. |
| Dual-Guard | Use as conceptual comparison if code/checkpoint is unavailable, especially for global provenance anchor vs local content anchor taxonomy. |

SCE-LocGuard should be evaluated under both Stage 10A settings:

- unconstrained diffusion re-rendering / watermark laundering stress test
- localized-composited AIGC local editing evaluation

This avoids unfairly penalizing local tamper localization methods under a global reconstruction attack, while still preserving the hard attack as a robustness stress test.

## Future Experiment Table Fields

Recommended fields for Stage 10B and paper tables:

| Field | Meaning |
|---|---|
| IoU | Pixel-level overlap between predicted and GT tamper masks. |
| Dice | Pixel-level F1-style localization score. |
| precision | Fraction of predicted tamper pixels that are correct. |
| recall | Fraction of GT tamper pixels recovered. |
| bit accuracy | Robust branch bit-level recovery accuracy. |
| payload recovery | Exact or structured payload recovery success. |
| auth success | Checksum/authentication validation success. |
| F3 unknown | Unknown change-type rate after F3 report aggregation. |
| reports/image | Average number of final audit reports per image. |
| GT coverage | GT tamper area covered by retained report regions. |
| predicted area retained | Predicted tamper area retained after report filtering/aggregation. |
| semantic capsule availability | Whether semantic capsule fields are available and recovered. |
| audit report availability | Whether the method can produce localized semantic/audit reports, not only masks. |

## Reporting Guidance

For the paper, avoid claiming that Stage 10A unconstrained diffusion is a failure of strict local editing robustness. The correct interpretation is:

- EditGuard-style robust payload and fragile localization are vulnerable to global generative reconstruction.
- When the unmasked watermarked region is preserved, localized-composite AIGC editing remains compatible with the current dual-branch SCE-LocGuard pipeline.
- Therefore, SCE-LocGuard should report both a global watermark-laundering stress test and a fair local AIGC edit evaluation.

No training, VLM, 128-bit payload expansion, watermark embedding modification, dataset upload, model upload, output upload, log upload, or checkpoint upload is required by this alignment note.
