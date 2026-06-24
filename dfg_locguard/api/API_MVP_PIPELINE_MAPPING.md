# API MVP Pipeline Mapping

This document maps the path-based API wrapper to the existing SCE-LocGuard /
DFG-LocGuard research pipeline. The wrapper does not train models, enable VLMs,
expand the 64-bit payload, or change watermark embedding.

| API endpoint | Underlying script/function | Level | Required inputs | Produced outputs | Known limitation |
|---|---|---|---|---|---|
| `GET /health` | `dfg_locguard.api.server.health` | real | none | service status JSON | Does not probe model availability; use capabilities for that. |
| `GET /api/v1/capabilities` | `SCELocGuardService.capabilities()` | real | project root | runtime capability JSON | Checks local checkpoint/CUDA/cache only; it does not load large models. |
| `POST /api/v1/watermark/embed` | `generate_editguard_outputs_stage1.image_to_editguard_batch`, `stage8c.build_payload_bits`, `Model_VSN.image_hiding()` | real pipeline if `checkpoints/clean.pth`, `test_editguard.yml`, PyTorch, and CUDA are available | host image path, owner id, semantic metadata, output dir | `watermarked.png`, `payload.json`, `embed_report.json` | Payload packing reuses the existing Stage8C P2 64-bit compact capsule. If runtime prerequisites are missing, returns `not_implemented`. |
| `POST /api/v1/attack/aigc` | `run_stage10a_real_aigc_attack_subset.load_diffusers_pipe`, `generate_with_diffusers`; localized composite implemented in service | real pipeline if cached `stable-diffusion-v1-5/stable-diffusion-inpainting` and CUDA are available | watermarked image, attack type, optional mask/prompt, mode | `attack_mask.png`, `diffusion_output.png`, `attacked.png`, `attack_report.json` | Does not download model. If cache is unavailable, returns `model_unavailable`. |
| `POST /api/v1/watermark/verify` | `generate_editguard_outputs_stage1.image_to_editguard_batch`, `Model_VSN.image_recovery()` | real pipeline if checkpoint/config/CUDA are available | suspect image path only | `predicted_mask.png`, optional `overlay.png`, `verify_report.json` | Blind verification uses the fixed localization template and does not access original image or GT mask. `bit_accuracy` is unavailable in blind mode because ground-truth payload bits are not API input. |
| `GET /api/v1/watermark/report/{job_id}` | `SCELocGuardService.export_report()` | real | job id | stored job report JSON | Only returns reports from API jobs known to the local job index. |

## Reused Research Components

- Single-image watermark embedding: `code/models/IBSN.py::Model_VSN.image_hiding()`
- Blind mask and robust payload recovery: `code/models/IBSN.py::Model_VSN.image_recovery()`
- 64-bit semantic capsule packing: `dfg_locguard/scripts/run_stage8c_robust_payload_semantic_capsule_audit.py`
- Input tensor conversion and EditGuard loading: `dfg_locguard/scripts/generate_editguard_outputs_stage1.py`
- AIGC inpainting attack generation: `dfg_locguard/scripts/run_stage10a_real_aigc_attack_subset.py`
- Localized-composite construction: `attacked = mask * diffusion_output + (1 - mask) * watermarked_source`

## Not Yet Full Stage8E/F

The API MVP verification report is a lightweight single-image forensic report:

- predicted mask from the fragile localization branch
- recovered 64-bit payload bits
- checksum/auth validation
- connected-component region summaries
- coarse rule-based change type / severity

It does not run the full Stage8E report-quality audit or Stage8F multi-strategy
aggregation inside the request path. Those remain batch research scripts.
