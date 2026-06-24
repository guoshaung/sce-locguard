# SCE-LocGuard API MVP

This directory exposes a lightweight API / SDK wrapper around the current
SCE-LocGuard / DFG-LocGuard research pipeline. It does not change the
watermarking algorithm, model weights, training code, or existing experiment
outputs.

## Current API Status

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | real | Liveness check only. |
| `GET /api/v1/capabilities` | real | Checks checkpoint/config/CUDA and cached inpainting model availability. |
| `POST /api/v1/watermark/embed` | real if runtime is available | Reuses `Model_VSN.image_hiding()` and Stage8C P2 64-bit capsule packing. |
| `POST /api/v1/attack/aigc` | real if model cache is available | Reuses cached Stable Diffusion inpainting and supports localized-composite mode. |
| `POST /api/v1/watermark/verify` | real if runtime is available | Reuses `Model_VSN.image_recovery()` in blind mode, without original image or GT mask. |
| `GET /api/v1/watermark/report/{job_id}` | real | Exports reports created by API jobs. |

If a runtime prerequisite is missing, the API returns `not_implemented` or
`model_unavailable`. It never fabricates masks, payload status, or forensic
reports.

## Invariants

- No training.
- No VLM.
- No 128-bit payload.
- No watermark embedding modification.
- Verification is blind and does not access the original image.
- GT masks are only for evaluation and are not API inputs.
- The robust payload remains a 64-bit semantic capsule / copyright/auth payload.
- The current semantic report is coarse-grained and rule-based.

## Start The Server

From the project root:

```bash
uvicorn dfg_locguard.api.server:app --host 0.0.0.0 --port 8000
```

Optionally set an explicit project root:

```bash
export SCE_LOCGUARD_PROJECT_ROOT=/data/watermark_exps
uvicorn dfg_locguard.api.server:app --host 0.0.0.0 --port 8000
```

## Check Capabilities First

```bash
curl http://127.0.0.1:8000/api/v1/capabilities
```

Example response:

```json
{
  "embed_real_pipeline_available": true,
  "attack_real_pipeline_available": true,
  "verify_real_pipeline_available": true,
  "supports_64bit_capsule": true,
  "supports_vlm": false,
  "supports_single_image": true,
  "notes": ["EditGuard checkpoint/config/CUDA are available."]
}
```

The main project should check this endpoint before running the full flow.

## Main Project Flow

```text
AI generation -> embed watermark -> AIGC attack -> blind verify -> report
```

1. The main project generates an image with its own AI module.
2. Call `POST /api/v1/watermark/embed`.
3. Call `POST /api/v1/attack/aigc` on the returned watermarked image.
4. Call `POST /api/v1/watermark/verify` on the returned attacked image.
5. Call `GET /api/v1/watermark/report/{job_id}` to export the report.

## JSON Requests

### Embed / Register Image

```bash
curl -X POST http://127.0.0.1:8000/api/v1/watermark/embed \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/ai_generated.png",
    "owner_id": "test_owner",
    "semantic_metadata": {"scene": "document", "risk": "medium"},
    "output_dir": "/tmp/sce_api_jobs",
    "strict": true
  }'
```

Expected fields:

```json
{
  "status": "ok",
  "job_id": "embed_...",
  "implementation_level": "real_pipeline",
  "watermarked_image_path": "/tmp/sce_api_jobs/embed_.../embed/watermarked.png",
  "payload_id": "...",
  "capsule_bits": "010101...",
  "auth_hash": "...",
  "psnr": 36.7,
  "message": "Watermarked image generated with existing EditGuard image_hiding() pipeline."
}
```

If runtime prerequisites are missing, `status` is `not_implemented` and
`implementation_level` is `skeleton`.

### AIGC Attack

```bash
curl -X POST http://127.0.0.1:8000/api/v1/attack/aigc \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/tmp/sce_api_jobs/embed_.../embed/watermarked.png",
    "attack_type": "inpainting",
    "mode": "localized_composite",
    "output_dir": "/tmp/sce_api_jobs",
    "strict": true
  }'
```

Expected fields:

```json
{
  "status": "ok",
  "job_id": "attack_...",
  "implementation_level": "real_pipeline",
  "attack_type": "inpainting",
  "mode": "localized_composite",
  "source_image_path": ".../source.png",
  "attacked_image_path": ".../attacked.png",
  "diffusion_output_path": ".../diffusion_output.png",
  "mask_path": ".../attack_mask.png",
  "outside_mask_preserved": true,
  "message": "AIGC attack generated with cached diffusers inpainting model..."
}
```

If the cached model is missing, `status` is `model_unavailable`.

### Blind Verify

```bash
curl -X POST http://127.0.0.1:8000/api/v1/watermark/verify \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/tmp/sce_api_jobs/attack_.../attack/attacked.png",
    "output_dir": "/tmp/sce_api_jobs",
    "return_overlay": true,
    "strict": true
  }'
```

Expected fields:

```json
{
  "status": "ok",
  "job_id": "verify_...",
  "implementation_level": "real_pipeline",
  "auth_status": "valid",
  "payload_recovered": true,
  "capsule_recovered": true,
  "bit_accuracy": null,
  "predicted_mask_path": ".../predicted_mask.png",
  "overlay_path": ".../overlay.png",
  "report_json_path": ".../verify_report.json",
  "attack_regime": "localized_tamper_candidate",
  "reports": [
    {
      "region_id": 1,
      "bbox": [120, 140, 260, 300],
      "area_ratio": 0.05,
      "change_type": "texture_change",
      "severity": "high",
      "auth_status": "valid",
      "confidence": 0.8
    }
  ],
  "message": "Blind verification completed with existing EditGuard image_recovery()."
}
```

`bit_accuracy` is `null` in blind mode because the API does not receive the
ground-truth payload bits.

## CLI Commands

```bash
python -m dfg_locguard.api.cli capabilities
```

```bash
python -m dfg_locguard.api.cli embed \
  --image path/to/input.png \
  --owner_id test_owner \
  --out output_dir
```

```bash
python -m dfg_locguard.api.cli attack \
  --image path/to/watermarked.png \
  --attack_type inpainting \
  --mode localized_composite \
  --out output_dir
```

```bash
python -m dfg_locguard.api.cli verify \
  --image path/to/attacked.png \
  --out output_dir
```

One-shot demo:

```bash
python -m dfg_locguard.api.cli demo \
  --image path/to/input.png \
  --owner_id test_owner \
  --attack_type inpainting \
  --mode localized_composite \
  --out output_dir
```

The demo saves:

- `demo_embed_response.json`
- `demo_attack_response.json`
- `demo_verify_response.json`
- `demo_summary.json`

If a real stage is unavailable, demo stops explicitly with
`not_implemented` or `model_unavailable`.

## Python Integration

```python
from dfg_locguard.api import SCELocGuardService, EmbedRequest, AIGCAttackRequest, VerifyRequest

service = SCELocGuardService(project_root="/data/watermark_exps")
caps = service.capabilities()

embed = service.embed(EmbedRequest(
    image_path="/path/to/ai_generated.png",
    owner_id="test_owner",
    semantic_metadata={"risk": "medium"},
    output_dir="/tmp/sce_api_jobs",
))

attack = service.attack_aigc(AIGCAttackRequest(
    image_path=embed.watermarked_image_path,
    attack_type="inpainting",
    mode="localized_composite",
    output_dir="/tmp/sce_api_jobs",
))

verify = service.verify(VerifyRequest(
    image_path=attack.attacked_image_path,
    output_dir="/tmp/sce_api_jobs",
))
```

## Mapping

See `API_MVP_PIPELINE_MAPPING.md` for the exact mapping from API endpoints to
the underlying research scripts/functions.
