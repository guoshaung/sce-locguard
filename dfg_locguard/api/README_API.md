# SCE-LocGuard API Wrapper

This directory exposes a lightweight API / SDK wrapper around the current
SCE-LocGuard / DFG-LocGuard research pipeline. It does not change the
watermarking algorithm, model weights, training code, or existing experiment
outputs.

## Current Scope

- The API is a wrapper around the existing research scripts.
- Verification is blind and does not access the original image.
- GT masks are only for evaluation and are not API inputs.
- The robust payload remains a 64-bit semantic capsule / authentication payload.
- The current semantic report is coarse-grained and rule-based.
- No VLM is used.
- No 128-bit payload is enabled.
- No watermark embedding logic is modified.

The first version is intentionally conservative. It validates paths, creates
job records, exposes stable request/response schemas, and returns
`not_implemented_single_image_pipeline` when a true single-image embed/verify
pipeline is not yet connected. It does not pretend that incomplete production
functionality is already implemented.

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

## Endpoints

### Health

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "service": "sce-locguard",
  "algorithm_mode": "wrapper",
  "single_image_pipeline": "not_implemented_single_image_pipeline"
}
```

### Embed / Register Image

```bash
curl -X POST http://127.0.0.1:8000/api/v1/watermark/embed \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.png",
    "owner_id": "test_owner",
    "semantic_metadata": {"scene": "document", "risk": "medium"},
    "output_dir": "/tmp/sce_api_jobs"
  }'
```

Example response:

```json
{
  "status": "not_implemented_single_image_pipeline",
  "job_id": "embed_20260624T000000Z_abcd1234",
  "watermarked_image_path": null,
  "payload_id": "b82e6fd...",
  "capsule_bits": "010101...",
  "auth_hash": "a13c...",
  "message": "Input image registered and API job created, but the current research code does not yet expose a real single-image embed pipeline."
}
```

### Verify Tampered Image

```bash
curl -X POST http://127.0.0.1:8000/api/v1/watermark/verify \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/tampered.png",
    "mode": "blind",
    "output_dir": "/tmp/sce_api_jobs"
  }'
```

Example response:

```json
{
  "status": "not_implemented_single_image_pipeline",
  "job_id": "verify_20260624T000000Z_abcd1234",
  "auth_status": "not_evaluated",
  "payload_recovered": false,
  "capsule_recovered": false,
  "predicted_mask_path": null,
  "overlay_path": null,
  "report_json_path": "/tmp/sce_api_jobs/verify_.../verify_report.json",
  "attack_regime": "unknown",
  "reports": [],
  "message": "Input image validated and blind verification job created, but the current research code does not yet expose a real single-image verification pipeline."
}
```

### Export Report

```bash
curl http://127.0.0.1:8000/api/v1/watermark/report/verify_20260624T000000Z_abcd1234
```

## CLI Usage

```bash
python -m dfg_locguard.api.cli verify \
  --image path/to/tampered.png \
  --out output_dir
```

```bash
python -m dfg_locguard.api.cli embed \
  --image path/to/image.png \
  --owner_id test_owner \
  --metadata_json '{"scene":"document","risk":"medium"}' \
  --out output_dir
```

```bash
python -m dfg_locguard.api.cli report \
  --job_id verify_20260624T000000Z_abcd1234
```

## External Module Integration

External systems should call:

- `POST /api/v1/watermark/embed` to register or embed an image.
- `POST /api/v1/watermark/verify` to run blind verification on a suspect image.
- `GET /api/v1/watermark/report/{job_id}` to fetch the generated forensic report.

For Python integration, use:

```python
from dfg_locguard.api import SCELocGuardService, VerifyRequest

service = SCELocGuardService(project_root="/data/watermark_exps")
response = service.verify(
    VerifyRequest(
        image_path="/path/to/tampered.png",
        mode="blind",
        output_dir="/tmp/sce_api_jobs",
    )
)
```

## Implementation Notes

The current research pipeline has strong batch evaluation scripts for Stage9C,
Stage10A, and Stage10B. A production single-image embed/verify path still needs
to be connected to the wrapper. Until that is done, the wrapper is safe for API
contract integration and orchestration tests, but not a claim of complete
single-image forensic inference.
