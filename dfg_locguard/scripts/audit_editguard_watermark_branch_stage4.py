from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


KEY_FILES = [
    "code/models/IBSN.py",
    "code/models/modules/Inv_arch.py",
    "code/models/bitnetwork/Encoder_U.py",
    "code/models/bitnetwork/Decoder_U.py",
    "code/models/modules/loss.py",
    "code/models/networks.py",
    "code/models/__init__.py",
    "code/train.py",
    "code/train_bit.py",
    "code/options/train_editguard_image.yml",
    "code/options/train_editguard_bit.yml",
    "code/options/test_editguard.yml",
]

KEY_PATTERNS = {
    "Model_VSN.optimize_parameters": ("code/models/IBSN.py", r"^\s*def optimize_parameters\("),
    "Model_VSN.test": ("code/models/IBSN.py", r"^\s*def test\("),
    "Model_VSN.image_hiding": ("code/models/IBSN.py", r"^\s*def image_hiding\("),
    "Model_VSN.image_recovery": ("code/models/IBSN.py", r"^\s*def image_recovery\("),
    "Model_VSN.load_test": ("code/models/IBSN.py", r"^\s*def load_test\("),
    "VSN.__init__": ("code/models/modules/Inv_arch.py", r"^\s*def __init__\(self, opt"),
    "VSN.forward": ("code/models/modules/Inv_arch.py", r"^\s*def forward\(self, x, x_h=None"),
    "InvBlock.forward": ("code/models/modules/Inv_arch.py", r"^\s*def forward\(self, x1, x2, rev=False"),
    "InvNN.forward": ("code/models/modules/Inv_arch.py", r"^\s*def forward\(self, x, x_h, rev=False"),
    "DW_Encoder.forward": ("code/models/bitnetwork/Encoder_U.py", r"^\s*def forward\(self, x, watermark\)"),
    "DW_Decoder.forward": ("code/models/bitnetwork/Decoder_U.py", r"^\s*def forward\(self, x\)"),
    "ReconstructionLoss": ("code/models/modules/loss.py", r"^class ReconstructionLoss"),
    "ReconstructionMsgLoss": ("code/models/modules/loss.py", r"^class ReconstructionMsgLoss"),
    "train.main": ("code/train.py", r"^\s*def main\("),
    "train_bit.main": ("code/train_bit.py", r"^\s*def main\("),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Stage 4 audit of EditGuard watermark branches.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage4_watermark_branch_audit")
    return parser.parse_args()


def find_line(path: Path, pattern: str) -> int | None:
    if not path.exists():
        return None
    rx = re.compile(pattern)
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if rx.search(line):
            return idx
    return None


def file_status(project_root: Path) -> dict[str, Any]:
    return {
        rel: {
            "exists": (project_root / rel).exists(),
            "path": str((project_root / rel).resolve()),
        }
        for rel in KEY_FILES
    }


def function_locations(project_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (rel, pattern) in KEY_PATTERNS.items():
        line = find_line(project_root / rel, pattern)
        out[name] = {"file": rel, "line": line, "exists": line is not None}
    return out


def checkpoint_status(project_root: Path) -> dict[str, Any]:
    checkpoint_dir = project_root / "checkpoints"
    files = []
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*")):
            if path.is_file():
                files.append({"name": path.name, "bytes": path.stat().st_size, "path": str(path.resolve())})
    return {"checkpoint_dir": str(checkpoint_dir.resolve()), "files": files}


def dataset_status(project_root: Path) -> dict[str, Any]:
    dataset_root = project_root / "dataset"
    train_paths = [
        Path("/userhome/train2017"),
        Path("/userhome/train2017.txt"),
        project_root / "dataset" / "train2017",
        project_root / "dataset" / "train2017.txt",
    ]
    return {
        "dataset_root": str(dataset_root.resolve()),
        "available_dataset_dirs": sorted(path.name for path in dataset_root.iterdir() if path.is_dir()) if dataset_root.exists() else [],
        "official_train_paths_found": [
            str(path) for path in train_paths if path.exists()
        ],
    }


def md_link(project_root: Path, rel: str, line: int | None = None) -> str:
    path = (project_root / rel).resolve()
    suffix = f":{line}" if line else ""
    label = f"{rel}{suffix}"
    return f"[{label}]({path.as_posix()}{suffix})"


def build_report(project_root: Path, status: dict[str, Any], functions: dict[str, Any], checkpoints: dict[str, Any], datasets: dict[str, Any]) -> str:
    f = functions

    def loc(name: str) -> str:
        item = f[name]
        return md_link(project_root, item["file"], item["line"])

    checkpoint_names = ", ".join(item["name"] for item in checkpoints["files"]) or "none"
    train_path_note = ", ".join(datasets["official_train_paths_found"]) or "not found in current workspace"
    relevant_files = "\n".join(
        f"- {md_link(project_root, rel)}: {'exists' if data['exists'] else 'missing'}"
        for rel, data in status.items()
    )
    relevant_functions = "\n".join(
        f"- `{name}`: {md_link(project_root, item['file'], item['line']) if item['line'] else 'not found'}"
        for name, item in functions.items()
    )

    return f"""# Stage 4.0 Watermark Branch Audit

This is a read-only audit. It does not modify the model, does not train, and does not connect SAM/CLIP/DINO/Stable Diffusion.

## relevant_files

{relevant_files}

## relevant_functions

{relevant_functions}

## image_hiding_flow

`Model_VSN.image_hiding()` is located at {loc("Model_VSN.image_hiding")}.

Observed flow:

1. `feed_data()` stores `data['LQ']` as `self.ref_L` and `data['GT']` as `self.real_H`.
2. `image_hiding()` extracts `self.host` from `self.real_H`, extracts `self.secret` from `self.ref_L`, transforms the secret with DWT, and reads the copyright bit vector from `self.mes`.
3. It calls `self.netG(x=dwt(host), x_h=dwt(secret), message=message)`.
4. In `VSN.forward(..., rev=False)` at {loc("VSN.forward")}, image mode first calls `self.irn(x, x_h, rev=False)` to produce `out_y`; then applies inverse wavelet transform `iwt(out_y)`.
5. The copyright bit branch then calls `self.bitencoder(out_y, message)` and returns this final encoded image as `container`.
6. `image_hiding()` clamps `container` and converts it to an image.

The implementation is therefore closer to a sequential nonlinear composition:

```text
fragile_carrier = IWT(IRN(DWT(host), DWT(secret)))
container       = bitencoder(fragile_carrier, copyright_bits)
```

It is not explicitly implemented as a clean additive decomposition:

```text
I_w = I + W_robust + W_fragile
```

However, for research modification, one can approximate `W_fragile` as `fragile_carrier - host` before the bit encoder.

## image_recovery_flow

`Model_VSN.image_recovery()` is located at {loc("Model_VSN.image_recovery")}.

Observed flow:

1. The input image has already been loaded through `feed_data()` into `self.real_H`.
2. `image_recovery()` sets `template = self.secret.reshape(...)` from the expected localization secret image.
3. It sends the received image through quantization and then calls `self.netG(x=y, rev=True)`.
4. In `VSN.forward(..., rev=True)`:
   - `self.bitdecoder(x)` recovers the copyright message.
   - `x = dwt(x)`.
   - `self.pm(x)` predicts a latent secret branch for reverse invertible recovery.
   - `self.irn(x, out_z_new, rev=True)` reconstructs host and secret-like outputs.
5. `image_recovery()` applies IWT to `out_x_h`, gets `rec_loc`, and computes:

```text
residual        = abs(template - rec_loc)
binary_residual = residual > threshold
pred_mask       = sum(binary_residual over RGB channels)
```

So the localization mask decoder is not a separate segmentation head. It is a residual-thresholding procedure over reconstructed fragile secret image content.

## robust_branch_location

The robust copyright watermark branch is the bit encoder/decoder pair:

- `self.bitencoder = DW_Encoder(...)` and `self.bitdecoder = DW_Decoder(...)` are constructed in `VSN.__init__` at {loc("VSN.__init__")}.
- Embedding happens in `DW_Encoder.forward()` at {loc("DW_Encoder.forward")}; the message is expanded at multiple scales and fused into the image.
- Recovery happens in `DW_Decoder.forward()` at {loc("DW_Decoder.forward")}; it produces the recovered bit vector.
- In image mode training, `l_msg` is computed/logged but is not included in the image-mode loss, so the robust bit branch is effectively kept fixed by optimizer selection.

## fragile_branch_location

The fragile localization watermark branch is the invertible image/secret pathway:

- `self.irn = InvNN(...)` in `VSN.__init__`.
- `InvNN.forward()` at {loc("InvNN.forward")} chains `InvBlock.forward()` blocks.
- `InvBlock.forward()` at {loc("InvBlock.forward")} mixes host branch `x1` and secret branch `x2`.
- The reverse localization support module is `self.pm = PredictiveModuleMIMO_prompt(...)`; in reverse mode it predicts `out_z` from the received image and passes it into `self.irn(..., rev=True)`.
- The final predicted localization mask is produced in `Model_VSN.image_recovery()` from `abs(template - rec_loc) > threshold`.

There is no explicit `fragile_watermark_tensor` variable. The fragile signal is implicit in the IRN-transformed carrier and later recovered through `out_x_h`.

## trainable_modules

Training scripts exist:

- `code/train.py` for image-mode training.
- `code/train_bit.py` for bit-mode training.

Optimizer selection in `Model_VSN.__init__`:

- `mode == "image"` optimizes parameters whose names start with `module.irn` or `module.pm`.
- `mode == "bit"` optimizes parameters whose names start with `module.bitencoder` or `module.bitdecoder`.

Image-mode losses in `optimize_parameters()`:

```text
l_forw_fit = reconstruction(container, host)
l_back_rec = reconstruction(recovered_host, host)
l_center_x = reconstruction(recovered_secret, secret)
loss = l_forw_fit * 2 + l_back_rec + l_center_x * 4
```

`l_msg` is computed in image mode but not added to the loss. Bit-mode training uses:

```text
loss = l_msg * lambda_msg + l_forw_fit
```

## checkpoint_usage

Available checkpoint files: {checkpoint_names}.

`Model_VSN.load_test()` at {loc("Model_VSN.load_test")} loads a checkpoint into `self.netG`. Training configs can also use `path.pretrain_model_G`, but current training YAMLs leave it blank.

## whether_fragile_embedding_can_be_modified

Yes, but not as a simple standalone additive tensor in the current code.

Best interpretation:

- Robust copyright branch: `bitencoder/bitdecoder`.
- Fragile localization branch: `irn + pm + residual threshold recovery`.
- Current architecture separates trainable parameter groups by mode, so image-mode training can update localization branch modules while leaving bit encoder/decoder out of the optimizer.
- The actual watermarked image is produced after `bitencoder`, so a fragile-only modification should be inserted before `bitencoder` if the copyright branch must remain unchanged.

## recommended_insertion_point_for_A_sem

Recommended insertion point for a future semantic spatial strength map `A_sem`:

```text
code/models/modules/Inv_arch.py
VSN.forward(..., rev=False, mode="image")
after:  out_y = iwt(out_y)
before: encoded_image = self.bitencoder(out_y, message)
```

Conceptually:

```python
host_img = iwt(x)              # x is DWT(host)
fragile_residual = out_y - host_img
out_y = host_img + A_sem * fragile_residual
encoded_image = self.bitencoder(out_y, message)
```

This keeps the robust bit encoder as the last stage and targets the fragile carrier before copyright embedding. It will still require training-time care because scaling the fragile residual changes reverse recovery behavior and may indirectly affect bit recovery through changed bitencoder input.

Alternative insertion point:

- Modulate `x_h` before `self.irn(x, x_h, rev=False)`.

This is riskier because it changes the secret branch distribution inside an invertible coupling network and may break the learned reverse pathway more severely.

## training_feasibility

- Can the code retrain EditGuard? **Yes, training scripts and options exist.**
- Is the current workspace immediately ready for official full retraining? **Not fully.** The YAML training data paths point to `/userhome/train2017` and `/userhome/train2017.txt`; detected train paths: `{train_path_note}`.
- Does current workspace have inference checkpoints? **Yes**, `clean.pth` and `degrade.pth`.
- Is there explicit localization mask loss? **No.** The localization signal is trained through secret-image reconstruction (`l_center_x`) rather than BCE/IoU/Dice mask loss against `valAGE-Set-Mask`.
- Can only the localization branch be fine-tuned? **Likely yes with code/config care.** Image mode already optimizes `irn` and `pm`, not `bitencoder/bitdecoder`.
- Can the robust branch be frozen? **Yes.** It is already excluded from the image-mode optimizer, but a safer future implementation should set `requires_grad=False` and possibly eval mode for `bitencoder/bitdecoder`.
- Can copyright branch remain unchanged? **Likely yes**, if future changes operate before `bitencoder` and bit branch weights are frozen.

## risk_points

1. The system is not additive; `I_w = I + W_robust + W_fragile` is only a useful mental model, not the actual implementation.
2. The robust branch is applied after the fragile carrier, so changing the fragile carrier changes the bit encoder input distribution.
3. No explicit mask decoder or mask loss exists; localization is residual-threshold based.
4. `pm` is part of reverse localization recovery; changing embedding without retraining `pm` may hurt localization.
5. Many predicted masks fragment into small connected components, so future training losses should consider region compactness or semantic consistency, but this audit does not implement that.
6. Current training configs reference unavailable original training data paths.
7. Image-mode optimizer excludes bit branch parameters but does not explicitly set `requires_grad=False`; gradients can still be computed through the bit branch unless explicitly frozen.
8. The code imports diffusion pipelines at module import time in `IBSN.py`; future clean training scripts may need dependency guards, but Stage 4 does not change this.
9. `hidebit`/`bithide` naming is inconsistent across YAMLs; future config work should normalize carefully.
10. Any `A_sem` insertion must preserve output range, quantization behavior, and bit recovery stability.
"""


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    status = file_status(project_root)
    functions = function_locations(project_root)
    checkpoints = checkpoint_status(project_root)
    datasets = dataset_status(project_root)
    report = build_report(project_root, status, functions, checkpoints, datasets)
    (output_dir / "watermark_branch_audit.md").write_text(report, encoding="utf-8")
    (output_dir / "watermark_branch_audit_index.json").write_text(
        json.dumps(
            {
                "project_root": str(project_root),
                "files": status,
                "functions": functions,
                "checkpoints": checkpoints,
                "datasets": datasets,
                "report": str((output_dir / "watermark_branch_audit.md").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved report: {output_dir / 'watermark_branch_audit.md'}")
    print(f"Saved index: {output_dir / 'watermark_branch_audit_index.json'}")


if __name__ == "__main__":
    main()
