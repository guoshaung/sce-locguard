from __future__ import annotations

from pathlib import Path

import numpy as np

from src.modules.common import load_mask


class EditGuardAdapter:
    """Adapter for EditGuard / OmniGuard outputs.

    Version 0 supports local PNG masks only.
    """

    def load_pred_mask(self, path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
        return load_mask(path, size=size)

    def load_confidence_map(self, path: str | Path | None, size: tuple[int, int] | None = None):
        if path is None or str(path).strip() == "":
            return None
        mask = load_mask(path, size=size)
        return mask.astype(np.float32)

    def load_copyright_result(self, path: str | Path | None):
        if path is None or str(path).strip() == "":
            return None
        path = Path(path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip()

