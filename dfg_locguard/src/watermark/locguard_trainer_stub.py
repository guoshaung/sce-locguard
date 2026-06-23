from __future__ import annotations


class LocGuardTrainerStub:
    """Placeholder for future localization-watermark decoder training."""

    def __init__(self, config=None):
        self.config = config or {}

    def train(self):
        raise NotImplementedError(
            "Version 0 does not train a new watermark network. "
            "Future versions will use diffusion guidance maps only during training."
        )

