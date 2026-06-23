from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.watermark.locguard_trainer_stub import LocGuardTrainerStub


def main() -> None:
    trainer = LocGuardTrainerStub()
    try:
        trainer.train()
    except NotImplementedError as exc:
        print(exc)


if __name__ == "__main__":
    main()
