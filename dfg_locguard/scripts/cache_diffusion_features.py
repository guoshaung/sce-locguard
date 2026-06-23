from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dfg.cache_features import cache_placeholder_features
from src.dfg.diffusion_feature_extractor import DiffusionFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache placeholder diffusion features for future training.")
    parser.add_argument("--image-dir", default=str(PROJECT_ROOT / "data" / "original"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "features"))
    args = parser.parse_args()
    image_paths = sorted(Path(args.image_dir).glob("*.png"))
    saved = cache_placeholder_features(image_paths, Path(args.out_dir), DiffusionFeatureExtractor(enabled=False))
    print(f"Saved placeholder feature files: {len(saved)}")


if __name__ == "__main__":
    main()
