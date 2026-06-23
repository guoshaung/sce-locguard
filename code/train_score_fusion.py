from __future__ import annotations

import argparse

from semantic_guard.score_fusion import FEATURE_NAMES, train_fusion_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small score-fusion model for SemanticGuard SSE.")
    parser.add_argument("--csv", required=True, help="Training CSV with feature columns and label/severity_score.")
    parser.add_argument("--out", default="checkpoints/semantic_guard_score_fusion.joblib")
    parser.add_argument("--model", default="logistic", choices=["logistic", "mlp"])
    args = parser.parse_args()
    summary = train_fusion_model(args.csv, args.out, args.model)
    print("Required feature columns:", ", ".join(FEATURE_NAMES))
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
