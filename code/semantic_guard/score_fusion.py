from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import ObjectRegion, SCCResult


FEATURE_NAMES = [
    "object_tamper_ratio",
    "image_tamper_ratio",
    "semantic_change",
    "category_weight",
    "tsa_confidence",
    "pixel_delta",
]


def feature_vector(region: ObjectRegion, scc: SCCResult, category_weight: float, image_area: int) -> np.ndarray:
    return np.asarray(
        [
            float(region.tamper_ratio),
            float(region.tampered_area / max(1, image_area)),
            float(scc.change_score),
            float(category_weight),
            float(region.confidence),
            float(scc.pixel_delta),
        ],
        dtype=np.float32,
    )


def train_fusion_model(csv_path: str | Path, output_path: str | Path, model_type: str = "logistic") -> dict[str, object]:
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = pd.read_csv(csv_path)
    missing = [name for name in FEATURE_NAMES if name not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing feature columns: {missing}")

    x = data[FEATURE_NAMES].to_numpy(dtype=np.float32)
    if "severity_score" in data.columns:
        y = data["severity_score"].to_numpy(dtype=np.float32)
        stratify = None
        is_regression = True
    elif "label" in data.columns:
        y = data["label"].to_numpy()
        stratify = y if len(set(y)) > 1 else None
        is_regression = False
    else:
        raise ValueError("CSV must include either 'label' severity classes or 'severity_score'.")

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )

    if is_regression:
        estimator = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, random_state=42) if model_type == "mlp" else Ridge()
    else:
        estimator = (
            MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
            if model_type == "mlp"
            else LogisticRegression(max_iter=2000, class_weight="balanced")
        )

    model = make_pipeline(StandardScaler(), estimator)
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    metric_name = "mae" if is_regression else "accuracy"
    metric_value = float(mean_absolute_error(y_val, pred)) if is_regression else float(accuracy_score(y_val, pred))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return {
        "output_path": str(output_path),
        "model_type": model_type,
        "target": "severity_score" if is_regression else "label",
        metric_name: metric_value,
        "feature_names": FEATURE_NAMES,
    }
