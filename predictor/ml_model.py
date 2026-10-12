"""
Machine-learning inference module.

Loads the serialized classifier (trained in notebooks/wine_quality_analysis.ipynb)
once at server start-up and exposes a single ``predict_quality`` function used by
the prediction view. Keeping this logic isolated from views.py mirrors the report's
architecture: "A machine-learning module loads the serialized classifier and
exposes a prediction function."
"""

import json
import pickle
import threading

import pandas as pd
from django.conf import settings

_lock = threading.Lock()
_model = None
_scaler = None
_metadata = None


def _load():
    """Lazily load the model, scaler and metadata (thread-safe, loaded once)."""
    global _model, _scaler, _metadata
    if _model is not None:
        return

    with _lock:
        if _model is not None:  # double-checked locking
            return

        if not settings.ML_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Trained model not found at "
                f"{settings.ML_MODEL_PATH}. Run notebooks/wine_quality_analysis.ipynb "
                "(or notebooks/train_script.py) first to generate it."
            )

        with open(settings.ML_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)

        with open(settings.ML_METADATA_PATH, "r") as f:
            _metadata = json.load(f)

        if _metadata.get("uses_scaler") and settings.ML_SCALER_PATH.exists():
            with open(settings.ML_SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)


def get_metadata():
    _load()
    return _metadata


def predict_quality(feature_dict):
    """
    Run inference for a single wine sample.

    Parameters
    ----------
    feature_dict : dict
        Mapping of the 11 feature names (as in ``model_metadata.json``
        ``feature_order``) to numeric values.

    Returns
    -------
    int
        The predicted wine-quality class.
    """
    _load()

    feature_order = _metadata["feature_order"]
    ordered_df = pd.DataFrame([[feature_dict[name] for name in feature_order]], columns=feature_order)

    features_for_model = ordered_df
    if _metadata.get("uses_scaler") and _scaler is not None:
        features_for_model = _scaler.transform(ordered_df)

    prediction = _model.predict(features_for_model)[0]
    return int(prediction)


def predict_quality_with_confidence(feature_dict):
    """Same as predict_quality but also returns per-class probabilities, when available."""
    _load()

    feature_order = _metadata["feature_order"]
    ordered_df = pd.DataFrame([[feature_dict[name] for name in feature_order]], columns=feature_order)

    features_for_model = ordered_df
    if _metadata.get("uses_scaler") and _scaler is not None:
        features_for_model = _scaler.transform(ordered_df)

    prediction = int(_model.predict(features_for_model)[0])

    confidence = None
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(features_for_model)[0]
        class_labels = list(_model.classes_)
        confidence = {
            int(cls): round(float(p), 4) for cls, p in zip(class_labels, proba)
        }

    return prediction, confidence
