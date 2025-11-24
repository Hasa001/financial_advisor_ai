# backend/fastapi_app/model_loader.py
import os
import json
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "upgraded_catboost_model.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.json")

_model_cache = None
_feature_cols_cache = None

def load_model():
    """
    Loads ML model + feature list in correct order.
    Cached so repeated calls are instant.
    Returns: (model, feature_cols)
    """
    global _model_cache, _feature_cols_cache

    # Already loaded → return cache
    if _model_cache is not None and _feature_cols_cache is not None:
        return _model_cache, _feature_cols_cache

    # Load feature list
    if not os.path.exists(FEATURE_LIST_PATH):
        raise FileNotFoundError(f"Missing feature_list.json at {FEATURE_LIST_PATH}")

    with open(FEATURE_LIST_PATH, "r") as f:
        _feature_cols_cache = json.load(f)

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model file {MODEL_PATH}. Place upgraded_catboost_model.pkl in backend/models/"
        )

    try:
        _model_cache = joblib.load(MODEL_PATH)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Model unpickle failed due to missing dependency: {e}. "
            f"You must install catboost:"
            f"    pip install catboost"
        )

    return _model_cache, _feature_cols_cache
