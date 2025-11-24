import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "upgraded_catboost_model.pkl")
FEATURE_LIST_PATH = os.path.join(BASE_DIR, "models", "feature_list.json")

# Load CatBoost model
model = joblib.load(MODEL_PATH)

# Load feature list
with open(FEATURE_LIST_PATH, "r") as f:
    feature_cols = json.load(f)
