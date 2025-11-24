# backend/fastapi_app/preprocess.py
import pandas as pd
from .model_loader import load_model

def preprocess_input(df: pd.DataFrame):
    model, feature_cols = load_model()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    df2 = df[feature_cols].copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0)
    return df2
