# backend/fastapi_app/ml_predictor.py
import os
import pandas as pd
from .chart_generator import fetch_ohlcv
from .feature_engineering import compute_features
from .model_loader import load_model

def run_ml_prediction(ticker: str):
    """
    Fetch recent OHLC -> build full feature set -> run ML model -> return prediction.
    Returns dict: {"direction":"UP"/"DOWN"/"UNKNOWN", "probability": float}
    """
    try:
        model, feature_cols = load_model()
    except Exception as e:
        print("ML load error:", e)
        return {"direction":"UNKNOWN","probability":0.0}

    try:
        # Fetch sufficient history for rolling features (90 days)
        df_raw = fetch_ohlcv(ticker, period="90d", interval="1d")

        # Normalize column names to lowercase expected by feature_engineering
        # yfinance sometimes gives 'Open', 'High', etc.
        col_map = {}
        for c in df_raw.columns:
            lc = c.lower()
            if lc == "open":
                col_map[c] = "open"
            if lc == "high":
                col_map[c] = "high"
            if lc == "low":
                col_map[c] = "low"
            if lc == "close":
                col_map[c] = "close"
            if lc == "volume":
                col_map[c] = "volume"
        df_raw = df_raw.rename(columns=col_map)

        # Ensure required columns present
        for required in ["open","high","low","close","volume"]:
            if required not in df_raw.columns:
                raise ValueError(f"Required column {required} missing from OHLC data")

        # compute features
        df_feat = compute_features(df_raw, ticker_name=ticker)
        if df_feat.empty:
            raise ValueError("No rows after computing features")

        latest = df_feat.iloc[[-1]].copy()

        # Ensure all expected features exist in same order
        missing = [c for c in feature_cols if c not in latest.columns]
        if missing:
            raise ValueError(f"Missing feature columns required by model: {missing}")

        X = latest[feature_cols]

        pred = model.predict(X)[0]
        try:
            prob = float(model.predict_proba(X)[0].max())
        except Exception:
            prob = 0.0

        direction = "UP" if int(pred) == 1 else "DOWN"
        return {"direction": direction, "probability": prob}

    except Exception as e:
        print("ML prediction error:", e)
        return {"direction":"UNKNOWN","probability":0.0}
