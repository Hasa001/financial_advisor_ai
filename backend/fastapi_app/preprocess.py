import pandas as pd
from fastapi_app.model_loader import feature_cols

def preprocess_input(df: pd.DataFrame):

    # Ensure all required features exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Reorder columns to match training dataset
    df = df[feature_cols]

    return df
