# backend/fastapi_app/feature_engineering.py
import numpy as np
import pandas as pd
import ta

def compute_features(df: pd.DataFrame, ticker_name="INDEX") -> pd.DataFrame:
    """
    df must contain: date, open, high, low, close, volume
    Returns a processed DataFrame with all ML-required features.
    """
    df = df.copy()

    # Basic returns
    df["return_1d"] = df["close"].pct_change()
    df["return_7d"] = df["close"].pct_change(7)

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

    # Rolling statistics
    for w in [3, 5, 7, 10, 14]:
        df[f"roll_mean_{w}"] = df["close"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["close"].rolling(w).std()
        df[f"roll_vol_{w}"] = df["volume"].rolling(w).std()

    # ROC
    df["roc_5"] = df["close"].pct_change(5)
    df["roc_10"] = df["close"].pct_change(10)

    # Momentum
    df["momentum"] = df["close"] - df["close"].shift(10)

    # Technical Indicators
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()

    # ADX
    df["adx_14"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()

    # Weighted MA (wma_20)
    def wma(x):
        weights = np.arange(1, len(x)+1)
        return (x * weights).sum() / weights.sum()
    df["wma_20"] = df["close"].rolling(20).apply(lambda x: wma(x) if len(x) > 0 else np.nan, raw=True)

    # ticker_id numeric encoding
    if "ticker_id" not in df.columns:
        df["ticker_id"] = hash(ticker_name) % 1000

    # Drop rows with NaN and return
    df = df.dropna().reset_index(drop=True)
    return df
