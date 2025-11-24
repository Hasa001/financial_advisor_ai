# backend/fastapi_app/chart_generator.py
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

def fetch_ohlcv(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}")

    # Remove multi-index column issues
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Fallback to Adj Close
    if "Close" not in df.columns or df["Close"].isna().all():
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            raise ValueError("No usable Close column found.")

    # Force numeric close
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError("No usable Close values after cleanup.")

    # Ensure Date exists
    if "Date" not in df.columns:
        raise ValueError("Missing Date column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["Date"])

    return df


def plot_line_chart(df: pd.DataFrame, title: str = None) -> io.BytesIO:
    if df.empty:
        raise ValueError("Cannot plot empty DataFrame.")

    # Convert to numpy arrays (FIXES your error)
    x = df["Date"].values
    y = df["Close"].values.astype(float)

    # Ensure both are 1D
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Date or Close column is not 1D even after cleanup.")

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y, linewidth=1.8, color="blue")
    plt.fill_between(x, y, alpha=0.08)

    plt.title(title or "Price Movement")
    plt.xlabel("Date")
    plt.xticks(rotation=25)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    return buf


def generate_chart_bytes(ticker: str, period: str = "1mo", interval: str = "1d"):
    df = fetch_ohlcv(ticker, period=period, interval=interval)
    title = f"{ticker} ({period})"
    return plot_line_chart(df, title)
