# backend/rag/ingestion/ingest_pipeline.py
import os
from dotenv import load_dotenv
load_dotenv()
from backend.fastapi_app.rss_fetcher import fetch_rss_entries
from .rolling_ingest import run_rolling_ingestion
import pandas as pd

def run_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    run_rolling_ingestion(df)

def run_from_rss():
    df_entries = fetch_rss_entries()
    import pandas as pd
    df = pd.DataFrame(df_entries)
    run_rolling_ingestion(df)

if __name__ == "__main__":
    # default: ingest live RSS into azure (last 30 days)
    run_from_rss()
