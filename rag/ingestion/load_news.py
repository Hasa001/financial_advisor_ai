# backend/rag/ingestion/load_news.py
import pandas as pd

def load_news_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "content" not in df.columns:
        if "title" in df.columns and "summary" in df.columns:
            df = df.dropna(subset=["title","summary"]).copy()
            df["content"] = df["title"].astype(str) + " - " + df["summary"].astype(str)
        else:
            raise ValueError("CSV must contain 'content' or 'title'+'summary'")
    df = df.reset_index(drop=True)
    if "url" not in df.columns: df["link"] = ""
    if "date" not in df.columns: df["published"] = ""
    return df[["content","url","date"]]
