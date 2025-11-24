# rag/ingestion/rolling_ingest.py
import os
import datetime
import pandas as pd
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType
)
from dateutil import parser as dateparser

# Environment variables
ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX = os.getenv("AZURE_SEARCH_INDEX", "market-news-index")

# ----------------------------------------------------------------------
#   AUTO-CREATE INDEX IF NOT EXISTS
# ----------------------------------------------------------------------
def ensure_index_exists():
    """
    Creates the index if it does not exist. Safe to call every ingestion run.
    """
    index_client = SearchIndexClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
        api_version="2023-11-01"
    )

    try:
        # Check if index exists
        existing = index_client.get_index(INDEX)
        print(f"[INFO] Index '{INDEX}' already exists.")
        return
    except Exception:
        print(f"[INFO] Index '{INDEX}' not found. Creating...")

    # Create a minimal TEXT-SEARCH-ONLY index (no vectors needed)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="url", type=SearchFieldDataType.String),
        SimpleField(name="date", type=SearchFieldDataType.String),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
    ]

    index = SearchIndex(name=INDEX, fields=fields)
    index_client.create_or_update_index(index)
    print(f"[SUCCESS] Created Azure Search Index: {INDEX}")


# ----------------------------------------------------------------------
#   SEARCH CLIENT
# ----------------------------------------------------------------------
def get_client():
    return SearchClient(
        endpoint=ENDPOINT,
        index_name=INDEX,
        credential=AzureKeyCredential(KEY),
        api_version="2023-11-01"
    )


# ----------------------------------------------------------------------
#   DATE HELPERS
# ----------------------------------------------------------------------
def to_utc_naive(dt_str):
    """Convert RSS datetime → UTC tz-naive for filtering."""
    try:
        if not dt_str:
            return None
        dt = dateparser.parse(dt_str)
        if dt is None:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return dt
    except:
        return None


# ----------------------------------------------------------------------
#   DELETE OLD NEWS (> 30 DAYS)
# ----------------------------------------------------------------------
def delete_old_documents(days=30):
    client = get_client()

    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    to_delete = []

    results = client.search(search_text="", top=1000)

    for doc in results:
        try:
            pub = to_utc_naive(doc.get("date", ""))
            if pub and pub < cutoff:
                to_delete.append({"id": doc["id"]})
        except:
            pass

    if to_delete:
        print(f"[INFO] Deleting {len(to_delete)} outdated documents...")
        client.delete_documents(to_delete)
    else:
        print("[INFO] No old documents to delete.")


# ----------------------------------------------------------------------
#   UPLOAD NEW NEWS
# ----------------------------------------------------------------------
def upload_recent_news(df: pd.DataFrame):
    client = get_client()
    docs = []
    timestamp = int(datetime.datetime.utcnow().timestamp())

    for i, row in df.iterrows():
        docs.append({
            "id": f"news_{i}_{timestamp}",
            "content": f"{row.get('title','')} - {row.get('summary','')}",
            "url": row.get("link", ""),
            "date": row.get("published", "")
        })

    if docs:
        print(f"[INFO] Uploading {len(docs)} new documents...")
        client.upload_documents(docs)
        print("[SUCCESS] Upload completed.")
    else:
        print("[INFO] No documents to upload.")


# ----------------------------------------------------------------------
#   MAIN INGESTION FUNCTION
# ----------------------------------------------------------------------
def run_rolling_ingestion(df: pd.DataFrame):
    # STEP 1 — Ensure index exists
    ensure_index_exists()

    # STEP 2 — Fix date column names
    if "published" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "published"})

    if "summary" not in df.columns and "content" in df.columns:
        df["summary"] = df["content"]

    # STEP 3 — Parse published date
    df["published_dt"] = df["published"].apply(to_utc_naive)
    df = df.dropna(subset=["published_dt"])

    now = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(days=30)

    df_recent = df[df["published_dt"] >= cutoff]
    print(f"[INFO] Filtered {len(df_recent)} articles from last 30 days.")

    # STEP 4 — Delete old docs from Azure
    delete_old_documents(days=30)

    # STEP 5 — Upload new docs
    upload_recent_news(df_recent)
