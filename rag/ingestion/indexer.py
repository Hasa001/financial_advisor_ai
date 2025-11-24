# backend/fastapi_app/indexer.py
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType
)

def create_news_index(index_name: str):
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    if not endpoint or not key:
        raise ValueError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_ADMIN_KEY")
    client = SearchIndexClient(endpoint, AzureKeyCredential(key), api_version="2023-11-01")
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="url", type=SearchFieldDataType.String),
        SimpleField(name="date", type=SearchFieldDataType.String),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True)
    ]
    index = SearchIndex(name=index_name, fields=fields)
    client.create_or_update_index(index)
    print(f"Index '{index_name}' created or updated (text-only).")
