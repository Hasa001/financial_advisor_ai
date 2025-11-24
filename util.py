from dotenv import load_dotenv
load_dotenv()
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import os

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = "market_news_index"

client = SearchIndexClient(endpoint, AzureKeyCredential(key))
idx = client.get_index(index_name)

for f in idx.fields:
    print(f.name, f.type, getattr(f, "vector_search_dimensions", None))
