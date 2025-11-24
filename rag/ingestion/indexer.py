from dotenv import load_dotenv
load_dotenv()


from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os

EMBED_DIM=768

def create_news_index(index_name: str):
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

    if not endpoint or not admin_key:
        raise ValueError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_ADMIN_KEY")

    client = SearchIndexClient(
        endpoint,
        AzureKeyCredential(admin_key),
        api_version="2023-11-01"     # CRITICAL FOR VECTOR SUPPORT
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="url", type=SearchFieldDataType.String),
        SimpleField(name="date", type=SearchFieldDataType.String),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name="news-profile"
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                kind="hnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="news-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ]
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )

    client.create_or_update_index(index)
    print(f"Index '{index_name}' created successfully.")



def upload_news(index_name: str, docs: list):
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

    client = SearchClient(endpoint, index_name, AzureKeyCredential(admin_key))

    # Upload in safe batches of 100 docs
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        client.upload_documents(batch)
        print(f"Uploaded {len(batch)} docs")

