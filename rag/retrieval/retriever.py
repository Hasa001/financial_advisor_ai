from azure.search.documents import SearchClient
import os

def fetch_relevant_news(index_name: str, query_vector, k=5):
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

    client = SearchClient(endpoint, index_name, admin_key)

    results = client.search(
        search_text=None,
        vectors=[{
            "value": query_vector,
            "k": k,
            "fields": "content_vector"
        }],
        select=["content", "url", "date"]
    )

    return [doc for doc in results]
