from rag.ingestion.load_news import load_news_csv
from rag.ingestion.embedder import get_embedding_model, embed_texts
from rag.ingestion.indexer import create_news_index, upload_news


def run_ingestion(csv_path: str, index_name: str):
    print("Loading news...")
    df = load_news_csv(csv_path)

    print("Loading embedding model...")
    model = get_embedding_model()

    print("Generating embeddings...")
    vectors = embed_texts(model, df["content"].tolist())
    print(type(vectors[0][0]))

    print("Creating index...")
    create_news_index(index_name)

    print("Uploading documents...")
    docs = []
    for i in range(len(df)):
        row = df.iloc[i]
        docs.append({
            "id": str(i),
            "content": row["content"],
            "url": row["link"],
            "date": str(row["published"]),
            "content_vector": vectors[i]
        })

    print(f"DOCUMENTS------{docs}")

    upload_news(index_name, docs)

    print("Ingestion complete.")

if __name__ == "__main__":
    CSV_PATH = "data/news_raw.csv"  # Path to your news CSV
    INDEX_NAME = "market_news_index"        # Desired Azure Search index name

    run_ingestion(CSV_PATH, INDEX_NAME)