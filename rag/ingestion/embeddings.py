# rag/ingestion/embeddings.py
from sentence_transformers import SentenceTransformer

_model = None
def get_embedding_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed_texts(model, texts):
    raw = model.encode(texts, convert_to_numpy=True)
    return [[float(x) for x in vec] for vec in raw]
