# query_embed.py
from sentence_transformers import SentenceTransformer
from typing import List
import os

# lazy singleton for embedding model
_EMBED_MODEL = None

def get_embedding_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL

def embed_texts_pyfloat(model, texts: List[str]):
    """
    Return list[list[float]] with Python float elements (required by Azure Search).
    """
    raw = model.encode(texts, convert_to_numpy=True)
    vectors = [[float(x) for x in vec] for vec in raw]
    return vectors
