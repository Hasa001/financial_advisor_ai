# backend/fastapi_app/query_embed.py
from sentence_transformers import SentenceTransformer
from typing import List
import threading

# thread-safe lazy loader
_model = None
_model_lock = threading.Lock()

def get_embedding_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(model_name)
    return _model

def embed_texts_pyfloat(model, texts: List[str]):
    """
    Return list[list[float]] with pure Python float elements.
    """
    raw = model.encode(texts, convert_to_numpy=True)
    # convert each numpy vector to list of python floats
    vectors = [[float(x) for x in vec] for vec in raw]
    return vectors
