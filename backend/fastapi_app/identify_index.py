# backend/fastapi_app/identify_index.py
import re
from rapidfuzz import process, fuzz
from .ticker_db import CANONICAL, CANDIDATES
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import threading

_sem_model = None
_sem_lock = threading.Lock()

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\^]", " ", text)
    return re.sub(r"\s+", " ", text)

def get_sem_model():
    global _sem_model
    if _sem_model is None:
        with _sem_lock:
            if _sem_model is None:
                _sem_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _sem_model

def detect_index(query: str, fuzzy_threshold: float = 70.0, use_semantic: bool = True):
    q = normalize_text(query)
    if not q:
        return None, None, 0.0, "no_match"

    tokens = q.split()
    for t in tokens:
        if t in CANONICAL:
            return CANONICAL[t], t, 100.0, "exact_token"

    match, score, _ = process.extractOne(q, CANDIDATES, scorer=fuzz.WRatio)
    if score >= fuzzy_threshold:
        return CANONICAL[match], match, float(score), "fuzzy"

    if use_semantic:
        try:
            model = get_sem_model()
            q_emb = model.encode([q], convert_to_numpy=True)
            cand_embs = model.encode(CANDIDATES, convert_to_numpy=True)
            sims = (cand_embs @ q_emb.T).squeeze() / (norm(cand_embs, axis=1) * norm(q_emb))
            best_idx = int(np.argmax(sims))
            score = float(sims[best_idx]) * 100.0
            if score >= 30.0:
                best = CANDIDATES[best_idx]
                return CANONICAL[best], best, score, "semantic"
        except Exception:
            pass

    return None, None, 0.0, "no_match"
