# advisor_service.py
import os
from typing import List, Dict, Any
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from .query_embed import get_embedding_model, embed_texts_pyfloat
from .model_loader import model, feature_cols
import numpy as np

# For LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Config
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "market-news-index")
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")  # change if needed
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Azure client factory
def get_search_client():
    if not SEARCH_ENDPOINT or not SEARCH_KEY:
        raise RuntimeError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_ADMIN_KEY env vars")
    cred = AzureKeyCredential(SEARCH_KEY)
    return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=cred, api_version=os.getenv("AZURE_SEARCH_API_VERSION","2023-11-01"))

# LLM load (lazy)
_LLM = None
_TOKENIZER = None
def load_llm(model_name: str = None):
    global _LLM, _TOKENIZER
    mname = model_name or LLM_MODEL_NAME
    if _LLM is not None and _TOKENIZER is not None:
        return _TOKENIZER, _LLM

    # fallback CPU-friendly config if GPU not present
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(mname, use_fast=True)
        # try to load with device_map auto
        _LLM = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=torch.float16, device_map="auto")
    except Exception:
        # CPU fallback: small models only recommended
        _TOKENIZER = AutoTokenizer.from_pretrained(mname, use_fast=True)
        _LLM = AutoModelForCausalLM.from_pretrained(mname)
    return _TOKENIZER, _LLM

def predict_ml(input_df):
    """
    Accepts a pandas DataFrame already preprocessed and ordered by feature_cols.
    Returns (direction:int, confidence:float)
    """
    # model.predict expects same columns; ensure
    X = input_df[feature_cols]
    direction = int(model.predict(X)[0])
    # some models may not have predict_proba
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        prob = 0.0
    return direction, prob

def fetch_relevant_news(query_vector: List[float], k: int = TOP_K) -> List[Dict[str, Any]]:
    client = get_search_client()
    results = client.search(
        search_text=None,
        vectors=[{
            "value": query_vector,
            "k": k,
            "fields": "content_vector"
        }],
        select=["id","content","url","date"]
    )
    # convert to simple list of dicts
    docs = []
    for r in results:
        docs.append({
            "id": r.get("id"),
            "content": r.get("content"),
            "url": r.get("url"),
            "date": r.get("date")
        })
    return docs

def synthesize_advice(prediction: str, confidence: float, news_docs: List[Dict[str,Any]], user_query: str = None) -> str:
    tokenizer, llm = load_llm()
    news_summary = "\n".join([f"- {d['content']}" for d in news_docs]) if news_docs else "No relevant news found."
    prompt = f"""
You are a helpful and cautious financial advisor.

ML Signal:
- Direction: {prediction}
- Confidence: {confidence:.2f}

User query: {user_query if user_query else 'N/A'}

Top relevant news:
{news_summary}

Please provide:
1) A short plain-English summary of what the ML signal + news suggests.
2) A conservative action recommendation (entry/exit/hold) with concise reasoning.
3) Any caveats or risks.

Be brief and avoid giving specific buy/sell orders or exact position sizes.
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(llm.parameters()).device
    inputs = {k:v.to(device) for k,v in inputs.items()}
    out = llm.generate(**inputs, max_new_tokens=300, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text
