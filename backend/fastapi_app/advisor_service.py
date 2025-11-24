# backend/fastapi_app/advisor_service.py
"""
GPU-enabled DeepSeek integration using Transformers (quantization supported).
Text-only Azure Search retrieval + RSS merging.
Provides functions used by router: fetch_news_for_index, predict_ml, synthesize_advice, generate_advice_for_query
"""
from .ml_predictor import run_ml_prediction

import os, threading
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import torch
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from transformers import AutoTokenizer, AutoModelForCausalLM

from .model_loader import load_model
from .chart_generator import fetch_ohlcv
from .rss_fetcher import fetch_rss_entries, filter_rss_by_query

# CONFIG
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "market-news-index")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
LLM_DEVICE = os.getenv("LLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
LLM_QUANT = os.getenv("LLM_QUANT", "").lower()
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))

_tokenizer = None
_model = None
_llm_lock = threading.Lock()

def get_search_client() -> SearchClient:
    if not SEARCH_ENDPOINT or not SEARCH_KEY:
        raise RuntimeError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_ADMIN_KEY")
    return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))

def fetch_relevant_news_text(query: str, k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    client = get_search_client()
    results = client.search(search_text=query, top=k)
    docs = []
    for r in results:
        docs.append({"id": r.get("id"), "content": r.get("content"), "url": r.get("url"), "date": r.get("date")})
    return docs

def fetch_news_for_index(index_name: str, user_query: Optional[str] = None, k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    azure_query = f"{user_query or ''} {index_name} market RBI monetary policy"
    azure_news = fetch_relevant_news_text(azure_query, k=k)
    live = filter_rss_by_query(fetch_rss_entries(), user_query or index_name, top_k=k)
    live_formatted = [{"id": f"rss_{i}", "content": f"{a['title']} - {a['summary']}", "url": a["link"], "date": a.get("published","")} for i,a in enumerate(live)]
    # merge + dedupe
    combined = azure_news + live_formatted
    seen = set(); unique=[]
    for it in combined:
        sig = (it["content"] or "")[:120]
        if sig not in seen:
            seen.add(sig)
            unique.append(it)
    return unique[:k]

# ---------------- ML wrapper ----------------
def predict_ml(input_df: pd.DataFrame) -> Tuple[int, float]:
    model, feature_cols = load_model()
    X = input_df[feature_cols]
    direction = int(model.predict(X)[0])
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        prob = 0.0
    return direction, prob

# ---------------- LLM loader + generation ----------------
def load_llm_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    with _llm_lock:
        if _tokenizer is not None and _model is not None:
            return _tokenizer, _model
        print(f"[LLM] Loading {LLM_MODEL_NAME} device={LLM_DEVICE} quant={LLM_QUANT}")
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True)
        load_kwargs = {"trust_remote_code": True}
        try:
            if LLM_DEVICE == "cuda" and torch.cuda.is_available():
                if LLM_QUANT == "8bit":
                    load_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
                    _model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, **load_kwargs)
                elif LLM_QUANT == "4bit":
                    load_kwargs.update({"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16, "device_map": "auto"})
                    _model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, **load_kwargs)
                else:
                    _model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float16, device_map="auto", **load_kwargs)
            else:
                if LLM_QUANT in ("8bit","4bit"):
                    load_kwargs.update({"load_in_8bit": True, "device_map": "cpu"})
                    _model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, **load_kwargs)
                else:
                    _model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="cpu")
        except Exception as e:
            print("[LLM] primary load failed:", e)
            _model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="cpu")
        if LLM_DEVICE == "cuda" and torch.cuda.is_available():
            try:
                _model.to("cuda")
            except Exception:
                pass
        print("[LLM] loaded.")
        return _tokenizer, _model

def _shorten(s: Optional[str], n:int=400)->str:
    if not s: return ""
    return s if len(s)<=n else s[:n-3]+"..."

def synthesize_advice(prediction_label: str, confidence: float, news_docs: List[Dict[str, Any]], user_query: Optional[str] = None) -> Dict[str, Any]:
    tokenizer, model = load_llm_model()
    news_text = "\n".join([f"- {_shorten(n['content'],400)}" for n in (news_docs or [])]) or "No relevant news found."
    prompt = f"""
You are an expert cautious financial analyst. Answer in the exact format below.

User question:
{user_query or 'N/A'}

ML signal: {prediction_label} (confidence={confidence:.2f})

Top relevant news:
{news_text}

Required output exactly:
DIRECTION: <UP|DOWN|NEUTRAL>
SUMMARY: <1-2 sentences>
RECOMMENDATION: <one short phrase>
RISK: <one short sentence>
"""
    device = "cuda" if (LLM_DEVICE=="cuda" and torch.cuda.is_available()) else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    gen_kwargs = dict(max_new_tokens=LLM_MAX_TOKENS, temperature=0.0, do_sample=False)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    direction="NEUTRAL"; summary=""; recommendation=""; risk=""
    for line in raw.splitlines():
        ln=line.strip()
        if not ln: continue
        up=ln.upper()
        if up.startswith("DIRECTION:"):
            v=ln.split(":",1)[1].strip().upper()
            if v in ("UP","DOWN","NEUTRAL"): direction=v
        elif up.startswith("SUMMARY:"):
            summary=ln.split(":",1)[1].strip()
        elif up.startswith("RECOMMENDATION:"):
            recommendation=ln.split(":",1)[1].strip()
        elif up.startswith("RISK:"):
            risk=ln.split(":",1)[1].strip()
    if direction=="NEUTRAL":
        lraw=raw.lower()
        if any(x in lraw for x in ["likely to rise","bullish","upward","positive","buy"]): direction="UP"
        if any(x in lraw for x in ["likely to fall","bearish","downward","negative","sell"]): direction="DOWN"
    return {"raw_text": raw, "direction": direction, "summary": summary or _shorten(raw,300), "recommendation": recommendation, "risk": risk}


def generate_advice_for_query(
    user_query: str,
    ticker: str,
    index_name: str,
    features_df: Optional[pd.DataFrame] = None,
    k: int = RAG_TOP_K
):
    # 1) Fetch relevant news
    news = fetch_news_for_index(index_name=index_name, user_query=user_query, k=k)

    # 2) Default ML values
    ml_label = "Unknown"
    ml_conf = 0.0

    # 3) If frontend provided feature_df → use old ML prediction
    if features_df is not None:
        try:
            dirv, prob = predict_ml(features_df)
            ml_label = "UP" if dirv == 1 else "DOWN"
            ml_conf = float(prob)
        except Exception as e:
            print("Direct ML prediction error:", e)
            ml_label = "Unknown"
            ml_conf = 0.0
    else:
        # 4) Otherwise use automatic ML
        try:
            from .ml_predictor import run_ml_prediction
            ml_res = run_ml_prediction(ticker)
            ml_label = ml_res.get("direction", "Unknown")
            ml_conf = float(ml_res.get("probability", 0.0))
        except Exception as e:
            print("Auto ML error:", e)
            ml_label = "Unknown"
            ml_conf = 0.0

    # 5) Call LLM with correct parameter names
    llm_result = synthesize_advice(
        prediction_label=ml_label,
        confidence=ml_conf,
        news_docs=news,
        user_query=user_query
    )

    return {
        "news": news,
        "ml_label": ml_label,
        "ml_confidence": ml_conf,
        "llm": llm_result
    }


