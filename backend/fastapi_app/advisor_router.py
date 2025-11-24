# advisor_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from .preprocess import preprocess_input
from .advisor_service import (
    get_embedding_model, embed_texts_pyfloat, predict_ml,
    fetch_relevant_news, synthesize_advice, get_search_client
)
from .query_embed import get_embedding_model as _get_embedding_model

router = APIRouter()

class AdvisorRequest(BaseModel):
    # include only the raw features that your preprocess expects
    # keep same fields you used for PredictionRequest earlier
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    # ... add all engineered features you require OR accept a preprocessed payload
    # to simplify, we accept a preprocessed row as mapping
    features: dict
    user_query: str = None

class AdvisorResponse(BaseModel):
    prediction: str
    direction: int
    confidence: float
    news: list
    advice: str

@router.post("/advisor", response_model=AdvisorResponse)
def advisor_endpoint(req: AdvisorRequest):
    # Build DataFrame from features (client must send all required feature columns)
    df = pd.DataFrame([req.features])
    # Preprocess (reorders columns and validates)
    try:
        processed = preprocess_input(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocess error: {e}")

    # ML predict
    direction, confidence = predict_ml(processed)
    prediction = "UP" if direction == 1 else "DOWN"

    # create query vector (embed user_query OR content of ticker as fallback)
    embed_model = _get_embedding_model()
    query_text = req.user_query if req.user_query else f"{req.ticker} market news"
    query_vector = embed_texts_pyfloat(embed_model, [query_text])[0]

    # fetch news
    news_docs = fetch_relevant_news(query_vector, k=5)

    # synthesize advice
    advice = synthesize_advice(prediction, confidence, news_docs, req.user_query)

    return AdvisorResponse(
        prediction=prediction,
        direction=direction,
        confidence=confidence,
        news=news_docs,
        advice=advice
    )
