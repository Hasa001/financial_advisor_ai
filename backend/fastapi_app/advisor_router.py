# backend/fastapi_app/advisor_router.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
from .identify_index import detect_index
from .chart_generator import generate_chart_bytes, fetch_ohlcv
from .preprocess import preprocess_input
from .advisor_service import generate_advice_for_query, fetch_news_for_index, fetch_relevant_news_text
from .ticker_db import CANONICAL

router = APIRouter()

@router.post("/identify")
def identify(req: dict):
    q = req.get("query","")
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    ticker, matched, score, method = detect_index(q)
    return {"query": q, "ticker": ticker, "matched": matched, "score": score, "method": method}

@router.get("/chart_by_choice")
def chart_by_choice(choice: str = Query(...), period: str = Query("1mo")):
    key = choice.lower().strip()
    ticker = CANONICAL.get(key)
    if not ticker:
        t,_,_,_ = detect_index(choice)
        ticker = t
    if not ticker:
        return JSONResponse({"error":"Cannot map choice to ticker"}, status_code=400)
    try:
        buf = generate_chart_bytes(ticker, period=period)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": f"Chart generation failed: {e}"}, status_code=500)

@router.get("/news_by_choice")
def news_by_choice(choice: str = Query(...), k: int = Query(6)):
    q = f"{choice} market RBI monetary policy"
    try:
        docs = fetch_relevant_news_text(q, k=k)
        return {"choice": choice, "news": docs}
    except Exception as e:
        return JSONResponse({"error": f"News fetch failed: {e}"}, status_code=500)

@router.post("/ask")
def ask(req: dict):
    q = req.get("query","")
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    ticker, matched, score, method = detect_index(q)
    if not ticker:
        return JSONResponse({"error":"Could not detect index from query."}, status_code=400)
    index_name = matched.upper() if matched else ticker
    try:
        df = fetch_ohlcv(ticker)
        latest = df.iloc[-1]
        daily_change_pct = float((latest["Close"]-latest["Open"])/latest["Open"])
    except Exception as e:
        return JSONResponse({"error": f"Live data fetch failed: {e}"}, status_code=500)

    # optional features for ML
    features_df = None
    ml_prediction=None; ml_conf=None
    if "features" in req:
        try:
            df_feat = pd.DataFrame([req["features"]])
            processed = preprocess_input(df_feat)
            features_df = processed
        except Exception as e:
            return JSONResponse({"error": f"Feature preprocessing failed: {e}"}, status_code=400)

    try:
        result = generate_advice_for_query(user_query=q, ticker=ticker, index_name=index_name, features_df=features_df, k=int(req.get("k",6)))
    except Exception as e:
        return JSONResponse({"error": f"Generation failed: {e}"}, status_code=500)

    return {
        "index_detected": index_name,
        "ticker": ticker,
        "detection_method": method,
        "detection_confidence": score,
        "latest_close": float(latest["Close"]),
        "daily_change_pct": daily_change_pct,
        "ml_label": result.get("ml_label"),
        "ml_confidence": result.get("ml_confidence"),
        "news": result.get("news"),
        "llm": result.get("llm")
    }
