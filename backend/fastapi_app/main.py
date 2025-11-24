# main.py
from fastapi import FastAPI
from .model_loader import model, feature_cols
from .preprocess import preprocess_input
from .advisor_router import router as advisor_router

app = FastAPI(title="Financial Advisor AI", version="1.0")

app.include_router(advisor_router, prefix="/api")

@app.get("/")
def root():
    return {"status":"ok", "service":"financial-advisor-ai backend"}
