# backend/fastapi_app/main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from .advisor_router import router as advisor_router

app = FastAPI(title="Financial Advisor AI", version="1.0")
app.include_router(advisor_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok", "service": "financial-advisor-ai backend"}
