# backend/fastapi_app/ticker_db.py
INDEXES = {
    "sensex": {"ticker": "^BSESN", "aliases": ["sensex","bse","bse sensex","bombay stock exchange","bombay"]},
    "nifty": {"ticker": "^NSEI", "aliases": ["nifty","nse","nifty 50","nse nifty"]},
    "bank nifty": {"ticker": "^NSEBANK", "aliases": ["bank nifty","banknifty"]},
    "bse": {"ticker": "^BSESN", "aliases": ["bse"]},
    "nse": {"ticker": "^NSEI", "aliases": ["nse"]},
}

CANONICAL = {}
for k,v in INDEXES.items():
    CANONICAL[k] = v["ticker"]
    for a in v["aliases"]:
        CANONICAL[a] = v["ticker"]

CANDIDATES = list(CANONICAL.keys())
