# backend/fastapi_app/rss_fetcher.py
import feedparser, re
from typing import List, Dict
from datetime import datetime
import time
import os
RSS_FEEDS =  [

    # ---------------- INDIA: TOP FINANCIAL SOURCES ----------------
    # Economic Times
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://economictimes.indiatimes.com/markets/forex/rssfeeds/15619437.cms",
    "https://economictimes.indiatimes.com/markets/commodities/rssfeeds/2146841.cms",
    "https://economictimes.indiatimes.com/wealth/rssfeeds/837555174.cms",
    "https://economictimes.indiatimes.com/news/rssfeeds/1715249553.cms",

    # Moneycontrol
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    "https://www.moneycontrol.com/rss/sensex.xml",
    "https://www.moneycontrol.com/rss/nifty.xml",
    "https://www.moneycontrol.com/rss/technicals.xml",
    "https://www.moneycontrol.com/rss/business.xml",
    "https://www.moneycontrol.com/rss/economy.xml",
    "https://www.moneycontrol.com/rss/worldmarkets.xml",

    # LiveMint
    "https://www.livemint.com/rss/markets",
    "https://www.livemint.com/rss/money",
    "https://www.livemint.com/rss/industry",
    "https://www.livemint.com/rss/opinion",

    # Business Standard
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.business-standard.com/rss/finance-102.rss",
    "https://www.business-standard.com/rss/economy-policy-101.rss",
    "https://www.business-standard.com/rss/companies-104.rss",

    # NDTV Business
    "https://www.ndtv.com/business/rss/business",
    "https://www.ndtv.com/business/rss/markets",
    "https://www.ndtv.com/business/rss/latest",

    # The Hindu Business Line
    "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "https://www.thehindubusinessline.com/economy/feeder/default.rss",
    "https://www.thehindubusinessline.com/portfolio/feeder/default.rss",

    # Financial Express
    "https://www.financialexpress.com/feed/",
    "https://www.financialexpress.com/market/feed/",
    "https://www.financialexpress.com/economy/feed/",

    # Business Today
    "https://www.businesstoday.in/rssfeeds/?id=86",
    "https://www.businesstoday.in/rssfeeds/?id=93",
    "https://www.businesstoday.in/rssfeeds/?id=99",

    # The Hindu - Business
    "https://www.thehindu.com/business/feeder/default.rss",
    "https://www.thehindu.com/business/Economy/feeder/default.rss",
    "https://www.thehindu.com/business/markets/feeder/default.rss",

    # Deccan Herald
    "https://www.deccanherald.com/rss/business",
    "https://www.deccanherald.com/rss/business/economy",
    "https://www.deccanherald.com/rss/business/markets",

    # India Today
    "https://www.indiatoday.in/rss/1206514",

    # Times of India Business
    "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",


    # ---------------- INTERNATIONAL MARKET SOURCES ----------------

    # Reuters India / World Markets
    "https://www.reuters.com/finance/rss",
    "https://www.reuters.com/business/finance/rss",
    "https://www.reuters.com/markets/rss",
    "https://www.reuters.com/world/india/rss",

    # Bloomberg (RSSHub mirrors)
    "https://rsshub.app/bloomberg/markets",
    "https://rsshub.app/bloomberg/economics",
    "https://rsshub.app/bloomberg/opinion",

    # CNBC
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://www.cnbc.com/id/19746125/device/rss/rss.html",
    "https://www.cnbc.com/id/20409666/device/rss/rss.html",

    # Financial Times (FT Markets)
    "https://www.ft.com/markets/rss",
    "https://www.ft.com/companies/rss",
    "https://www.ft.com/world/rss",

    # WSJ
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",

    # MarketWatch
    "https://www.marketwatch.com/rss/topstories",
    "https://www.marketwatch.com/rss/marketpulse",

    # Yahoo Finance
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NSEI&region=IN&lang=en-IN",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^BSESN&region=IN&lang=en-IN",
    "https://finance.yahoo.com/news/rss",

    # Investing.com (RSSHub)
    "https://rsshub.app/investing/stock/india",
    "https://rsshub.app/investing/indices/sensex",
    "https://rsshub.app/investing/indices/nifty-50",

    # CNN Business
    "http://rss.cnn.com/rss/money_latest.rss",

    # NYT Business
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",

    # The Guardian
    "https://www.theguardian.com/world/economy/rss",
    "https://www.theguardian.com/business/rss",
]

def fetch_rss_entries() -> List[Dict]:
    articles=[]
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                articles.append({
                    "title": e.get("title",""),
                    "summary": e.get("summary",""),
                    "link": e.get("link",""),
                    "published": e.get("published","")
                })
        except Exception as e:
            print("RSS parse error", e)
    return articles

def filter_rss_by_query(entries: List[Dict], user_query: str, top_k: int = 6) -> List[Dict]:
    words = set(re.findall(r"\w+", (user_query or "").lower()))
    scores=[]
    for e in entries:
        text = (e.get("title","")+" "+e.get("summary","")).lower()
        score = sum(1 for w in words if w in text)
        if score>0:
            scores.append((score,e))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [e for s,e in scores[:top_k]]
