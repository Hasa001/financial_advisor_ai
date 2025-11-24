import pandas as pd
import yfinance as yf
import feedparser
import ta
import numpy as np
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker symbol from Yahoo Finance.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data.
    """
    df = yf.download(f"^{ticker}",start=start_date, end=end_date)

    df.to_csv(f"data/{ticker}_data.csv")

    preprocess_for_model(ticker)
    return f"Data for {ticker} from {start_date} to {end_date} saved to data/{ticker}_data.csv"

def fetch_rss_feed(feed_urls):
    """
    Fetch and parse an RSS feed.

    Parameters:
    feed_url (str): The URL of the RSS feed.

    Returns:
    feedparser.FeedParserDict: The parsed RSS feed.
    """
    articles = []
    for url in feed_urls:
        feed= feedparser.parse(url)
        
        for entry in feed.entries:
            print(f"Fetching feed from---------------: {entry}")
            articles.append({
                'title': entry.title,
                'link': entry.link,
                'published': entry.published,
                'summary': entry.summary
            })
    df_news = pd.DataFrame(articles)
    df_news.to_csv("data/news.csv",index=False)
    return f"RSS feed data saved to data/news.csv"

def preprocess_for_model(df, ticker_name="SENSEX"):
    """
    df must already contain: date, open, high, low, close, volume
    Returns a single-row dataframe with ALL features required by ML model.
    """

    df = df.copy()

    # Base returns
    df["return_1d"] = df["close"].pct_change()
    df["return_7d"] = df["close"].pct_change(7)

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

    # Rolling means
    for w in [3,5,7,10,14]:
        df[f"roll_mean_{w}"] = df["close"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["close"].rolling(w).std()
        df[f"roll_vol_{w}"] = df["volume"].rolling(w).std()

    # ROC
    df["roc_5"] = df["close"].pct_change(5)
    df["roc_10"] = df["close"].pct_change(10)

    # Momentum
    df["momentum"] = df["close"] - df["close"].shift(10)

    # Technical Indicators
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()

    # ADX
    df["adx_14"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()

    # Weighted MA
    df["wma_20"] = df["close"].rolling(20).apply(lambda x: (x * np.arange(1,21)).sum() / np.arange(1,21).sum(), raw=True)

    # ticker_id (encode manually)
    df["ticker_id"] = hash(ticker_name) % 1000

    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Example usage
    ticker1 = "NSEI"
    ticker2= "NSEBANK"
    ticker3= "BSESN"
    ticker_list = [ticker1, ticker2, ticker3]
    start_date = "2010-01-01"
    end_date = "2025-12-31"
    urls = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://economictimes.indiatimes.com/markets/forex/rssfeeds/15619437.cms",
    "https://economictimes.indiatimes.com/markets/commodities/rssfeeds/2146841.cms",

    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    "https://www.moneycontrol.com/rss/sensex.xml",
    "https://www.moneycontrol.com/rss/nifty.xml",
    "https://www.moneycontrol.com/rss/technicals.xml",

    "https://www.livemint.com/rss/markets",
    "https://www.livemint.com/rss/money",
    "https://www.livemint.com/rss/industry",

    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.business-standard.com/rss/finance-102.rss",
    "https://www.business-standard.com/rss/economy-policy-101.rss",

    "https://www.ndtv.com/business/rss/business",
    "https://www.ndtv.com/business/rss/markets",

    "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "https://www.thehindubusinessline.com/economy/feeder/default.rss",

    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NSEI&region=IN&lang=en-IN",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=RELIANCE.NS&region=IN&lang=en-IN",

    "https://www.deccanherald.com/rss/business",

    # Global / Optional
    "https://rsshub.app/bloomberg/markets",
    "https://rsshub.app/investing/stock/india",
    "https://www.ft.com/markets/rss",
    "https://www.cnbc.com/id/15839069/device/rss/rss.html"
    ]
    # fetch_stock_data(ticker1, start_date, end_date)
    # fetch_stock_data(ticker2, start_date, end_date)
    # fetch_stock_data(ticker3, start_date, end_date)
    output= map(fetch_stock_data, ticker_list, [start_date]*3, [end_date]*3)
    for out in output:
        print(out)
    fetch_rss_feed(urls)
    





 