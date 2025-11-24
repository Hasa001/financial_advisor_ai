import pandas as pd
import yfinance as yf
import feedparser
import ta
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

    preprocess_stock_data(ticker)
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
    df_news.to_csv("data/news_raw.csv",index=False)
    return f"RSS feed data saved to data/news_raw.csv"

def preprocess_stock_data(ticker):
    """
    Preprocess stock data for machine learning.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    pd.DataFrame: A DataFrame containing the preprocessed stock data.
    """
    df = pd.read_csv(f"data/{ticker}_data.csv", skiprows=2)
    df.columns = ["date", "close", "high", "low", "open", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.reset_index(drop=True, inplace=True)
    df["return_1d"] = df["close"].pct_change()
    df["return_7d"] = df["close"].pct_change(7)

    # RSI
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # EMA
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()

    df["target_return_1d"] = df["close"].pct_change().shift(-1)
    df["target_direction"] = (df["target_return_1d"] > 0).astype(int)
    df = df.dropna().reset_index(drop=True)

    df.to_csv(f"data/{ticker}_data.csv", index=False)

    return f"Preprocessed data for {ticker} saved to data/{ticker}_ml_ready.csv"

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
    





 