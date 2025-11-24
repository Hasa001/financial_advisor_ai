import requests

url = "http://127.0.0.1:8000/api/advisor"

payload = {
  "ticker": "^BSESN",
  "features": {
    "open": 41634.51,
    "high": 41649.28,
    "low": 41328.44,
    "close": 41626.64,
    "volume": 5300,

    "rsi_14": 45.3,
    "ema_20": 41222.12,
    "ema_50": 40500.55,

    "macd": 30.5,
    "macd_signal": 28.3,
    "macd_hist": 2.2,

    "bb_high": 42000.5,
    "bb_mid": 41500.3,
    "bb_low": 41000.1,

    "return_1d": 0.0023,
    "return_7d": 0.0134,
    "ticker_id": 1,

    "close_lag_1": 41500.22,
    "close_lag_2": 41480.10,
    "close_lag_3": 41390.50,
    "close_lag_5": 41200.75,

    "volume_lag_1": 4700,
    "volume_lag_2": 4500,
    "volume_lag_3": 6000,
    "volume_lag_5": 5100,

    "roll_mean_3": 41400.2,
    "roll_mean_5": 41350.9,
    "roll_mean_7": 41310.5,
    "roll_mean_10": 41280.1,
    "roll_mean_14": 41255.8,

    "roll_std_3": 150.2,
    "roll_std_5": 180.5,
    "roll_std_7": 200.3,
    "roll_std_10": 210.1,
    "roll_std_14": 220.9,

    "roll_vol_3": 0.009,
    "roll_vol_5": 0.011,
    "roll_vol_7": 0.012,
    "roll_vol_10": 0.013,
    "roll_vol_14": 0.014,

    "roc_5": 0.004,
    "roc_10": 0.007,
    "momentum": 120,
    "adx_14": 25.9,
    "wma_20": 41200.4
  },
  "user_query": "What is the short-term outlook for Indian markets?"
}


res = requests.post(url, json=payload)
print(res.json())
