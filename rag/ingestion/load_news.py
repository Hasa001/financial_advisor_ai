# import pandas as pd

# def load_news_csv(path: str):
#     """
#     CSV must contain: title, description, url, date
#     """
#     df = pd.read_csv(path)
#     df = df.dropna(subset=["title", "summary"])
#     df["content"] = df["title"] + " - " + df["summary"]
#     return df

import pandas as pd

def load_news_csv(path: str):
    df = pd.read_csv(path)

    # Remove rows missing title or description
    df = df.dropna(subset=["title", "summary"]).copy()

    # Reset index to ensure 0..N-1 continuous
    df = df.reset_index(drop=True)

    # Create content field
    df["content"] = df["title"] + " - " + df["summary"]

    return df

