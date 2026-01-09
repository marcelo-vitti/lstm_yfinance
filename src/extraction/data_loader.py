import pandas as pd

from src.utils.config import SYMBOL

url = f"https://stooq.com/q/d/l/?s={SYMBOL}&i=d"

df = pd.read_csv(url)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

df.to_csv(f"data/raw/{SYMBOL}.csv")
# python -m src.extraction.test
