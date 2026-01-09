SYMBOL = "V.US"
START_DATE = "2008-01-01"
END_DATE = "2025-01-01"

FEATURES = ["Open", "High", "Low", "Close", "Volume", "log_return"]
TARGET_COL = FEATURES.index("log_return")

LOOKBACK = 30
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

EPOCHS = 50
BATCH_SIZE = 32
SEED = 42
