import numpy as np
import joblib

from tensorflow.keras.models import load_model

from src.config import LOOKBACK


MODEL_PATH = "models/lstm_visa_vus.h5"
SCALER_PATH = "data/trusted/scaler.pkl"
METADATA_PATH = "models/metadata.pkl"


model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)


def closes_to_log_returns(closes: list[float]) -> np.ndarray:
    closes = np.array(closes, dtype=np.float32)
    return np.log(closes[1:] / closes[:-1])


def predict_from_close(closes: list[float]) -> tuple[float, float]:

    if len(closes) != LOOKBACK + 1:
        raise ValueError(f"Expected {LOOKBACK + 1} close prices, got {len(closes)}")

    log_returns = closes_to_log_returns(closes)

    x = log_returns.reshape(-1, 1)
    x_scaled = scaler.transform(x)
    X = x_scaled.reshape(1, LOOKBACK, 1)
    y_scaled = model.predict(X, verbose=0)

    predicted_log_return = scaler.inverse_transform(y_scaled)[0][0]
    last_price = closes[-1]

    predicted_price = last_price * np.exp(predicted_log_return)

    return float(predicted_log_return), float(predicted_price)
