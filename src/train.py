import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    SYMBOL,
    FEATURES,
    TRAIN_SPLIT,
    LOOKBACK,
    TARGET_COL,
    EPOCHS,
    BATCH_SIZE,
    SEED,
    VAL_SPLIT,
)
from src.dataset import create_sequences
from src.model import build_lstm
from src.preprocessing import scale_train_test
from src.tests.validation import validate_raw_data


df = pd.read_csv(f"data/raw/{SYMBOL}.csv", index_col=0, parse_dates=True)

validate_raw_data(df)

df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna().reset_index(drop=True)
print("📋 Dados carregados:")
df = df[FEATURES].dropna()
print(df.head())
print(df["log_return"].describe())

split_idx = int(len(df) * TRAIN_SPLIT)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

train_scaled, test_scaled, scaler = scale_train_test(train_df, test_df)

X_train, y_train = create_sequences(train_scaled, LOOKBACK, TARGET_COL)
X_test, y_test = create_sequences(test_scaled, LOOKBACK, TARGET_COL)

split_idx = int(len(X_train) * (1 - VAL_SPLIT))

X_tr = X_train[:split_idx]
y_tr = y_train[:split_idx]

X_val = X_train[split_idx:]
y_val = y_train[split_idx:]

np.random.seed(SEED)
tf.random.set_seed(SEED)

model = build_lstm((LOOKBACK, X_train.shape[2]))

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

y_pred_scaled = model.predict(X_test)
print(f"Predictions (scaled): {y_pred_scaled[:5]}")


def inverse_close(y_scaled, scaled_data, scaler, target_col):
    dummy = np.zeros((len(y_scaled), scaled_data.shape[1]))
    dummy[:, target_col] = y_scaled[:, 0]
    return scaler.inverse_transform(dummy)[:, target_col]


y_test_real = inverse_close(
    y_test.reshape(-1, 1), test_scaled[LOOKBACK:], scaler, TARGET_COL
)

y_pred_real = inverse_close(y_pred_scaled, test_scaled[LOOKBACK:], scaler, TARGET_COL)

y_val_pred_scaled = model.predict(X_val)

val_scaled = train_scaled[LOOKBACK + split_idx :]


y_val_real = inverse_close(y_val.reshape(-1, 1), val_scaled, scaler, TARGET_COL)

y_val_pred_real = inverse_close(y_val_pred_scaled, val_scaled, scaler, TARGET_COL)

mae = mean_absolute_error(y_val_real, y_val_pred_real)

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

mape = np.mean(np.abs((y_val_real - y_val_pred_real) / y_val_real)) * 100

print("📊 Avaliação no conjunto de validação")
print(f"MAE : {mae} %")
print(f"RMSE: {rmse} %")
print(y_val.min(), y_val.max())
print(y_val.std())
print(f"MAPE: {mape:.2f} %")

naive_pred = y_val_real[:-1]
naive_true = y_val_real[1:]

naive_rmse = np.sqrt(mean_squared_error(naive_true, naive_pred))
print(f"Naive RMSE: {naive_rmse:.2f}")

model.save("models/lstm_visa_vus.h5")

joblib.dump(scaler, "data/trusted/scaler.pkl")
