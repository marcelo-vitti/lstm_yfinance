import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    SYMBOL,
    FEATURES,
    TRAIN_SPLIT,
    TARGET_COL,
    EPOCHS,
    SEED,
    VAL_SPLIT,
)
from src.dataset import create_sequences
from src.model import build_lstm
from src.preprocessing import scale_train_test
from src.tests.validation import validate_raw_data

param_grid = {
    "lookback": [20, 30, 60],
    "units": [32, 64, 128],
    "dropout": [0.1, 0.2, 0.3],
    "batch_size": [16, 32],
    "learning_rate": [1e-3, 1e-4],
}

df = pd.read_csv(f"data/raw/{SYMBOL}.csv", index_col=0, parse_dates=True)
validate_raw_data(df)

df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna().reset_index(drop=True)
df = df[FEATURES]

print(df.head())
print(df["log_return"].describe())

split_idx = int(len(df) * TRAIN_SPLIT)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

train_scaled, test_scaled, scaler = scale_train_test(train_df, test_df)

np.random.seed(SEED)
tf.random.set_seed(SEED)

results = []

for lookback in param_grid["lookback"]:

    X_train, y_train = create_sequences(train_scaled, lookback, TARGET_COL)

    X_test, y_test = create_sequences(test_scaled, lookback, TARGET_COL)

    split_val = int(len(X_train) * (1 - VAL_SPLIT))

    X_tr, y_tr = X_train[:split_val], y_train[:split_val]
    X_val, y_val = X_train[split_val:], y_train[split_val:]

    for units in param_grid["units"]:
        for dropout in param_grid["dropout"]:
            for lr in param_grid["learning_rate"]:
                for batch_size in param_grid["batch_size"]:

                    print(
                        f"Training | lb={lookback} u={units} "
                        f"d={dropout} lr={lr} bs={batch_size}"
                    )

                    model = build_lstm(
                        input_shape=(lookback, X_train.shape[2]),
                        units=units,
                        dropout=dropout,
                        learning_rate=lr,
                    )

                    model.fit(
                        X_tr,
                        y_tr,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=batch_size,
                        verbose=0,
                    )

                    y_val_pred = model.predict(X_val)

                    mae = mean_absolute_error(y_val, y_val_pred)

                    results.append(
                        {
                            "lookback": lookback,
                            "units": units,
                            "dropout": dropout,
                            "learning_rate": lr,
                            "batch_size": batch_size,
                            "val_mae": mae,
                        }
                    )

results_df = pd.DataFrame(results)
best = results_df.sort_values("val_mae").iloc[0]

print("🏆 Best configuration:")
print(best)

best_lookback = int(best.lookback)

X_train_full, y_train_full = create_sequences(train_scaled, best_lookback, TARGET_COL)

X_test, y_test = create_sequences(test_scaled, best_lookback, TARGET_COL)

model = build_lstm(
    input_shape=(best_lookback, X_train_full.shape[2]),
    units=int(best.units),
    dropout=float(best.dropout),
    learning_rate=float(best.learning_rate),
)

model.fit(
    X_train_full,
    y_train_full,
    epochs=EPOCHS,
    batch_size=int(best.batch_size),
    verbose=1,
)

y_pred_scaled = model.predict(X_test)


def inverse_close(y_scaled, scaled_data, scaler, target_col):
    dummy = np.zeros((len(y_scaled), scaled_data.shape[1]))
    dummy[:, target_col] = y_scaled[:, 0]
    return scaler.inverse_transform(dummy)[:, target_col]


y_test_real = inverse_close(
    y_test.reshape(-1, 1),
    test_scaled[best_lookback:],
    scaler,
    TARGET_COL,
)

y_pred_real = inverse_close(
    y_pred_scaled,
    test_scaled[best_lookback:],
    scaler,
    TARGET_COL,
)

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
print(f"RMSE (test): {rmse:.4f}")

naive_pred = y_test_real[:-1]
naive_true = y_test_real[1:]

naive_rmse = np.sqrt(mean_squared_error(naive_true, naive_pred))
print(f"Naive RMSE: {naive_rmse:.4f}")

model.save("models/lstm_visa_vus.h5")
joblib.dump(scaler, "data/trusted/scaler.pkl")

results_df.to_csv("models/grid_results.csv", index=False)
