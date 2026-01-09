import numpy as np


def create_sequences(data, lookback, target_col):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(data[i, target_col])
    return np.array(X), np.array(y)
