import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm(input_shape, units, dropout, learning_rate):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(units, input_shape=input_shape),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model
