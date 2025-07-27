import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


def prepare_lstm_data(df, n_steps=30):
    """
    Prepares the data for LSTM model training.
    :param df: DataFrame with columns ['ds', 'y']
    :param n_steps: Number of past days to use for prediction
    :return: X, y arrays for training and testing, scaled df, and scaler
    """
    df = df.copy()
    df = df.sort_values("ds")
    df = df.set_index("ds")

    scaler = MinMaxScaler()
    df["y_scaled"] = scaler.fit_transform(df[["y"]])

    X, y = [], []
    for i in range(n_steps, len(df)):
        X.append(df["y_scaled"].values[i - n_steps:i])
        y.append(df["y_scaled"].values[i])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, df, scaler


def train_lstm_model(X, y):
    """
    Builds and trains a simple LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    return model


def predict_future_sales(model, df, scaler, n_future=30, n_steps=30):
    """
    Predicts future sales using the trained LSTM model.
    :param model: Trained LSTM model
    :param df: Scaled dataframe with 'y_scaled'
    :param scaler: Scaler used for inverse transform
    :param n_future: Days to forecast
    :param n_steps: Number of steps used for training
    """
    last_sequence = df["y_scaled"].values[-n_steps:]
    future_predictions = []

    for _ in range(n_future):
        input_seq = np.reshape(last_sequence, (1, n_steps, 1))
        predicted = model.predict(input_seq, verbose=0)
        future_predictions.append(predicted[0][0])
        last_sequence = np.append(last_sequence[1:], predicted[0][0])

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_future)
    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat_lstm": scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    })

    return forecast
