import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

def run_lstm_forecast(df, product_name, epochs=20):
    df = df.copy()
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df = df[df['Product_Name'] == product_name]

    df = df.groupby('Order_Date').agg({'Sales': 'sum'}).reset_index()
    df = df.set_index('Order_Date').resample('D').sum().fillna(0)

    if len(df) < 60:
        print(f"Not enough data for '{product_name}' to forecast.")
        return None, None, None

    data = df['Sales'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    window_size = 30
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)

    # Predict future
    future_steps = 10
    input_seq = data_scaled[-window_size:]
    predictions = []

    for _ in range(future_steps):
        pred = model.predict(input_seq.reshape(1, window_size, 1), verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_steps)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

    # Evaluate on historical data
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Sales'], label='Historical Sales')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='LSTM Forecast', color='orange')
    plt.title(f"{product_name} - Sales Forecast (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()

    return fig, {'mape': mape}, forecast_df