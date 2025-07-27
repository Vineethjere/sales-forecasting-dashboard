import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def run_prophet(df, product_name, future_periods=30):
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])

    df = df[df['Product_Name'] == product_name]
    daily_sales = df.groupby('Order_Date')['Sales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']

    if len(daily_sales) < 2:
        print(f"Not enough data for {product_name}")
        return None, None, None

    model = Prophet()
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)

    # Metrics
    actual = daily_sales['y'].values
    predicted = model.predict(daily_sales)['yhat'].values
    mape = mean_absolute_percentage_error(actual, predicted) * 100

    # Plotting
    fig = plt.figure(figsize=(10, 4))
    model.plot(forecast, ax=fig.gca())
    plt.title(f"{product_name} - Sales Forecast (Prophet)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()

    forecast_out = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
    return fig, {'mape': mape}, forecast_out