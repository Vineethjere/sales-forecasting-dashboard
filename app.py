# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from forecast_with_prophet import run_prophet
from lstm_forecast import run_lstm_forecast

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# --- Theme Toggle ---
theme_toggle = st.sidebar.radio("🌓 Theme", ["Dark", "Light"])
if theme_toggle == "Light":
    page_bg = "#ffffff"
    text_color = "#000000"
    card_bg = "#f0f2f6"
    info_bg = "#e1f0ff"
    warn_bg = "#fff4e6"
    df_bg = "#ffffff"
    df_text = "#000000"
else:
    page_bg = "#0e1117"
    text_color = "#f1f1f1"
    card_bg = "#262730"
    info_bg = "#173b5a"
    warn_bg = "#4d3b00"
    df_bg = "#1e1e1e"
    df_text = "#f1f1f1"

# --- Theme Styling ---
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {page_bg};
            color: {text_color};
        }}
        .stDownloadButton > button {{
            background-color: {card_bg};
            color: {text_color};
            border-radius: 6px;
            padding: 0.5em 1em;
            transition: 0.2s;
        }}
        .stDownloadButton > button:hover {{
            background-color: #6c9df3;
            color: white;
        }}
        .metric-box {{
            background-color: {card_bg};
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }}
        .custom-info {{
            background-color: {info_bg};
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .custom-warning {{
            background-color: {warn_bg};
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .block-title {{
            font-size: 22px;
            font-weight: 600;
            margin-top: 20px;
        }}
        section.main > div {{
            padding: 1.5rem;
        }}
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown(f"<h1 style='color:{text_color};'>📦 Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{text_color};'>Forecast future sales by SKU using <strong>Prophet</strong> and <strong>LSTM</strong>.</p>", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/SampleSuperstore.csv", encoding="ISO-8859-1", parse_dates=['Order Date'])
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df = df.rename(columns={"Order_Date": "Order_Date", "Product_Name": "Product_Name"})
    return df

df = load_data()

# --- Sidebar Options ---
st.sidebar.header("⚙️ Model Configuration")
product_list = df['Product_Name'].unique()
selected_product = st.sidebar.selectbox("🔎 Select Product (SKU)", product_list)

date_min = df['Order_Date'].min()
date_max = df['Order_Date'].max()

start_date = st.sidebar.date_input("📅 Start Date", date_min)
end_date = st.sidebar.date_input("📅 End Date", date_max)

# Model Hyperparameters
epochs = st.sidebar.slider("🔁 LSTM Epochs", min_value=5, max_value=100, value=20, step=5)
future_periods = st.sidebar.slider("📆 Prophet Forecast Days", min_value=7, max_value=180, value=30, step=7)

# --- Filter Data ---
filtered_df = df[(df['Product_Name'] == selected_product) &
                 (df['Order_Date'] >= pd.to_datetime(start_date)) &
                 (df['Order_Date'] <= pd.to_datetime(end_date))]

# --- Validate Data ---
st.markdown(f"<p style='color: {text_color};'>✅ Number of records for <strong>{selected_product}</strong>: <code>{len(filtered_df)}</code></p>", unsafe_allow_html=True)

if len(filtered_df) < 2:
    st.markdown("<div class='custom-warning'>⚠️ Not enough data for this selection. Try another product or a wider date range.</div>", unsafe_allow_html=True)
else:
    tab1, tab2 = st.tabs(["📈 Prophet Forecast", "🔁 LSTM Forecast"])

    with tab1:
        st.subheader("📈 Prophet Forecast")
        if len(filtered_df) < 20:
            st.markdown("<div class='custom-info'>ℹ️ Prophet may produce unreliable results with fewer than 20 data points.</div>", unsafe_allow_html=True)
        else:
            fig1, prophet_metrics, forecast_df = run_prophet(filtered_df, selected_product, future_periods)
            if fig1:
                st.pyplot(fig1)
                st.markdown(f"<div class='metric-box'>📊 <strong>Prophet MAPE:</strong> <code>{prophet_metrics['mape']:.2f}%</code></div>", unsafe_allow_html=True)
                st.download_button("📥 Download Prophet Forecast", forecast_df.to_csv(index=False), file_name="prophet_forecast.csv")

    with tab2:
        st.subheader("🔁 LSTM Forecast")
        if len(filtered_df) < 60:
            st.markdown("<div class='custom-info'>ℹ️ LSTM may produce unreliable results. Minimum 60 data points recommended.</div>", unsafe_allow_html=True)
        else:
            fig2, lstm_metrics, forecast_df = run_lstm_forecast(filtered_df, selected_product, epochs)
            if fig2:
                st.pyplot(fig2)
                mape_val = lstm_metrics['mape']
                if mape_val < 1e5:
                    st.markdown(f"<div class='metric-box'>📊 <strong>LSTM MAPE:</strong> <code>{mape_val:.2f}%</code></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='custom-warning'>⚠️ LSTM MAPE not reliable due to insufficient data.</div>", unsafe_allow_html=True)
                st.download_button("📥 Download LSTM Forecast", forecast_df.to_csv(index=False), file_name="lstm_forecast.csv")

    # --- Show Raw Data ---
    with st.expander("📄 Show Raw Daily Sales Data"):
        st.dataframe(filtered_df.style.set_properties(
            **{'background-color': df_bg, 'color': df_text}))
