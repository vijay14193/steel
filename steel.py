import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import csv

# Step 1: Download steel price data
def fetch_steel_price_data():
    ticker = "SLX"  # Steel ETF
    end_date = datetime.datetime.today()
    data = yf.download(ticker, start="2010-01-01", end=end_date)
    df = data[['Close']].rename(columns={'Close': 'Steel_Price'})
    df.reset_index(inplace=True)
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/steel_data.csv", index=False)
    return df

# Step 2: Train LSTM model
def train_lstm(df):
    df = df[pd.to_numeric(df['Steel_Price'], errors='coerce').notnull()]
    df['Steel_Price'] = df['Steel_Price'].astype(float)

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Steel_Price']])
    joblib.dump(scaler, 'scaler.save')

    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=3)])
    model.save('lstm_model.h5')

# Step 3: Predict next day price
def predict_next_day():
    model = load_model('lstm_model.h5')
    scaler = joblib.load('scaler.save')

    df = pd.read_csv("data/steel_data.csv")
    last_60_days = df['Steel_Price'].values[-60:].reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60_days)
    X_input = last_60_scaled.reshape((1, 60, 1))

    next_scaled = model.predict(X_input)[0][0]
    predicted_price = scaler.inverse_transform([[next_scaled]])[0][0]
    return round(predicted_price, 2)

# Step 4: Log prediction
def log_prediction(predicted_price):
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    prediction_row = [tomorrow, predicted_price]
    log_path = "predictions.csv"
    file_exists = os.path.exists(log_path)

    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Predicted_Price'])
        writer.writerow(prediction_row)

# ---- Streamlit UI ----
st.set_page_config(page_title="Steel Price Predictor", layout="centered")
st.title("📈 Steel Price Predictor (LSTM)")
st.markdown("Uses SLX ETF price to predict next day steel price")

# Buttons
if st.button("🔄 Fetch Latest Steel Data"):
    df = fetch_steel_price_data()
    st.success("✅ Data fetched and saved!")
    st.dataframe(df.tail())

if st.button("🧠 Train LSTM Model"):
    df = pd.read_csv("data/steel_data.csv")
    train_lstm(df)
    st.success("✅ Model trained and saved!")

if st.button("📅 Predict Tomorrow’s Price"):
    predicted = predict_next_day()
    log_prediction(predicted)
    st.success(f"📌 Predicted Steel Price for tomorrow: ₹{predicted}")

# Show past predictions
if os.path.exists("predictions.csv"):
    st.subheader("📜 Past Predictions")
    pred_df = pd.read_csv("predictions.csv")
    st.line_chart(pred_df.set_index("Date")["Predicted_Price"])
    st.dataframe(pred_df.tail())