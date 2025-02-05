import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as data
import keras
from keras.src.layers import Dense, Dropout, LSTM
from keras.src.models import Sequential

# from keras.src.models import load_model
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, send_file
import os
import streamlit as st

plt.style.use("fivethirtyeight")

# Load the model
# model = load_model("stock_dl_model.h5")

# Streamlit App
st.title("Stock Price Prediction with Technical Indicators")

# User input for stock symbol
stock = st.text_input("Enter Stock Symbol", value="POWERGRID.NS")

# Define the start and end dates
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2024, 10, 1)

# Fetch data
df = yf.download(stock, start=start, end=end)

# Compute Technical Indicators
# 1. Exponential Moving Averages
ema20 = df["Close"].ewm(span=20, adjust=False).mean()
ema50 = df["Close"].ewm(span=50, adjust=False).mean()
ema100 = df["Close"].ewm(span=100, adjust=False).mean()
ema200 = df["Close"].ewm(span=200, adjust=False).mean()


# 2. RSI (Relative Strength Index)
def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


df["RSI"] = compute_rsi(df["Close"])


# 3. MACD (Moving Average Convergence Divergence)
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


df["MACD"], df["Signal_Line"] = compute_macd(df["Close"])

# 4. Bollinger Bands
window = 20
std_dev = df["Close"].rolling(window).std()
df["Upper_Band"] = ema20 + (std_dev * 2)
df["Lower_Band"] = ema20 - (std_dev * 2)

# Data Splitting
data_training = pd.DataFrame(df["Close"][0 : int(len(df) * 0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70) :])

# Scaling Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare Data for Prediction
past_100_days = data_training.tail(100)
final_df = pd.concat([data_testing, past_100_days], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Make Predictions
# y_predicted = model.predict(x_test)

# Inverse Scaling
scale_factor = 1 / scaler.scale_[0]
# y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot: Closing Price & EMA
st.subheader("Closing Price with EMAs")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df["Close"], "y", label="Closing Price")
ax1.plot(ema20, "g", label="EMA 20")
ax1.plot(ema50, "r", label="EMA 50")
ax1.legend()
st.pyplot(fig1)

# Plot: RSI
st.subheader("RSI (Relative Strength Index)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df["RSI"], "b", label="RSI")
ax2.axhline(70, linestyle="--", color="r")
ax2.axhline(30, linestyle="--", color="g")
ax2.legend()
st.pyplot(fig2)

# Plot: MACD
st.subheader("MACD Indicator")
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(df["MACD"], "r", label="MACD")
ax3.plot(df["Signal_Line"], "b", label="Signal Line")
ax3.legend()
st.pyplot(fig3)

# Plot: Bollinger Bands
st.subheader("Bollinger Bands")
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(df["Close"], "y", label="Closing Price")
ax4.plot(df["Upper_Band"], "g", linestyle="dashed", label="Upper Band")
ax4.plot(df["Lower_Band"], "r", linestyle="dashed", label="Lower Band")
ax4.legend()
st.pyplot(fig4)

# Prediction vs Original
# st.subheader("Prediction vs Original Trend")
# fig5, ax5 = plt.subplots(figsize=(12, 6))
# ax5.plot(y_test, "g", label="Original Price", linewidth=1)
# ax5.plot(y_predicted, "r", label="Predicted Price", linewidth=1)
# ax5.legend()
# st.pyplot(fig5)

# Download Dataset
csv_file_path = f"{stock}_dataset.csv"
df.to_csv(csv_file_path)
st.subheader("Download Dataset")
st.download_button(label="Download CSV", data=df.to_csv(), file_name=csv_file_path)
