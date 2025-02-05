import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as data
import keras
from keras.src.layers import Dense, Dropout, LSTM
from keras.src.models import Sequential
from keras.src.models import load_model
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, send_file
import os
import streamlit as st

plt.style.use("fivethirtyeight")

# Load the model (make sure your model is in the correct path)
model = load_model("stock_dl_model.keras")

# Streamlit App
st.title("Stock Price Prediction")

# User input for stock symbol
stock = st.text_input("Enter Stock Symbol", value="POWERGRID.NS")

# Define the start and end dates for stock data
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2024, 10, 1)

# Fetch data from Yahoo Finance
df = yf.download(stock, start=start, end=end)

# Descriptive data
data_desc = df.describe()

# Display Data Description
st.subheader(f"Descriptive Data for {stock}")
st.write(data_desc)

# Exponential Moving Averages
ema20 = df.Close.ewm(span=20, adjust=False).mean()
ema50 = df.Close.ewm(span=50, adjust=False).mean()
ema100 = df.Close.ewm(span=100, adjust=False).mean()
ema200 = df.Close.ewm(span=200, adjust=False).mean()

# Data splitting
data_training = pd.DataFrame(df["Close"][0 : int(len(df) * 0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70) :])

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare data for prediction
past_100_days = data_training.tail(100)
final_df = pd.concat([data_testing, past_100_days], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)

# Inverse scaling for predictions
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
st.subheader("Closing Price vs Time (20 & 50 Days EMA)")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df.Close, "y", label="Closing Price")
ax1.plot(ema20, "g", label="EMA 20")
ax1.plot(ema50, "r", label="EMA 50")
ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

# Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
st.subheader("Closing Price vs Time (100 & 200 Days EMA)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df.Close, "y", label="Closing Price")
ax2.plot(ema100, "g", label="EMA 100")
ax2.plot(ema200, "r", label="EMA 200")
ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# Plot 3: Prediction vs Original Trend
st.subheader("Prediction vs Original Trend")
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(y_test, "g", label="Original Price", linewidth=1)
ax3.plot(y_predicted, "r", label="Predicted Price", linewidth=1)
ax3.set_title("Prediction vs Original Trend")
ax3.set_xlabel("Time")
ax3.set_ylabel("Price")
ax3.legend()
st.pyplot(fig3)

# Save dataset as CSV
csv_file_path = f"{stock}_dataset.csv"
df.to_csv(csv_file_path)

# Provide download link for the CSV file
st.subheader("Download Dataset")
st.write(f"Download the dataset for {stock} as a CSV file.")
st.download_button(label="Download CSV", data=df.to_csv(), file_name=csv_file_path)


# import tensorflow as tf

# print(tf.__version__)


# import sys

# print(sys.version)

# import keras

# print(keras.__version__)
