import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Title of the app
st.title("Stock Price Prediction App")

# Sample stocks from Nifty 50 and Nifty Bank (with .NS suffix for yfinance)
nifty_50_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Consultancy Services": "TCS.NS"
}

nifty_bank_stocks = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS"
}

# Combine stock options
stock_options = {"Manual Input": "Manual"} | nifty_50_stocks | nifty_bank_stocks

# User inputs
stock_choice = st.sidebar.selectbox("Select Stock or Manual Input:", stock_options.keys())
if stock_choice == "Manual Input":
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, RELIANCE.NS):", "AAPL").upper()
else:
    stock_symbol = stock_options[stock_choice]

days_to_predict = st.sidebar.slider("Days to Predict (0-10):", 0, 10, 5)
model_choice = st.sidebar.selectbox("Select Prediction Model:", ["Linear Regression", "SVR", "Both"])

# Fetch stock data
@st.cache_data
def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="1y")  # 1 year of historical data
    return df

# Prepare data for modeling
def prepare_data(df, days):
    df = df[['Close']].copy()
    df['Target'] = df['Close'].shift(-days)  # Shift target by 'days' for prediction
    df = df.dropna()
    X = np.array(range(len(df))).reshape(-1, 1)  # Use index as feature (time)
    y = df['Target'].values
    return X, y, df

# Train and predict with selected model(s)
def train_and_predict(X, y, days, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    
    if model_type in ["Linear Regression", "Both"]:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_score = lr_model.score(X_test, y_test)
        last_index = X[-1][0]
        future_X = np.array(range(last_index + 1, last_index + days + 1)).reshape(-1, 1)
        lr_predictions = lr_model.predict(future_X)
        results['Linear Regression'] = (lr_predictions, lr_score)
    
    if model_type in ["SVR", "Both"]:
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
        svr_model.fit(X_train, y_train)
        svr_score = svr_model.score(X_test, y_test)
        last_index = X[-1][0]
        future_X = np.array(range(last_index + 1, last_index + days + 1)).reshape(-1, 1)
        svr_predictions = svr_model.predict(future_X)
        results['SVR'] = (svr_predictions, svr_score)
    
    return results

# Main logic
if stock_symbol:
    try:
        # Fetch and prepare data
        df = fetch_data(stock_symbol)
        st.write(f"Predicting for {stock_symbol}")
        st.line_chart(df['Close'])
        
        X, y, df_prepared = prepare_data(df, days_to_predict)
        
        # Train models and predict
        results = train_and_predict(X, y, days_to_predict, model_choice)
        
        # Display results
        future_dates = pd.date_range(start=df.index[-1], periods=days_to_predict + 1, freq='B')[1:]
        pred_df = pd.DataFrame(index=future_dates)
        
        for model, (predictions, score) in results.items():
            st.write(f"{model} Accuracy (R² Score): {score:.4f}")
            pred_df[model] = predictions
        
        st.write(f"Predicted Prices for the next {days_to_predict} days:")
        st.write(pred_df)
        
        # Plot predictions
        fig, ax = plt.subplots()
        if 'Linear Regression' in results:
            ax.plot(pred_df['Linear Regression'], label='Linear Regression', linestyle='--', color='blue')
        if 'SVR' in results:
            ax.plot(pred_df['SVR'], label='SVR', linestyle='--', color='orange')
        ax.set_title(f"{stock_symbol} Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {str(e)}. Please check the stock symbol or try again.")

# Run the app
if __name__ == "__main__":
    st.write("Select a stock or enter a symbol, and adjust settings in the sidebar to see predictions!")