import os
import joblib
import subprocess
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def fetch_stock_data(ticker):
    """
    Fetches stock data from Yahoo Finance, calculates moving averages,
    and saves it to a CSV file if not already saved.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: Stock data with added moving averages (MA_10, MA_50).
    """
    file_path = f"data/{ticker}_stock_data.csv"  # Path to save the data

    if os.path.exists(file_path):
        print(f"Loading existing data for {ticker} from {file_path}")
        # Load data from the CSV file
        data = pd.read_csv(file_path, header=[0, 1], index_col=0)
    else:
        print(f"Downloading new data for {ticker}...")
        # Download stock data using Yahoo Finance
        data = yf.download(ticker)
        data["MA_10"] = data["Close"].rolling(window=10).mean() 
        data["MA_50"] = data["Close"].rolling(window=50).mean() 
        data = data.dropna()
        
        os.makedirs('data', exist_ok=True)
        data.to_csv(file_path)  

    return data

def train_model(data):
    """
    Trains a linear regression model on the stock data to predict the next day's closing price.

    Args:
        data (pd.DataFrame): Stock data including closing prices and moving averages.

    Returns:
        model: Trained linear regression model.
        X_test_sorted (pd.DataFrame): Sorted test set features.
        y_test_sorted (pd.Series): Sorted test set target values.
        predictions (np.ndarray): Predictions made by the model.
        model_filename (str): The path to the saved model.
    """
    X = data[["Close", "MA_10", "MA_50"]]
    y = data["Close"].shift(-1)
    X = X[:-1]
    y = y.dropna()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    X_test_sorted = X_test.sort_index()
    y_test_sorted = y_test.loc[X_test_sorted.index]
    
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    model_filename = 'model/model.pkl'
    os.makedirs('model', exist_ok=True) 
    joblib.dump(model, model_filename)

    predictions = model.predict(X_test_sorted)
    
    mse = mean_squared_error(y_test_sorted, predictions)
    r2 = r2_score(y_test_sorted, predictions)

    return model, X_test_sorted, y_test_sorted, predictions, model_filename


def simulate_trading(ticker, initial_balance, days=30):
    """
    Simulates a trading strategy using a linear regression model and stock predictions.

    Args:
        ticker (str): Stock ticker symbol.
        initial_balance (float): Starting balance for the simulation.
        days (int, optional): Number of days to simulate trading. Defaults to 30.

    Returns:
        tuple: Final balance, profit, the updated stock data with predictions, and trade log.
    """
    data = fetch_stock_data(ticker)  # Function to fetch stock data (make sure you have this implemented)
    model, X_test, y_test, predictions, model_filename = train_model(data)

    balance = initial_balance
    position = 0
    predicted_prices = []
    trade_log = []

    for i in range(min(len(X_test), days)):
        current_row = X_test.iloc[-(i + 1)]
        current_price = current_row["Close"]
        predicted_price = predictions[-(i + 1)]

        current_price = (
            current_price.item()
            if isinstance(current_price, pd.Series)
            else current_price
        )
        predicted_price = (
            predicted_price.item()
            if isinstance(predicted_price, pd.Series)
            else predicted_price
        )

        if pd.notna(predicted_price):
            predicted_prices.append(predicted_price)

            if predicted_price > current_price and balance >= current_price:
                shares_to_buy = int(balance // current_price)
                if shares_to_buy > 0:
                    position += shares_to_buy
                    balance -= shares_to_buy * current_price
                    trade_log.append({
                        "action": "Buy",
                        "date": str(X_test.index[-(i + 1)]),
                        "price": f"{current_price:.2f}",
                        "shares": str(shares_to_buy)
                    })

            elif predicted_price < current_price and position > 0:
                balance += position * current_price
                trade_log.append({
                    "action": "Sell",
                    "date": str(X_test.index[-(i + 1)]),
                    "price": f"{current_price:.2f}",
                    "shares": str(position)
                })
                position = 0
        else:
            predicted_prices.append(current_price)

    data.loc[X_test.index[-len(predicted_prices) :], "Predicted"] = predicted_prices[
        ::-1
    ]

    final_balance = balance + (
        position * X_test.iloc[min(len(X_test), days) - 1]["Close"]
    )
    profit = final_balance - initial_balance

    return final_balance, profit, data, trade_log
