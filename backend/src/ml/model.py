import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def fetch_stock_data(ticker):
    """
    Fetches stock data from Yahoo Finance and calculates moving averages.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: Stock data with added moving averages (MA_10, MA_50).
    """
    data = yf.download(ticker)
    data["MA_10"] = data["Close"].rolling(window=10).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    data = data.dropna()
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
    predictions = model.predict(X_test_sorted)
    mse = mean_squared_error(y_test_sorted, predictions)
    r2 = r2_score(y_test_sorted, predictions)

    return model, X_test_sorted, y_test_sorted, predictions