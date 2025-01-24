import base64
import matplotlib.pyplot as plt
from io import BytesIO
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Define the input and output data models
class TradeRequest(BaseModel):
    ticker: str
    initial_balance: float
    days: int = 30  # Added days parameter with default value of 30


class TradeResponse(BaseModel):
    final_balance: float
    profit: float
    plot_url: str


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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    X_test_sorted = X_test.sort_index()
    y_test_sorted = y_test.loc[X_test_sorted.index]
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test_sorted)
    mse = mean_squared_error(y_test_sorted, predictions)
    r2 = r2_score(y_test_sorted, predictions)

    return model, X_test_sorted, y_test_sorted, predictions


def simulate_trading(ticker, initial_balance, days=30):
    """
    Simulates a trading strategy using a linear regression model and stock predictions.

    Args:
        ticker (str): Stock ticker symbol.
        initial_balance (float): Starting balance for the simulation.
        days (int, optional): Number of days to simulate trading. Defaults to 30.

    Returns:
        tuple: Final balance, profit, and the updated stock data with predictions.
    """
    data = fetch_stock_data(ticker)
    model, X_test, y_test, predictions = train_model(data)

    balance = initial_balance
    position = 0
    predicted_prices = []

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

            elif predicted_price < current_price and position > 0:
                balance += position * current_price
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

    return final_balance, profit, data


def create_plot(data, days):
    """
    Creates a plot for the actual and predicted stock prices for the last 'days' days.

    Args:
        data (pd.DataFrame): Stock data with actual and predicted prices.
        days (int): Number of days to plot.

    Returns:
        str: Base64 encoded image of the plot.
    """
    data_to_plot = data[-days:]

    plt.figure(figsize=(10, 5))
    plt.plot(data_to_plot.index, data_to_plot["Close"], label="Actual Price")
    plt.plot(
        data_to_plot.index,
        data_to_plot["Predicted"],
        label="Predicted Price",
        linestyle="--",
    )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price vs Predicted Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return f"data:image/png;base64,{plot_base64}"


@app.post("/trade", response_model=TradeResponse)
async def trade(request: TradeRequest):
    """
    Endpoint to simulate trading based on the linear regression model.

    Args:
        request (TradeRequest): Request containing the stock ticker, initial balance, and number of days.

    Returns:
        TradeResponse: Final balance, profit, and plot URL.
    """
    final_balance, profit, data = simulate_trading(
        request.ticker, request.initial_balance, request.days
    )
    plot_url = create_plot(data, request.days)
    return TradeResponse(final_balance=final_balance, profit=profit, plot_url=plot_url)
