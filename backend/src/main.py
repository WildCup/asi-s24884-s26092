from fastapi import FastAPI
from models.models import TradeRequest, TradeResponse
from ml.model import fetch_stock_data, train_model
from utils import create_plot

import pandas as pd

app = FastAPI()

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
    data = fetch_stock_data(ticker)
    model, X_test, y_test, predictions = train_model(data)

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


@app.post("/trade", response_model=TradeResponse)
async def trade(request: TradeRequest):
    """
    Simulates a trading strategy using a linear regression model based on the given stock ticker, initial balance, 
    and the number of days. It returns the final balance, profit, plot of actual vs predicted prices, and a detailed
    trade log of buy and sell actions.

    Args:
        request (TradeRequest): Request containing:
            - `ticker`: Stock ticker symbol (e.g., "AAPL").
            - `initial_balance`: The starting balance for the trading simulation (float).
            - `days`: The number of days to simulate the trading (int, default is 30).

    Returns:
        TradeResponse: Response containing:
            - `final_balance`: The balance at the end of the simulation after all trades (float).
            - `profit`: The profit/loss made during the simulation (float).
            - `plot_url`: A URL to a plot showing actual vs predicted stock prices for the simulated days (string, base64 encoded image).
            - `trade_log`: A detailed log of all buy and sell actions, including the date, price, and number of shares involved (List[Dict[str, Union[str, float]]]).
    """
    final_balance, profit, data, trade_log = simulate_trading(
        request.ticker, request.initial_balance, request.days
    )
    plot_url = create_plot(data, request.days)
    return TradeResponse(final_balance=final_balance, profit=profit, plot_url=plot_url, trade_log=trade_log)
