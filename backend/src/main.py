from fastapi import FastAPI
from src.models.models import TradeRequest, TradeResponse
from src.ml.model import simulate_trading
from src.utils import create_plot, validate_ticker
from fastapi import HTTPException

import pandas as pd

app = FastAPI()


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
    is_valid = validate_ticker(request.ticker)
    if is_valid:
        final_balance, profit, data, trade_log = simulate_trading(
            request.ticker, request.initial_balance, request.days
        )
        plot_url = create_plot(data, request.days)
        return TradeResponse(final_balance=final_balance, profit=profit, plot_url=plot_url, trade_log=trade_log)
    else:
        raise HTTPException(status_code=400, detail=f"{request.ticker} is not a valid ticker.")