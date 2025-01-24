from fastapi import FastAPI
from models.models import TradeRequest, TradeResponse
from ml.model import simulate_trading
from utils import create_plot

app = FastAPI()

@app.post("/trade", response_model=TradeResponse)
async def trade(request: TradeRequest):
    """
    Endpoint to simulate trading based on the linear regression model.

    Args:
        request (TradeRequest): Request containing the stock ticker, initial balance, and number of days.

    Returns:
        TradeResponse: Final balance, profit, and plot URL.
    """
    final_balance, profit, data = simulate_trading(request.ticker, request.initial_balance, request.days)
    plot_url = create_plot(data, request.days)
    return TradeResponse(final_balance=final_balance, profit=profit, plot_url=plot_url)