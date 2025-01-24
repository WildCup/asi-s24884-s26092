from pydantic import BaseModel

class TradeRequest(BaseModel):
    ticker: str
    initial_balance: float
    days: int = 30  # Added days parameter with default value of 30

class TradeResponse(BaseModel):
    final_balance: float
    profit: float
    plot_url: str
