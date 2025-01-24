from pydantic import BaseModel
from typing import List, Dict

class TradeRequest(BaseModel):
    ticker: str
    initial_balance: float
    days: int = 30  # Added days parameter with default value of 30

class TradeLogEntry(BaseModel):
    action: str
    date: str
    price: str
    shares: str

class TradeResponse(BaseModel):
    final_balance: float
    profit: float
    plot_url: str
    trade_log: List[TradeLogEntry]