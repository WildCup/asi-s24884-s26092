import base64
import matplotlib.pyplot as plt
from io import BytesIO

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
