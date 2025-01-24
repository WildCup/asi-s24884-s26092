import streamlit as st
import requests
import base64

# Backend URL (adjust the URL accordingly)
API_URL = "http://127.0.0.1:8000/trade"


def fetch_trade_data(ticker, initial_balance, days):
    try:
        # Send POST request to backend with the number of days
        response = requests.post(
            API_URL,
            json={"ticker": ticker, "initial_balance": initial_balance, "days": days},
        )
        response.raise_for_status()  # Check if the request was successful
        return response.json()  # Return the response in JSON format
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching trade data: {str(e)}")
        return None


# Streamlit frontend
st.title("Stock Trading Simulation")

# User inputs
ticker = st.text_input("Enter stock ticker:", "AAPL")
initial_balance = st.number_input(
    "Enter initial balance (USD):", value=10000.0, min_value=1.0
)

# Slider for number of days
days = st.slider(
    "Select number of days for simulation:", min_value=1, max_value=365, value=30
)

# Example stock tickers as plain text
with st.expander("See details of example stock tickers"):
    st.text("AAPL - Apple Inc.")
    st.text("GOOGL - Alphabet Inc. (Google)")
    st.text("AMZN - Amazon.com, Inc.")
    st.text("MSFT - Microsoft Corporation")
    st.text("TSLA - Tesla, Inc.")
    st.text("NFLX - Netflix, Inc.")

# Trigger trade simulation
if st.button("Run Trade Simulation"):
    trade_data = fetch_trade_data(ticker, initial_balance, days)

    if trade_data:
        # Display the final balance and profit
        st.write(f"Final Balance: ${trade_data['final_balance']:.2f}")
        st.write(f"Profit: ${trade_data['profit']:.2f}")

        # Display the plot (base64 encoded image)
        plot_url = trade_data.get("plot_url")
        if plot_url:
            # Embed the image using base64 string
            try:
                plot_image = base64.b64decode(
                    plot_url.split(",")[1]
                )  # Decode the base64 string (remove data URL part)
                st.image(
                    plot_image,
                    caption="Stock Price & Predicted Prices",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        else:
            st.error("No plot data found.")
