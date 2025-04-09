import sys
import os

# Ensure required packages are installed
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import streamlit as st
except ImportError:
    os.system(f"{sys.executable} -m pip install yfinance pandas numpy matplotlib scipy streamlit")
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import streamlit as st

# Ensure compatibility with Streamlit
plt.switch_backend('Agg')

# Centered App Title
st.markdown("<h1 style='text-align: center;'>📈 Smart Portfolio Optimization</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Optimize your investment portfolio using Modern Portfolio Theory.</div>", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("")


# Initialize Session State for Review Log
if "review_log" not in st.session_state:
    st.session_state.review_log = []


# User Input Fields (Centered)
st.markdown("<b>Enter your stock tickers:</b>", unsafe_allow_html=True)
tickers = st.text_area("", value="AAPL, MSFT, GOOGL, AMZN, TSLA").strip()


st.markdown("<b>Enter your investment amount:</b>", unsafe_allow_html=True)
principal = st.number_input("", min_value=0.0, value=100.0, step=100.0)


# Centering and resizing the Optimize Portfolio button
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)
optimize_button = st.button("Optimize Portfolio")


# Function to Fetch Stock Data
def fetch_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)['Close']


# Function to Calculate Portfolio Performance
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility


# Function to Optimize Portfolio
def optimize_portfolio(returns, risk_free_rate):
    num_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=(returns, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# Function to Calculate Negative Sharpe Ratio
def negative_sharpe_ratio(weights, returns, risk_free_rate):
    p_ret, p_vol = portfolio_performance(weights, returns)
    return -(p_ret - risk_free_rate) / p_vol


# Function to Plot Efficient Frontier
def plot_efficient_frontier(returns, risk_free_rate):
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, returns)
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility
    return results, weights_record


# Main Functionality
# Main Functionality
if optimize_button:
    ticker_list = [ticker.strip() for ticker in tickers.split(',') if ticker.strip()]

    if not ticker_list:
        st.warning("⚠️ Please enter at least one valid stock ticker.")
    else:
            start_date = '2020-01-01'
    end_date = '2024-01-01'
    valid_tickers = []
    invalid_tickers = []

    for ticker in ticker_list:
        df = fetch_data(ticker, start_date, end_date)
        if df.empty or df.isnull().all().all():
            invalid_tickers.append(ticker)
        else:
            valid_tickers.append(ticker)

    if invalid_tickers:
        st.error(f"⚠️ No valid data found. The following ticker(s) are invalid or returned no data: {', '.join(invalid_tickers)}")
    else:
        data = fetch_data(valid_tickers, start_date, end_date)
        returns = data.pct_change().dropna()

        if data.empty or data.isnull().all().all():
            st.warning("⚠️ No valid data found for the given tickers. Please check your ticker symbols.")
        else:
            returns = data.pct_change().dropna()

            # Fetch Risk-Free Rate (10-year US Treasury yield)
            treasury_yield = yf.Ticker("^TNX")
            treasury_data = treasury_yield.history(start="2020-01-01", end="2024-01-01")

            if treasury_data.empty:
                st.warning("⚠️ Unable to retrieve risk-free rate data. Try again later.")
            else:
                risk_free_rate = treasury_data['Close'].iloc[-1] / 100

                # Optimize Portfolio
                optimized_result = optimize_portfolio(returns, risk_free_rate)
                optimized_weights = optimized_result.x
                optimized_return, optimized_volatility = portfolio_performance(optimized_weights, returns)
                optimized_sharpe_ratio = (optimized_return - risk_free_rate) / optimized_volatility

                # Sort Allocations in Descending Order
                sorted_indices = np.argsort(optimized_weights)[::-1]
                sorted_tickers = [ticker_list[i] for i in sorted_indices]
                sorted_weights = optimized_weights[sorted_indices]

                # Pie Chart - Portfolio Allocation
                st.header("📊 Optimized Portfolio Allocation")
                fig, ax = plt.subplots()
                cmap = plt.get_cmap('Greens')
                colors = cmap(np.linspace(0.3, 0.7, len(sorted_tickers)))

                labels = [f"{ticker} {w*100:.1f}%" if w >= 0.1 else "" for ticker, w in zip(sorted_tickers, sorted_weights)]
                ax.pie(sorted_weights, labels=labels, startangle=140, colors=colors)
                legend_labels = [f"{ticker}: {w*100:.1f}%" for ticker, w in zip(sorted_tickers, sorted_weights)]
                ax.legend(legend_labels, title="Stock Allocations", loc="center left", bbox_to_anchor=(1, 0.5))
                ax.axis('equal')
                st.pyplot(fig)

                # Allocation Breakdown
                st.markdown("<b>Portfolio Breakdown</b>", unsafe_allow_html=True)
                st.write("These are the optimized allocations for your investment:")

                allocation = pd.DataFrame({
                    'Ticker': sorted_tickers,
                    'Allocation (%)': np.round(sorted_weights * 100, 2),
                    'Amount ($)': np.round(sorted_weights * principal, 2)
                })
                st.dataframe(allocation)

                # Save the input + results to session state for review log
                st.session_state.review_log.insert(0, {
                    "Tickers": tickers,
                    "Principal": principal,
                    "Allocation": allocation
                })

                # Efficient Frontier & Risk-Return Graph
                st.header("📈 Risk-Return Analysis")
                results, weights_record = plot_efficient_frontier(returns, risk_free_rate)
                max_sharpe_idx = np.argmax(results[2])
                sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap='Blues', marker='o')
                ax.scatter(sdp, rp, marker='*', color='r', s=200, label='Max Sharpe Ratio Portfolio')
                ax.set_xlabel('Risk (Volatility)')
                ax.set_ylabel('Expected Return')
                ax.legend()
                plt.colorbar(scatter, label='Sharpe Ratio')
                st.pyplot(fig)

                # Explanation of Markowitz Theory & Sharpe Ratio
                st.header("📊 Understanding Your Portfolio")
                st.write(
                    "This portfolio is optimized using **Markowitz Modern Portfolio Theory (MPT)**, "
                    "which finds the best combination of stocks to maximize return while minimizing risk. "
                    "The **Efficient Frontier graph** above shows different portfolios' risk-return tradeoff."
                )

                st.write(
                    "The **Sharpe Ratio** measures how much excess return you earn per unit of risk. "
                    "A **higher Sharpe Ratio** means a better risk-adjusted return:"
                )
                st.markdown(
                    "- 🔴 **< 1.0:** Weak risk-adjusted returns\n"
                    "- 🟡 **1.0 - 2.0:** Moderate returns\n"
                    "- 🟢 **> 2.0:** Strong risk-adjusted returns"
                )

                st.markdown(
                    "Your portfolio's **Sharpe Ratio** is **{:.2f}**, which indicates its expected performance.".format(optimized_sharpe_ratio)
                )





# Sidebar Review Log
st.sidebar.header("📜 Review Log")
with st.sidebar.expander("View All Previous Allocations", expanded=False):
    for i, entry in enumerate(st.session_state.review_log, start=1):
        st.write(f"### 📌 Entry {len(st.session_state.review_log) - i + 1}:")
        st.write(f"**Tickers:** {entry['Tickers']}")
        st.write(f"**Principal:** ${entry['Principal']}")
        st.dataframe(entry["Allocation"])
        st.markdown("---")


# Footer
st.write("---")
st.markdown("<div style='text-align: center;'>Developed by: Paige Spencer, Ian Ortega, Nabil Othman, Chris Giamis</div>", unsafe_allow_html=True)

