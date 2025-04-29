# Import necessary system and JSON handling modules
import sys
import os
import json


# --- Package Installation Check ---
# Try importing required packages. If not installed, install them using pip.
try:
    import yfinance as yf  # for fetching financial data
    import pandas as pd    # for data manipulation
    import numpy as np     # for numerical operations
    import matplotlib.pyplot as plt  # for plotting graphs
    from scipy.optimize import minimize  # for portfolio optimization
    import streamlit as st  # for creating the web app
except ImportError:
    # If any import fails, install the required packages via pip
    os.system(f"{sys.executable} -m pip install yfinance pandas numpy matplotlib scipy streamlit")
    # Re-import after installation
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import streamlit as st


# Use non-interactive backend for matplotlib to avoid GUI issues in Streamlit
plt.switch_backend('Agg')


# --- Constants ---
USERS_FILE = "users.json"  # File for storing user account information and review logs


# --- Utility Functions ---


# Load users from the JSON file if it exists
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}  # Return empty dict if file doesn't exist
    with open(USERS_FILE, "r") as f:
        return json.load(f)  # Load and return JSON data


# Save users back to the JSON file
def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


# --- Authentication Logic ---


# Load existing users into memory
users = load_users()


# Initialize session state variables if they don't exist
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None


# Display log in/create account UI if user isn't logged in
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>📈 Smart Portfolio Optimization</h1>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("<b>🔐 Log In or Create Account</b>", unsafe_allow_html=True)


    # Let user choose between log iin or account creation
    auth_choice = st.radio("Choose action:", ["Log In", "Create Account"])


    # Input fields for username and password
    username = st.text_input("Username (max 30 characters)")
    password = st.text_input("Password (max 15 characters)", type="password")


    # Style submit button to be full-width
    st.markdown("""
         <style>
            .stButton>button {
                width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)


    # Submit button for authentication
    if st.button("Submit"):
        # Validate input length
        if len(username) > 30 or len(password) > 15:
            st.error("Username and Password must be 15 characters or fewer.")
        elif auth_choice == "Log In":
            # Check if username and password match
            if username in users and users[username]["password"] == password:
                st.success("Log in successful! Click Submit again to finish logging in :)")
                st.session_state.logged_in = True
                st.session_state.username = username
                # Initialize review log in session
                if "review_log" not in st.session_state:
                    st.session_state.review_log = users[username].get("review_log", [])
            else:
                st.error("Invalid username or password.")
        else:  # If creating a new account
            if username in users:
                st.error("Username already exists.")
            else:
                # Save new account to users file
                users[username] = {
                    "password": password,
                    "review_log": []
                }
                save_users(users)
                st.success("Account created successfully! Please log in.")
    # Stop script here until log in is completed
    st.stop()


# --- Main App (After Log In) ---


# Title and description for the main app interface
st.markdown("<h1 style='text-align: center;'>📈 Smart Portfolio Optimization</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Optimize your investment portfolio using Modern Portfolio Theory.</div>", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("")


# Initialize review log in session if not already present
if "review_log" not in st.session_state:
    st.session_state.review_log = []


# --- User Input Section ---


# Input for stock tickers
st.markdown("<b>Enter your stock tickers:</b>", unsafe_allow_html=True)
tickers = st.text_area("", value="AAPL, MSFT, GOOGL, AMZN, TSLA").strip()  # default example tickers


# Input for investment amount
st.markdown("<b>Enter your investment amount:</b>", unsafe_allow_html=True)
principal = st.number_input("", min_value=0.0, value=100.0, step=100.0)


# Style submit button to be full-width
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)


# Button to trigger portfolio optimization
optimize_button = st.button("Optimize Portfolio")


# --- Portfolio Functions ---


# Fetch historical closing prices for given tickers and date range
def fetch_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)['Close']


# Calculate portfolio return and volatility given asset weights and return data
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility
    return portfolio_return, portfolio_volatility


# Optimize portfolio to maximize Sharpe Ratio using numerical optimization
def optimize_portfolio(returns, risk_free_rate):
    num_assets = len(returns.columns)  # Number of stocks
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short-selling allowed
    # Minimize the negative Sharpe ratio
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=(returns, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# Calculate negative Sharpe ratio for use in optimization
def negative_sharpe_ratio(weights, returns, risk_free_rate):
    p_ret, p_vol = portfolio_performance(weights, returns)
    return -(p_ret - risk_free_rate) / p_vol  # Sharpe Ratio = (Return - Risk-Free Rate) / Volatility


# Generate data for plotting efficient frontier
def plot_efficient_frontier(returns, risk_free_rate):
    num_portfolios = 10000  # Number of random portfolios to simulate
    results = np.zeros((3, num_portfolios))  # Store volatility, return, and Sharpe ratio
    weights_record = []  # Store weights for each simulated portfolio
    for i in range(num_portfolios):
        weights = np.random.random(len(returns.columns))  # Random weights
        weights /= np.sum(weights)  # Normalize to sum to 1
        weights_record.append(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, returns)
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe Ratio
    return results, weights_record  # Return data for plotting frontier


# --- Main Portfolio Logic ---
if optimize_button:  # Trigger the portfolio optimization process when the user clicks the button
    # Parse the user-input tickers, removing extra spaces and filtering out empty entries
    ticker_list = [ticker.strip() for ticker in tickers.split(',') if ticker.strip()]
   
    if not ticker_list:
        st.warning("⚠️ Please enter at least one valid stock ticker.")  # Warn if no tickers are provided
    else:
        # Define analysis time window
        start_date = '2020-01-01'
        end_date = '2024-01-01'
       
        # Lists to track valid and invalid tickers
        valid_tickers = []
        invalid_tickers = []


        # Validate tickers by attempting to fetch data
        for ticker in ticker_list:
            df = fetch_data(ticker, start_date, end_date)
            if df.empty or df.isnull().all().all():
                invalid_tickers.append(ticker)  # Add to invalid if no data found
            else:
                valid_tickers.append(ticker)  # Add to valid if data exists


        if invalid_tickers:
            # Show error if any invalid tickers are detected
            st.error(f"⚠️ No valid data found. Invalid tickers: {', '.join(invalid_tickers)}")
        else:
            # Fetch data for valid tickers and compute daily returns
            data = fetch_data(valid_tickers, start_date, end_date)
            returns = data.pct_change().dropna()


            # Fetch 10-year Treasury yield for risk-free rate
            treasury_yield = yf.Ticker("^TNX")
            treasury_data = treasury_yield.history(start="2020-01-01", end="2024-01-01")
           
            if treasury_data.empty:
                st.warning("⚠️ Unable to retrieve risk-free rate data.")  # Warn if risk-free data unavailable
            else:
                # Use latest yield as risk-free rate (convert % to decimal)
                risk_free_rate = treasury_data['Close'].iloc[-1] / 100


                # Optimize portfolio weights using MPT
                optimized_result = optimize_portfolio(returns, risk_free_rate)
                optimized_weights = optimized_result.x  # Optimal weights for each asset


                # Calculate performance metrics
                optimized_return, optimized_volatility = portfolio_performance(optimized_weights, returns)
                optimized_sharpe_ratio = (optimized_return - risk_free_rate) / optimized_volatility


                # Sort tickers and weights in descending order for display
                sorted_indices = np.argsort(optimized_weights)[::-1]
                sorted_tickers = [ticker_list[i] for i in sorted_indices]
                sorted_weights = optimized_weights[sorted_indices]


                # Display allocation pie chart
                st.header("📊 Optimized Portfolio Allocation")
                fig, ax = plt.subplots()
                cmap = plt.get_cmap('Greens')
                colors = cmap(np.linspace(0.3, 0.7, len(sorted_tickers)))


                # Labels only for weights ≥ 0.1%
                labels = [f"{ticker} {w*100:.1f}%" if w >= 0.001 else "" for ticker, w in zip(sorted_tickers, sorted_weights)]
                ax.pie(sorted_weights, labels=labels, startangle=140, colors=colors)
               
                # Create legend for the pie chart
                legend_labels = [f"{ticker}: {w*100:.1f}%" for ticker, w in zip(sorted_tickers, sorted_weights)]
                ax.legend(legend_labels, title="Stock Allocations", loc="center left", bbox_to_anchor=(1, 0.5))
                ax.axis('equal')  # Equal aspect ratio for perfect circle
                st.pyplot(fig)


                # Show table of portfolio allocations
                st.markdown("<b>Portfolio Breakdown</b>", unsafe_allow_html=True)
                allocation = pd.DataFrame({
                    'Ticker': sorted_tickers,
                    'Allocation (%)': np.round(sorted_weights * 100, 2),
                    'Amount ($)': np.round(sorted_weights * principal, 2)
                })
                st.dataframe(allocation)


                # Save allocation to session state
                new_entry = {
                    "Tickers": tickers,
                    "Principal": principal,
                    "Allocation": allocation.to_dict()
                }
                st.session_state.review_log.insert(0, new_entry)  # Add to top of log


                # Persist data to user file
                users = load_users()
                users[st.session_state.username]["review_log"] = st.session_state.review_log
                save_users(users)


                # Show efficient frontier plot with risk-return
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


                # Provide explanation of MPT and Sharpe Ratio
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


# --- Sidebar Review Log Section ---
st.sidebar.header("📜 Review Log")
st.write("")  # Spacing
st.write("")
st.write("")
st.write("")
st.write("---")


# Create placeholder for log out button to stay at bottom
logout_placeholder = st.empty()


# --- Logout Button ---
with logout_placeholder:
    if st.button("Log Out", key="log out"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]  # Clear session data
        st.rerun()  # Restart the app


# --- Expandable Sidebar for Review History ---
with st.sidebar.expander("View All Previous Allocations", expanded=False):
    for i, entry in enumerate(st.session_state.review_log, start=1):
        st.write(f"### 📌 Entry {len(st.session_state.review_log) - i + 1}:")
        st.write(f"**Tickers:** {entry['Tickers']}")
        st.write(f"**Principal:** ${entry['Principal']}")
        st.dataframe(pd.DataFrame(entry["Allocation"]))  # Show allocation breakdown
        st.markdown("---")


# --- Footer ---
st.write("---")
st.markdown("<div style='text-align: center;'>Developed by: Paige Spencer, Ian Ortega, Nabil Othman, Chris Giamis</div>", unsafe_allow_html=True)
