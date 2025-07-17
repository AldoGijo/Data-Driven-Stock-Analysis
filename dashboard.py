import os
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DATA_DIR = "data"
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SECTOR_MAPPING_FILE = os.path.join(DATA_DIR, "sector_mapping.csv")

# === UTILITY FUNCTIONS ===
@st.cache_data
def load_all_stock_data():
    stock_data = []
    for file in os.listdir(PROCESSED_DATA_DIR):
        if (
            file.endswith(".csv")
            and not file.startswith("top_")
            and not file.startswith("sector_")
            and file != "sector_mapping.csv"
        ):
            file_path = os.path.join(PROCESSED_DATA_DIR, file)
            df = pd.read_csv(file_path)

            if {'date', 'symbol', 'close'}.issubset(df.columns):
                df['date'] = pd.to_datetime(df['date'])
                stock_data.append(df)
            else:
                st.warning(f"Skipped file: {file} — missing required columns.")
    return pd.concat(stock_data, ignore_index=True) if stock_data else pd.DataFrame()

@st.cache_data
def load_sector_mapping():
    try:
        df = pd.read_csv(SECTOR_MAPPING_FILE)
        return df[['symbol', 'sector']]
    except FileNotFoundError:
        st.error(f"Sector file not found: {SECTOR_MAPPING_FILE}")
        return pd.DataFrame()

def calculate_daily_return(df):
    df = df.sort_values(by="date")
    df["daily_return"] = df["close"].pct_change()
    return df

# === MAIN APP ===
st.title(" Data-Driven Stock Analysis Dashboard")
st.markdown("Interactive dashboard analyzing Nifty 50 stock trends using Python, Pandas, and Plotly.")

# Load data
df = load_all_stock_data()
sector_map = load_sector_mapping()

if df.empty:
    st.stop()

# Sidebar: Symbol selection
symbols = df["symbol"].unique().tolist()
selected_symbols = st.sidebar.multiselect("Select stocks to view", options=symbols, default=symbols[:5])
filtered_df = df[df["symbol"].isin(selected_symbols)]

st.markdown(f"**Currently showing data for:** {', '.join(selected_symbols)}")

# Top Gainers & Losers (Annual Return)
st.header(" Top 10 Gainers & Losers (Annual Return)")
returns = filtered_df.copy()
returns = calculate_daily_return(returns)
annual_returns = returns.groupby("symbol")["daily_return"].mean() * 252
top_gainers = annual_returns.sort_values(ascending=False).head(10)
top_losers = annual_returns.sort_values().head(10)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Gainers")
    st.bar_chart(top_gainers)

with col2:
    st.subheader("Top 10 Losers")
    st.bar_chart(top_losers)

# Market Summary
st.header(" Market Summary")
latest_date = filtered_df["date"].max()
latest_prices = filtered_df[filtered_df["date"] == latest_date]
previous_prices = filtered_df[filtered_df["date"] == filtered_df["date"].unique()[-2]]

merged = pd.merge(latest_prices, previous_prices, on="symbol", suffixes=("_latest", "_prev"))
merged["change"] = merged["close_latest"] - merged["close_prev"]

green = (merged["change"] > 0).sum()
red = (merged["change"] <= 0).sum()
avg_price = merged["close_latest"].mean()
avg_volume = merged["volume_latest"].mean()

st.metric(" Green Stocks", green)
st.metric(" Red Stocks", red)
st.metric(" Avg Price", f"{avg_price:.2f}")
st.metric(" Avg Volume", f"{avg_volume:.0f}")

# Volatility Analysis
st.header(" Top 10 Most Volatile Stocks")
volatility = returns.groupby("symbol")["daily_return"].std().sort_values(ascending=False).head(10)
st.bar_chart(volatility)

# Cumulative Returns (Top 5 Performing)
st.header("Cumulative Returns (Top 5 Performing Stocks)")
cum_returns_df = pd.DataFrame()
for symbol in top_gainers.head(5).index:
    stock = filtered_df[filtered_df["symbol"] == symbol].copy()
    stock = calculate_daily_return(stock)
    stock["cum_return"] = (1 + stock["daily_return"]).cumprod()
    cum_returns_df[symbol] = stock.set_index("date")["cum_return"]

st.line_chart(cum_returns_df)

# Sector-wise Average Return
st.header("Sector-wise Performance")
if not sector_map.empty:
    df_returns = filtered_df.copy()
    df_returns = calculate_daily_return(df_returns)
    df_returns = df_returns.merge(sector_map, on="symbol", how="left")
    sector_perf = df_returns.groupby("sector")["daily_return"].mean() * 252
    sector_df_plot = sector_perf.reset_index()
    sector_df_plot.columns = ['sector', 'avg_return']

    fig = px.bar(sector_df_plot, x="sector", y="avg_return",
                 title="Average Annual Return by Sector",
                 labels={"sector": "Sector", "avg_return": "Avg Annual Return"})
    st.plotly_chart(fig)
else:
    st.warning(" Sector mapping data not available.")

# Correlation Heatmap
st.header(" Stock Price Correlation Heatmap")
pivot_df = filtered_df.pivot(index="date", columns="symbol", values="close")
corr = pivot_df.pct_change().corr()

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Monthly Gainers & Losers
st.header(" Monthly Gainers & Losers")
monthly_returns = filtered_df.copy()
monthly_returns["month"] = monthly_returns["date"].dt.to_period("M")
monthly_returns = calculate_daily_return(monthly_returns)
monthly_summary = monthly_returns.groupby(["month", "symbol"])["daily_return"].mean().reset_index()
latest_month = monthly_summary["month"].max()

latest_month_df = monthly_summary[monthly_summary["month"] == latest_month]
top_5_monthly = latest_month_df.sort_values(by="daily_return", ascending=False).head(5)
bottom_5_monthly = latest_month_df.sort_values(by="daily_return").head(5)

col3, col4 = st.columns(2)
with col3:
    st.subheader(f"Top 5 in {latest_month}")
    st.dataframe(top_5_monthly)

with col4:
    st.subheader(f"Bottom 5 in {latest_month}")
    st.dataframe(bottom_5_monthly)

st.caption(" Built using Python, Pandas, Plotly, and Streamlit.")
