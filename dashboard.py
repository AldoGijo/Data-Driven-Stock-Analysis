import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DATA_DIR = "data"
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# === UTILITY FUNCTIONS ===
@st.cache_data
def load_all_stock_data():
    stock_data = []
    if not os.path.isdir(PROCESSED_DATA_DIR):
        st.error(f"Processed data directory not found: {PROCESSED_DATA_DIR}")
        return pd.DataFrame()

    for file in os.listdir(PROCESSED_DATA_DIR):
        if (
            file.endswith(".csv")
            and not file.startswith("top_")
            and not file.startswith("sector_")
        ):
            file_path = os.path.join(PROCESSED_DATA_DIR, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                st.warning(f"Could not read {file}: {e}")
                continue

            # require columns
            if {"date", "symbol", "close"}.issubset(df.columns):
                df["date"] = pd.to_datetime(df["date"])
                if "volume" not in df.columns:
                    df["volume"] = np.nan
                # normalize symbol strings and remove common suffixes
                df["symbol"] = (
                    df["symbol"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .str.replace(".NS", "", regex=False)
                    .str.replace(".BO", "", regex=False)
                )
                stock_data.append(df)
            else:
                st.warning(f"Skipped file: {file} — missing required columns (need date, symbol, close).")

    return pd.concat(stock_data, ignore_index=True) if stock_data else pd.DataFrame()

def calculate_daily_return(df):
    """Calculate daily returns per symbol safely."""
    if df.empty:
        return df
    df = df.sort_values(["symbol", "date"]).copy()
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()
    return df

# === MAIN APP ===
st.set_page_config(page_title=" Stock Dashboard", layout="wide")
st.title("Data-Driven Stock Analysis Dashboard")

# Load data
df = load_all_stock_data()

if df.empty:
    st.warning("No stock files found in data/processed/. Place CSVs there and try again.")
    st.stop()

# Sidebar filters
st.sidebar.header(" Filter Options")
symbols = sorted(df["symbol"].astype(str).unique().tolist())
selected_symbols = st.sidebar.multiselect("Select stocks:", options=symbols, default=symbols[:5])

min_date = df["date"].min().date()
max_date = df["date"].max().date()
selected_range = st.sidebar.date_input(
    "Select date range:", [min_date, max_date], min_value=min_date, max_value=max_date
)

if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
    start_dt = pd.to_datetime(selected_range[0])
    end_dt = pd.to_datetime(selected_range[1])
else:
    start_dt = pd.to_datetime(min_date)
    end_dt = pd.to_datetime(max_date)

# Filter dataframe
filtered_df = df[(df["symbol"].isin(selected_symbols)) & (df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
if filtered_df.empty:
    st.warning(" No stock data matches your filter criteria (symbol/date). Adjust filters and try again.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs([" Overview", "Visualizations", " Data View"])

# ---------------------- TAB 1: OVERVIEW ----------------------
with tab1:
    st.subheader(" Top 10 Gainers & Losers (Annual Return) — Full Dataset")

    full_returns = calculate_daily_return(df.copy())
    annual_returns_full = full_returns.groupby("symbol")["daily_return"].mean().dropna() * 252 * 100

    if annual_returns_full.empty:
        st.info("Not enough data in the full dataset to compute annual returns.")
    else:
        top_gainers = annual_returns_full.sort_values(ascending=False).head(10)
        top_losers = annual_returns_full.sort_values().head(10)

        col1, col2 = st.columns(2)
        with col1:
            fig_g = px.bar(
                top_gainers.sort_values(),
                x=top_gainers.values,
                y=top_gainers.index,
                orientation="h",
                labels={"x": "Annual Return (%)", "y": "Symbol"},
                title="Top 10 Gainers (Overall)",
            )
            st.plotly_chart(fig_g, use_container_width=True)

        with col2:
            fig_l = px.bar(
                top_losers.sort_values(),
                x=top_losers.values,
                y=top_losers.index,
                orientation="h",
                labels={"x": "Annual Return (%)", "y": "Symbol"},
                title="Top 10 Losers (Overall)",
            )
            st.plotly_chart(fig_l, use_container_width=True)

    # Market summary (filtered)
    st.subheader(" Market Summary (Filtered)")
    unique_dates = filtered_df["date"].drop_duplicates().sort_values()
    latest_date = unique_dates.iloc[-1]
    prev_date = unique_dates.iloc[-2] if len(unique_dates) >= 2 else latest_date

    latest_prices = filtered_df[filtered_df["date"] == latest_date]
    previous_prices = filtered_df[filtered_df["date"] == prev_date]

    if latest_prices.empty or previous_prices.empty:
        st.info("Not enough date range in filtered data to compute market summary.")
    else:
        merged = pd.merge(
            latest_prices[["symbol", "close", "volume"]],
            previous_prices[["symbol", "close", "volume"]],
            on="symbol",
            suffixes=("_latest", "_prev"),
        )
        merged["change"] = merged["close_latest"] - merged["close_prev"]

        green = int((merged["change"] > 0).sum())
        red = int((merged["change"] <= 0).sum())
        avg_price = float(merged["close_latest"].mean())
        avg_volume = float(merged["volume_latest"].mean()) if "volume_latest" in merged.columns else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Green Stocks", green)
        c2.metric("Red Stocks", red)
        c3.metric("Avg Price", f"{avg_price:.2f}")
        c4.metric("Avg Volume", f"{avg_volume:.0f}" if not np.isnan(avg_volume) else "N/A")

# ---------------------- TAB 2: VISUALIZATIONS ----------------------
with tab2:
    st.subheader(" Volatility Analysis (Top 10) — Filtered")
    returns_filtered = calculate_daily_return(filtered_df.copy())
    volatility = returns_filtered.groupby("symbol")["daily_return"].std().dropna().sort_values(ascending=False).head(10)

    if volatility.empty:
        st.info("Not enough data to compute volatility for the selected filters.")
    else:
        fig_v = px.bar(
            volatility.sort_values(),
            x=volatility.values,
            y=volatility.index,
            orientation="h",
            labels={"x": "Volatility (Std Dev)", "y": "Symbol"},
            title="Top 10 Volatile Stocks (Filtered)",
        )
        st.plotly_chart(fig_v, use_container_width=True)

    st.subheader(" Cumulative Returns (Top 5 Performing — Overall)")
    if not annual_returns_full.empty:
        top5 = annual_returns_full.sort_values(ascending=False).head(5).index.tolist()
    else:
        top5 = filtered_df["symbol"].unique()[:5].tolist()

    cum_df = pd.DataFrame()
    for sym in top5:
        s = df[df["symbol"] == sym].copy()
        s = calculate_daily_return(s)
        s["cum_return"] = (1 + s["daily_return"]).cumprod()
        s = s.groupby("date", as_index=False).last()
        if "cum_return" in s.columns:
            series = s.set_index("date")["cum_return"].rename(sym)
            cum_df = pd.concat([cum_df, series], axis=1)

    if cum_df.empty:
        st.info("Not enough data to plot cumulative returns.")
    else:
        st.line_chart(cum_df.sort_index())

    st.subheader(" Stock Price Correlation Heatmap (Filtered)")
    pivot_df = filtered_df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()

    if pivot_df.shape[1] < 2:
        st.info("Not enough symbols (columns) to compute correlation heatmap.")
    else:
        returns_pct = pivot_df.pct_change()
        corr = returns_pct.corr(min_periods=1)
        if corr.isnull().all().all():
            st.info("Not enough overlapping data across symbols to compute correlation.")
        else:
            fig2, ax2 = plt.subplots(figsize=(max(6, corr.shape[0] * 0.5), max(4, corr.shape[0] * 0.5)))
            sns.heatmap(corr.fillna(0), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)

    st.subheader(" Monthly Gainers & Losers (Filtered)")
    monthly_returns = calculate_daily_return(filtered_df.copy())
    monthly_returns["month"] = monthly_returns["date"].dt.to_period("M")
    monthly_summary = monthly_returns.groupby(["month", "symbol"])["daily_return"].mean().reset_index()

    if monthly_summary.empty:
        st.info("No monthly summary could be computed for this selection.")
    else:
        latest_month = monthly_summary["month"].max()
        latest_month_df = monthly_summary[monthly_summary["month"] == latest_month]
        if latest_month_df.empty:
            st.info("No data for the latest month in the selected range.")
        else:
            top_5_monthly = latest_month_df.sort_values(by="daily_return", ascending=False).head(5)
            bottom_5_monthly = latest_month_df.sort_values(by="daily_return", ascending=True).head(5)

            col7, col8 = st.columns(2)
            with col7:
                fig_topm = px.bar(
                    top_5_monthly.sort_values(by="daily_return"),
                    x="daily_return",
                    y="symbol",
                    orientation="h",
                    labels={"daily_return": "Avg Daily Return", "symbol": "Symbol"},
                    title=f"Top 5 in {latest_month}",
                )
                st.plotly_chart(fig_topm, use_container_width=True)
            with col8:
                fig_botm = px.bar(
                    bottom_5_monthly.sort_values(by="daily_return"),
                    x="daily_return",
                    y="symbol",
                    orientation="h",
                    labels={"daily_return": "Avg Daily Return", "symbol": "Symbol"},
                    title=f"Bottom 5 in {latest_month}",
                )
                st.plotly_chart(fig_botm, use_container_width=True)

# ---------------------- TAB 3: DATA VIEW ----------------------
with tab3:
    st.subheader(" Filtered Stock Data")
    st.dataframe(filtered_df.sort_values(["symbol", "date"]).reset_index(drop=True), use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(" Download CSV", csv, "filtered_stocks.csv", "text/csv")

st.caption("Built with Python, Pandas, Plotly, and Streamlit.")
