import streamlit as st
st.set_page_config(
    page_title="📈 Advanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# SARIMA and Prophet imports
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📈 Advanced Stock Market Analysis Dashboard</h1>',
            unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🎛️ Dashboard Controls")

# Stock selection
default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
selected_stock = st.sidebar.selectbox(
    "Select Stock Symbol",
    default_stocks,
    index=0
)

# Date range selection
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input(
    "Start Date",
    value=datetime.now() - timedelta(days=365)
)
end_date = col2.date_input(
    "End Date",
    value=datetime.now()
)

# Analysis options
st.sidebar.subheader("📊 Analysis Options")
show_technical = st.sidebar.checkbox("Technical Indicators", value=True)
show_volume = st.sidebar.checkbox("Volume Analysis", value=True)
show_predictions = st.sidebar.checkbox("Price Predictions", value=False)

# Forecast horizon and model
forecast_days = st.sidebar.slider("Forecast Days", 1, 90, 30)
forecast_model = st.sidebar.selectbox("Forecast Model", ["SARIMA", "Prophet"], index=0)

@st.cache_data
def load_stock_data(symbol, start, end):
    """Load stock data with caching"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

    return df

# Load and process data
with st.spinner(f"Loading {selected_stock} data..."):
    stock_data = load_stock_data(selected_stock, start_date, end_date)

if stock_data is not None and not stock_data.empty:
    # Calculate technical indicators
    stock_data = calculate_technical_indicators(stock_data)

    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)

    # Key metrics
    current_price = stock_data['Close'].iloc[-1]
    prev_price = stock_data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}")
    col2.metric("Change %", f"{price_change_pct:+.2f}%")
    col3.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")
    col4.metric("52W High", f"${stock_data['High'].max():.2f}")

    # Price chart with technical indicators
    fig_price = go.Figure()

    # Candlestick chart
    fig_price.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='OHLC'
    ))

    if show_technical:
        # Moving averages
        fig_price.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['SMA_20'],
            name='SMA 20', line=dict(color='orange', width=1)
        ))
        fig_price.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['SMA_50'],
            name='SMA 50', line=dict(color='red', width=1)
        ))

        # Bollinger Bands
        fig_price.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['BB_upper'],
            name='BB Upper', line=dict(color='gray', dash='dash')
        ))
        fig_price.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['BB_lower'],
            name='BB Lower', line=dict(color='gray', dash='dash'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
        ))

    fig_price.update_layout(
        title=f"{selected_stock} Stock Price Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600
    )

    st.plotly_chart(fig_price, use_container_width=True)

    # === Forecasting Section ===
    def sarima_forecast(df, periods):
        df = df.copy()
        df = df.asfreq('B')
        df['Close'].interpolate(inplace=True)
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 5)
        model = SARIMAX(df['Close'], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.get_forecast(steps=periods)
        forecast_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=periods, freq='B')
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecast': forecast.predicted_mean,
            'Lower': forecast.conf_int()['lower Close'],
            'Upper': forecast.conf_int()['upper Close']
        })
        return forecast_df

    def prophet_forecast(df, periods):
        prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_df = forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'})
        return forecast_df

    def plot_with_forecast(df, forecast_df, model_name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name=f'{model_name} Forecast'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Upper'], mode='lines', name='Upper CI',
            line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Lower'], mode='lines', name='Lower CI',
            fill='tonexty', line=dict(width=0), showlegend=False
        ))
        fig.update_layout(title=f"{selected_stock} Stock Price with {model_name} {forecast_days}-Day Forecast",
                          xaxis_title="Date", yaxis_title="Price")
        return fig

    if show_predictions:
        if forecast_model == "SARIMA":
            forecast_df = sarima_forecast(stock_data[['Close']].copy(), forecast_days)
        else:
            forecast_df = prophet_forecast(stock_data[['Close']].copy(), forecast_days)
        fig_forecast = plot_with_forecast(stock_data, forecast_df, forecast_model)
        st.plotly_chart(fig_forecast, use_container_width=True)

    # Technical indicators dashboard
    if show_technical:
        col1, col2 = st.columns(2)

        with col1:
            # MACD
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.1,
                                   subplot_titles=['MACD', 'RSI'])

            fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'],
                                        name='MACD', line=dict(color='blue')), row=1, col=1)
            fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD_signal'],
                                        name='Signal', line=dict(color='red')), row=1, col=1)

            # RSI
            fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'],
                                        name='RSI', line=dict(color='purple')), row=2, col=1)
            fig_macd.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_macd.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig_macd.update_layout(height=500, title="Technical Indicators")
            st.plotly_chart(fig_macd, use_container_width=True)

        with col2:
            # Volume analysis
            if show_volume:
                fig_volume = go.Figure()

                # Volume bars with color coding
                colors = ['red' if close < open else 'green'
                         for close, open in zip(stock_data['Close'], stock_data['Open'])]

                fig_volume.add_trace(go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    marker_color=colors,
                    name='Volume',
                    opacity=0.7
                ))

                fig_volume.update_layout(
                    title="Volume Analysis",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=500
                )

                st.plotly_chart(fig_volume, use_container_width=True)

    # Correlation analysis for multiple stocks
    st.subheader("📊 Market Correlation Analysis")
    correlation_stocks = st.multiselect(
        "Select stocks for correlation analysis",
        default_stocks,
        default=['AAPL', 'GOOGL', 'MSFT']
    )

    if len(correlation_stocks) > 1:
        correlation_data = {}
        for stock in correlation_stocks:
            stock_info = yf.Ticker(stock)
            hist = stock_info.history(start=start_date, end=end_date)
            correlation_data[stock] = hist['Close']

        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()

        fig_corr = px.imshow(
            correlation_matrix,
            title="Stock Correlation Heatmap",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Summary statistics
    st.subheader("📈 Summary Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Price Statistics**")
        price_stats = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
        st.dataframe(price_stats)
    with col2:
        st.write("**Technical Indicator Values**")
        current_rsi = stock_data['RSI'].iloc[-1]
        current_macd = stock_data['MACD'].iloc[-1]

        if current_rsi > 70:
            rsi_signal = "🔴 Overbought"
        elif current_rsi < 30:
            rsi_signal = "🟢 Oversold"
        else:
            rsi_signal = "🟡 Neutral"

        st.metric("RSI", f"{current_rsi:.2f}", help=rsi_signal)
        st.metric("MACD", f"{current_macd:.4f}")

        # Trading signals
        st.write("**Trading Signals**")
        if stock_data['Close'].iloc[-1] > stock_data['SMA_20'].iloc[-1]:
            st.success("🟢 Price above SMA 20 - Bullish")
        else:
            st.error("🔴 Price below SMA 20 - Bearish")

else:
    st.error("Failed to load stock data. Please check the symbol and try again.")

# Footer
st.markdown("---")
st.markdown(
    "**Disclaimer**: This dashboard is for educational purposes only. "
    "Always consult with financial advisors before making investment decisions."
)
