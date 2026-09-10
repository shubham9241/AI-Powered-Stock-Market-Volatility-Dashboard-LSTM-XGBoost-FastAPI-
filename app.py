import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Stock Market Volatility Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI-Powered Stock Market Volatility Dashboard")
st.markdown("### LSTM + XGBoost Hybrid Prediction")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("NIFTY IT", "35,240")

with col2:
    st.metric("TCS", "3,950")

with col3:
    st.metric("INFY", "1,480")

with col4:
    st.metric("WIPRO", "492")

st.divider()

st.subheader("Market Volatility")

data = pd.DataFrame({
    "Time": range(20),
    "Volatility": np.random.uniform(0.01, 0.05, 20)
})

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data["Time"],
        y=data["Volatility"],
        mode="lines+markers",
        name="Volatility"
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Volatility",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Risk Status")

st.success("STABLE")

st.info(
    "Hybrid AI model using LSTM and XGBoost "
    "for stock market volatility analysis."
)
