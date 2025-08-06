import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px


path = "important-files/trade_results_summary.csv"
df = pd.read_csv(path)
profits = df["Profit/Loss"].dropna().tolist()
path_v1 = "important-files/todays_fills.csv"
df1= pd.read_csv(path_v1)   


if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

st.set_page_config(page_title="Trading Dashboard Performance", layout="wide")
st.title("ML Trading Bot Performance")

col1, col2, col3 = st.columns(3)

with col1:
    total_profit= df["Profit/Loss"].sum()
    st.metric("Total Profit: ", f"${total_profit}")

with col2:
    total_trades = len(df)
    st.metric("Total trades", total_trades)

with col3:
    win_rate = round((df['Profit/Loss']>0).mean() * 100,2)
    st.metric("Win Rate: ", f"{win_rate}%")


st.subheader("Cumulative Profit over time")
daily = df.groupby(df['Date'].dt.date)["Profit/Loss"].sum().cumsum()

st.subheader("📋 Trade Details")
st.dataframe(df.sort_values(by="Date", ascending=False))


st.subheader("📭 Today's filled Trades")
st.dataframe(df1.sort_values(by="Date", ascending=False))


st.subheader("🏆 Top Performing Symbols")
top = df.groupby("Symbol")["Profit/Loss"].sum().sort_values(ascending=False).reset_index()
if not top.empty:
    fig = px.bar(top, x="Symbol", y = "Profit/Loss", color ="Profit/Loss", color_continuous_scale="Tealrose", title="Total Profit/Loss by Symbol")
    fig.update_layout(xaxis_title="symbols", yaxis_title="Total Profit/Loss")
    st.plotly_chart(fig, use_container_width = True)
else:
    st.info("no filled trade with profit/loss")
    
st.subheader("🗓️ Avg Profit by Day of Week")
df['Day'] = pd.to_datetime(df['Date']).dt.day_name()
avg_by_day = df.groupby('Day')["Profit/Loss"].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday'])
fig = px.bar(avg_by_day, title="Average Profit by Weekday", labels={'value': 'Average Profit'}, color=avg_by_day.values)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Performance Charts")

col1,col2 = st.columns(2)

with col1:

    st.subheader("📈 Equity Curve")
    equity_curve = df.groupby(df['Date'].dt.date)["Profit/Loss"].sum().cumsum().reset_index()
    fig = px.line(equity_curve, x='Date', y='Profit/Loss', title='Cumulative Profit/Loss Over Time')
    fig.update_traces(line_color='#00cc96')
    st.plotly_chart(fig, use_container_width=True)


with col2:
    st.subheader("📊 Profit Distribution")
    if profits:
        fig = ff.create_distplot([profits], group_labels = ["Profit/Loss"], bin_size = 1)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.info("no filled trade with profit/loss")



st.markdown("----")
st.subheader("📈 Simulated Performance (Scaled by Price)")

def simulated_price(price):
    if price <=5:
        return 5
    elif price <=15:
        return 3
    else:
        return 2

df['Simulated Qty'] = df['Entry Price'].apply(simulated_price)
df['Simulated Profit/Loss'] = df['Profit/Loss'] * df['Simulated Qty']
total_scaled_return = df['Simulated Profit/Loss'].sum()
st.metric("Simulated Total Return (Scaled)", f"${total_scaled_return:.2f}")

st.dataframe(df[['Date',"Symbol","Entry Price","Filled Price","Qty","Simulated Qty", "Profit/Loss","Simulated Profit/Loss"]])
st.subheader("📊 Simulated Profit by Symbol (Scaled)")
simulated_by_symbol = df.groupby("Symbol")['Simulated Profit/Loss'].sum().sort_values(ascending=False)
st.bar_chart(simulated_by_symbol,color=["#fd0"])
