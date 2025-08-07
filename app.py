# file: app.py
import streamlit as st
from datetime import datetime
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go

# ===========================================================
# Constants
BASE_EPSILON = 1e-4


# ===========================================================

# Configuration Init
st.set_page_config(page_title="Bonding Curve Simulator", layout="wide")
st.title("42: Twin Bonding Curve Simulatoooor")

# Initialize state
if 'logs' not in st.session_state:
    st.session_state.logs = []
for token in ['A', 'B', 'C']:
    if f'reserve_{token}' not in st.session_state:
        st.session_state[f'reserve_{token}'] = 0 # circulating shares
        st.session_state[f'usdc_reserve_{token}'] = 0.0
if 'usdc_reserve' not in st.session_state:
    st.session_state.usdc_reserve = 0.0


# ===========================================================
# Functions

# Buy bonding curve
def buy_curve(x):
    return x**(1/4) + x / 400

# Sell bonding curve
def sell_curve(x):
    return ((x - 500)/40) / ((8 + ((x - 500)/80)**2)**0.5) + (x - 500)/300 + 3.6

# Buy delta
def buy_delta(x):
    return (640 * x**(5/4) + x**2) / 800

# Sell delta
def sell_delta(x):
    return 1.93333 * x + 0.00166667 * x**2 + 2 * math.sqrt(301200 - 1000 * x + x**2)

# Calculate prices and deltas
def get_all_metrics(x, quantity):
    new_x = x + quantity
    buy_price = buy_curve(new_x)
    sell_price = sell_curve(x)
    buy_amt_delta = buy_delta(new_x) - buy_delta(x)
    sell_amt_delta = sell_delta(x) - sell_delta(x - quantity) if x - quantity >= 0 else 0
    return buy_price, sell_price, buy_amt_delta, sell_amt_delta


# Input UI
input_mode = st.radio("Select Input Mode", ["Quantity", "USDC"], horizontal=True)

quantity = 0
usdc_input = 0.0
if input_mode == "Quantity":
    quantity = st.number_input("Enter Quantity", min_value=1, step=1)
else:
    usdc_input = st.number_input("Enter USDC Amount", min_value=0.0, step=0.1)


# Estimation via USDC amount specified
# buy
def find_quantity_for_usdc(reserve, usdc_amount):
    low, high = 0, 10000  # Arbitrary upper bound
    epsilon = BASE_EPSILON
    while low < high:
        mid = (low + high) / 2
        delta = buy_delta(reserve + mid) - buy_delta(reserve)
        if abs(delta - usdc_amount) < epsilon:
            return round(mid)
        elif delta < usdc_amount:
            low = mid + epsilon
        else:
            high = mid - epsilon
    return round(low)

#sell -> to double check on formula
def find_quantity_for_sell_usdc(reserve, usdc_amount):
    low, high = 0, reserve
    epsilon = BASE_EPSILON
    while low < high:
        mid = (low + high) / 2
        delta = sell_delta(reserve) - sell_delta(reserve - mid)
        if abs(delta - usdc_amount) < epsilon:
            return round(mid)
        elif delta < usdc_amount:
            low = mid + epsilon
        else:
            high = mid - epsilon
    return round(low)

# ===========================================================
# Start of App Logic

st.subheader("Buy/Sell Controls")

cols = st.columns(3)
for i, token in enumerate(['A', 'B', 'C']):
    with cols[i]:
        st.markdown(f"### Outcome {token}")

        reserve = st.session_state[f'reserve_{token}']
        buy_col, sell_col,_,_ = st.columns(4)
        
        with buy_col:
            if st.button(f"Buy {token}"):
                if input_mode == "USDC":
                    quantity = find_quantity_for_usdc(reserve, usdc_input)

                buy_price, _, buy_amt_delta, _ = get_all_metrics(reserve, quantity)
           
                # Update both overall & token reserves
                st.session_state.usdc_reserve += buy_amt_delta
                st.session_state[f'usdc_reserve_{token}'] += buy_amt_delta

                # update circulating share reserves (token & overall)
                st.session_state[f'reserve_{token}'] += quantity
                total_market = sum(st.session_state[f'reserve_{t}'] for t in ['A', 'B', 'C'])

                payout_share = round(total_market / st.session_state[f'reserve_{token}'], 4) if st.session_state[f'reserve_{token}'] > 0 else 0

                st.session_state.logs.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Action": "Buy",
                    "Outcome": token,
                    "Quantity": quantity,
                    "Buy Price": round(buy_price, 4),
                    "Sell Price": "-",
                    "BuyAmt_Delta": round(buy_amt_delta, 4),
                    "SellAmt_Delta": "-",
                    "Token Cir. Shares": st.session_state[f'reserve_{token}'],
                    "Token MCAP": st.session_state[f'usdc_reserve_{token}'],
                    "Overall USDC Reserve": round(st.session_state.usdc_reserve, 2),
                    "Payout/Share": payout_share
                })
        with sell_col:
            if st.button(f"Sell {token}"):
                reserve = st.session_state[f'reserve_{token}']

                if input_mode == "USDC":
                    quantity = find_quantity_for_sell_usdc(reserve, usdc_input)
                    
                if reserve >= quantity:
                    _, sell_price, _, sell_amt_delta = get_all_metrics(reserve, quantity)

                    # Update both overall & token reserves
                    st.session_state.usdc_reserve -= sell_amt_delta
                    st.session_state[f'usdc_reserve_{token}'] -= sell_amt_delta

                    # Update circulating shares (token & overall market)
                    st.session_state[f'reserve_{token}'] -= quantity
                    total_market = sum(st.session_state[f'reserve_{t}'] for t in ['A', 'B', 'C'])

                    payout_share = round(total_market / st.session_state[f'reserve_{token}'], 4) if st.session_state[f'reserve_{token}'] > 0 else 0
                    st.session_state.logs.append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Action": "Sell",
                        "Outcome": token,
                        "Quantity": quantity,
                        "Buy Price": "-",
                        "Sell Price": round(sell_price, 4),
                        "BuyAmt_Delta": "-",
                        "SellAmt_Delta": round(sell_amt_delta, 4),
                        "Token Cir. Shares": st.session_state[f'reserve_{token}'],
                        "Token MCAP": st.session_state[f'usdc_reserve_{token}'],
                        "Overall USDC Reserve": round(st.session_state.usdc_reserve, 2),
                        "Payout/Share": payout_share
                    })
                else:
                    st.warning(f"Insufficient qty to sell.")

# to include the amount is 'locked up' too
# Get rid of overall market size

# Display logs
if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs)

    reserve_A = st.session_state['reserve_A']
    reserve_B = st.session_state['reserve_B']
    reserve_C = st.session_state['reserve_C']
    total_market = reserve_A + reserve_B + reserve_C

    odds = {
        'A': round(1 / (reserve_A / total_market), 4) if reserve_A > 0 else '-',
        'B': round(1 / (reserve_B / total_market), 4) if reserve_B > 0 else '-',
        'C': round(1 / (reserve_C / total_market), 4) if reserve_C > 0 else '-',
    }

    st.subheader("Overall Market Metrics")
    res_cols_1 = st.columns(5)
    with res_cols_1[0]:
        st.metric("Total USDC Reserve", round(st.session_state.usdc_reserve, 2))


    token_cols = st.columns(3)
    for i, token in enumerate(['A', 'B', 'C']):
        with token_cols[i]:
            st.subheader(f'Outcome {token}')
            reserve = st.session_state[f'reserve_{token}']
            mcap = round(st.session_state[f'usdc_reserve_{token}'],2)
            price = round(buy_curve(reserve), 2)

            sub_cols = st.columns(4)
            with sub_cols[0]:
                st.metric(f"Total Shares {token}", reserve)
            with sub_cols[1]:
                st.metric(f"Price {token}", price)
            with sub_cols[2]:
                st.metric(f"MCAP {token}", mcap)

    st.write(f"Odds A: {odds['A']} | Odds B: {odds['B']} | Odds C: {odds['C']}")

    st.subheader("Transaction Log")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Payout/Share Trend")
    df_viz = df[df['Payout/Share'] != '-'].copy()
    df_viz['Payout/Share'] = pd.to_numeric(df_viz['Payout/Share'], errors='coerce')
    fig = px.line(df_viz, x="Time", y="Payout/Share", color="Outcome", markers=True, title="Payout/Share Over Time")
    st.plotly_chart(fig, use_container_width=True)


    # Outlining Historical Token Circulating Shares Trend
    st.subheader("📊 Circulating Token Shares Over Time")
    df_reserve = df[df['Action'].isin(['Buy', 'Sell'])].copy()
    df_reserve['Time'] = pd.to_datetime(df_reserve['Time'])

    # Reconstruct reserves over time
    reserve_log = []
    running_reserves = {'A': 0, 'B': 0, 'C': 0}
    for _, row in df_reserve.iterrows():
        token = row['Outcome']
        qty = row['Quantity']
        if row['Action'] == 'Buy':
            running_reserves[token] += qty
        elif row['Action'] == 'Sell':
            running_reserves[token] -= qty
        reserve_log.append({
            'Time': row['Time'],
            'Shares A': running_reserves['A'],
            'Shares B': running_reserves['B'],
            'Shares C': running_reserves['C']
        })

    df_reserve_reconstructed = pd.DataFrame(reserve_log)
    df_reserve = df_reserve_reconstructed.melt(id_vars='Time', var_name='Token', value_name='Reserve')

    fig_stack = px.area(df_reserve, x="Time", y="Reserve", color="Token", title="Token Shares vs. Time")
    st.plotly_chart(fig_stack, use_container_width=True)

    # Tabbed bonding curve visualization with markers
    st.subheader("🔁 Bonding Curves by Outcome")
    tab1, tab2, tab3 = st.tabs([
        "Outcome A",
        "Outcome B",
        "Outcome C"
    ])
    x_vals = list(range(1, 1001))
    buy_vals = [buy_curve(x) for x in x_vals]
    sell_vals = [sell_curve(x) for x in x_vals]

    for token, tab in zip(['A', 'B', 'C'], [tab1, tab2, tab3]):
        reserve = st.session_state[f'reserve_{token}']
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=x_vals, y=buy_vals, mode='lines', name='Buy Curve', line=dict(color='green')))
        fig_curve.add_trace(go.Scatter(x=x_vals, y=sell_vals, mode='lines', name='Sell Curve', line=dict(color='red')))
        fig_curve.add_trace(go.Scatter(x=[reserve], y=[buy_curve(reserve)], mode='markers+text',
                                       name=f'{token} Buy Point', text=[f"{token}: {reserve}"],
                                       textposition="top center", marker=dict(size=10, color='green')))
        fig_curve.add_trace(go.Scatter(x=[reserve], y=[sell_curve(reserve)], mode='markers+text',
                                       name=f'{token} Sell Point', text=[f"{token}: {reserve}"],
                                       textposition="bottom center", marker=dict(size=10, color='red')))
        fig_curve.add_trace(go.Scatter(x=[reserve, reserve], y=[0, buy_curve(reserve)], mode='lines',
                                       line=dict(color='green', dash='dot'), showlegend=False))
        fig_curve.add_trace(go.Scatter(x=[0, reserve], y=[buy_curve(reserve), buy_curve(reserve)], mode='lines',
                                       line=dict(color='green', dash='dot'), showlegend=False))
        fig_curve.add_trace(go.Scatter(x=[reserve, reserve], y=[0, sell_curve(reserve)], mode='lines',
                                       line=dict(color='red', dash='dot'), showlegend=False))
        fig_curve.add_trace(go.Scatter(x=[0, reserve], y=[sell_curve(reserve), sell_curve(reserve)], mode='lines',
                                       line=dict(color='red', dash='dot'), showlegend=False))
        fig_curve.update_layout(title=f'{token} Price vs Shares', xaxis_title='Shares', yaxis_title='Price')
        tab.plotly_chart(fig_curve, use_container_width=True)
else:
    st.info("No actions taken yet. Use the buttons above to add rows.")
