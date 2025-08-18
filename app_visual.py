# file: app.py
import streamlit as st
from datetime import datetime
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ===========================================================
# Constants
BASE_EPSILON = 1e-4
OUTCOMES = ['A', 'B', 'C']
MAX_SHARES = 4000

# ===========================================================
# Functions


# ===========================================================
# Curve Sets (plug your own functions here)


# #1 Model: Baseline
def baseline_buy_curve(x):      # your current model
    return x**(1/4) + x / 400

def baseline_sell_curve(x):     # your current model
    return ((x - 500)/40) / ((8 + ((x - 500)/80)**2)**0.5) + (x - 500)/300 + 3.6

def baseline_buy_delta(x):      # integral of baseline_buy_curve (your current)
    return (640 * x**(5/4) + x**2) / 800

def baseline_sell_delta(x):     # integral of baseline_sell_curve (your current)
    return 1.93333 * x + 0.00166667 * x**2 + 2 * math.sqrt(301200 - 1000 * x + x**2)

# #2 Model: Baseline
# Example alt set 1: smoother convex buy, piecewise sell (edit freely)
def alt1_buy_curve(x):
    return 0.02 * np.sqrt(x + 1) + (x / 500)

def alt1_sell_curve(x):
    # gentle slope at start, steeper later
    return 0.015 * x + 0.00015 * (x**2) / (1 + 0.0005 * x)

def alt1_buy_delta(x):
    # ∫ alt1_buy_curve dx
    return 0.02 * (2/3) * (x + 1)**(3/2) + (x**2) / 1000

def alt1_sell_delta(x):
    # primitive of alt1_sell_curve (approx closed form for demo)
    return 0.015 * x**2 / 2 + 0.00015 * (x**3) / (3 + 0.0015 * x)  # simple smooth approx

# #3 Model: Baseline
# Example alt set 2: linear buy, cubic-ish sell (edit freely)
def alt2_buy_curve(x):
    return 0.03 * x + 2.5

def alt2_sell_curve(x):
    return 2.2 + 0.02 * x + 1e-6 * x**3

def alt2_buy_delta(x):
    return 0.03 * x**2 / 2 + 2.5 * x

def alt2_sell_delta(x):
    return 2.2 * x + 0.02 * x**2 / 2 + 1e-6 * x**4 / 4

CURVE_SETS = {
    "Baseline (current)": {
        "buy_curve": baseline_buy_curve,
        "sell_curve": baseline_sell_curve,
        "buy_delta": baseline_buy_delta,
        "sell_delta": baseline_sell_delta,
        "max_shares": 10000,    # per-set override if you like
    },
    "Alt 1 (smooth convex buy)": {
        "buy_curve": alt1_buy_curve,
        "sell_curve": alt1_sell_curve,
        "buy_delta": alt1_buy_delta,
        "sell_delta": alt1_sell_delta,
        "max_shares": 10000,
    },
    "Alt 2 (linear buy, cubic sell)": {
        "buy_curve": alt2_buy_curve,
        "sell_curve": alt2_sell_curve,
        "buy_delta": alt2_buy_delta,
        "sell_delta": alt2_sell_delta,
        "max_shares": 10000,
    },
}

# Configuration Init
st.set_page_config(page_title="Bonding Curve Simulator", layout="wide")
st.title("42: Twin Bonding Curve Simulatoooor")

with st.sidebar:
    st.header("Mode")
    app_mode = st.radio("Choose view", ["Simulator", "Curve Viewer"], index=0)

with st.sidebar:
    st.header("Curve Settings")
    curve_choice = st.selectbox("Curve set", list(CURVE_SETS.keys()), index=0)


    st.subheader("Marker quantity")
    qty_mode = st.radio(
        "Marker source",
        ["Preset list", "Custom amount", 
        #  "Use live reserve (A)"
         ],
        index=0
    )

    _active = CURVE_SETS[curve_choice]

    # Preset option
    if qty_mode == "Preset list":
        preset_qty = st.selectbox(
            "Preset share quantity (for charts)",
            [100, 1000, 2500, 5000, 7500, 10000],
            index=1
        )

    # Custom option
    if qty_mode == "Custom amount":
        # 0 to MAX_SHARES inclusive
        custom_qty = st.slider(
            "Custom share quantity (q)",
            min_value=0,
            max_value=int(_active.get("max_shares", MAX_SHARES)),
            value=min(1000, int(_active.get("max_shares", MAX_SHARES))),
            step=1
        )

    # (Optional) keep this for legacy logic, but it will be derived from qty_mode
    use_preset_for_markers = (qty_mode == "Preset list")


# # Curve set & visualization controls
# with st.sidebar:
#     st.header("Curve Settings")
#     curve_choice = st.selectbox("Curve set", list(CURVE_SETS.keys()), index=0)
#     preset_qty = st.selectbox("Preset share quantity (for charts)", [100, 1000, 2500, 5000, 7500, 10000], index=1)
#     use_preset_for_markers = st.toggle("Use preset quantity for chart markers (instead of live reserves)", value=True)

st.caption(f"Active curve set: **{curve_choice}**")
# Bind active curve functions for the rest of the app
_active = CURVE_SETS[curve_choice]
buy_curve     = _active["buy_curve"]
sell_curve    = _active["sell_curve"]
buy_delta     = _active["buy_delta"]
sell_delta    = _active["sell_delta"]
MAX_SHARES    = _active.get("max_shares", MAX_SHARES)  # keep your original as fallback


# Chart configuration
with st.sidebar:
    st.header("Chart Display")
    show_point_labels = st.toggle("Show price labels at marker", value=True)

    show_curve_labels = st.toggle("Show inline price labels along curve", value=False)
    if show_curve_labels:
        label_step = st.select_slider(
            "Label every N shares",
            options=[50, 100, 250, 500, 1000, 2000],
            value=500
        )

    st.subheader("Axes")
    # X-axis controls
    x_mode = st.radio("X-axis range", ["Auto", "Custom"], horizontal=True)
    if x_mode == "Custom":
        x_min = st.number_input("X min (shares)", value=1, step=1, min_value=1, max_value=MAX_SHARES-1)
        x_max = st.number_input("X max (shares)", value=MAX_SHARES, step=1, min_value=x_min+1, max_value=MAX_SHARES)
    else:
        x_min, x_max = 1, MAX_SHARES

    # Y-axis controls (use a rough default ceiling from current curve set)
    approx_ymax = max(buy_curve(MAX_SHARES-1), sell_curve(MAX_SHARES-1))
    y_mode = st.radio("Y-axis range", ["Auto", "Custom"], horizontal=True)
    if y_mode == "Custom":
        y_min = st.number_input("Y min (price)", value=0.0, step=0.1)
        y_max = st.number_input("Y max (price)", value=float(approx_ymax * 1.1), step=0.1, min_value=y_min + 0.1)
    else:
        y_min = y_max = 0, 10




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
# Start of App Logic

# ---------- Shared helpers ----------
def get_all_metrics(x, quantity):
    new_x = x + quantity
    buy_price = buy_curve(new_x)
    sell_price = sell_curve(x)
    buy_amt_delta = buy_delta(new_x) - buy_delta(x)
    sell_amt_delta = sell_delta(x) - sell_delta(x - quantity) if x - quantity >= 0 else 0
    return buy_price, sell_price, buy_amt_delta, sell_amt_delta

def render_curve_viewer():
    st.subheader("🔁 Bonding Curves Visualisation")

    token_label = "Outcome"  # cosmetic label only

    # # Build x-range (respect custom axis controls)
    # if x_mode == "Custom":
    #     x_vals = list(range(int(x_min), int(x_max)))
    # else:
    #     x_vals = list(range(1, MAX_SHARES))

        # Marker quantity: preset, custom, or live (A)
    if qty_mode == "Preset list":
        marker_reserve = int(preset_qty)
    elif qty_mode == "Custom amount":
        marker_reserve = int(custom_qty)
    else:  # Use live reserve (A)
        marker_reserve = int(st.session_state.get('reserve_A', 0))

    # Build x-range (respect custom axis controls)
    if x_mode == "Custom":
        x_vals = list(range(int(x_min), int(x_max)))  # [x_min, x_max)
    else:
        x_vals = list(range(0, MAX_SHARES))           # [0, MAX_SHARES)

    # Clamp marker into current visible range to avoid out-of-bounds labels
    if x_vals:
        marker_reserve = int(np.clip(marker_reserve, x_vals[0], x_vals[-1]))
    else:
        marker_reserve = 0
        
    # Curves
    buy_vals = [buy_curve(x) for x in x_vals]
    sell_vals = [sell_curve(x) for x in x_vals]

    
    pb = float(buy_curve(marker_reserve))
    ps = float(sell_curve(marker_reserve))

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=x_vals, y=buy_vals, mode='lines', name='Buy Curve'))
    fig_curve.add_trace(go.Scatter(x=x_vals, y=sell_vals, mode='lines', name='Sell Curve'))

    # Marker with compact text
    fig_curve.add_trace(go.Scatter(
        x=[marker_reserve], y=[pb],
        mode='markers+text', name='Buy Point',
        text=[f"Shares: {marker_reserve}<br>Price: {pb:.2f}"],
        textposition="top right", marker=dict(size=10)
    ))
    fig_curve.add_trace(go.Scatter(
        x=[marker_reserve], y=[ps],
        mode='markers+text', name='Sell Point',
        text=[f"Shares: {marker_reserve}<br>Price: {ps:.2f}"],
        textposition="bottom right", marker=dict(size=10)
    ))

    # Crosshairs
    fig_curve.add_trace(go.Scatter(
        x=[marker_reserve, marker_reserve], y=[0, pb],
        mode='lines', line=dict(dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[0, marker_reserve], y=[pb, pb],
        mode='lines', line=dict(dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[marker_reserve, marker_reserve], y=[0, ps],
        mode='lines', line=dict(dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[0, marker_reserve], y=[ps, ps],
        mode='lines', line=dict(dash='dot'), showlegend=False
    ))

    # Optional inline labels
    if show_curve_labels:
        xs = list(range(int(x_vals[0]), int(x_vals[-1]) + 1, label_step))
        fig_curve.add_trace(go.Scatter(
            x=xs, y=[buy_curve(x) for x in xs],
            mode='markers+text', text=[f"{buy_curve(x):.2f}" for x in xs],
            textposition="top center", name="Buy labels", showlegend=False, marker=dict(size=5)
        ))
        fig_curve.add_trace(go.Scatter(
            x=xs, y=[sell_curve(x) for x in xs],
            mode='markers+text', text=[f"{sell_curve(x):.2f}" for x in xs],
            textposition="bottom center", name="Sell labels", showlegend=False, marker=dict(size=5)
        ))

    fig_curve.update_layout(
        title=f'{token_label} Price vs Shares — {curve_choice}',
        xaxis_title='Shares', yaxis_title='Price', hovermode='x unified'
    )

    # Axes ranges if custom
    if x_mode == "Custom":
        fig_curve.update_xaxes(range=[int(x_min), int(x_max)])
    if y_mode == "Custom":
        fig_curve.update_yaxes(range=[float(y_min), float(y_max)])

    st.plotly_chart(fig_curve, use_container_width=True)

    # --- Metrics under the chart ---
    q = int(marker_reserve)
    pb = float(buy_curve(q))
    ps = float(sell_curve(q))
    try:
        mcap_0_to_q = float(buy_delta(q) - buy_delta(0))
    except Exception:
        mcap_0_to_q = float('nan')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Quantity", f"{q:,}", border=True)
    m2.metric("Buy price", f"{pb:.2f}", border=True)
    m3.metric("Sell price", f"{ps:.2f}", border=True)
    m4.metric("Theoretical MCAP (∫ buy 0→q)", f"{mcap_0_to_q:,.2f}", border=True)

    if use_preset_for_markers:
        st.caption("Metrics computed at the preset quantity. Toggle in the sidebar to use live reserves instead.")


def render_simulator():
    # --- Input UI ---
    input_mode = st.radio("Select Input Mode", ["Quantity", "USDC"], horizontal=True)
    quantity = 0
    usdc_input = 0.0
    if input_mode == "Quantity":
        quantity = st.number_input("Enter Quantity", min_value=1, step=1)
    else:
        usdc_input = st.number_input("Enter USDC Amount", min_value=0.0, step=0.1)

    st.subheader("Buy/Sell Controls")

    cols = st.columns(3)
    for i, token in enumerate(['A', 'B', 'C']):
        with cols[i]:
            st.markdown(f"### Outcome {token}")

            reserve = st.session_state[f'reserve_{token}']
            buy_col, sell_col, _, _ = st.columns(4)

            with buy_col:
                if st.button(f"Buy {token}"):
                    if input_mode == "USDC":
                        quantity = find_quantity_for_usdc(reserve, usdc_input)

                    buy_price, _, buy_amt_delta, _ = get_all_metrics(reserve, quantity)

                    # Update both overall & token reserves
                    st.session_state.usdc_reserve += buy_amt_delta
                    st.session_state[f'usdc_reserve_{token}'] += buy_amt_delta

                    # update circulating share reserves
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

                        st.session_state.usdc_reserve -= sell_amt_delta
                        st.session_state[f'usdc_reserve_{token}'] -= sell_amt_delta

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

    # --- Display logs and analytics ---
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
        for i, tkn in enumerate(OUTCOMES):
            with token_cols[i]:
                st.subheader(f'Outcome {tkn}')
                reserve = st.session_state[f'reserve_{tkn}']
                mcap = round(st.session_state[f'usdc_reserve_{tkn}'], 2)
                price = round(buy_curve(reserve), 2)

                sub_cols = st.columns(4)
                with sub_cols[0]:
                    st.metric(f"Total Shares {tkn}", reserve)
                with sub_cols[1]:
                    st.metric(f"Price {tkn}", price)
                with sub_cols[2]:
                    st.metric(f"MCAP {tkn}", mcap)

        # Odds
        sub_cols_2 = st.columns(len(OUTCOMES))
        for i, tkn in enumerate(OUTCOMES):
            with sub_cols_2[i]:
                st.metric(f"Odds {tkn}", f"{odds[tkn]}x" if odds[tkn] != '-' else "-")

        st.subheader("Transaction Log")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Payout/Share Trend")
        df_viz = df[df['Payout/Share'] != '-'].copy()
        df_viz['Payout/Share'] = pd.to_numeric(df_viz['Payout/Share'], errors='coerce')
        fig = px.line(df_viz, x="Time", y="Payout/Share", color="Outcome", markers=True, title="Payout/Share Over Time")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Circulating Token Shares Over Time")
        df_reserve = df[df['Action'].isin(['Buy', 'Sell'])].copy()
        df_reserve['Time'] = pd.to_datetime(df_reserve['Time'])

        # Reconstruct reserves over time
        reserve_log = []
        running_reserves = {'A': 0, 'B': 0, 'C': 0}
        for _, row in df_reserve.iterrows():
            tkn = row['Outcome']
            qty = row['Quantity']
            running_reserves[tkn] += qty if row['Action'] == 'Buy' else -qty
            reserve_log.append({
                'Time': row['Time'],
                'Shares A': running_reserves['A'],
                'Shares B': running_reserves['B'],
                'Shares C': running_reserves['C']
            })

        df_reserve_reconstructed = pd.DataFrame(reserve_log)
        df_reserve_melt = df_reserve_reconstructed.melt(id_vars='Time', var_name='Token', value_name='Reserve')
        fig_stack = px.area(df_reserve_melt, x="Time", y="Reserve", color="Token", title="Token Shares vs. Time")
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("No actions taken yet. Use the buttons above to add rows.")


# ---------- Mode switch ----------
if app_mode == "Simulator":
    render_simulator()
else:
    render_curve_viewer()
