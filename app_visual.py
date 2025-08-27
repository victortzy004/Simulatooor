# file: app_visual.py
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
MAX_SHARES = 5000000 # 5M

SHARE_DISPLAY_OPTIONS = [50000, 100000, 250000, 500000, 1000000]
PRESET_SHARE_QUANTITY_OPTIONS = [10000, 25000, 50000, 100000, 250000, 500000, 1000000, 25000000]
APP_MODES = ["Curve Viewer", "Simulator"]

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


# #1 Model: Baseline
def secondary_buy_curve(x):      # your current model
    return (x**(1/3)/1000) + 0.1

def secondary_buy_delta(x):
    return (3.0/4000.0) * (x**(4.0/3.0)) + x/10.0

# =========================
# NumPy (vectorized) versions
# =========================
def _cbrt(x):
    # vectorized real cube root
    x = np.asarray(x, dtype=float)
    return np.cbrt(x)

# ---- FLAT MODE ----
def flat_buy_curve_np(x):
    return _cbrt(x) / 1000.0 + 0.1

def flat_buy_delta_np(x, C: float = 0.0):
    x = np.asarray(x, dtype=float)
    return (3.0/4000.0) * (np.abs(x) ** (4.0/3.0)) + 0.1 * x + C

def flat_sell_curve_np(x):
    x = np.asarray(x, dtype=float)
    t = (x - 500_000.0) / 1_000_000.0
    return 1.0 / (4.0 * (0.8 + np.exp(-t))) - 0.05

def flat_sell_delta_np(x, C: float = 0.0):
    x = np.asarray(x, dtype=float)
    t = (x - 500_000.0) / 1_000_000.0
    return 312_500.0 * np.log1p(0.8 * np.exp(t)) - 0.05 * x + C


# ---- STEEP MODE ----
def steep_buy_curve_np(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(0.0, x)) / 5000.0 + 0.01

def steep_buy_delta_np(x, C: float = 0.0):
    x = np.asarray(x, dtype=float)
    return (2.0/15000.0) * (np.maximum(0.0, x) ** 1.5) + 0.01 * x + C

def steep_sell_curve_np(x):
    x = np.asarray(x, dtype=float)
    t = (x - 500_000.0) / 1_000_000.0
    return 1.0 / (6.0 * (0.4 + np.exp(-t))) - 0.08 + x / 80_000_000.0

def steep_sell_delta_np(x, C: float = 0.0):
    x = np.asarray(x, dtype=float)
    t = (x - 500_000.0) / 1_000_000.0
    return (1_250_000.0/3.0) * np.log1p(0.4 * np.exp(t)) - 0.08 * x + (x**2) / 160_000_000.0 + C


# ---- MEDIUM MODE ----
def medium_buy_curve_np(x):
    x = np.asarray(x, dtype=float)
    return (np.maximum(0.0, x) ** (4.0/9.0)) / 3000.0 + 0.05

def medium_buy_delta_np(x, C: float = 0.0):
    x = np.asarray(x, dtype=float)
    return (3.0/13000.0) * (np.maximum(0.0, x) ** (13.0/9.0)) + 0.05 * x + C

def medium_sell_curve_np(x):
    x = np.asarray(x, dtype=float)
    t = (x - 500_000.0) / 1_000_000.0
    return 1.0 / (5.0 * (0.6 + np.exp(-t))) - 0.06 + x / 150_000_000.0

def medium_sell_delta_np(x, C: float = 0.0):
    x = np.asarray(x, dtype=float)
    t = (x - 500_000.0) / 1_000_000.0
    return (1_000_000.0/3.0) * np.log1p(0.6 * np.exp(t)) - 0.06 * x + (x**2) / 300_000_000.0 + C



CURVE_SETS = {
    "Baseline (Simulatoor default)": {
        "buy_curve": baseline_buy_curve,
        "sell_curve": baseline_sell_curve,
        "buy_delta": baseline_buy_delta,
        "sell_delta": baseline_sell_delta,
        "max_shares": 1000000,    # per-set override if you like
    },
     "Flat":   {"buy_curve": flat_buy_curve_np,   "buy_delta": flat_buy_delta_np,
               "sell_curve": flat_sell_curve_np, "sell_delta": flat_sell_delta_np, "max_shares": MAX_SHARES},
    "Steep":  {"buy_curve": steep_buy_curve_np,  "buy_delta": steep_buy_delta_np,
               "sell_curve": steep_sell_curve_np,"sell_delta": steep_sell_delta_np,"max_shares": MAX_SHARES},
    "Medium": {"buy_curve": medium_buy_curve_np, "buy_delta": medium_buy_delta_np,
               "sell_curve": medium_sell_curve_np,"sell_delta": medium_sell_delta_np,"max_shares": MAX_SHARES},
}


# Default existing simulator curve
SIMULATOR_BUY_CURVE = flat_buy_curve_np
SIMULATOR_BUY_DELTA = flat_buy_delta_np
SIMULATOR_SELL_CURVE = flat_sell_curve_np
SIMULATOR_SELL_DELTA = flat_sell_delta_np

# Search Functions
def qty_from_buy_usdc(reserve: int, usd: float) -> int:
    if usd <= 0:
        return 0
    # initial guess: linear approximation
    q = usd / max(SIMULATOR_BUY_CURVE(reserve), 1e-9)
    q = max(0.0, min(q, MAX_SHARES - reserve))

    for _ in range(12):
        f  = (SIMULATOR_BUY_DELTA(reserve + q) - SIMULATOR_BUY_DELTA(reserve)) - usd
        fp = max(SIMULATOR_BUY_CURVE(reserve + q), 1e-9)  # df/dq
        step = f / fp
        q -= step
        # clamp
        if q < 0.0: q = 0.0
        if q > (MAX_SHARES - reserve): q = float(MAX_SHARES - reserve)
        if abs(step) < 1e-6:
            break
    return int(q)

def qty_from_sell_usdc(reserve: int, usd: float) -> int:
    if usd <= 0:
        return 0
    # initial guess: linear approx using current sell price
    q = usd / max(SIMULATOR_SELL_CURVE(reserve), 1e-9)
    q = max(0.0, min(q, float(reserve)))

    for _ in range(12):
        f  = (SIMULATOR_SELL_DELTA(reserve) - SIMULATOR_SELL_CURVE(reserve - q)) - usd
        fp = max(SIMULATOR_SELL_CURVE(reserve - q), 1e-9)  # df/dq = price at (reserve - q)
        step = f / fp
        q -= step
        if q < 0.0: q = 0.0
        if q > reserve: q = float(reserve)
        if abs(step) < 1e-6:
            break
    return int(q)

# Configuration Init
st.set_page_config(page_title="42 DPM Visualizer", layout="wide")
st.title("42: Twin Bonding Curve Static Visualizooor")

with st.sidebar:
    st.header("Mode")
    app_mode = st.radio("Choose view", APP_MODES, index=0)

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
            PRESET_SHARE_QUANTITY_OPTIONS,
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
            options=SHARE_DISPLAY_OPTIONS,
            value=SHARE_DISPLAY_OPTIONS[3]
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
    buy_price = SIMULATOR_BUY_CURVE(new_x)
    sell_price = SIMULATOR_SELL_CURVE(x)
    buy_amt_delta = SIMULATOR_BUY_DELTA(new_x) - SIMULATOR_BUY_DELTA(x)
    sell_amt_delta = SIMULATOR_SELL_DELTA(x) - SIMULATOR_SELL_DELTA(x - quantity) if x - quantity >= 0 else 0
    return buy_price, sell_price, buy_amt_delta, sell_amt_delta


# --- smart sampler: dense around reserve, sparse elsewhere ---
def _curve_samples(max_shares: int, reserve: int, dense_pts: int = 1500, sparse_pts: int = 600) -> np.ndarray:
    # 0-based domain (so q=0 works)
    if max_shares <= 0:
        return np.array([0], dtype=int)

    # Dense window = ±3% of domain (capped at ±50k)
    half = min(int(0.03 * max_shares), 50_000)
    lo = max(0, reserve - half)
    hi = min(max_shares, reserve + half)

    dense = np.linspace(lo, hi, num=max(2, dense_pts), dtype=int)

    left = np.array([], dtype=int)
    if lo > 0:
        # logspace from 1→lo (avoid log10(0)); include 0 manually after unique
        left = np.unique(np.logspace(0, np.log10(max(1, lo)), num=max(2, sparse_pts // 2), base=10.0)).astype(int)
        left = np.concatenate([np.array([0], dtype=int), left[left < lo]])

    right = np.array([], dtype=int)
    if hi < max_shares:
        # logspace from hi+1 → max_shares
        start = max(hi + 1, 1)  # avoid log10(0)
        right = np.unique(np.logspace(np.log10(start), np.log10(max_shares), num=max(2, sparse_pts // 2), base=10.0)).astype(int)
        right = right[(right > hi) & (right <= max_shares)]

    xs = np.unique(np.concatenate([left, dense, right]))
    return xs


# --- robust vectorized evaluation (handles math- or numpy-based curve fns) ---
def _eval_curve(fn, xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    try:
        y = fn(xs)                       # if fn is numpy-aware
        return np.asarray(y, dtype=float)
    except Exception:
        # scalar fallback
        return np.fromiter((fn(float(v)) for v in xs), dtype=float, count=len(xs))


# Cache the heavy sampling+eval by (curve_key, MAX_SHARES, reserve, x_lo, x_hi, dense/sparse)
@st.cache_data(show_spinner=False)
def get_curve_series(curve_key: str, max_shares: int, reserve: int, x_lo: int, x_hi: int,
                     dense_pts: int = 1500, sparse_pts: int = 600):
    xs_all = _curve_samples(max_shares, reserve, dense_pts=dense_pts, sparse_pts=sparse_pts)
    # keep only what’s visible
    mask = (xs_all >= x_lo) & (xs_all <= x_hi)
    xs = xs_all[mask]
    # use the *currently bound* active curves (from your CURVE_SETS selection)
    ys_buy = _eval_curve(buy_curve, xs)
    ys_sell = _eval_curve(sell_curve, xs)
    return xs, ys_buy, ys_sell

# Core Feature #1: Bonding curve visualisooor with different curves
def render_curve_viewer():
    st.subheader("🔁 Bonding Curves Visualisation")

    # Marker quantity: preset, custom, or live (A)
    if qty_mode == "Preset list":
        marker_reserve = int(preset_qty)
    elif qty_mode == "Custom amount":
        marker_reserve = int(custom_qty)
    else:  # Use live reserve (A)
        marker_reserve = int(st.session_state.get('reserve_A', 0))

    # Visible X-range for plotting (inclusive)
    if x_mode == "Custom":
        x_lo, x_hi = int(x_min), int(x_max)
    else:
        x_lo, x_hi = 0, int(MAX_SHARES)

    # Clamp marker into current visible range
    marker_reserve = int(np.clip(marker_reserve, x_lo, x_hi))

    # Smart-sampled, cached series for the active curve *and* visible range
    xs, ys_buy, ys_sell = get_curve_series(
        curve_key=curve_choice,
        max_shares=int(MAX_SHARES),
        reserve=marker_reserve,
        x_lo=x_lo, x_hi=x_hi,
        dense_pts=1500, sparse_pts=600
    )

    # Prices at marker (use the active bound functions)
    pb = float(buy_curve(marker_reserve))
    ps = float(sell_curve(marker_reserve))

    # Plot
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=xs, y=ys_buy, mode='lines', name='Buy Curve'))
    fig_curve.add_trace(go.Scatter(x=xs, y=ys_sell, mode='lines', name='Sell Curve'))

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
        x=[marker_reserve, marker_reserve], y=[min(ys_buy.min(), ys_sell.min(), 0), pb],
        mode='lines', line=dict(dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[x_lo, marker_reserve], y=[pb, pb],
        mode='lines', line=dict(dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[marker_reserve, marker_reserve], y=[min(ys_buy.min(), ys_sell.min(), 0), ps],
        mode='lines', line=dict(dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[x_lo, marker_reserve], y=[ps, ps],
        mode='lines', line=dict(dash='dot'), showlegend=False
    ))

    # Optional inline labels along the curves
    if show_curve_labels:
        label_xs = np.arange(x_lo, x_hi + 1, int(label_step))
        label_buy = _eval_curve(buy_curve, label_xs)
        label_sell = _eval_curve(sell_curve, label_xs)
        fig_curve.add_trace(go.Scatter(
            x=label_xs, y=label_buy,
            mode='markers+text',
            text=[f"{v:.2f}" for v in label_buy],
            textposition="top center", name="Buy labels", showlegend=False, marker=dict(size=5)
        ))
        fig_curve.add_trace(go.Scatter(
            x=label_xs, y=label_sell,
            mode='markers+text',
            text=[f"{v:.2f}" for v in label_sell],
            textposition="bottom center", name="Sell labels", showlegend=False, marker=dict(size=5)
        ))

    fig_curve.update_layout(
        title=f'Price vs Shares — {curve_choice}',
        xaxis_title='Shares',
        yaxis_title='Price',
        hovermode='x unified'
    )

    # Axes ranges if custom (Y stays as-is unless you set custom)
    if x_mode == "Custom":
        fig_curve.update_xaxes(range=[x_lo, x_hi])
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

    if qty_mode == "Preset list":
        st.caption("Metrics computed at the preset quantity.")
    elif qty_mode == "Custom amount":
        st.caption("Metrics computed at the custom quantity slider value.")
    else:
        st.caption("Metrics computed at the live reserve of outcome A.")


# Core Feature #2: Simulator with 3 Outcome Tokens
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
                        quantity =qty_from_buy_usdc(reserve, usdc_input)

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
                        quantity = qty_from_sell_usdc(reserve, usdc_input)

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

        outcome_reserves = {
            t: int(st.session_state.get(f"reserve_{t}", 0))
            for t in OUTCOMES  # OUTCOMES = ['A','B','C']
        }


        total_market = sum(outcome_reserves.values())
        odds = {
            t: (round(1 / (outcome_reserves[t] / total_market), 4)
                if outcome_reserves[t] > 0 and total_market > 0 else '-')
            for t in OUTCOMES
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
                price = round(SIMULATOR_BUY_CURVE(reserve), 2)

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


        # Trend #1: Payout/Share Historical Visualisation
        st.subheader("📈 Payout/Share Trend")
        df_viz = df[df['Payout/Share'] != '-'].copy()
        df_viz['Payout/Share'] = pd.to_numeric(df_viz['Payout/Share'], errors='coerce')
        fig = px.line(df_viz, x="Time", y="Payout/Share", color="Outcome", markers=True, title="Payout/Share Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.divider()


    #     # Trend #2: Bonding Curves by Outcome (optimized)
    #     st.subheader("🔁 Bonding Curves by Outcome")
    #     tabs = st.tabs(OUTCOMES)

    #     outcome_reserves = {
    #         t: int(st.session_state.get(f"reserve_{t}", 0))
    #         for t in OUTCOMES  # OUTCOMES = ['A','B','C']
    #     }


    #     for token, tab in zip(OUTCOMES, tabs):
    #         reserve = int(outcome_reserves.get(token, 0))

    #         # point annotations at the *exact* reserve using your scalar functions
    #         buy_price_now = float(SIMULATOR_BUY_CURVE(reserve))
    #         sell_price_now = float(SIMULATOR_SELL_CURVE(reserve))

    #         # smart-sampled, cached series around this reserve
    #         xs, buy_vals, sell_vals = get_curve_series(MAX_SHARES, reserve)

    #         fig_curve = go.Figure()
    #         fig_curve.add_trace(go.Scattergl(
    #             x=xs, y=buy_vals, mode='lines', name='Buy Curve', line=dict(color='green'
    #         )))
    #         fig_curve.add_trace(go.Scattergl(
    #             x=xs, y=sell_vals, mode='lines', name='Sell Curve', line=dict(color='red'
    #         )))

    #         # Buy point annotation
    #         fig_curve.add_trace(go.Scatter(
    #             x=[reserve], y=[buy_price_now], mode='markers+text',
    #             name=f'{token} Buy Point',
    #             text=[f"Shares: {reserve}<br>Price: {buy_price_now:.4f}"],
    #             textposition="top right",
    #             marker=dict(size=10, color='green'),
    #             showlegend=False
    #         ))

    #         # Sell point annotation
    #         fig_curve.add_trace(go.Scatter(
    #             x=[reserve], y=[sell_price_now], mode='markers+text',
    #             name=f'{token} Sell Point',
    #             text=[f"Shares: {reserve}<br>Price: {sell_price_now:.4f}"],
    #             textposition="bottom right",
    #             marker=dict(size=10, color='red'),
    #             showlegend=False
    #         ))

    #         # Dashed helper lines (use max with 0 to avoid negative bottoms in view)
    #         y0_buy = max(0.0, min(buy_vals.min(), sell_vals.min(), buy_price_now, sell_price_now))
    #         fig_curve.add_trace(go.Scatter(
    #                 x=[reserve, reserve], y=[y0_buy, buy_price_now], mode='lines',
    #                 line=dict(dash='dot'), showlegend=False
    #             ))
    #         fig_curve.add_trace(go.Scatter(
    #                 x=[xs.min(), reserve], y=[buy_price_now, buy_price_now], mode='lines',
    #                 line=dict(dash='dot'), showlegend=False
    #             ))
    #         fig_curve.add_trace(go.Scatter(
    #                 x=[reserve, reserve], y=[y0_buy, sell_price_now], mode='lines',
    #                 line=dict(dash='dot'), showlegend=False
    #             ))
    #         fig_curve.add_trace(go.Scatter(
    #                 x=[xs.min(), reserve], y=[sell_price_now, sell_price_now], mode='lines',
    #                 line=dict(dash='dot'), showlegend=False
    #             ))

    #         fig_curve.update_layout(
    #             title=f'{token} Price vs Shares',
    #             xaxis_title='Shares',
    #             yaxis_title='Price',
    #             hovermode="x unified",
    #             # uirevision="curves",  # keeps view on widget changes
    #             # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    #         )

    #         tab.plotly_chart(fig_curve, use_container_width=True, key=f"chart_curve_{token}")
    #     st.divider()

    #     # Trend #3: Historical circulating token shares
    #     st.subheader("📊 Circulating Token Shares Over Time")
    #     df_reserve = df[df['Action'].isin(['Buy', 'Sell'])].copy()
    #     df_reserve['Time'] = pd.to_datetime(df_reserve['Time'])

    #     # Reconstruct reserves over time
    #     reserve_log = []
    #     running_reserves = {'A': 0, 'B': 0, 'C': 0}
    #     for _, row in df_reserve.iterrows():
    #         tkn = row['Outcome']
    #         qty = row['Quantity']
    #         running_reserves[tkn] += qty if row['Action'] == 'Buy' else -qty
    #         reserve_log.append({
    #             'Time': row['Time'],
    #             'Shares A': running_reserves['A'],
    #             'Shares B': running_reserves['B'],
    #             'Shares C': running_reserves['C']
    #         })

    #     df_reserve_reconstructed = pd.DataFrame(reserve_log)
    #     df_reserve_melt = df_reserve_reconstructed.melt(id_vars='Time', var_name='Token', value_name='Reserve')
    #     fig_stack = px.area(df_reserve_melt, x="Time", y="Reserve", color="Token", title="Token Shares vs. Time")
    #     st.plotly_chart(fig_stack, use_container_width=True)
    # else:
    #     st.info("No actions taken yet. Use the buttons above to add rows.")
    # Trend #2: Bonding Curves by Outcome (optimized)
    st.subheader("🔁 Bonding Curves by Outcome")
    tabs = st.tabs(OUTCOMES)

    # per-outcome reserves from session state
    outcome_reserves = {t: int(st.session_state.get(f"reserve_{t}", 0)) for t in OUTCOMES}

    # visible X range
    if x_mode == "Custom":
        x_lo, x_hi = int(x_min), int(x_max)
    else:
        x_lo, x_hi = 0, int(MAX_SHARES)

    for token, tab in zip(OUTCOMES, tabs):
        reserve = outcome_reserves.get(token, 0)
        reserve_clamped = int(np.clip(reserve, x_lo, x_hi))

        # smart-sampled, cached series for THIS curve set & outcome
        xs, buy_vals, sell_vals = get_curve_series(
            curve_key=f"{curve_choice}:{token}",
            max_shares=int(MAX_SHARES),
            reserve=reserve_clamped,
            x_lo=x_lo,
            x_hi=x_hi,
            dense_pts=1200,
            sparse_pts=500,
        )

        # prices at the exact reserve using active curve funcs
        pb = float(buy_curve(reserve))
        ps = float(sell_curve(reserve))

        # plot
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scattergl(x=xs, y=buy_vals, mode='lines', name='Buy Curve'))
        fig_curve.add_trace(go.Scattergl(x=xs, y=sell_vals, mode='lines', name='Sell Curve'))

        # marker annotations
        fig_curve.add_trace(go.Scatter(
            x=[reserve_clamped], y=[pb], mode='markers+text', name='Buy Point',
            text=[f"Shares: {reserve}<br>Price: {pb:.4f}"], textposition="top right",
            marker=dict(size=10), showlegend=False
        ))
        fig_curve.add_trace(go.Scatter(
            x=[reserve_clamped], y=[ps], mode='markers+text', name='Sell Point',
            text=[f"Shares: {reserve}<br>Price: {ps:.4f}"], textposition="bottom right",
            marker=dict(size=10), showlegend=False
        ))

        # crosshair helpers
        y_floor = float(min(np.nanmin(buy_vals), np.nanmin(sell_vals), 0.0))
        fig_curve.add_trace(go.Scatter(x=[reserve_clamped, reserve_clamped], y=[y_floor, pb],
                                    mode='lines', line=dict(dash='dot'), showlegend=False))
        fig_curve.add_trace(go.Scatter(x=[x_lo, reserve_clamped], y=[pb, pb],
                                    mode='lines', line=dict(dash='dot'), showlegend=False))
        fig_curve.add_trace(go.Scatter(x=[reserve_clamped, reserve_clamped], y=[y_floor, ps],
                                    mode='lines', line=dict(dash='dot'), showlegend=False))
        fig_curve.add_trace(go.Scatter(x=[x_lo, reserve_clamped], y=[ps, ps],
                                    mode='lines', line=dict(dash='dot'), showlegend=False))

        fig_curve.update_layout(
            title=f'{token} Price vs Shares — {curve_choice}',
            xaxis_title='Shares',
            yaxis_title='Price',
            hovermode="x unified",
        )

        # respect custom axes
        if x_mode == "Custom":
            fig_curve.update_xaxes(range=[x_lo, x_hi])
        if y_mode == "Custom":
            fig_curve.update_yaxes(range=[float(y_min), float(y_max)])

        tab.plotly_chart(fig_curve, use_container_width=True, key=f"chart_curve_{token}")

    st.divider()


# ---------- Mode switch ----------
if app_mode == "Simulator":
    render_simulator()
else:
    render_curve_viewer()
