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
MAX_SHARES = 50000000  # 50M

# Time-scaling constants for redemption spread
T_KINK_DEFAULT = 0.67       # default point where time-scaling starts to kick in
T_GROWTH_DEFAULT = 3.0      # default exponent; aggressiveness of late-stage tax growth

# These are the *current* values used in the formula (can be overridden by sliders)
T_KINK = T_KINK_DEFAULT
T_GROWTH = T_GROWTH_DEFAULT

# This will be overridden by the Streamlit slider each run
TIME_PHASE = 0.0    # 0.0 = just opened, 1.0 = at redemption

SHARE_DISPLAY_OPTIONS = [500000, 1000000, 2500000, 5000000]
PRESET_SHARE_QUANTITY_OPTIONS = [100000, 500000,1000000, 2500000, 5000000, 7500000, 10000000, 25000000, 35000000, 45000000]
APP_MODES = ["Curve Viewer", "Simulator"]

CURVE_VIEW_MODES = ["Buy Curve", "Sell Spread", "Effective Sell (Net)"]

# ===========================================================
# NumPy (vectorized) versions

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


# ---- STEEP MODE ----
def steep_buy_curve_np(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(0.0, x)) / 5000.0 + 0.01

def steep_buy_delta_np(x, C: float = 0.0):
    x = np.asarray(x, dtype=float)
    return (2.0/15000.0) * (np.maximum(0.0, x) ** 1.5) + 0.01 * x + C


# ---- MEDIUM MODE ----
def medium_buy_curve_np(x):
    x = np.asarray(x, dtype=float)
    return (np.maximum(0.0, x) ** (4.0/9.0)) / 3000.0 + 0.05

def medium_buy_delta_np(x, C: float = 0.0):
    x = np.asarray(x, dtype=float)
    return (3.0/13000.0) * (np.maximum(0.0, x) ** (13.0/9.0)) + 0.05 * x + C


# ===== Sale Tax Helpers (new) =====
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def sale_tax_rate(q: int, C: int, p: float | None = None) -> float:
    """
    q: number of shares sold in *this order*
    C: total circulating shares (reserve) before the sale
    p: market phase in [0, 1] (0 = just started, 1 = at redemption).
       If None, uses the global TIME_PHASE (set by UI).

    Base (size-of-order) component:
        base = 1.15 - 1.3/(1 + e^(4*(q/C) - 2))

    Then scaled by:
        s(C): supply-based multiplier
            1.00  if C < 1,000,000
            1.25  if 1,000,000 <= C < 10,000,000
            1.50  if C >= 10,000,000

        t(p): time-based multiplier
            t(p) = (1 + max(0, p - T_KINK)) ** T_GROWTH

    Final tax is clamped to [0, 1].
    """
    if C <= 0 or q <= 0:
        return 0.0

    # If UI didn't explicitly pass a phase, use the current slider state
    if p is None:
        p = TIME_PHASE

    # Clamp inputs
    X = _clamp01(float(q) / float(C))       # fraction of supply this order is selling
    p = _clamp01(float(p))                  # market phase 0..1

    # --- Base size-of-order component (unchanged) ---
    base = 1.15 - 1.3 / (1.0 + math.e ** (4.0 * X - 2.0))

    # --- Supply multiplier s(C) ---
    if C < 1_000_000:
        s_mult = 1.0
    elif C < 10_000_000:
        s_mult = 1.25
    else:
        s_mult = 1.5

    # --- Time multiplier t(p) ---
    # No effect until phase passes T_KINK, then grows super-linearly
    excess_phase = max(0.0, p - T_KINK)
    t_mult = (1.0 + excess_phase) ** T_GROWTH

    tax = base * s_mult * t_mult
    return _clamp01(tax)

def sell_proceeds_net(reserve: int, q: int) -> float:
    """Net USDC user receives after the order-level sale tax."""
    if q <= 0 or reserve <= 0:
        return 0.0
    q = min(q, reserve)
    gross = buy_delta(reserve) - buy_delta(reserve - q)
    tax = sale_tax_rate(q, reserve)
    net = gross * (1.0 - tax)
    return max(0.0, float(net))

def current_marginal_sell_price_after_tax(reserve: int) -> float:
    """
    A single-share 'instant' effective price (for UI display).
    Uses q=1 to estimate marginal price after tax.
    """
    if reserve <= 0:
        return 0.0
    gross1 = buy_delta(reserve) - buy_delta(reserve - 1)
    tax1 = sale_tax_rate(1, reserve)
    return max(0.0, float(gross1 * (1.0 - tax1)))

def sell_gross_from_bonding(reserve: int, q: int) -> float:
    """Bonding-curve gross proceeds (no tax), using your buy integral as the primitive."""
    if q <= 0 or reserve <= 0:
        return 0.0
    q = min(q, reserve)
    return float(buy_delta(reserve) - buy_delta(reserve - q))

# vectorized tax (for charts)
_sale_tax_rate_vec = np.vectorize(sale_tax_rate, otypes=[float])

def metrics_from_qty(x: int, q: int):
    """
    Returns (in order):
      buy_price         = buy spot after adding q
      sell_price        = marginal 1-share sell price after tax (display only)
      buy_amt_delta     = USDC to buy q
      sell_amt_delta    = USDC received to sell q after tax
      sell_tax_rate_used= order-level tax applied for selling q out of reserve x (0..1)
    """
    q = int(max(0, q))
    new_x = x + q

    buy_price = buy_curve(new_x)
    sell_price = current_marginal_sell_price_after_tax(x)

    buy_amt_delta = buy_delta(new_x) - buy_delta(x)

    # Clamp sell qty to circulating reserve for proceeds + tax computation
    q_eff = min(q, x if x > 0 else 0)
    if q_eff > 0 and x > 0:
        tax_used = sale_tax_rate(q_eff, x)
        sell_amt_delta = sell_proceeds_net(x, q_eff)
    else:
        tax_used = 0.0
        sell_amt_delta = 0.0

    return buy_price, sell_price, buy_amt_delta, sell_amt_delta, float(tax_used)


def qty_from_buy_usdc(reserve: int, usd: float) -> int:
    if usd <= 0:
        return 0
    # initial guess: linear approximation
    q = usd / max(buy_curve(reserve), 1e-9)
    q = max(0.0, min(q, MAX_SHARES - reserve))

    for _ in range(12):
        f  = (buy_delta(reserve + q) - buy_delta(reserve)) - usd
        fp = max(buy_curve(reserve + q), 1e-9)  # df/dq
        step = f / fp
        q -= step
        # clamp
        if q < 0.0: q = 0.0
        if q > (MAX_SHARES - reserve): q = float(MAX_SHARES - reserve)
        if abs(step) < 1e-6:
            break
    return int(q)

def qty_from_sell_usdc(reserve: int, usd: float) -> int:
    if usd <= 0.0 or reserve <= 0:
        return 0

    # initial guess using current *net* marginal price
    p0_net = max(current_marginal_sell_price_after_tax(reserve), 1e-12)
    q_guess = min(reserve, usd / p0_net)

    # robust binary search on net proceeds
    lo, hi = 0, int(reserve)
    # tighten bounds around the guess to speed up
    lo = max(0, int(q_guess * 0.25))
    hi = min(int(reserve), max(lo, int(q_guess * 1.75)))

    while lo < hi:
        mid = (lo + hi + 1) // 2
        net = sell_proceeds_net(reserve, mid)
        if net <= usd:
            lo = mid
        else:
            hi = mid - 1
    return int(lo)

def format_usdc_compact(value: float) -> str:
    """Compact money: $999.99, $1.2K, $12.3M, $4B, $5T (no trailing .0)."""
    sign = "-" if value < 0 else ""
    v = abs(float(value))

    units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for i, (scale, suf) in enumerate(units):
        if v >= scale:
            q = v / scale
            # adaptive decimals
            dec = 2 if q < 10 else (1 if q < 100 else 0)
            s = f"{q:.{dec}f}"

            # If rounding hit 1000 of current unit, bump to the next unit
            if float(s) >= 1000 and i > 0:
                next_scale, next_suf = units[i - 1]
                q = v / next_scale
                dec = 2 if q < 10 else (1 if q < 100 else 0)
                s = f"{q:.{dec}f}"
                suf = next_suf

            # strip trailing zeros / dot
            s = s.rstrip("0").rstrip(".")
            return f"{sign}${s}{suf}"

    # < 1,000 — show standard money
    return f"{sign}${v:,.2f}"



CURVE_SETS = {
    "Default": {
        "buy_curve": flat_buy_curve_np,
        "buy_delta": flat_buy_delta_np,
        "sell_tax_curve": _sale_tax_rate_vec,
        "max_shares": MAX_SHARES
    },
    # "Steep": {
    #     "buy_curve": steep_buy_curve_np,
    #     "buy_delta": steep_buy_delta_np,
    #     "sell_tax_curve": _sale_tax_rate_vec,
    #     "max_shares": MAX_SHARES
    # },
    # "Medium": {
    #     "buy_curve": medium_buy_curve_np,
    #     "buy_delta": medium_buy_delta_np,
    #     "sell_tax_curve": _sale_tax_rate_vec,
    #     "max_shares": MAX_SHARES
    # },
}



# Configuration Init
st.set_page_config(page_title="Bonding Curve Simulator", layout="wide")
st.title("42: Twin Bonding Curve Simulatoooor")

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

# with st.sidebar:
#     st.header("Market Phase (Time)")
#     TIME_PHASE = st.slider(
#         "Progress toward deadline",
#         min_value=0.0,
#         max_value=1.0,
#         value=0.0,
#         step=0.01,
#         help=(
#             "0.00 = market just opened, 1.00 = at redemption. "
#             "Controls the time-based multiplier t(p) for the sell spread."
#         ),
#     )
#     st.caption(
#         f"Current phase p = **{TIME_PHASE:.2f}** → "
#         f"time multiplier t(p) = (1 + max(0, p - {T_KINK}))^{T_GROWTH}"
#     )
with st.sidebar:
    st.header("Market Phase (Time)")
    TIME_PHASE = st.slider(
        "Progress toward deadline",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help=(
            "0.00 = market just opened, 1.00 = at redemption. "
            "Controls the time-based multiplier t(p) for the sell spread."
        ),
    )

    # --- Advanced controls for the time multiplier t(p) ---
    with st.expander("Advanced time-scaling (t(p))", expanded=False):
        t_kink_slider = st.slider(
            "t_kink (phase threshold)",
            min_value=0.0,
            max_value=1.0,
            value=float(T_KINK_DEFAULT),
            step=0.01,
            help="Phase p at which the time-based tax starts growing (> t_kink increases spread).",
        )

        t_growth_slider = st.slider(
            "t_growth (growth exponent)",
            min_value=1.0,
            max_value=5.0,
            value=float(T_GROWTH_DEFAULT),
            step=0.1,
            help="Exponent controlling how sharply tax increases once p > t_kink.",
        )

        # Optional reset button back to defaults
        if st.button("Reset to defaults"):
            t_kink_slider = T_KINK_DEFAULT
            t_growth_slider = T_GROWTH_DEFAULT

    # Override the globals used inside sale_tax_rate
    T_KINK = t_kink_slider
    T_GROWTH = t_growth_slider

    # Small summary of the current settings
    st.caption(
        f"Current phase p = **{TIME_PHASE:.2f}**  \n"
        f"t_kink = **{T_KINK:.2f}**, t_growth = **{T_GROWTH:.2f}**  \n"
        f"t(p) = (1 + max(0, p - t_kink))^t_growth"
    )


st.caption(f"Active curve set: **{curve_choice}**")
# Bind active curve functions for the rest of the app
_active = CURVE_SETS[curve_choice]
buy_curve     = _active["buy_curve"]
sell_tax_curve    = _active["sell_tax_curve"]
buy_delta     = _active["buy_delta"]
# sell_delta    = _active["sell_delta"]
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
    approx_ymax = max(buy_curve(MAX_SHARES-1), 0)
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

# --- robust vectorized evaluation for 2-arg curves like sale_tax_rate(q, C) ---
def _eval_curve_2arg(fn, xs: np.ndarray, C: float) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    # Try vectorized call first (works for np.vectorize or true numpy funcs)
    try:
        y = fn(xs, C)
        return np.asarray(y, dtype=float)
    except TypeError:
        # Fallback to scalar calls
        return np.fromiter((fn(float(v), float(C)) for v in xs), dtype=float, count=len(xs))

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
    # two-arg evaluator for sale tax: sale_tax_rate(q, C=reserve)
    ys_sell = _eval_curve_2arg(sell_tax_curve, xs, reserve)
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

    # for view_mode in CURVE_VIEW_MODES:
    view_mode_tabs = st.tabs(CURVE_VIEW_MODES)  # token-level tabs only
    

    with view_mode_tabs[0]:
        # Prices at marker (use the active bound functions)
        pb = float(buy_curve(marker_reserve))

        # Plot
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=xs, y=ys_buy, mode='lines', name='Buy Curve'))

        # Marker with compact text
        fig_curve.add_trace(go.Scatter(
            x=[marker_reserve], y=[pb],
            mode='markers+text', name='Buy Point',
            text=[f"Shares: {marker_reserve}<br>Price: {pb:.2f}"],
            textposition="top right", marker=dict(size=10)
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

        # Optional inline labels along the curves
        if show_curve_labels:
            label_xs = np.arange(x_lo, x_hi + 1, int(label_step))
            label_buy = _eval_curve(buy_curve, label_xs)

            fig_curve.add_trace(go.Scatter(
                x=label_xs, y=label_buy,
                mode='markers+text',
                text=[f"{v:.2f}" for v in label_buy],
                textposition="top center", name="Buy labels", showlegend=False, marker=dict(size=5)
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

    with view_mode_tabs[1]:
        steps = 200
        X = np.linspace(0.0, 1.0, steps + 1)
        q_grid = (X * marker_reserve).astype(int)

        # vectorized tax rate for selling q out of current reserve
        tax_y = _sale_tax_rate_vec(q_grid, marker_reserve)

        fig_tax = go.Figure()
        
        fig_tax.add_trace(go.Scattergl(
            x=X * 100.0, 
            y=tax_y, mode='lines', 
            marker=dict(size=10),  
            name='Spread Rate', 
            showlegend=True
        ))
        
        fig_tax.update_layout(
        title='Sell Spread vs % of Supply Sold (per order)',
        xaxis_title='% of Current Supply Sold in Order',
        yaxis_title='Spread Rate',
        hovermode="x unified",   # nice unified hover; vertical guide
        spikedistance=-1         # show spikes whenever the mouse is in the plot
        )

        # X spikes (to x-axis)
        fig_tax.update_xaxes(
        showspikes=True,
        spikemode="across",      # draw across the plot area
        spikesnap="cursor",      # follow the cursor position
        spikedash="dot"          # dotted line
        )


        st.plotly_chart(fig_tax, use_container_width=True, key=f"sell_spread_curve")
    
    with view_mode_tabs[2]:  # "Effective Sell (Net)"
        # up to 10% of supply (at least 1)
        q_max = max(1, marker_reserve // 10)
        q_axis = np.linspace(1, q_max, 200).astype(int)

        # gross proceeds via integral difference; then apply order-level tax
        bd_C      = np.vectorize(buy_delta,  otypes=[float])(np.full_like(q_axis, marker_reserve))
        bd_C_minQ = np.vectorize(buy_delta,  otypes=[float])(marker_reserve - q_axis)
        gross     = bd_C - bd_C_minQ

        tax = _sale_tax_rate_vec(q_axis, np.full_like(q_axis, marker_reserve))
        net = np.maximum(0.0, gross * (1.0 - tax))
        avg_net = net / np.maximum(1, q_axis)

        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scattergl(
            x=q_axis, 
            y=avg_net, 
            mode='lines', 
            name=f'Share Avg Net Sell Price',
            showlegend=True
        ))
        fig_eff.update_layout(
            title=f'Effective Avg Net Sell Price vs Quantity — Share',
            xaxis_title='Quantity sold in a single order',
            yaxis_title='Avg Net Sell Price (USDC/share)',
            hovermode="x unified"
        )
        st.plotly_chart(fig_eff, use_container_width=True, key=f"effective_sell_curve")


    # ===============

 

    # --- Metrics under the chart ---
    q = int(marker_reserve)
    pb = float(buy_curve(q))
    # ps = float(sell_tax_curve(q))
    try:
        mcap_0_to_q = float(buy_delta(q) - buy_delta(0))
    except Exception:
        mcap_0_to_q = float('nan')

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Quantity", f"{q:,}", border=True)
    m2.metric("Buy price", f"{pb:.2f}", border=True)
    m3.metric("Theoretical MCAP (∫ buy 0→q)", f"{format_usdc_compact(mcap_0_to_q)}", border=True)

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
        usdc_str = st.text_input("Enter USDC Amount", key="usdc_input_raw", placeholder="0.00")
        try:
            usdc_input = float(usdc_str) if usdc_str.strip() else 0.0
            if usdc_input < 0:
                st.warning("USDC must be ≥ 0.")
                usdc_input = 0.0
        except ValueError:
            st.warning("Enter a valid number, e.g. 123.45")
            usdc_input = 0.0
    st.subheader("Buy/Sell Controls")
    

    cols = st.columns(3)
    for i, token in enumerate(['A', 'B', 'C']):

        with cols[i]:
            st.markdown(f"### Outcome {token}")

            reserve = st.session_state[f'reserve_{token}']
            price_now = round(buy_curve(reserve), 2)
            mcap_now = round(reserve, 2)
            st.text(f"Current Price: ${price_now}")
            st.text(f"MCAP: {format_usdc_compact(mcap_now)}")
            # --- Preview section (live estimate) ---
            # derive est quantities from current inputs (without executing)
            if input_mode == "Quantity":
                est_q_buy = quantity
                est_q_sell = quantity
            else:
                est_q_buy = qty_from_buy_usdc(reserve, usdc_input)
                est_q_sell = qty_from_sell_usdc(reserve, usdc_input)

            # compute deltas for estimates (clamp to non-negative)
            est_q_buy = max(0, int(est_q_buy))
            est_q_sell = max(0, int(est_q_sell))

            # buy estimate
            if est_q_buy > 0:
                _, _, est_buy_cost, _, _ = metrics_from_qty(reserve, est_q_buy)
            else:
                est_buy_cost = 0.0

            # sell estimate + tax %
            if est_q_sell > 0:
                _, _, _, est_sell_proceeds, est_tax_rate = metrics_from_qty(reserve, est_q_sell)
            else:
                est_sell_proceeds = 0.0
                est_tax_rate = 0.0

            st.caption(
                f"Est. Buy Cost ({est_q_buy}x sh): **{est_buy_cost:,.2f} USDC**  \n"
                f"Est. Sell Proceeds ({est_q_sell}x sh): **{est_sell_proceeds:,.2f} USDC**  \n"
                f"Order Tax on Sell: **{est_tax_rate*100:.2f}%**"
            )

            buy_col, sell_col, _, _ = st.columns(4)

            with buy_col:
                if st.button(f"Buy {token}"):
                    if input_mode == "USDC":
                        quantity =qty_from_buy_usdc(reserve, usdc_input)

                    buy_price, _, buy_amt_delta, _, _= metrics_from_qty(reserve, quantity)

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
                        "Sell Price": None,
                        "BuyAmt_Delta": round(buy_amt_delta, 4),
                        "SellAmt_Delta": None,
                        "Token Cir. Shares": st.session_state[f'reserve_{token}'],
                        "Token MCAP": st.session_state[f'usdc_reserve_{token}'],
                        "Overall USDC Reserve": round(st.session_state.usdc_reserve, 2),
                        "Payout/Share": payout_share
                    })
                     # Immediately re-run so the header stats (price/MCAP) reflect updated reserves
                    st.rerun()

            with sell_col:
                if st.button(f"Sell {token}"):
                    reserve = st.session_state[f'reserve_{token}']

                    if input_mode == "USDC":
                        quantity = qty_from_sell_usdc(reserve, usdc_input)

                    if reserve >= quantity:
                        _, sell_price, _, sell_amt_delta, _ = metrics_from_qty(reserve, quantity)

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
                            "Buy Price": None,
                            "Sell Price": round(sell_price, 4),
                            "BuyAmt_Delta": None,
                            "SellAmt_Delta": round(sell_amt_delta, 4),
                            "Token Cir. Shares": st.session_state[f'reserve_{token}'],
                            "Token MCAP": st.session_state[f'usdc_reserve_{token}'],
                            "Overall USDC Reserve": round(st.session_state.usdc_reserve, 2),
                            "Payout/Share": payout_share
                        })
                        st.rerun()
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

        # Keep df Arrow-friendly
        numeric_cols = [
            "Quantity", "Buy Price", "Sell Price",
            "BuyAmt_Delta", "SellAmt_Delta",
            "Token Cir. Shares", "Token MCAP",
            "Overall USDC Reserve", "Payout/Share"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        
        # (Optional) nice number formats while still using "-" for NaN
        styler = df.style.format({
            "Quantity": "{:,.0f}",
            "Token Cir. Shares": "{:,.0f}",
            "Buy Price": "{:,.4f}",
            "Sell Price": "{:,.4f}",
            "BuyAmt_Delta": "{:,.2f}",
            "SellAmt_Delta": "{:,.2f}",
            "Token MCAP": "{:,.2f}",
            "Overall USDC Reserve": "{:,.2f}",
            "Payout/Share": "{:,.4f}",
        }, na_rep="-")
        st.dataframe(styler, use_container_width=True)

        # Trend #1: Payout/Share Historical Visualisation
        st.subheader("📈 Payout/Share Trend")
        df_viz = df[df['Payout/Share'] != '-'].copy()
        df_viz["Payout/Share"] = pd.to_numeric(df_viz["Payout/Share"], errors="coerce")
        df_viz = df_viz.dropna(subset=["Payout/Share"])
        fig = px.line(df_viz, x="Time", y="Payout/Share", color="Outcome", markers=True, title="Payout/Share Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

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
        with tab:
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
            
            # One radio to switch sub-graphs (no nested tabs)
            view = st.radio(
                "View",
                CURVE_VIEW_MODES,
                horizontal=True,
                key=f"view_{token}",
            )
            if view == "Buy Curve":
                # point annotations at current reserve
                buy_price_now = float(buy_curve(reserve))
                sell_net_now = float(current_marginal_sell_price_after_tax(reserve))  # if you have this

                # smart-sampled series (buy + sell curves)
                x_lo, x_hi = 0, int(MAX_SHARES)
                xs, buy_vals, sell_net_vals = get_curve_series(
                        curve_key=curve_choice,
                            max_shares=int(MAX_SHARES),
                            reserve=reserve,
                            x_lo=x_lo, x_hi=x_hi,
                            dense_pts=1500, sparse_pts=600
                                )

                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scattergl(
                    x=xs, y=buy_vals, mode='lines', name='Buy Curve'
                ))
                # If you want to show the net sell (1-share) curve too, uncomment:
                # fig_curve.add_trace(go.Scattergl(
                #     x=xs, y=sell_net_vals, mode='lines', name='Sell (net, 1 share)'
                # ))

                # Buy point annotation
                fig_curve.add_trace(go.Scatter(
                    x=[reserve], y=[buy_price_now], mode='markers+text',
                    name=f'{token} Buy Point',
                    text=[f"Shares: {reserve}<br>Buy: {buy_price_now:.4f}"],
                    textposition="top right",
                    marker=dict(size=10),
                    showlegend=False
                ))

                # helper lines
                y0 = max(
                    0.0,
                    min(
                        float(np.nanmin(buy_vals)),
                        float(np.nanmin(sell_net_vals)),
                        buy_price_now,
                        sell_net_now,
                    )
                )
                fig_curve.add_trace(go.Scatter(
                    x=[reserve, reserve], y=[y0, buy_price_now],
                    mode='lines', line=dict(dash='dot'), showlegend=False
                ))
                fig_curve.add_trace(go.Scatter(
                    x=[xs.min(), reserve], y=[buy_price_now, buy_price_now],
                    mode='lines', line=dict(dash='dot'), showlegend=False
                ))

                fig_curve.update_layout(
                    title=f'{token} — Buy vs Sell (net, 1 share)',
                    xaxis_title='Shares (reserve)',
                    yaxis_title='Price',
                    hovermode="x unified",
                )

                st.plotly_chart(fig_curve, use_container_width=True, key=f"buy_curve_sim_{token}")

            elif view == "Sell Spread":
                if reserve <= 0:
                    st.info("No circulating shares yet — Sell spread curve will show once there is supply.")
                else:
                    steps = 200
                    X = np.linspace(0.0, 1.0, steps + 1)
                    q_grid = (X * reserve).astype(int)

                    # vectorized tax rate for selling q out of current reserve
                    tax_y = _sale_tax_rate_vec(q_grid, reserve)

                    fig_tax = go.Figure()
                    fig_tax.add_trace(go.Scattergl(
                        x=X * 100.0, y=tax_y, mode='lines', marker=dict(size=10),  name='Sale Tax Rate'
                    ))
                    
                    fig_tax.update_layout(
                    title='Sell Spread vs % of Supply Sold (per order)',
                    xaxis_title='% of Current Supply Sold in Order',
                    yaxis_title='Spread Rate',
                    hovermode="x unified",   # nice unified hover; vertical guide
                    spikedistance=-1         # show spikes whenever the mouse is in the plot
                    )

                    # X spikes (to x-axis)
                    fig_tax.update_xaxes(
                    showspikes=True,
                    spikemode="across",      # draw across the plot area
                    spikesnap="cursor",      # follow the cursor position
                    spikedash="dot"          # dotted line
                    )

                    # Y spikes (to y-axis) + percent ticks
                    fig_tax.update_yaxes(
                    tickformat=".0%",
                    range=[0, 1],
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    spikedash="dot"
                    )
                    # fig_tax.update_layout(
                    #     title='Sale Tax vs % of Supply Sold (per order)',
                    #     xaxis_title='% of Current Supply Sold in Order',
                    #     yaxis_title='Tax Rate',
                    #     hovermode="x unified"
                    # )
                    # # Format Y as percentages
                    # fig_tax.update_yaxes(tickformat=".0%", range=[0, 1])
                    st.plotly_chart(fig_tax, use_container_width=True, key=f"sell_spread_curve_{token}")

            else:  # "Effective Sell (Net)"
                C = reserve
                st.markdown(f"**{token}**")
                if C <= 0:
                    st.info("No circulating shares.")
                else:
                    # up to 10% of supply (at least 1)
                    q_max = max(1, C // 10)
                    q_axis = np.linspace(1, q_max, 200).astype(int)

                    # gross proceeds via integral difference; then apply order-level tax
                    bd_C      = np.vectorize(buy_delta,  otypes=[float])(np.full_like(q_axis, C))
                    bd_C_minQ = np.vectorize(buy_delta,  otypes=[float])(C - q_axis)
                    gross     = bd_C - bd_C_minQ

                    tax = _sale_tax_rate_vec(q_axis, np.full_like(q_axis, C))
                    net = np.maximum(0.0, gross * (1.0 - tax))
                    avg_net = net / np.maximum(1, q_axis)

                    fig_eff = go.Figure()
                    fig_eff.add_trace(go.Scattergl(
                        x=q_axis, y=avg_net, mode='lines', name=f'{token} Avg Net Sell Price'
                    ))
                    fig_eff.update_layout(
                        title=f'Effective Avg Net Sell Price vs Quantity — {token}',
                        xaxis_title='Quantity sold in a single order',
                        yaxis_title='Avg Net Sell Price (USDC/share)',
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_eff, use_container_width=True, key=f"effective_sell_{token}")

   
    st.divider()


# ---------- Mode switch ----------
if app_mode == "Simulator":
    render_simulator()
else:
    render_curve_viewer()
