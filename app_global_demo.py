# file: app.py
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from contextlib import closing

# to change -> question, resolution note and tokens.
# ===========================================================
# Constants
DEFAULT_DECIMAL_PRECISION = 2
BASE_EPSILON = 1e-4
MARKET_DURATION_DAYS = 5
END_TS = "2025-09-07 00:00"
DB_PATH = "app.db"
MAX_SHARES = 5000000 #5M
STARTING_BALANCE = 50000.0 #50k
MARKET_QUESTION = "Price of Ethereum by 7th Sept?"
RESOLUTION_NOTE = (
    'This market will resolve according to the final "Close" price of the '
    'Binance 1-minute candle for ETH/USDT at 12:00 UTC.'
)
TOKENS = ["<4300", "4300-4700", ">4700"]
# MARKET_QUESTION = "Will the total crypto market cap be larger than NVIDIA's market cap by the 7th Sept?"

# RESOLUTION_NOTE = (
#     'This market will resolve to "YES" if the total cryptocurrency market capitalization '
#     'as reported by CoinGecko is greater than the market capitalization of NVIDIA (NVDA) '
#     'as reported by Yahoo Finance at the resolution timestamp. '
#     'It will resolve to "NO" otherwise. '
#     'If either source is unavailable or shows materially inconsistent data, the admins will use reasonable judgment to determine resolution.'
# )
# TOKENS = ["<4200", "4200-4600", ">4600"]
# TOKENS = ["YES", "NO"] 

# Whitelisted usernames and admin reset control
WHITELIST = {"admin", "rui", "haoye", "leo", "steve", "wenbo", "sam", "sharmaine", "mariam", "henry", "guard", "victor", "toby"}

# Inflection Points
EARLY_QUANTITY_POINT = 270
MID_QUANTITY_POINT = 630

# ==== Points System (tunable) ====
TRADE_POINTS_PER_USD = 10.0   # Buy & Sell volume → 10 pts per $1 traded
PNL_POINTS_PER_USD   = 5.0    # Only for positive PnL → 5 pts per $1 profit
EARLY_MULTIPLIER     = 1.5    # Applies to Buy $ when reserve < EARLY_QUANTITY_POINT
MID_MULTIPLIER       = 1.25   # Applies to Buy $ when reserve < MID_QUANTITY_POINT
LATE_MULTIPLIER      = 1.0    # Applies to Buy $ when reserve >= MID_QUANTITY_POINT

PHASE_MULTIPLIERS = {
    "early": EARLY_MULTIPLIER,
    "mid": MID_MULTIPLIER,
    "late": LATE_MULTIPLIER,
}
# ===========================================================
# Streamlit Setup
st.set_page_config(page_title="42:Simulator", layout="wide")
st.title("42:Simulator — Global")

st.subheader(f":blue[{MARKET_QUESTION}]")

def st_display_market_status(active):
    if active:
        st.badge("Active", icon=":material/check:", color="green")
    else:
        st.markdown(":gray-badge[Inactive]")

# ===========================================================
# Math Helpers (curves)
def buy_curve(x: float) -> float:
    return (x**(1/3)/1000) + 0.1
    # return x**(1/4) + x / 400

def sell_curve(x: float) -> float:
    t = (x - 500000.0) / 1_000_000.0
    p = 1.0 / (4.0 * (0.8 + math.exp(-t))) - 0.05
    return max(p,0.0)
    # return ((x - 500)/40) / ((8 + ((x - 500)/80)**2)**0.5) + (x - 500)/300 + 3.6

def buy_delta(x: float) -> float:
    return (3.0/4000.0) * (x**(4.0/3.0)) + x/10.0
    # return (640 * x**(5/4) + x**2) / 800

def sell_delta(x: float) -> float:
    """
    ∫ y dx = 312500 * ln(1 + 0.8 * e^{(x-500000)/1e6}) - 0.05*x + C, C=0
    """
    t = (x - 500000.0) / 1_000_000.0
    # use log1p for better numerical stability
    return 312_500.0 * math.log1p(0.8 * math.exp(t)) - 0.05 * x
    # return 1.93333 * x + 0.00166667 * x**2 + 2 * math.sqrt(301200 - 1000 * x + x**2)


# ===== Sale Tax Helpers (new) =====
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def sale_tax_rate(q: int, C: int) -> float:
    """
    q: number of shares sold in *this order*
    C: total circulating shares (reserve) before the sale

    Tax = min(1, (1.15 - 1.3/(1 + e^(4*(q/C) - 2))))

    We clamp to [0,1] to keep it sane even if the scaling makes it big.
    """
    if C <= 0 or q <= 0:
        return 0.0
    X = q / float(C)  # fraction of supply this order is selling
    base = 1.15 - 1.3 / (1.0 + math.e ** (4.0 * X - 2.0))
    # scale = C * math.e ** ((C / 100000.0 - 1.0) / 10000.0)
    # tax = base * scale
    tax = base
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

# ===========================================================
# DB Helpers
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Make SQLite play nice for multiple sessions
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=3000;")  # 3s
    return conn

def ensure_meta():
    with closing(get_conn()) as conn, conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS app_meta (
            id INTEGER PRIMARY KEY CHECK (id=1),
            version INTEGER NOT NULL
        );""")
        # Seed single-row meta
        c.execute("INSERT OR IGNORE INTO app_meta (id, version) VALUES (1, 0);")

def bump_version():
    with closing(get_conn()) as conn, conn:
        conn.execute("UPDATE app_meta SET version = version + 1 WHERE id=1;")

def get_version():
    with closing(get_conn()) as conn:
        r = conn.execute("SELECT version FROM app_meta WHERE id=1").fetchone()
        return r["version"] if r else 0

# Call this once at startup (near your other setup/DB init)
ensure_meta()

def init_db():
    with closing(get_conn()) as conn, conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            balance REAL NOT NULL,
            created_at TEXT NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS market (
            id INTEGER PRIMARY KEY CHECK (id=1),
            start_ts TEXT NOT NULL,
            end_ts TEXT NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS reserves (
            token TEXT PRIMARY KEY,
            shares INTEGER NOT NULL,
            usdc REAL NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            shares INTEGER NOT NULL,
            PRIMARY KEY (user_id, token),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            token TEXT NOT NULL,
            qty INTEGER NOT NULL,
            buy_price REAL,
            sell_price REAL,
            buy_delta REAL,
            sell_delta REAL,
            balance_after REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )""")
        # Seed market if missing
        c.execute("SELECT COUNT(*) FROM market")
        if c.fetchone()[0] == 0:
            start = datetime.utcnow()
            end = start + timedelta(days=MARKET_DURATION_DAYS)
            c.execute("INSERT INTO market(id,start_ts,end_ts) VALUES(1,?,?)", (start.isoformat(), end.isoformat()))
        # Seed reserves A/B/C
        for t in TOKENS:
            c.execute("INSERT OR IGNORE INTO reserves(token,shares,usdc) VALUES(?,?,?)", (t, 0, 0.0))
        # Create indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_tx_user_time ON transactions(user_id, ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tx_time ON transactions(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_holdings_user ON holdings(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reserves_token ON reserves(token)")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)")

def ensure_market_resolution_columns():
    with closing(get_conn()) as conn, conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(market)").fetchall()}
        if "winner_token" not in cols:
            conn.execute("ALTER TABLE market ADD COLUMN winner_token TEXT")
        if "resolved" not in cols:
            conn.execute("ALTER TABLE market ADD COLUMN resolved INTEGER DEFAULT 0")
        if "resolved_ts" not in cols:
            conn.execute("ALTER TABLE market ADD COLUMN resolved_ts TEXT")


def compute_cost_basis(user_id: int) -> dict:
    """
    Returns {token: {'shares': int, 'cost': float, 'avg_cost': float}}
    using running-average cost basis from the transactions table.
    - On Buy: add cost and shares.
    - On Sell: reduce cost proportional to average cost per share.
    """
    basis = {t: {'shares': 0, 'cost': 0.0, 'avg_cost': 0.0} for t in TOKENS}
    with closing(get_conn()) as conn:
        txu = pd.read_sql_query(
            """
            SELECT action, token, qty, buy_delta AS b, sell_delta AS s
            FROM transactions
            WHERE user_id=? AND action IN ('Buy','Sell')
            ORDER BY ts ASC
            """,
            conn, params=(user_id,)
        )

    for _, r in txu.iterrows():
        t = r['token']
        q = int(r['qty'] or 0)
        if r['action'] == 'Buy':
            add_cost = float(r['b'] or 0.0)
            basis[t]['cost'] += add_cost
            basis[t]['shares'] += q
        elif r['action'] == 'Sell':
            # reduce cost by avg cost per share * shares sold
            sh = basis[t]['shares']
            if sh > 0 and q > 0:
                avg = basis[t]['cost'] / sh
                used = min(q, sh)
                basis[t]['cost'] -= avg * used
                basis[t]['shares'] -= used

    for t in TOKENS:
        sh = basis[t]['shares']
        basis[t]['avg_cost'] = (basis[t]['cost'] / sh) if sh > 0 else 0.0
    return basis


# For points computation
def compute_user_points(tx_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      User, VolumePoints, PnLPoints, TotalPoints
    Volume points:
      - 10 pts per $1 traded (Buy or Sell)
      - Buy $ gets a phase multiplier based on *reserve before* the Buy:
          < EARLY_QUANTITY_POINT → 1.5x
          < MID_QUANTITY_POINT   → 1.25x
          >= MID_QUANTITY_POINT  → 1.0x
    PnL points:
      - 5 pts per $1 profit (only positive PnL), using the latest portfolio PnL snapshot
        we compute below when we build the leaderboard.
    """
    # Map id -> username
    id_to_user = dict(zip(users_df["id"], users_df["username"]))

    # Running reserves per token to know the *phase at time of each buy*
    reserves_state = {t: 0 for t in TOKENS}

    # Per-user accumulators
    vol_points = {uid: 0.0 for uid in users_df["id"]}
    # We'll add PnL points later (when we know latest PnL)

    # Process chronologically
    tdf = tx_df.copy()
    tdf["Time"] = pd.to_datetime(tdf["Time"])
    tdf = tdf.sort_values("Time")

    for _, r in tdf.iterrows():
        uid  = int(r["user_id"])
        act  = r["Action"]
        tkn  = r["Outcome"]
        qty  = int(r["Quantity"])

        if act == "Buy":
            # Dollars traded for this buy
            buy_usd = float(r["BuyAmt_Delta"] or 0.0)

            # Determine phase by reserve BEFORE adding qty
            r_before = reserves_state.get(tkn, 0)
            if r_before < EARLY_QUANTITY_POINT:
                mult = EARLY_MULTIPLIER
            elif r_before < MID_QUANTITY_POINT:
                mult = MID_MULTIPLIER
            else:
                mult = LATE_MULTIPLIER

            vol_points[uid] += buy_usd * TRADE_POINTS_PER_USD * mult

            # Update state
            reserves_state[tkn] = r_before + qty

        elif act == "Sell":
            sell_usd = float(r["SellAmt_Delta"] or 0.0)
            vol_points[uid] += sell_usd * TRADE_POINTS_PER_USD

            # Update state (post-sell)
            reserves_state[tkn] = reserves_state.get(tkn, 0) - qty

        # "Resolve" does not change volume points

    # Build dataframe (PnL points will be joined later)
    out = pd.DataFrame({
        "user_id": list(vol_points.keys()),
        "User":    [id_to_user[uid] for uid in vol_points.keys()],
        "VolumePoints": [vol_points[uid] for uid in vol_points.keys()]
    })

    return out


def compute_points_timeline(txp: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a timeline of points per user at each transaction timestamp.
    TotalPoints(t) = CumulativeVolumePoints(t) + max(PnL(t), 0) * PNL_POINTS_PER_USD
    PnL(t) is instantaneous (not cumulative) based on reconstructed portfolio valuation.

    Phase rules (for BUY only), decided on reserve *before* applying the buy:
        reserve < EARLY_QUANTITY_POINT -> early
        EARLY_QUANTITY_POINT <= reserve < MID_QUANTITY_POINT -> mid
        reserve >= MID_QUANTITY_POINT -> late
    """

    # Defensive copy & types
    df = txp.copy()
    if df.empty:
        return pd.DataFrame(columns=["Time","User","VolumePointsCum","PnLPointsInstant","TotalPoints"])

    df["Time"] = pd.to_datetime(df["Time"])

    # States to reconstruct reserves + user portfolios
    reserves_state = {t: 0 for t in TOKENS}
    user_state = {
        int(r.id): {"username": r.username, "balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKENS}}
        for _, r in users_df.iterrows()
    }

    # Cumulative volume points per user
    vol_points = {int(r.id): 0.0 for _, r in users_df.iterrows()}

    rows = []

    for _, r in df.iterrows():
        uid = int(r["user_id"])
        act = r["Action"]
        tkn = r["Outcome"]
        qty = int(r["Quantity"])

        # --- Volume points increment (if any) ---
        add_points = 0.0
        if act == "Buy":
            buy_usd = float(r["BuyAmt_Delta"] or 0.0)
            # Phase by reserve BEFORE buy
            reserve_before = reserves_state[tkn]
            if reserve_before < EARLY_QUANTITY_POINT:
                mult = PHASE_MULTIPLIERS["early"]
            elif reserve_before < MID_QUANTITY_POINT:
                mult = PHASE_MULTIPLIERS["mid"]
            else:
                mult = PHASE_MULTIPLIERS["late"]
            add_points = buy_usd * TRADE_POINTS_PER_USD * mult

        elif act == "Sell":
            sell_usd = float(r["SellAmt_Delta"] or 0.0)
            # Sells: volume points = 10 pts / $1 (no multiplier)
            add_points = sell_usd * TRADE_POINTS_PER_USD

        # accumulate
        vol_points[uid] += float(add_points)

        # --- Apply the transaction to portfolio/reserves reconstruction ---
        if act == "Buy":
            delta = float(r["BuyAmt_Delta"] or 0.0)
            user_state[uid]["balance"] -= delta
            user_state[uid]["holdings"][tkn] += qty
            reserves_state[tkn] += qty

        elif act == "Sell":
            delta = float(r["SellAmt_Delta"] or 0.0)
            user_state[uid]["balance"] += delta
            user_state[uid]["holdings"][tkn] -= qty
            reserves_state[tkn] -= qty

        elif act == "Resolve":
            # credit this user's payout (logged as SellAmt_Delta)
            payout = float(r["SellAmt_Delta"] or 0.0)
            user_state[uid]["balance"] += payout

            # wipe all holdings and reserves
            for u_id in user_state:
                user_state[u_id]["holdings"] = {t: 0 for t in TOKENS}
            reserves_state = {t: 0 for t in TOKENS}

        # --- Price snapshot (use Buy curve for valuation consistency) ---
        if act == "Resolve":
            prices = {t: 0.0 for t in TOKENS}
        else:
            prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}

        # --- Build a row per user at this timestamp ---
        ts = r["Time"]
        for u_id, s in user_state.items():
            pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
            pnl_instant = max(pv - STARTING_BALANCE, 0.0)
            pnl_points = pnl_instant * PNL_POINTS_PER_USD
            total_points = vol_points[u_id] + pnl_points

            rows.append({
                "Time": ts,
                "user_id": u_id,
                "User": s["username"],
                "VolumePointsCum": vol_points[u_id],
                "PnLPointsInstant": pnl_points,
                "TotalPoints": total_points,
            })

    return pd.DataFrame(rows)


init_db()
ensure_market_resolution_columns()


@st.cache_data(show_spinner=False)
def load_tx_and_users(cache_key:int):
    with closing(get_conn()) as conn:
        txp = pd.read_sql_query("""SELECT ...""", conn)
        users_df = pd.read_sql_query("SELECT id, username FROM users ORDER BY id", conn)
    return txp, users_df

# Use the max tx timestamp as an invalidation key (so we don’t need to bump on every write)
def latest_tx_epoch() -> int:
    with closing(get_conn()) as conn:
        r = conn.execute("SELECT COALESCE(MAX(ts),'1970-01-01T00:00:00') AS mx FROM transactions").fetchone()
    return int(pd.to_datetime(r["mx"]).value)  # nanoseconds

# cache_key = latest_tx_epoch()
# txp, users_df = load_tx_and_users(cache_key)

def compute_points_timeline_fast(txp: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    if txp.empty:
        return pd.DataFrame(columns=["Time","User","VolumePointsCum","PnLPointsInstant","TotalPoints"])

    U = len(users_df); T = len(TOKENS)
    uid_index = {int(u.id): i for _, u in users_df.iterrows()}
    token_index = {t: i for i, t in enumerate(TOKENS)}

    balances = np.full(U, STARTING_BALANCE, dtype=np.float64)
    holdings = np.zeros((U, T), dtype=np.int64)
    reserves = np.zeros(T, dtype=np.int64)
    vol_points = np.zeros(U, dtype=np.float64)

    rows = []
    for _, r in txp.sort_values("Time").iterrows():
        u = uid_index[int(r["user_id"])]
        k = token_index[r["Outcome"]]
        act = r["Action"]; q = int(r["Quantity"])

        if act == "Buy":
            usd = float(r["BuyAmt_Delta"] or 0.0)
            # phase multiplier by reserve BEFORE buy
            rb = reserves[k]
            mult = EARLY_MULTIPLIER if rb < EARLY_QUANTITY_POINT else (MID_MULTIPLIER if rb < MID_QUANTITY_POINT else LATE_MULTIPLIER)
            vol_points[u] += usd * TRADE_POINTS_PER_USD * mult

            balances[u] -= usd
            holdings[u, k] += q
            reserves[k] += q

        elif act == "Sell":
            usd = float(r["SellAmt_Delta"] or 0.0)
            vol_points[u] += usd * TRADE_POINTS_PER_USD

            balances[u] += usd
            holdings[u, k] -= q
            reserves[k] -= q

        elif act == "Resolve":
            usd = float(r["SellAmt_Delta"] or 0.0)
            balances[u] += usd
            holdings[:, :] = 0
            reserves[:] = 0

        # prices for this snapshot
        if act == "Resolve":
            prices = np.zeros(T, dtype=np.float64)
        else:
            # buy price for valuation (your original choice)
            # vectorize across tokens
            px = np.array([buy_curve(int(res)) for res in reserves], dtype=np.float64)
            prices = px

        # vectorized PV + points
        pv = balances + (holdings @ prices)
        pnl_points = np.maximum(pv - STARTING_BALANCE, 0.0) * PNL_POINTS_PER_USD
        total = vol_points + pnl_points

        # append as one block
        ts = pd.to_datetime(r["Time"])
        rows.extend({
            "Time": ts,
            "user_id": int(users_df.iloc[i].id),
            "User": users_df.iloc[i].username,
            "VolumePointsCum": float(vol_points[i]),
            "PnLPointsInstant": float(pnl_points[i]),
            "TotalPoints": float(total[i]),
        } for i in range(U))

    return pd.DataFrame(rows)


# ===========================================================
# Sidebar user auth with whitelist (persist to session, rerun)
with st.sidebar:
    st.header("User")
    username_input = st.text_input("Enter username", key="username_input")
    if username_input and username_input not in WHITELIST:
        st.error("Username not whitelisted.")
    join_disabled = bool(username_input) and (username_input not in WHITELIST)

    if st.button("Join / Load", disabled=join_disabled):
        if not username_input:
            st.warning("Please enter a username.")
        else:
            with closing(get_conn()) as conn, conn:
                c = conn.cursor()
                # enforce max 10 users at create time
                row = c.execute("SELECT id, balance FROM users WHERE username=?", (username_input,)).fetchone()
                if not row:
                    cnt = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
                    if cnt >= 20:
                        st.error("Max 20 users reached. Try another time.")
                        st.stop()
                    c.execute(
                        "INSERT INTO users(username,balance,created_at) VALUES(?,?,?)",
                        (username_input, STARTING_BALANCE, datetime.utcnow().isoformat()),
                    )
                    user_id = c.lastrowid
                    for t in TOKENS:
                        c.execute("INSERT OR IGNORE INTO holdings(user_id,token,shares) VALUES(?,?,0)", (user_id, t))
                else:
                    user_id = row["id"]
            st.session_state.user_id = int(user_id)
            st.session_state.username = username_input
            st.rerun()

    if "user_id" in st.session_state and st.button("Logout", type="primary"):
        for k in ("user_id", "username"):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# Refresh session balance snapshot
if "user_id" in st.session_state:
    with closing(get_conn()) as conn:
        bal_row = conn.execute("SELECT balance FROM users WHERE id=?", (st.session_state.user_id,)).fetchone()
        if bal_row:
            st.session_state.balance = float(bal_row["balance"])


# ===========================================================

# Admin-only reset market button (use session username)
def reseed_reserves(c):
    """Wipe all reserves rows and re-seed zeroed rows for current TOKENS."""
    # why: remove stale tokens from previous sessions so totals reset to 0 correctly
    c.execute("DELETE FROM reserves")
    for t in TOKENS:
        c.execute("INSERT INTO reserves(token, shares, usdc) VALUES(?,?,?)", (t, 0, 0.0))


# Admin-only reset market button (use session username)
if 'user_id' in st.session_state and st.session_state.get("username") == "admin":
    st.sidebar.subheader("Admin Controls")
    wipe_users = st.sidebar.checkbox("Reset all users (wipe accounts)", value=False)
    if st.sidebar.button("Reset Market"):
        with closing(get_conn()) as conn, conn:
            c = conn.cursor()
            start = datetime.utcnow()
            end = start + timedelta(days=MARKET_DURATION_DAYS)
            # end = END_TS
            # reset market window + clear resolution flags
            c.execute("""
                UPDATE market
                SET start_ts=?,
                    end_ts=?,
                    winner_token=NULL,
                    resolved=0,
                    resolved_ts=NULL
                WHERE id=1
            """, (start.isoformat(), end.isoformat()))
            # reset reserves
            for t in TOKENS:
                c.execute("UPDATE reserves SET shares=0, usdc=0 WHERE token=?", (t,))

            # *** Key fix: wipe & re-seed reserves so Total USDC Reserve goes to 0 ***
            reseed_reserves(c)

            if wipe_users:
                # full wipe of user state
                c.execute("DELETE FROM holdings")
                c.execute("DELETE FROM transactions")
                c.execute("DELETE FROM users")
                # also clear any admin session info so UI is consistent
                for k in ("user_id", "username", "balance"):
                    if k in st.session_state:
                        del st.session_state[k]
            else:
                # keep users; reset balances/holdings and clear txs
                c.execute("UPDATE users SET balance=?", (STARTING_BALANCE,))
                c.execute("UPDATE holdings SET shares=0")
                c.execute("DELETE FROM transactions")

        bump_version()  # notify all sessions
        st.success("Market has been reset." + (" All users wiped." if wipe_users else ""))
        st.rerun()


# --- Admin: Resolve Parimutuel Market ---
# resolved = int(m["resolved"] or 0)
if 'user_id' in st.session_state and st.session_state.get("username") == "admin":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resolve Market")
        # Fetch resolved state once to control the UI
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT COALESCE(resolved, 0) AS r FROM market WHERE id=1").fetchone()
    is_resolved = bool(row and int(row["r"]) == 1)
    winner = st.sidebar.selectbox("Winning outcome", TOKENS, key="winner_select")

    btn_disabled = (winner is None) or is_resolved
    if st.sidebar.button("Resolve Now", disabled=btn_disabled , type="secondary"):
        # Do resolution
        with closing(get_conn()) as conn, conn:
            c = conn.cursor()

            # guard: already resolved?
            row = c.execute("SELECT resolved FROM market WHERE id=1").fetchone()
            if row and int(row["resolved"] or 0) == 1:
                st.sidebar.warning("Market already resolved.")
            else:
                # Total USDC pool
                tot_pool = float(c.execute("SELECT SUM(usdc) AS s FROM reserves").fetchone()["s"] or 0.0)

                # Total winning shares
                win_shares = int(c.execute("SELECT shares FROM reserves WHERE token=?", (winner,)).fetchone()["shares"] or 0)

                # Build payouts map (user_id -> payout)
                payouts = {}
                if win_shares > 0 and tot_pool > 0:
                    for r in c.execute("SELECT user_id, shares FROM holdings WHERE token=? AND shares>0", (winner,)).fetchall():
                        uid = int(r["user_id"])
                        u_shares = int(r["shares"])
                        payout = tot_pool * (u_shares / win_shares)
                        payouts[uid] = payouts.get(uid, 0.0) + payout

                # Pay users
                for uid, amt in payouts.items():
                    # add to balance
                    bal = float(c.execute("SELECT balance FROM users WHERE id=?", (uid,)).fetchone()["balance"] or 0.0)
                    new_bal = bal + float(amt)
                    c.execute("UPDATE users SET balance=? WHERE id=?", (new_bal, uid))
                    # log a resolve tx (per user)
                    c.execute("""
                        INSERT INTO transactions (ts,user_id,action,token,qty,buy_price,sell_price,buy_delta,sell_delta,balance_after)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (datetime.utcnow().isoformat(), uid, "Resolve", winner, 0, None, None, None, float(amt), new_bal))

                # Zero out all holdings & reserves after settlement
                c.execute("UPDATE holdings SET shares=0")
                c.execute("UPDATE reserves SET shares=0, usdc=0")

                # Mark market resolved & end now
                c.execute("UPDATE market SET winner_token=?, resolved=1, resolved_ts=?, end_ts=? WHERE id=1",
                          (winner, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))

        bump_version()
        st.success(f"Market resolved. Winner: {winner}. Payout pool: ${tot_pool:,.2f}")
        st.rerun()

# Winner banner (if resolved)
with closing(get_conn()) as conn:
    row = conn.execute("SELECT resolved, winner_token FROM market WHERE id=1").fetchone()
    if row and int(row["resolved"] or 0) == 1:
        st.success(f"✅ Resolved — Winner: {row['winner_token']}")

# ===========================================================
# Market status
with closing(get_conn()) as conn:
    cur = conn.cursor()
    cur.execute("SELECT start_ts,end_ts,winner_token,resolved FROM market WHERE id=1")
    m = cur.fetchone()
    market_start = datetime.fromisoformat(m["start_ts"])
    market_end_db = datetime.fromisoformat(m["end_ts"])
    resolved_flag = int(m["resolved"] or 0)

now = datetime.utcnow()
hard_end = datetime.fromisoformat(END_TS)   # <-- enforce constant cutoff

# Market is active only if:
# 1) within DB market window
# 2) before hard END_TS cutoff
# 3) not resolved
active = (market_start <= now <= market_end_db) and (now <= hard_end) and (resolved_flag == 0)

    # Sidebar market status row
with st.sidebar:
    status_cols = st.columns(2)
    with status_cols[0]:
        st.write("**Market Status:**")
    with status_cols[1]:
       st_display_market_status(active)


st.sidebar.markdown(f"Start (UTC): {market_start:%Y-%m-%d %H:%M}")
# st.sidebar.markdown(f"End (UTC): {market_end:%Y-%m-%d %H:%M}")
st.sidebar.markdown(f"End (UTC): {END_TS}")

if 'user_id' not in st.session_state:
    st.warning("Join with a username on the left to trade.")

# ===========================================================
# Trading UI

# Show market resolution / info section
with closing(get_conn()) as conn:
    market_row = conn.execute("""
        SELECT resolved, winner_token, resolved_ts, end_ts
        FROM market
        WHERE id=1
    """).fetchone()

if market_row:
    resolved_flag = int(market_row["resolved"] or 0)
    winner_token = market_row["winner_token"]
    resolved_ts = market_row["resolved_ts"]
    end_ts = market_row["end_ts"]

    if resolved_flag == 1 and winner_token:
        # Market resolved - show outcome highlight
        st.markdown(
            f"""
            <div style="
                background-color: orange;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #b2f2bb;
                margin-bottom: 1rem;
            ">
                <h4 style="margin-top: 0;">✅ Market Resolved</h4>
                <p>
                    The winning outcome is <b style="color: green;">{winner_token}</b>.
                    Settlement occurred on <b>{pd.to_datetime(resolved_ts).strftime('%Y-%m-%d %H:%M UTC')}</b>.
                </p>
                <p>All holdings in this outcome have been paid out proportionally from the total USDC pool.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
                f"""
                <div style="background-color: grey; padding: 1rem; border-radius: 0.5rem; border: 1px solid #a5b4fc; margin-bottom: 1rem;">
                    <h4 style="margin-top: 0;">📜 Resolution Rules</h4>
                    <p>
                        The market will resolve at <b>{END_TS}</b>.
                        The winning outcome will receive the <b>entire USDC pool</b>,
                        distributed <i>pro-rata</i> to holders based on their share count.
                    </p>
                    <p>
                        <b>Resolution Note:</b> {RESOLUTION_NOTE}
                    </p>
                    <p>
                        After resolution, all holdings will be cleared and balances updated automatically.
                    </p>
                    <h5>🔗 Resolution Sources/Resources:</h5>
                    <ul>
                        <li><a href="https://www.binance.com/en/trade/ETH_USDT?type=spot/" target="_blank">Binance International: ETH/USDT Spot Market</a></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.divider()

if 'user_id' in st.session_state:
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        # User state
        cur.execute("SELECT balance FROM users WHERE id=?", (st.session_state.user_id,))
        bal = float(cur.fetchone()[0] or 0.0)
        cur.execute("SELECT token, shares FROM holdings WHERE user_id=?", (st.session_state.user_id,))
        holdings_rows = cur.fetchall()
        user_holdings = {row[0]: int(row[1]) for row in holdings_rows} if holdings_rows else {}

        # Global reserves for pricing (buy price @ current circulating shares)
        cur.execute("SELECT token, shares FROM reserves")
        reserve_rows = cur.fetchall()
        global_reserves = {row[0]: int(row[1]) for row in reserve_rows} if reserve_rows else {}

    # Compute current buy price per token and holding values
    prices = {t: float(buy_curve(global_reserves.get(t, 0))) for t in TOKENS}
    holding_values = {t: user_holdings.get(t, 0) * prices[t] for t in TOKENS}
    holdings_total_value = sum(holding_values.values())
    portfolio_value = bal + holdings_total_value

    st.markdown("""
    <style>
        .pill {
            padding: 6px 10px; border-radius: 8px; font-size: 12px;
            border: 1px solid rgba(0,0,0,0.1); display: inline-block; margin-bottom: 6px;
            background: rgba(0,0,0,0.02);
        }
        .pill strong { font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Your Account")
        st.metric("Username", st.session_state.get("username",""))
  

        with st.container():
            overall_cols = st.columns(2)

            with overall_cols[0]:
                with st.container(border=True):
                    st.subheader("Portfolio")
                    ucols = st.columns(3)
                    with ucols[0]:
                        st.metric("Overall Value (USD)", f"{portfolio_value:,.2f}")
                    with ucols[1]:
                        st.metric("Balance (USDC)", f"{bal:,.2f}")
                    with ucols[2]:
                        st.metric("Shares Holdings Value (USD)", f"{holdings_total_value:,.2f}")

            # --- Points snapshot for connected user (live) ---
            # We reuse the points calculation helper to get Volume Points
            with closing(get_conn()) as conn:
                txp_pts = pd.read_sql_query("""
                    SELECT t.ts as Time, u.id as user_id, u.username as User, t.action as Action,
                        t.token as Outcome, t.qty as Quantity,
                        t.buy_delta as BuyAmt_Delta, t.sell_delta as SellAmt_Delta
                    FROM transactions t JOIN users u ON t.user_id=u.id
                    ORDER BY t.ts ASC
                """, conn)
                users_df_pts = pd.read_sql_query("SELECT id, username FROM users ORDER BY id", conn)

            my_vol_points = 0.0
            if not txp_pts.empty:
                points_live = compute_user_points(txp_pts, users_df_pts)
                me_row = points_live[points_live["user_id"] == st.session_state["user_id"]]
                if not me_row.empty:
                    my_vol_points = float(me_row["VolumePoints"].iloc[0])

            # PnL points use *current* portfolio snapshot in the account card
            my_pnl_now = max(portfolio_value - STARTING_BALANCE, 0.0)
            my_pnl_points = my_pnl_now * PNL_POINTS_PER_USD

            my_total_points = my_vol_points + my_pnl_points

            # Per-token breakdown
            with overall_cols[1]:
                with st.container(border=True):
                    st.subheader("Points")
                    # Display row
                    pcols = st.columns(3)
                    with pcols[0]:
                        st.metric("Volume Points", f"{my_vol_points:,.0f}")
                    with pcols[1]:
                        st.metric("PnL Points", f"{my_pnl_points:,.0f}")
                    with pcols[2]:
                        st.metric("Total Points", f"{my_total_points:,.0f}")


        # Per-token breakdown
        with st.container(border=True):
            st.subheader("Share Holdings")
            br_cols = st.columns(3)
            for i, token in enumerate(TOKENS):
                shares = user_holdings.get(token, 0)
                price = prices[token]
                val = holding_values[token]
                with br_cols[i]:
                    st.caption(f"Outcome :blue[{token}]")
                    st.text(f"Shares: {shares}")
                    st.text(f"Price: {price:.4f}")
                    st.text(f"Value: ${val:,.2f}")

# === Cost basis & per-token PnL table ===
user_id = st.session_state.get("user_id")
if user_id is not None:
    basis = compute_cost_basis(user_id)

    # pool stats for estimated payout if token wins now
    with closing(get_conn()) as conn:
        rows = conn.execute("SELECT token, shares, usdc FROM reserves").fetchall()
    reserves_now = {r["token"]: {"shares": int(r["shares"]), "usdc": float(r["usdc"])} for r in rows}
    total_pool_now = float(sum(r["usdc"] for r in reserves_now.values()))


    pnl_rows = []
    for t in TOKENS:
        sh = int(basis[t]['shares'])
        avg = float(basis[t]['avg_cost'])
        pos_cost = float(basis[t]['cost'])
        price_now_token = float(prices[t])  # rename from px -> price_now_token to prevent override
        mv = sh * price_now_token
        unreal = mv - pos_cost
        pnl_pct = (unreal / pos_cost * 100.0) if pos_cost > 0 else 0.0

        win_shares_now = max(1, int(reserves_now.get(t, {}).get('shares', 0)))
        est_payout_if_win = 0.0
        if total_pool_now > 0 and win_shares_now > 0 and sh > 0:
            est_payout_if_win = total_pool_now * (sh / win_shares_now)

        pnl_rows.append({
            "Token": t,
            "Shares": sh,
            "Avg Cost/Share": round(avg, DEFAULT_DECIMAL_PRECISION),
            "Position Cost": round(pos_cost, DEFAULT_DECIMAL_PRECISION),
            "Price Now": round(price_now_token, DEFAULT_DECIMAL_PRECISION),
            "Market Value": round(mv, DEFAULT_DECIMAL_PRECISION),
            "Unrealized PnL": round(unreal, DEFAULT_DECIMAL_PRECISION),
            "PnL %": round(pnl_pct, DEFAULT_DECIMAL_PRECISION),
            "Est. Payout if Wins": round(est_payout_if_win, DEFAULT_DECIMAL_PRECISION),
            "Est. Win PnL": round(est_payout_if_win - pos_cost, DEFAULT_DECIMAL_PRECISION)
        })

    pnl_df = pd.DataFrame(pnl_rows)

    st.markdown("**Per-Token Position & PnL**")
    st.dataframe(pnl_df, use_container_width=True)
    st.divider()

    

# Common Trading UI
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
cols = st.columns(4)

# Fetch current reserves (DB)
with closing(get_conn()) as conn:
    cur = conn.cursor()
    cur.execute("SELECT token, shares, usdc FROM reserves")
    reserves_map = {t: {"shares": s, "usdc": u} for t, s, u in cur.fetchall()}

for i, token in enumerate(TOKENS):

    
    with cols[i]:
        st.markdown(f"### Outcome :blue[{token}]")

        reserve = int(reserves_map[token]["shares"])  # global shares for token
        price_now = round(buy_curve(reserve), 2)
        mcap_now = round(reserves_map[token]["usdc"], 2)
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
            f"Sell Order Slippage: **{est_tax_rate*100:.2f}%**"
        )
        

        buy_col, sell_col = st.columns(2)


        # --- BUY ---
        with buy_col:
            disabled = ('user_id' not in st.session_state) or (not active)
            if st.button(f"Buy {token}", disabled=disabled):
                if 'user_id' not in st.session_state:
                    st.stop()
                # derive quantity if USDC mode
                q = quantity
                if input_mode == "USDC":
                    q = qty_from_buy_usdc(reserve, usdc_input)
                if q <= 0:
                    st.warning("Quantity computed as 0.")
                else:
                    bp, _, bdelta, _, _ = metrics_from_qty(reserve, q)
                    # check balance
                    with closing(get_conn()) as conn, conn:
                        c = conn.cursor()
                        c.execute("SELECT balance FROM users WHERE id=?", (st.session_state.user_id,))
                        bal = c.fetchone()[0]
                        if bal < bdelta:
                            st.error("Insufficient balance.")
                        else:
                            # update reserves
                            c.execute("UPDATE reserves SET shares=shares+?, usdc=usdc+? WHERE token=?", (q, bdelta, token))
                            # update user balance
                            new_bal = bal - bdelta
                            c.execute("UPDATE users SET balance=? WHERE id=?", (new_bal, st.session_state.user_id))
                            # update holdings
                            c.execute("INSERT INTO holdings(user_id,token,shares) VALUES(?,?,?) ON CONFLICT(user_id,token) DO UPDATE SET shares=shares+excluded.shares",
                                      (st.session_state.user_id, token, q))
                            # tx log
                            c.execute("""
                                INSERT INTO transactions(ts,user_id,action,token,qty,buy_price,buy_delta,sell_price,sell_delta,balance_after)
                                VALUES(?,?,?,?,?,?,?,?,?,?)
                            """, (
                                datetime.utcnow().isoformat(), st.session_state.user_id, 'Buy', token, q, bp, bdelta, None, None, new_bal
                            ))
                    st.rerun()

        # --- SELL ---
        with sell_col:
            disabled = ('user_id' not in st.session_state) or (not active)
            if st.button(f"Sell {token}", disabled=disabled):
                if 'user_id' not in st.session_state:
                    st.stop()
                q = quantity
                if input_mode == "USDC":
                    q = qty_from_sell_usdc(reserve, usdc_input)
                if q <= 0:
                    st.warning("Quantity computed as 0.")
                else:
                    # check user holdings
                    with closing(get_conn()) as conn, conn:
                        c = conn.cursor()
                        c.execute("SELECT shares FROM holdings WHERE user_id=? AND token=?", (st.session_state.user_id, token))
                        h = c.fetchone()
                        user_shares = h[0] if h else 0
                        if user_shares < q:
                            st.error("Insufficient shares to sell.")
                        else:
                            _, sp, _, sdelta, _ = metrics_from_qty(reserve, q)
                            # update reserves
                            c.execute("UPDATE reserves SET shares=shares-?, usdc=usdc-? WHERE token=?", (q, sdelta, token))
                            # update user balance (add USDC)
                            c.execute("SELECT balance FROM users WHERE id=?", (st.session_state.user_id,))
                            bal = c.fetchone()[0]
                            new_bal = bal + sdelta
                            c.execute("UPDATE users SET balance=? WHERE id=?", (new_bal, st.session_state.user_id))
                            # update holdings
                            c.execute("UPDATE holdings SET shares=shares-? WHERE user_id=? AND token=?", (q, st.session_state.user_id, token))
                            # tx log
                            c.execute("""
                                INSERT INTO transactions(ts,user_id,action,token,qty,buy_price,buy_delta,sell_price,sell_delta,balance_after)
                                VALUES(?,?,?,?,?,?,?,?,?,?)
                            """, (
                                datetime.utcnow().isoformat(), st.session_state.user_id, 'Sell', token, q, None, None, sp, sdelta, new_bal
                            ))
                    st.rerun()

# ===========================================================
# Dashboards (global)
with closing(get_conn()) as conn:
    cur = conn.cursor()
    cur.execute("SELECT token, shares, usdc FROM reserves")
    res_rows = cur.fetchall()

res_df = pd.DataFrame(res_rows, columns=["Token","Shares","USDC"])
res_tot_shares = int(res_df["Shares"].sum())
res_tot_usdc = float(res_df["USDC"].sum())

st.subheader("Overall Market Metrics")
mc_cols = st.columns(3)
with mc_cols[0]:
    st.metric("Total USDC Reserve", round(res_tot_usdc, 2))

with mc_cols[1]:
    st.text("Market Status")
    st_display_market_status(active)

with mc_cols[2]:
    st.metric("Users", int(pd.read_sql_query("SELECT COUNT(*) as n FROM users", get_conn())['n'][0]))

# Per-token metrics row
tok_cols = st.columns(3)
for i, token in enumerate(TOKENS):
    with tok_cols[i]:
        st.subheader(f'Outcome :blue[{token}]')
        row = res_df[res_df['Token']==token].iloc[0]
        reserve = int(row['Shares'])
        price = round(buy_curve(reserve), 3)
        mcap = round(float(row['USDC']), 2)
        sub_cols = st.columns(4)
        with sub_cols[0]:
            st.metric(f"Total Shares", reserve)
        with sub_cols[1]:
            st.metric(f"Price", f"${price}")
        with sub_cols[2]:
            st.metric(f"MCAP", format_usdc_compact(mcap))


# Odds based on circulating shares
total_market_shares = max(1, int(res_df['Shares'].sum()))


sub_cols_2 = st.columns(len(TOKENS))
for i, token in enumerate(TOKENS):
    s = int(res_df.loc[res_df['Token']==token, 'Shares'].iloc[0])
    odds_val = '-' if s == 0 else round(1 / (s / total_market_shares), 2)
    with sub_cols_2[i]:
        st.metric(f"Payout [{token}]", f"{odds_val}x" if odds_val != '-' else "-", border=True)
# ===========================================================
# Logs & Charts from DB
with closing(get_conn()) as conn:
    tx = pd.read_sql_query("""
        SELECT t.ts as Time, u.username as User, t.action as Action, t.token as Outcome, t.qty as Quantity,
               t.buy_price as `Buy Price`, t.sell_price as `Sell Price`,
               t.buy_delta as `BuyAmt_Delta`, t.sell_delta as `SellAmt_Delta`, t.balance_after as `Balance After`
        FROM transactions t JOIN users u ON t.user_id=u.id
        ORDER BY t.ts ASC
    """, conn)

# Create a display copy
tx_display = tx.copy()

# Format floats with 4 decimal places (or whatever you want)
float_cols = tx_display.select_dtypes(include=["float", "float64"]).columns
for col in float_cols:
    tx_display[col] = tx_display[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")


if not tx.empty:    
    st.subheader("Transaction Log")
    st.dataframe(tx_display, use_container_width=True)
    st.divider()

    # Payout/Share Trend (recomputed from shares across tokens at each tx)
    st.subheader("📈 Payout/Share Trend")
    # reconstruct shares timeline per token
    shares_timeline = []
    running = {t: 0 for t in TOKENS}
    for _, r in tx.iterrows():
        tkn, act, q = r['Outcome'], r['Action'], int(r['Quantity'])
        if act == 'Buy':
            running[tkn] += q
        else:
            running[tkn] -= q
        total = sum(running.values()) or 1
        payout = total / max(1, running[tkn])
        shares_timeline.append({
            'Time': pd.to_datetime(r['Time']), 'Outcome': tkn, 'Payout/Share': payout
        })
    ps_df = pd.DataFrame(shares_timeline)
    fig = px.line(ps_df, x="Time", y="Payout/Share", color="Outcome", markers=True,
              title="Payout/Share Over Time")
    st.plotly_chart(fig, use_container_width=True, key="chart_payout_share")
    # st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # Hide for now
    # # Circulating Shares Over Time
    # st.subheader("📊 Circulating Token Shares Over Time")
    # rows = []
    # running = {t: 0 for t in TOKENS}
    # for _, r in tx.iterrows():
    #     tkn, act, q = r['Outcome'], r['Action'], int(r['Quantity'])
    #     if act == 'Buy':
    #         running[tkn] += q
    #     else:
    #         running[tkn] -= q
    #     rows.append({'Time': pd.to_datetime(r['Time']), **{f'Shares {t}': running[t] for t in TOKENS}})
    # shares_df = pd.DataFrame(rows)
    # m = shares_df.melt(id_vars='Time', var_name='Token', value_name='Reserve')
    # fig_stack = px.area(m, x="Time", y="Reserve", color="Token", title="Token Shares vs. Time")
    # st.plotly_chart(fig_stack, use_container_width=True)

# ===========================================================

# --- smart sampler: dense around reserve, sparse elsewhere ---
def _curve_samples(max_shares: int, reserve: int, dense_pts: int = 1500, sparse_pts: int = 600) -> np.ndarray:
    if max_shares <= 1:
        return np.array([1], dtype=int)

    # Dense window = ±3% of domain (capped at ±50k)
    half = min(int(0.03 * max_shares), 50_000)
    lo = max(1, reserve - half)
    hi = min(max_shares, reserve + half)

    dense = np.linspace(lo, hi, num=max(2, dense_pts), dtype=int)

    left = np.array([], dtype=int)
    if lo > 1:
        left = np.unique(np.logspace(0, np.log10(lo), num=max(2, sparse_pts // 2), base=10.0)).astype(int)
        left = left[(left >= 1) & (left < lo)]

    right = np.array([], dtype=int)
    if hi < max_shares:
        right = np.unique(np.logspace(np.log10(hi + 1), np.log10(max_shares), num=max(2, sparse_pts // 2), base=10.0)).astype(int)
        right = right[(right > hi) & (right <= max_shares)]

    xs = np.unique(np.concatenate([left, dense, right]))
    return xs

# Vectorized curve evals
def buy_curve_np(x: np.ndarray) -> np.ndarray:
    # y = cbrt(x)/1000 + 0.1
    return np.cbrt(x) / 1000.0 + 0.1

def sell_marginal_net_np(x: np.ndarray) -> np.ndarray:
    # Effective *net* marginal price for selling 1 share at reserve x
    # = [gross (x→x-1)] * (1 - tax(q=1, C=x))
    x = x.astype(int)
    # gross price for 1 share = buy_delta(x) - buy_delta(x-1)
    bd = np.vectorize(buy_delta, otypes=[float])(x)
    bd_prev = np.vectorize(buy_delta, otypes=[float])(np.maximum(0, x - 1))
    gross_1 = bd - bd_prev
    tax_1 = _sale_tax_rate_vec(1, x)  # tax for selling 1 share from reserve x
    net_1 = gross_1 * (1.0 - tax_1)
    net_1[x <= 0] = 0.0
    return net_1

@st.cache_data(show_spinner=False)
def get_curve_series(max_shares: int, reserve: int, dense_pts: int = 1500, sparse_pts: int = 600):
    xs = _curve_samples(max_shares, reserve, dense_pts=dense_pts, sparse_pts=sparse_pts)
    return xs, buy_curve_np(xs), sell_marginal_net_np(xs)


# --- Bonding Curves by Outcome ---
st.subheader("🔁 Bonding Curves by Outcome")

token_tabs = st.tabs(TOKENS)  # token-level tabs only

# get latest reserves once
with closing(get_conn()) as conn:
    latest_reserves = {
        r["token"]: int(r["shares"])
        for r in conn.execute("SELECT token, shares FROM reserves").fetchall()
    }

for token, token_tab in zip(TOKENS, token_tabs):
    with token_tab:
        reserve = int(latest_reserves.get(token, 0))

        # One radio to switch sub-graphs (no nested tabs)
        view = st.radio(
            "View",
            ["Buy Curve", "Sell Spread", "Effective Sell (Net)"],
            horizontal=True,
            key=f"view_{token}",
        )

        if view == "Buy Curve":
            # point annotations at current reserve
            buy_price_now = float(buy_curve(reserve))
            sell_net_now = float(current_marginal_sell_price_after_tax(reserve))  # if you have this

            # smart-sampled series (buy + sell curves)
            xs, buy_vals, sell_net_vals = get_curve_series(MAX_SHARES, reserve)

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
            st.plotly_chart(fig_curve, use_container_width=True, key=f"chart_curve_{token}")

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


# ===========================================================


# Always fetch tx + users once
with closing(get_conn()) as conn:
    txp = pd.read_sql_query("""
        SELECT t.ts as Time, u.id as user_id, u.username as User, t.action as Action,
               t.token as Outcome, t.qty as Quantity, t.buy_delta as BuyAmt_Delta,
               t.sell_delta as SellAmt_Delta
        FROM transactions t JOIN users u ON t.user_id=u.id
        ORDER BY t.ts ASC
    """, conn)
    users_df = pd.read_sql_query("SELECT id, username FROM users ORDER BY id", conn)

if txp.empty:
    st.info("No transactions yet to compute history.")
else:
    # Historical charts: toggle between Portfolio Value and Points
    st.subheader("📊 History")

    # Tabs for Portfolio vs Points
    tab1, tab2 = st.tabs(["💼 Portfolio Value", "⭐ Points"])
    txp["Time"] = pd.to_datetime(txp["Time"])

    with tab1:

        # Reconstruct state over time (same as your current logic)
        reserves_state = {t: 0 for t in TOKENS}
        user_state = {
            int(r.id): {"username": r.username, "balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKENS}}
            for _, r in users_df.iterrows()
        }
        records = []

        for _, r in txp.iterrows():
            uid = int(r["user_id"])
            act = r["Action"]
            tkn = r["Outcome"]
            qty = int(r["Quantity"])

            if act == "Buy":
                delta = float(r["BuyAmt_Delta"] or 0.0)
                user_state[uid]["balance"] -= delta
                user_state[uid]["holdings"][tkn] += qty
                reserves_state[tkn] += qty

            elif act == "Sell":
                delta = float(r["SellAmt_Delta"] or 0.0)
                user_state[uid]["balance"] += delta
                user_state[uid]["holdings"][tkn] -= qty
                reserves_state[tkn] -= qty

            elif act == "Resolve":
                payout = float(r["SellAmt_Delta"] or 0.0)
                user_state[uid]["balance"] += payout
                for u_id in user_state:
                    user_state[u_id]["holdings"] = {t: 0 for t in TOKENS}
                reserves_state = {t: 0 for t in TOKENS}

            # Pricing snapshot
            if act == "Resolve":
                prices = {t: 0.0 for t in TOKENS}
            else:
                prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}

            # Snapshot every user at this event time
            for u_id, s in user_state.items():
                pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
                pnl = pv - STARTING_BALANCE
                records.append({"Time": r["Time"], "User": s["username"], "PortfolioValue": pv, "PnL": pnl})

        port_df = pd.DataFrame(records)
        fig_port = px.line(
            port_df, x="Time", y="PortfolioValue", color="User",
            title=f"Portfolio Value Over Time (Buy Price)"
        )
        st.plotly_chart(fig_port, use_container_width=True, key="portfolio_value_chart")

    with tab2:
        # ---- Points chart ----
        points_tl = compute_points_timeline(txp, users_df)

        if points_tl.empty:
            st.info("No points history to display yet.")
        else:
            # (Optional) keep leaderboard consistency by excluding admin
            points_plot_df = points_tl[points_tl["User"].str.lower() != "admin"].copy()

            fig_pts = px.line(
                points_plot_df,
                x="Time",
                y="TotalPoints",
                color="User",
                title="Total Points Over Time (Cumulative Volume + Instant PnL Points)",
                line_group="User",
            )
            st.plotly_chart(fig_pts, use_container_width=True, key='points_chart')

 # ===========================================================
# Historical portfolio visualization (toggle between buy and sell price)

with closing(get_conn()) as conn:
    txp = pd.read_sql_query("""
        SELECT t.ts as Time, u.id as user_id, u.username as User, t.action as Action,
               t.token as Outcome, t.qty as Quantity, t.buy_delta as BuyAmt_Delta,
               t.sell_delta as SellAmt_Delta
        FROM transactions t JOIN users u ON t.user_id=u.id
        ORDER BY t.ts ASC
    """, conn)
    users_df = pd.read_sql_query("SELECT id, username FROM users ORDER BY id", conn)

if not txp.empty:
    reserves_state = {t: 0 for t in TOKENS}
    user_state = {
        int(r.id): {"username": r.username, "balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKENS}}
        for _, r in users_df.iterrows()
    }
    records = []

    txp["Time"] = pd.to_datetime(txp["Time"])

    for _, r in txp.iterrows():
        uid = int(r["user_id"])
        act = r["Action"]
        tkn = r["Outcome"]
        qty = int(r["Quantity"])

        if act == "Buy":
            delta = float(r["BuyAmt_Delta"] or 0.0)
            user_state[uid]["balance"] -= delta
            user_state[uid]["holdings"][tkn] += qty
            reserves_state[tkn] += qty

        elif act == "Sell":
            delta = float(r["SellAmt_Delta"] or 0.0)
            user_state[uid]["balance"] += delta
            user_state[uid]["holdings"][tkn] -= qty
            reserves_state[tkn] -= qty

        elif act == "Resolve":
            # Apply this user's payout (logged in SellAmt_Delta)
            payout = float(r["SellAmt_Delta"] or 0.0)
            user_state[uid]["balance"] += payout

            # Zero ALL holdings (winners keep only credited payout; losers go to 0)
            for u_id in user_state:
                user_state[u_id]["holdings"] = {t: 0 for t in TOKENS}
            reserves_state = {t: 0 for t in TOKENS}

        # Pricing for valuation
        if act == "Resolve":
            # After resolution, holdings are worthless; prices treated as 0
            prices = {t: 0.0 for t in TOKENS}
        else:
            prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}

        # Snapshot all users at this event time
        for u_id, s in user_state.items():
            pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
            pnl = pv - STARTING_BALANCE
            records.append({"Time": r["Time"], "User": s["username"], "PortfolioValue": pv, "PnL": pnl})

    port_df2 = pd.DataFrame(records)  # <- use a fresh DF for this block
    st.divider()
    st.subheader("🏆 Leaderboard (Portfolio, PnL & Points)")

    # Latest portfolio snapshot per user (use port_df2, not the earlier port_df from the tab)
    latest = (
        port_df2.sort_values("Time")
        .groupby("User", as_index=False)
        .last()[["User", "PortfolioValue", "PnL"]]
    )

    # Exclude admin if you like
    latest = latest[latest["User"].str.lower() != "admin"]
    latest["PnL"] = latest["PortfolioValue"] - STARTING_BALANCE

    # ---- NEW: compute payout received at resolution from tx log ----
    with closing(get_conn()) as conn:
        payouts_df = pd.read_sql_query(
            """
            SELECT u.username AS User,
                COALESCE(SUM(t.sell_delta), 0.0) AS Payout
            FROM transactions t
            JOIN users u ON u.id = t.user_id
            WHERE t.action = 'Resolve'
            GROUP BY u.username
            """,
            conn,
        )

    latest = latest.merge(payouts_df, on="User", how="left")
    latest["Payout"] = pd.to_numeric(latest["Payout"], errors="coerce").fillna(0.0)

    # === Compute points === (unchanged)
    points_vol = compute_user_points(txp, users_df)
    pnl_points = latest[["User", "PnL"]].copy()
    pnl_points["PnLPoints"] = pnl_points["PnL"].clip(lower=0.0) * PNL_POINTS_PER_USD
    pnl_points = pnl_points[["User", "PnLPoints"]]
    pts = points_vol.merge(pnl_points, on="User", how="left")
    pts["PnLPoints"] = pts["PnLPoints"].fillna(0.0)
    pts["TotalPoints"] = pts["VolumePoints"] + pts["PnLPoints"]

    latest = latest.merge(
        pts[["User", "VolumePoints", "PnLPoints", "TotalPoints"]],
        on="User",
        how="left",
    ).fillna({"VolumePoints": 0.0, "PnLPoints": 0.0, "TotalPoints": 0.0})

    # ---- UI: let you sort by Payout to verify equal payouts after resolution ----
    metric_choice = st.radio(
        "Leaderboard metric (sort by):", ["Portfolio Value", "PnL", "Payout"], horizontal=True, key="lb_metric"
    )
    sort_key = {"Portfolio Value": "PortfolioValue", "PnL": "PnL", "Payout": "Payout"}[metric_choice]
    latest = latest.sort_values(sort_key, ascending=False)

    # Optional fairness check after resolution
    if resolved_flag == 1 and not payouts_df.empty and payouts_df["Payout"].nunique() == 1:
        st.caption("✅ All payouts are equal (same winning shares).")

    # Top cards + table (add Payout column)
    top_cols = st.columns(min(3, len(latest)))
    for i, (_, row) in enumerate(latest.head(3).iterrows()):
        with top_cols[i]:
            delta_val = f"${row['PnL']:,.2f}" if row['PnL'] >= 0 else f"-${abs(row['PnL']):,.2f}"
            st.metric(
                label=f"#{i+1} {row['User']}",
                value=f"${row['PortfolioValue']:,.2f}",
                delta=delta_val,
                border=True
            )
            st.caption(f"Payout: ${row['Payout']:,.2f}")
            st.caption(f"Points: {row['TotalPoints']:,.0f}")

    st.dataframe(
        latest[["User", "Payout", "PortfolioValue", "PnL", "VolumePoints", "PnLPoints", "TotalPoints"]],
        use_container_width=True
    )

else:
    st.info("No transactions yet to compute portfolio history.")