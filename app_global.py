




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

# ===========================================================
# Constants
BASE_EPSILON = 1e-4
MARKET_DURATION_DAYS = 5
END_TS = "2025-08-17 00:00"
DB_PATH = "app.db"
STARTING_BALANCE = 5000.0
MARKET_QUESTION = "Price of Ethereum by 17th Aug?"
RESOLUTION_NOTE = (
    'This market will resolve according to the final "Close" price of the '
    'Binance 1-minute candle for ETH/USDT at 12:00 UTC.'
)
TOKENS = ["<4200", "4200-4500", ">4500"]
# TOKENS = [f":blue[{TOKENS_TEXT[0]}]", f":blue[{TOKENS_TEXT[1]}]", f":blue[{TOKENS_TEXT[2]}]"]

# Whitelisted usernames and admin reset control
WHITELIST = {"admin", "rui", "haoye", "leo", "steve", "wenbo", "sam", "sharmaine", "mariam", "henry", "guard", "victor", "toby"}

# ===========================================================
# Streamlit Setup
st.set_page_config(page_title="42: Simulatoooor (Global)", layout="wide")
st.title("42: Twin Bonding Curve Simulatoooor — Global PVP")

st.subheader(f":blue[{MARKET_QUESTION}]")

def st_display_market_status(active):
    if active:
        st.badge("Active", icon=":material/check:", color="green")
    else:
        st.markdown(":gray-badge[Inactive]")

# ===========================================================
# Math Helpers (curves)

def buy_curve(x: float) -> float:
    return x**(1/4) + x / 400

def sell_curve(x: float) -> float:
    return ((x - 500)/40) / ((8 + ((x - 500)/80)**2)**0.5) + (x - 500)/300 + 3.6

def buy_delta(x: float) -> float:
    return (640 * x**(5/4) + x**2) / 800

def sell_delta(x: float) -> float:
    return 1.93333 * x + 0.00166667 * x**2 + 2 * math.sqrt(301200 - 1000 * x + x**2)

def metrics_from_qty(x: int, q: int):
    new_x = x + q
    buy_price = buy_curve(new_x)
    sell_price = sell_curve(x)
    buy_amt_delta = buy_delta(new_x) - buy_delta(x)
    sell_amt_delta = sell_delta(x) - sell_delta(x - q) if x - q >= 0 else 0.0
    return buy_price, sell_price, buy_amt_delta, sell_amt_delta

# Binary searches

def qty_from_buy_usdc(reserve: int, usdc_amount: float) -> int:
    low, high = 0.0, 10000.0
    eps = BASE_EPSILON
    while high - low > eps:
        mid = (low + high) / 2
        delta = buy_delta(reserve + mid) - buy_delta(reserve)
        if delta < usdc_amount:
            low = mid
        else:
            high = mid
    return max(0, int(round(low)))

def qty_from_sell_usdc(reserve: int, usdc_amount: float) -> int:
    low, high = 0.0, float(reserve)
    eps = BASE_EPSILON
    while high - low > eps:
        mid = (low + high) / 2
        delta = sell_delta(reserve) - sell_delta(reserve - mid)
        if delta < usdc_amount:
            low = mid
        else:
            high = mid
    return max(0, int(round(low)))

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

def ensure_market_resolution_columns():
    with closing(get_conn()) as conn, conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(market)").fetchall()}
        if "winner_token" not in cols:
            conn.execute("ALTER TABLE market ADD COLUMN winner_token TEXT")
        if "resolved" not in cols:
            conn.execute("ALTER TABLE market ADD COLUMN resolved INTEGER DEFAULT 0")
        if "resolved_ts" not in cols:
            conn.execute("ALTER TABLE market ADD COLUMN resolved_ts TEXT")

init_db()
ensure_market_resolution_columns()



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
                    if cnt >= 10:
                        st.error("Max 10 users reached. Try another time.")
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
if 'user_id' in st.session_state and st.session_state.get("username") == "admin":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resolve Market")
    winner = st.sidebar.selectbox("Winning outcome", TOKENS, key="winner_select")
    if st.sidebar.button("Resolve Now", disabled=not winner, type="secondary"):
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



# Market status
with closing(get_conn()) as conn:
    cur = conn.cursor()
    cur.execute("SELECT start_ts,end_ts,winner_token,resolved FROM market WHERE id=1")
    m = cur.fetchone()
    market_start = datetime.fromisoformat(m["start_ts"])
    market_end = datetime.fromisoformat(m["end_ts"])
    resolved_flag = int(m["resolved"] or 0)
    now = datetime.utcnow()
    active = (market_start <= now <= market_end) and (resolved_flag == 0)


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
        # Market still active - show rules/reminder
        st.markdown(
            f"""
            <div style="
                background-color: grey;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #a5b4fc;
                margin-bottom: 1rem;
            ">
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
            </div>
            """,
            unsafe_allow_html=True
        )

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
        ucols = st.columns(4)
        with ucols[0]:
            st.metric("Username", st.session_state.get("username",""))
        with ucols[1]:
            st.metric("Portfolio (USD)", f"{portfolio_value:,.2f}")
        with ucols[2]:
            st.metric("Balance (USDC)", f"{bal:,.2f}")
        with ucols[3]:
            st.metric("Shares Holdings Value (USD)", f"{holdings_total_value:,.2f}")

        # Per-token breakdown
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



# Common Trading UI
input_mode = st.radio("Select Input Mode", ["Quantity", "USDC"], horizontal=True)

quantity = 0
usdc_input = 0.0
if input_mode == "Quantity":
    quantity = st.number_input("Enter Quantity", min_value=1, step=1)
else:
    usdc_input = st.number_input("Enter USDC Amount", min_value=0.0, step=0.1)

st.subheader("Buy/Sell Controls")
cols = st.columns(3)

# Fetch current reserves (DB)
with closing(get_conn()) as conn:
    cur = conn.cursor()
    cur.execute("SELECT token, shares, usdc FROM reserves")
    reserves_map = {t: {"shares": s, "usdc": u} for t, s, u in cur.fetchall()}

for i, token in enumerate(TOKENS):
    with cols[i]:
        st.markdown(f"### Outcome :blue[{token}]")

        reserve = int(reserves_map[token]["shares"])  # global shares for token
        price_now = round(buy_curve(reserve), 4)
        mcap_now = round(reserves_map[token]["usdc"], 2)
        st.text(f"Current Price: {price_now}")
        st.text(f"MCAP: {mcap_now}")

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
                    bp, _, bdelta, _ = metrics_from_qty(reserve, q)
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
                            _, sp, _, sdelta = metrics_from_qty(reserve, q)
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
            st.metric(f"Total Shares {token}", reserve)
        with sub_cols[1]:
            st.metric(f"Price {token}", price)
        with sub_cols[2]:
            st.metric(f"MCAP {token}", mcap)


# Odds based on circulating shares
total_market_shares = max(1, int(res_df['Shares'].sum()))


sub_cols_2 = st.columns(len(TOKENS))
for i, token in enumerate(TOKENS):
    s = int(res_df.loc[res_df['Token']==token, 'Shares'].iloc[0])
    odds_val = '-' if s == 0 else round(1 / (s / total_market_shares), 2)
    with sub_cols_2[i]:
        st.metric(f"Odds {token}", f"{odds_val}x" if odds_val != '-' else "-", border=True)
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
    fig = px.line(ps_df, x="Time", y="Payout/Share", color="Outcome", markers=True, title="Payout/Share Over Time")
    st.plotly_chart(fig, use_container_width=True)
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
# Bonding Curves by Outcome with pointers
st.subheader("🔁 Bonding Curves by Outcome")
tab1, tab2, tab3 = st.tabs(TOKENS)
x_vals = list(range(1, 1001))
buy_vals = [buy_curve(x) for x in x_vals]
sell_vals = [sell_curve(x) for x in x_vals]

with closing(get_conn()) as conn:
    latest_reserves = {r["token"]: r["shares"] for r in conn.execute("SELECT token, shares FROM reserves").fetchall()}

for token, tab in zip(TOKENS, [tab1, tab2, tab3]):
    reserve = int(latest_reserves.get(token, 0))
    buy_price_now = buy_curve(reserve)
    sell_price_now = sell_curve(reserve)

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=x_vals, y=buy_vals, mode='lines', name='Buy Curve', line=dict(color='green')
    ))
    fig_curve.add_trace(go.Scatter(
        x=x_vals, y=sell_vals, mode='lines', name='Sell Curve', line=dict(color='red')
    ))

    # Buy point annotation
    fig_curve.add_trace(go.Scatter(
        x=[reserve], y=[buy_price_now], mode='markers+text',
        name=f'{token} Buy Point',
        text=[f"Shares: {reserve}<br>Price: {buy_price_now:.4f}"],
        textposition="top right",
        marker=dict(size=10, color='green'),
        showlegend=False
    ))

    # Sell point annotation
    fig_curve.add_trace(go.Scatter(
        x=[reserve], y=[sell_price_now], mode='markers+text',
        name=f'{token} Sell Point',
        text=[f"Shares: {reserve}<br>Price: {sell_price_now:.4f}"],
        textposition="bottom right",
        marker=dict(size=10, color='red'),
        showlegend=False
    ))

    # Dashed helper lines
    fig_curve.add_trace(go.Scatter(
        x=[reserve, reserve], y=[0, buy_price_now], mode='lines',
        line=dict(color='green', dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[0, reserve], y=[buy_price_now, buy_price_now], mode='lines',
        line=dict(color='green', dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[reserve, reserve], y=[0, sell_price_now], mode='lines',
        line=dict(color='red', dash='dot'), showlegend=False
    ))
    fig_curve.add_trace(go.Scatter(
        x=[0, reserve], y=[sell_price_now, sell_price_now], mode='lines',
        line=dict(color='red', dash='dot'), showlegend=False
    ))

    fig_curve.update_layout(
        title=f'{token} Price vs Shares',
        xaxis_title='Shares',
        yaxis_title='Price',
        hovermode="x unified"
    )

    tab.plotly_chart(fig_curve, use_container_width=True)

st.divider()
 # ===========================================================
# Historical portfolio visualization (toggle between buy and sell price)
price_mode = st.radio("Value holdings at:", ["Buy Price", "Mid Price", "Sell Price"], horizontal=True)
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
            if price_mode == "Buy Price":
                prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}
            elif price_mode == "Sell Price":
                prices = {t: sell_curve(reserves_state[t]) for t in TOKENS}
            else:
                prices = {t: buy_curve(reserves_state[t]) - sell_curve(reserves_state[t]) for t in TOKENS}

        # Snapshot all users at this event time
        for u_id, s in user_state.items():
            pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
            pnl = pv - STARTING_BALANCE
            records.append({"Time": r["Time"], "User": s["username"], "PortfolioValue": pv, "PnL": pnl})

    port_df = pd.DataFrame(records)
    fig_port = px.line(
        port_df, x="Time", y="PortfolioValue", color="User",
        title=f"Portfolio Value Over Time ({price_mode})"
    )
    st.subheader("💼 Portfolio Value History")
    st.plotly_chart(fig_port, use_container_width=True)

    # Leaderboard (final snapshot after last event)
    st.divider()
    st.subheader("🏆 Leaderboard (Portfolio & PnL)")
    latest = (
        port_df.sort_values("Time")
        .groupby("User", as_index=False)
        .last()[["User", "PortfolioValue", "PnL"]]
    )
        # Exclude admin from leaderboard
    latest = latest[latest["User"].str.lower() != "admin"]
    latest["PnL"] = latest["PortfolioValue"] - STARTING_BALANCE
    latest = latest.sort_values("PnL", ascending=False)

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

    st.dataframe(latest[["User", "PortfolioValue", "PnL"]], use_container_width=True)
else:
    st.info("No transactions yet to compute portfolio history.")