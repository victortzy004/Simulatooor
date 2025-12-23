from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import requests
import streamlit as st
import pandas as pd


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass(frozen=True)
class EvidenceRequest:
    binance_symbol: str
    timestamp_end_seconds: int


def print_resolution_evidence(e: dict) -> None:
    print("\n=== RESOLUTION EVIDENCE (Binance) ===")
    print(f"Symbol:          {e['symbol']}")
    print(f"Interval:        {e['interval']}")
    print(f"Candle open UTC: {e['resolution_candle_open_utc']}")
    print(f"Close price:     {e['close_price']}")
    print(f"Kline open ms:   {e['kline_open_time_ms']}")
    print(f"Kline close ms:  {e['kline_close_time_ms']}")
    print(f"Query URL:       {e['query_url']}")
    print("======================================\n")


def _dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    return int(dt.timestamp() * 1000)


def parse_ts_end(dt_str: str) -> int:
    """
    Accepts a human-readable datetime string and converts it to a UTC timestamp (seconds).

    Examples valid inputs:
    - "2025-09-15 23:59:00"
    - "2025-09-15T23:59:00Z"
    - "2025-09-15 23:59"
    - "2025-09-15T23:59"
    All interpreted as UTC unless an explicit offset is provided.
    """
    # Attempt ISO-8601 parse first (handles Z suffix)
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        # Fallback manual formats if needed
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(dt_str, fmt)
                # assume UTC if no tzinfo
                dt = dt.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                pass
        else:
            raise ValueError(f"Unrecognized datetime format: {dt_str}")

    # Ensure timezone-aware → convert to UTC timestamp
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp())


def _resolution_window_from_timestamp_end(timestamp_end_seconds: int) -> tuple[int, int, datetime]:
    """
    Given your market's timestamp_end (seconds), compute the 1m candle window
    that opens at 23:59:00 UTC on that same UTC date.

    startTime_ms = 23:59:00.000
    endTime_ms   = 23:59:59.999
    """
    end_dt = datetime.fromtimestamp(int(timestamp_end_seconds), tz=timezone.utc)

    # Your builder sets timestamp_end = 23:59:00 UTC, but we don't rely on that—
    # we recompute the 23:59 candle for that date anyway.
    candle_open = end_dt.replace(hour=23, minute=59, second=0, microsecond=0)
    candle_close = end_dt.replace(hour=23, minute=59, second=59, microsecond=999_000)

    return _dt_to_ms(candle_open), _dt_to_ms(candle_close), candle_open


def fetch_binance_close_price_evidence(
    *,
    binance_symbol: str,              # e.g. "BNBUSDT"
    timestamp_end_seconds: int,       # market end timestamp (seconds)
    session: Optional[requests.Session] = None,
    timeout_s: int = 15,
) -> Dict[str, Any]:
    """
    Returns evidence for resolution:
      - exact Binance query url
      - candle open time (UTC)
      - kline open/close times (ms)
      - close price (float)
      - raw kline payload
    """
    start_ms, end_ms, candle_open_dt = _resolution_window_from_timestamp_end(timestamp_end_seconds)

    params = {
        "symbol": binance_symbol,
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1,
    }

    sess = session or requests.Session()
    r = sess.get(BINANCE_KLINES_URL, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"No kline returned. symbol={binance_symbol} params={params} resp={data}")

    k = data[0]
    # [ open_time, open, high, low, close, volume, close_time, ... ]
    open_time_ms = int(k[0])
    close_time_ms = int(k[6])
    close_price = float(k[4])

    return {
        "symbol": binance_symbol,
        "interval": "1m",
        "resolution_candle_open_utc": candle_open_dt.isoformat(),
        "startTime_ms": start_ms,
        "endTime_ms": end_ms,
        "kline_open_time_ms": open_time_ms,
        "kline_close_time_ms": close_time_ms,
        "close_price": close_price,
        "raw": k,
        "query_url": r.url,
    }


def fetch_many_binance_close_price_evidence(
    *,
    symbols: Optional[Sequence[str]] = None,
    timestamp_end_seconds: Optional[int] = None,
    evidence_requests: Optional[Sequence[EvidenceRequest]] = None,
) -> List[Dict[str, Any]]:
    """
    Batch fetch resolution evidence.

    Use ONE of:
      1) symbols=[...], timestamp_end_seconds=...          (same end timestamp for all)
      2) evidence_requests=[EvidenceRequest(...), ...]     (per-symbol timestamps)

    Returns: list of evidence dicts (one per symbol), preserving input order.
    Each dict includes an extra key: "binance_symbol".
    """
    if evidence_requests is None:
        if not symbols or timestamp_end_seconds is None:
            raise ValueError("Provide either (evidence_requests=...) OR (symbols=... and timestamp_end_seconds=...).")
        reqs = [EvidenceRequest(s, int(timestamp_end_seconds)) for s in symbols]
    else:
        if symbols or timestamp_end_seconds is not None:
            raise ValueError("If evidence_requests=... is provided, do not pass symbols/timestamp_end_seconds.")
        reqs = list(evidence_requests)

    out: List[Dict[str, Any]] = []
    for r in reqs:
        try:
            ev = fetch_binance_close_price_evidence(
                binance_symbol=r.binance_symbol,
                timestamp_end_seconds=r.timestamp_end_seconds,
            )
            ev["binance_symbol"] = r.binance_symbol
            out.append(ev)
        except Exception as e:
            out.append(
                {
                    "binance_symbol": r.binance_symbol,
                    "timestamp_end_seconds": r.timestamp_end_seconds,
                    "error": str(e),
                }
            )
    return out


# -------------------------
# Streamlit Frontend
# -------------------------


def main() -> None:
    st.title("Binance Resolution Evidence Viewer")

    st.markdown(
        "Enter a **UTC market end datetime** and a list of Binance symbols. "
        "On click, this will fetch the **23:59 UTC 1m candle** for that date for each symbol."
    )

    default_symbols = "BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT"
    symbols_str = st.text_input("Symbols (comma-separated)", value=default_symbols)

    default_dt = "2025-12-22 23:59:00"
    dt_str = st.text_input("Market end datetime (UTC)", value=default_dt, help="e.g. 2025-12-22 23:59:00")

    if st.button("Fetch resolution evidence"):
        # Parse symbols
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            st.error("Please provide at least one symbol.")
            return

        # Parse datetime
        try:
            ts_end = parse_ts_end(dt_str)
        except ValueError as e:
            st.error(f"Error parsing datetime: {e}")
            return

        with st.spinner("Fetching klines from Binance..."):
            evidences = fetch_many_binance_close_price_evidence(
                symbols=symbols,
                timestamp_end_seconds=ts_end,
            )

        # Separate successes and errors
        success_rows: List[Dict[str, Any]] = []
        error_rows: List[Dict[str, Any]] = []

        for ev in evidences:
            if "error" in ev:
                error_rows.append(ev)
            else:
                success_rows.append(
                    {
                        "Symbol": ev["symbol"],
                        "Interval": ev["interval"],
                        "Candle open UTC": ev["resolution_candle_open_utc"],
                        "Close price": ev["close_price"],
                        # "Kline open ms": ev["kline_open_time_ms"],
                        # "Kline close ms": ev["kline_close_time_ms"],
                        # "Query URL": ev["query_url"],
                    }
                )

        if success_rows:
            st.subheader("Resolution Evidence")
            df = pd.DataFrame(success_rows)
            st.dataframe(df, use_container_width=True)

        if error_rows:
            st.subheader("Errors")
            for err in error_rows:
                st.error(
                    f"{err.get('binance_symbol')} "
                    f"(timestamp_end_seconds={err.get('timestamp_end_seconds')}): {err.get('error')}"
                )

        # Optional raw debug view
        with st.expander("Raw evidence payloads (debug)"):
            st.write(evidences)


if __name__ == "__main__":
    main()
