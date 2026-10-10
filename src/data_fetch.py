"""Pull a live option chain from Yahoo Finance via yfinance.

No API key, no registration. Yahoo's chain data is delayed/best-effort
and occasionally has stale or zero bid/ask on illiquid strikes - callers
are expected to filter before pricing (see surface.build_surface).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


@dataclass
class ChainSnapshot:
    ticker: str
    spot: float
    fetched_at: datetime
    calls: pd.DataFrame
    puts: pd.DataFrame
    expiry: str


def fetch_spot(ticker: str) -> float:
    t = yf.Ticker(ticker)
    fast = t.fast_info
    spot = fast.get("lastPrice") if hasattr(fast, "get") else fast["lastPrice"]
    if spot is None or spot <= 0:
        raise ValueError(f"Could not get a usable spot price for {ticker}: {spot!r}")
    return float(spot)


def list_expiries(ticker: str) -> list[str]:
    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries:
        raise ValueError(f"No listed option expiries found for {ticker}")
    return list(expiries)


def fetch_chain(ticker: str, expiry: str) -> ChainSnapshot:
    """Fetch one expiry's full call+put chain.

    Raises ValueError if the expiry isn't in the ticker's listed expiries,
    or if yfinance returns an empty chain (happens for thinly-traded names
    near expiry rollover).
    """
    t = yf.Ticker(ticker)
    available = t.options
    if expiry not in available:
        raise ValueError(f"{expiry} is not a listed expiry for {ticker}. Available: {available}")

    chain = t.option_chain(expiry)
    if chain.calls.empty and chain.puts.empty:
        raise ValueError(f"yfinance returned an empty chain for {ticker} {expiry}")

    spot = fetch_spot(ticker)
    return ChainSnapshot(
        ticker=ticker,
        spot=spot,
        fetched_at=datetime.now(timezone.utc),
        calls=chain.calls,
        puts=chain.puts,
        expiry=expiry,
    )


def years_to_expiry(expiry: str, as_of: datetime | None = None) -> float:
    """Calendar-day fraction of a year, using 365.25 days — fine for this project's
    tolerance; a trading-day calendar would be a refinement, not a correctness fix.
    """
    as_of = as_of or datetime.now(timezone.utc)
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(
        hour=21, minute=0, tzinfo=timezone.utc  # options expire ~4pm ET = ~21:00 UTC
    )
    delta_days = (expiry_dt - as_of).total_seconds() / 86400
    if delta_days <= 0:
        raise ValueError(f"Expiry {expiry} is not in the future relative to {as_of}")
    return delta_days / 365.25
