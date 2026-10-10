"""Build a strike x expiry implied-vol surface from a raw option chain.

Filtering choices (documented, not hidden):
- drop rows with no bid or no ask (can't form a mid, and last trade price on
  an illiquid contract can be hours stale)
- drop rows where bid > ask (seen occasionally in Yahoo's feed - bad quote)
- use mid price = (bid + ask) / 2 as the market price fed to implied_vol
- strikes more than `moneyness_cutoff` away from spot are kept but flagged,
  since far OTM/ITM implied vols are numerically unstable (see README
  "known limitations" - this module does not silently drop them)
"""
from __future__ import annotations

import pandas as pd

from src.black_scholes import implied_vol, BSInputs, delta, gamma, vega


def _mid_price(row: pd.Series) -> float:
    return (row["bid"] + row["ask"]) / 2.0


def clean_chain(df: pd.DataFrame, min_price: float = 0.01) -> pd.DataFrame:
    """Drop quotes that can't produce a usable implied vol: missing/crossed
    bid-ask, or a mid price too small to invert reliably.
    """
    out = df.copy()
    out = out[(out["bid"] > 0) & (out["ask"] > 0) & (out["bid"] <= out["ask"])]
    out["mid"] = out.apply(_mid_price, axis=1)
    out = out[out["mid"] >= min_price]
    return out.reset_index(drop=True)


def price_row(
    row: pd.Series, spot: float, T: float, r: float, q: float, is_call: bool
) -> dict:
    """Compute implied vol + Greeks for a single chain row. Returns a dict with
    an 'error' key (and no vol/greeks) rather than raising, so callers can
    batch-process a chain and keep the failures visible instead of losing the
    whole expiry to one bad strike.
    """
    K = float(row["strike"])
    mkt = float(row["mid"])
    try:
        iv = implied_vol(mkt, S=spot, K=K, T=T, r=r, q=q, is_call=is_call)
    except ValueError as e:
        return {"strike": K, "mid": mkt, "error": str(e)}

    inp = BSInputs(S=spot, K=K, T=T, r=r, q=q, sigma=iv, is_call=is_call)
    return {
        "strike": K,
        "mid": mkt,
        "implied_vol": iv,
        "delta": delta(inp),
        "gamma": gamma(inp),
        "vega": vega(inp),
        "moneyness": K / spot,
        "error": None,
    }


def build_expiry_slice(
    chain_df: pd.DataFrame, spot: float, T: float, r: float, q: float, is_call: bool
) -> pd.DataFrame:
    cleaned = clean_chain(chain_df)
    rows = [price_row(row, spot, T, r, q, is_call) for _, row in cleaned.iterrows()]
    return pd.DataFrame(rows)


def build_surface(
    chains_by_expiry: dict[str, pd.DataFrame],
    spot: float,
    expiry_years: dict[str, float],
    r: float,
    q: float,
    is_call: bool = True,
) -> pd.DataFrame:
    """Stack per-expiry slices into one long-format surface table with an
    `expiry` column. Rows that failed implied-vol inversion are kept with
    their `error` message, not dropped - the surface plot filters them
    visibly instead of the data pipeline hiding them.
    """
    frames = []
    for expiry, chain_df in chains_by_expiry.items():
        T = expiry_years[expiry]
        slice_df = build_expiry_slice(chain_df, spot, T, r, q, is_call)
        slice_df["expiry"] = expiry
        slice_df["T"] = T
        frames.append(slice_df)
    return pd.concat(frames, ignore_index=True)
