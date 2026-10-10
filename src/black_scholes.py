"""Black-Scholes pricing, Greeks, and implied volatility inversion.

European options, no dividends adjustment beyond a continuous yield q.
Rates and dividend yield are inputs, not fetched here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class BSInputs:
    S: float      # spot
    K: float      # strike
    T: float      # time to expiry in years
    r: float      # risk-free rate (continuous, annualized)
    q: float      # dividend yield (continuous, annualized)
    sigma: float  # volatility (annualized)
    is_call: bool = True


def _d1_d2(inp: BSInputs) -> tuple[float, float]:
    if inp.T <= 0 or inp.sigma <= 0:
        raise ValueError(f"T and sigma must be positive, got T={inp.T}, sigma={inp.sigma}")
    d1 = (math.log(inp.S / inp.K) + (inp.r - inp.q + 0.5 * inp.sigma ** 2) * inp.T) / (
        inp.sigma * math.sqrt(inp.T)
    )
    d2 = d1 - inp.sigma * math.sqrt(inp.T)
    return d1, d2


def price(inp: BSInputs) -> float:
    d1, d2 = _d1_d2(inp)
    disc_q = math.exp(-inp.q * inp.T)
    disc_r = math.exp(-inp.r * inp.T)
    if inp.is_call:
        return inp.S * disc_q * norm.cdf(d1) - inp.K * disc_r * norm.cdf(d2)
    return inp.K * disc_r * norm.cdf(-d2) - inp.S * disc_q * norm.cdf(-d1)


def delta(inp: BSInputs) -> float:
    d1, _ = _d1_d2(inp)
    disc_q = math.exp(-inp.q * inp.T)
    return disc_q * norm.cdf(d1) if inp.is_call else disc_q * (norm.cdf(d1) - 1)


def gamma(inp: BSInputs) -> float:
    d1, _ = _d1_d2(inp)
    disc_q = math.exp(-inp.q * inp.T)
    return disc_q * norm.pdf(d1) / (inp.S * inp.sigma * math.sqrt(inp.T))


def vega(inp: BSInputs) -> float:
    """Vega per 1.0 (100%) change in vol. Divide by 100 for per-1-vol-point convention."""
    d1, _ = _d1_d2(inp)
    disc_q = math.exp(-inp.q * inp.T)
    return inp.S * disc_q * norm.pdf(d1) * math.sqrt(inp.T)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Invert Black-Scholes for sigma via Brent's method on a bounded bracket.

    Raises ValueError if the market price is outside the arbitrage-free bounds
    for any positive vol (deep ITM/OTM or expired contracts produce this) or if
    brentq can't bracket a root within [lo, hi].
    """
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    disc_r = math.exp(-r * T)
    upper_bound = S if is_call else K * disc_r
    if market_price < intrinsic * disc_r - 1e-8 or market_price > upper_bound + 1e-8:
        raise ValueError(
            f"market_price={market_price} outside no-arbitrage bounds "
            f"[{intrinsic * disc_r:.4f}, {upper_bound:.4f}] for S={S}, K={K}, T={T}"
        )

    def objective(sigma: float) -> float:
        return price(BSInputs(S, K, T, r, q, sigma, is_call)) - market_price

    try:
        return brentq(objective, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError as e:
        raise ValueError(
            f"implied_vol failed to bracket a root in [{lo}, {hi}] for "
            f"S={S}, K={K}, T={T}, market_price={market_price}: {e}"
        ) from e
