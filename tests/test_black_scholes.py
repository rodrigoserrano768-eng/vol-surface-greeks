import math

import pytest

from src.black_scholes import BSInputs, price, delta, gamma, vega, implied_vol


def test_price_matches_known_textbook_value():
    # Hull, "Options, Futures and Other Derivatives" example: S=42, K=40, r=10%, sigma=20%, T=0.5, call
    inp = BSInputs(S=42, K=40, T=0.5, r=0.10, q=0.0, sigma=0.20, is_call=True)
    assert price(inp) == pytest.approx(4.76, abs=0.01)


def test_put_call_parity_holds():
    S, K, T, r, q, sigma = 100.0, 95.0, 1.0, 0.03, 0.01, 0.25
    call = price(BSInputs(S, K, T, r, q, sigma, is_call=True))
    put = price(BSInputs(S, K, T, r, q, sigma, is_call=False))
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_implied_vol_round_trips_from_known_price():
    inp = BSInputs(S=100, K=105, T=0.25, r=0.02, q=0.0, sigma=0.30, is_call=True)
    mkt_price = price(inp)
    recovered = implied_vol(mkt_price, S=100, K=105, T=0.25, r=0.02, q=0.0, is_call=True)
    assert recovered == pytest.approx(0.30, abs=1e-6)


def test_implied_vol_raises_on_price_below_intrinsic():
    # A call priced below its discounted intrinsic value is an arbitrage violation,
    # not a valid vol input — no sigma reproduces it.
    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_vol(market_price=0.01, S=100, K=50, T=1.0, r=0.0, q=0.0, is_call=True)


def test_implied_vol_raises_on_price_above_spot_for_call():
    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_vol(market_price=150, S=100, K=100, T=1.0, r=0.0, q=0.0, is_call=True)


def test_delta_call_between_zero_and_one():
    inp = BSInputs(S=100, K=100, T=1.0, r=0.02, q=0.0, sigma=0.2, is_call=True)
    d = delta(inp)
    assert 0.0 < d < 1.0


def test_delta_put_between_minus_one_and_zero():
    inp = BSInputs(S=100, K=100, T=1.0, r=0.02, q=0.0, sigma=0.2, is_call=False)
    d = delta(inp)
    assert -1.0 < d < 0.0


def test_gamma_positive_and_symmetric_for_call_and_put_same_strike():
    call = BSInputs(S=100, K=100, T=1.0, r=0.02, q=0.0, sigma=0.2, is_call=True)
    put = BSInputs(S=100, K=100, T=1.0, r=0.02, q=0.0, sigma=0.2, is_call=False)
    assert gamma(call) > 0
    assert gamma(call) == pytest.approx(gamma(put), abs=1e-10)


def test_vega_positive():
    inp = BSInputs(S=100, K=100, T=1.0, r=0.02, q=0.0, sigma=0.2, is_call=True)
    assert vega(inp) > 0


def test_zero_time_to_expiry_raises():
    with pytest.raises(ValueError):
        price(BSInputs(S=100, K=100, T=0.0, r=0.02, q=0.0, sigma=0.2, is_call=True))


def test_zero_vol_raises():
    with pytest.raises(ValueError):
        price(BSInputs(S=100, K=100, T=1.0, r=0.02, q=0.0, sigma=0.0, is_call=True))
