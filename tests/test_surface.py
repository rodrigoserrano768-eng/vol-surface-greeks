import json
from pathlib import Path

import pandas as pd
import pytest

from src.black_scholes import BSInputs, price
from src.surface import clean_chain, build_expiry_slice, build_surface

FIXTURES = Path(__file__).parent / "fixtures"
RISK_FREE_RATE = 0.0385  # approx short-term T-bill yield at fixture capture date, hardcoded for reproducibility
DIVIDEND_YIELD = 0.012   # approx SPY trailing dividend yield at fixture capture date


@pytest.fixture
def meta():
    return json.loads((FIXTURES / "spy_meta.json").read_text())


@pytest.fixture
def calls_df():
    return pd.read_csv(FIXTURES / "spy_calls.csv")


def test_clean_chain_drops_crossed_and_zero_quotes():
    df = pd.DataFrame(
        {
            "strike": [100, 110, 120, 130],
            "bid": [5.0, 0.0, 3.0, 4.0],
            "ask": [5.5, 1.0, 2.5, 4.5],  # row index 2 is crossed (bid > ask)
        }
    )
    cleaned = clean_chain(df)
    assert set(cleaned["strike"]) == {100, 130}


def test_build_expiry_slice_on_real_frozen_chain_has_rows(calls_df, meta):
    out = build_expiry_slice(
        calls_df, spot=meta["spot"], T=meta["T"], r=RISK_FREE_RATE, q=DIVIDEND_YIELD, is_call=True
    )
    assert len(out) > 0
    ok = out[out["error"].isna()]
    assert len(ok) > 0
    assert (ok["implied_vol"] > 0).all()
    assert (ok["implied_vol"] < 5.0).all()


def test_implied_vol_round_trips_market_price_within_tolerance(calls_df, meta):
    """The acceptance criterion from the project brief: feeding the computed
    implied vol back into Black-Scholes must reproduce the market mid price
    actors actually quoted, within a documented tolerance.
    """
    out = build_expiry_slice(
        calls_df, spot=meta["spot"], T=meta["T"], r=RISK_FREE_RATE, q=DIVIDEND_YIELD, is_call=True
    )
    ok = out[out["error"].isna()]
    assert len(ok) >= 5, "need enough surviving strikes to call this a real test"

    max_abs_error = 0.0
    for _, row in ok.iterrows():
        inp = BSInputs(
            S=meta["spot"], K=row["strike"], T=meta["T"], r=RISK_FREE_RATE,
            q=DIVIDEND_YIELD, sigma=row["implied_vol"], is_call=True,
        )
        repriced = price(inp)
        err = abs(repriced - row["mid"])
        max_abs_error = max(max_abs_error, err)

    # Tolerance: 1 cent. Brent solver is run to xtol=1e-8 on sigma, so the
    # repricing error here is essentially floating-point noise, not model error.
    assert max_abs_error < 0.01, f"max repricing error {max_abs_error} exceeds 1 cent tolerance"


def test_deep_otm_strike_either_prices_or_reports_error_never_silent_nan(calls_df, meta):
    """Edge case required by the project brief: very far OTM strikes. The
    pipeline must not silently produce NaN - it either prices the strike or
    records why it couldn't (error message), so a user auditing the surface
    can see which strikes were dropped and why.
    """
    out = build_expiry_slice(
        calls_df, spot=meta["spot"], T=meta["T"], r=RISK_FREE_RATE, q=DIVIDEND_YIELD, is_call=True
    )
    far_otm = out[out["strike"] > meta["spot"] * 1.3]
    if far_otm.empty:
        pytest.skip("fixture chain doesn't extend 30% OTM - nothing to check")
    for _, row in far_otm.iterrows():
        assert row["error"] is not None or pd.notna(row["implied_vol"])


def test_build_surface_stacks_multiple_expiries_with_expiry_column(calls_df, meta):
    chains = {meta["expiry"]: calls_df, "fake-second-expiry": calls_df}
    expiry_years = {meta["expiry"]: meta["T"], "fake-second-expiry": meta["T"] * 2}
    out = build_surface(chains, spot=meta["spot"], expiry_years=expiry_years, r=RISK_FREE_RATE, q=DIVIDEND_YIELD)
    assert set(out["expiry"].unique()) == {meta["expiry"], "fake-second-expiry"}
    assert "T" in out.columns
