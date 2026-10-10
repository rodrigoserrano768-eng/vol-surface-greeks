import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for CI/test runs - no display needed

import pandas as pd
import pytest

from src.surface import build_surface
from src.plotting import plot_heatmap, plot_surface_3d, _pivot_surface, _interpolated_grid

FIXTURES = Path(__file__).parent / "fixtures"
RISK_FREE_RATE = 0.0385
DIVIDEND_YIELD = 0.012


@pytest.fixture
def surface_df():
    meta = json.loads((FIXTURES / "spy_meta.json").read_text())
    calls = pd.read_csv(FIXTURES / "spy_calls.csv")
    chains = {meta["expiry"]: calls}
    expiry_years = {meta["expiry"]: meta["T"]}
    df = build_surface(chains, meta["spot"], expiry_years, RISK_FREE_RATE, DIVIDEND_YIELD)
    df.attrs["spot"] = meta["spot"]
    return df


def test_pivot_surface_moneyness_filter_narrows_strikes(surface_df):
    spot = surface_df.attrs["spot"]
    full = _pivot_surface(surface_df)
    filtered = _pivot_surface(surface_df, spot=spot, moneyness_band=(0.95, 1.05))
    assert len(filtered.index) <= len(full.index)


def test_interpolated_grid_values_stay_within_observed_vol_range(surface_df):
    """Interpolation must not overshoot the real quoted points - the whole
    point of linear interpolation between real strikes is that it can't
    invent a vol outside the range actually observed on that expiry.
    """
    spot = surface_df.attrs["spot"]
    ok = surface_df[surface_df["error"].isna()]
    grid = _interpolated_grid(surface_df, spot=spot)
    observed_min, observed_max = ok["implied_vol"].min(), ok["implied_vol"].max()
    finite = grid.values[~pd.isna(grid.values)]
    assert finite.min() >= observed_min - 1e-9
    assert finite.max() <= observed_max + 1e-9


def test_plot_heatmap_runs_without_error(surface_df):
    ax = plot_heatmap(surface_df, "SPY", spot=surface_df.attrs["spot"])
    assert ax is not None


def test_plot_heatmap_raises_if_nothing_priced():
    empty = pd.DataFrame({"strike": [100], "expiry": ["2099-01-01"], "T": [0.1], "error": ["fail"], "implied_vol": [None]})
    with pytest.raises(ValueError, match="No successfully priced rows"):
        plot_heatmap(empty, "TEST", spot=100)


def test_plot_surface_3d_raises_if_fewer_than_three_strikes_per_expiry():
    df = pd.DataFrame(
        {
            "strike": [100, 110],
            "expiry": ["2099-01-01", "2099-02-01"],
            "T": [0.1, 0.2],
            "error": [None, None],
            "implied_vol": [0.2, 0.25],
        }
    )
    with pytest.raises(ValueError, match=">= 3 priced strikes"):
        plot_surface_3d(df, "TEST", spot=100)
