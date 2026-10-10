"""Visualize the strike x expiry implied-vol surface.

Two views, both required by the project brief:
- a 2D heatmap (strike x expiry, color = implied vol) - easiest to read
  precisely
- a 3D surface - shows the shape (skew, term structure) at a glance
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3d projection


def _pivot_surface(
    df: pd.DataFrame, spot: float | None = None, moneyness_band: tuple[float, float] = (0.85, 1.15)
) -> pd.DataFrame:
    """Pivot the long-format surface table into strike (rows) x expiry (cols).

    Drops rows that failed implied-vol inversion - they have no vol to plot.
    If spot is given, also restricts to a moneyness band: strikes far OTM on
    short-dated expiries produce implied vols that are numerically unstable
    (near-zero time value, a one-cent bid/ask move swings the solved vol by
    tens of points) - real, not a bug, but they dominate the color scale and
    hide the actual surface shape. The full unfiltered data stays in the
    underlying DataFrame and CSV export; this is a plotting choice only.
    """
    ok = df[df["error"].isna()].copy()
    if ok.empty:
        raise ValueError("No successfully priced rows to plot - every strike failed inversion")
    if spot is not None:
        lo, hi = spot * moneyness_band[0], spot * moneyness_band[1]
        ok = ok[(ok["strike"] >= lo) & (ok["strike"] <= hi)]
        if ok.empty:
            raise ValueError(f"No strikes left within moneyness band {moneyness_band} of spot={spot}")
    pivot = ok.pivot_table(index="strike", columns="expiry", values="implied_vol", aggfunc="mean")
    # order columns by T so the expiry axis reads left-to-right as near-to-far
    expiry_order = ok.drop_duplicates("expiry").sort_values("T")["expiry"].tolist()
    return pivot[[c for c in expiry_order if c in pivot.columns]]


def plot_heatmap(
    df: pd.DataFrame, ticker: str, spot: float | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    pivot = _pivot_surface(df, spot=spot)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    # thin y-tick labels so they stay legible regardless of strike count
    n_strikes = len(pivot.index)
    step = max(1, n_strikes // 25)
    ytick_pos = list(range(0, n_strikes, step))
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"{pivot.index[i]:.0f}" for i in ytick_pos])
    ax.set_xlabel("Expiry")
    ax.set_ylabel("Strike")
    ax.set_title(f"{ticker} implied volatility surface (heatmap)")
    plt.colorbar(im, ax=ax, label="Implied vol")
    return ax


def plot_surface_3d(df: pd.DataFrame, ticker: str, spot: float | None = None) -> plt.Figure:
    """3D surface plot.

    Unlike the heatmap, plot_surface needs a complete rectangular grid - it
    interpolates across NaN gaps and produces spurious spikes where strikes
    are missing for some expiries (different expiries quote different strike
    increments, e.g. weeklies vs monthlies). So this view is restricted to
    the strikes quoted at every expiry in the window (set intersection) -
    fewer points than the heatmap shows, but no fabricated interpolation.
    """
    pivot = _pivot_surface(df, spot=spot)
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError(
            "No strike is quoted across every selected expiry - can't build a complete 3D grid. "
            "Use plot_heatmap instead, which tolerates partial coverage."
        )
    T_by_expiry = df.drop_duplicates("expiry").set_index("expiry")["T"]
    x_t = np.array([T_by_expiry[c] for c in pivot.columns])
    y_strikes = pivot.index.values
    X, Y = np.meshgrid(x_t, y_strikes)
    Z = pivot.values

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", antialiased=True)
    ax.set_xlabel("Time to expiry (years)")
    ax.set_ylabel("Strike")
    ax.set_zlabel("Implied vol")
    ax.set_title(f"{ticker} implied volatility surface")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Implied vol")
    return fig
