"""CLI entry point: fetch a live chain across several expiries for a ticker
and render the vol surface. Run with: python -m src.build_live_surface SPY
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from src.data_fetch import fetch_spot, list_expiries, fetch_chain, years_to_expiry
from src.surface import build_surface
from src.plotting import plot_heatmap, plot_surface_3d

# Both hardcoded rather than fetched live - a real desk would pull these from
# a rates curve (see Project 2), but adding that dependency here would be
# scope creep for a project whose point is the vol surface, not the rates
# curve. Documented in the README as a known simplification.
RISK_FREE_RATE = 0.0385
DIVIDEND_YIELD = 0.012


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--n-expiries", type=int, default=6)
    parser.add_argument("--out-prefix", default="surface")
    args = parser.parse_args()

    spot = fetch_spot(args.ticker)
    expiries = list_expiries(args.ticker)[: args.n_expiries]
    print(f"{args.ticker} spot={spot:.2f}, using {len(expiries)} expiries: {expiries}")

    chains = {}
    expiry_years = {}
    for exp in expiries:
        try:
            snap = fetch_chain(args.ticker, exp)
        except ValueError as e:
            print(f"  skipping {exp}: {e}", file=sys.stderr)
            continue
        chains[exp] = snap.calls
        expiry_years[exp] = years_to_expiry(exp)

    if not chains:
        print("No usable expiries fetched - aborting", file=sys.stderr)
        return 1

    surface_df = build_surface(chains, spot, expiry_years, RISK_FREE_RATE, DIVIDEND_YIELD)
    n_ok = surface_df["error"].isna().sum()
    n_total = len(surface_df)
    print(f"Priced {n_ok}/{n_total} contracts ({n_total - n_ok} failed inversion, see 'error' column)")

    ax = plot_heatmap(surface_df, args.ticker, spot=spot)
    ax.figure.tight_layout()
    ax.figure.savefig(f"{args.out_prefix}_heatmap.png", dpi=150)
    print(f"Saved {args.out_prefix}_heatmap.png")

    fig3d = plot_surface_3d(surface_df, args.ticker, spot=spot)
    fig3d.tight_layout()
    fig3d.savefig(f"{args.out_prefix}_3d.png", dpi=150)
    print(f"Saved {args.out_prefix}_3d.png")

    surface_df.to_csv(f"{args.out_prefix}_data.csv", index=False)
    print(f"Saved {args.out_prefix}_data.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
