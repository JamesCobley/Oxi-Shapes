#!/usr/bin/env python3
"""
Discrete_lattice.py
Construct and visualise the Oxi-Shapes discrete tropical lattice object
(site index lattice + bounded oxidation occupancy field).

This script intentionally imposes no relational structure (no edges / adjacency).
"""

import argparse
from math import ceil, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_unit(x: pd.Series, *, name: str = "value") -> pd.Series:
    """
    Express oxidation measurements on the unit interval [0,1] by unit conversion
    (percent -> fraction) where needed. Values outside [0,1] are categorically
    inadmissible for occupancy measurements and are treated as errors.
    """
    x = pd.to_numeric(x, errors="coerce")

    # Unit conversion if values appear to be percentages
    xn = x.dropna()
    if xn.size and xn.max() > 1.5:
        x = x / 100.0

    # Validate bounds (do not clip)
    bad = x.dropna()[(x.dropna() < 0) | (x.dropna() > 1)]
    if bad.size:
        raise ValueError(
            f"{name}: found {bad.size} values outside [0,1]. "
            f"Example(s): {bad.iloc[:5].tolist()}"
        )
    return x


def choose_site_id(df: pd.DataFrame) -> pd.Series:
    """
    Construct a stable per-site identifier from available columns.
    Prefer explicit site-level IDs when present. Otherwise concatenate a small
    set of stable columns. Falls back to row index only if necessary.

    NOTE: Uniqueness is asserted downstream. If duplicates exist, you must
    refine the identifier (e.g., include UniProt + residue position).
    """
    candidates = [
        "site_id", "SiteID", "SITE_ID",
        "protein_site", "ProteinSite",
        "cys_site", "CysSite", "cysteine_site",
        "uniprot", "UniProt",
        "protein", "Protein",
        "gene", "Gene",
        "position", "Position",
        "residue", "Residue",
        "peptide", "Peptide",
        "site", "Site", "SITE",
    ]
    present = [c for c in candidates if c in df.columns]
    if present:
        cols = present[:3]
        return df[cols].astype(str).agg("|".join, axis=1)
    return df.index.astype(str)


def assign_grid_xy(site_ids: np.ndarray):
    """
    Deterministic lattice indexing for visualisation:
    - Sort sites by site_id
    - Place sequentially on a near-square grid
    - Normalise coordinates to [0,1] for consistent aesthetics
    """
    order = np.argsort(site_ids)
    site_ids_sorted = site_ids[order]

    N = site_ids_sorted.size
    ncols = int(ceil(sqrt(N)))

    x = (np.arange(N) % ncols).astype(float)
    y = (np.arange(N) // ncols).astype(float)

    if x.max() > 0:
        x /= x.max()
    if y.max() > 0:
        y /= y.max()

    return order, x, y, site_ids_sorted


def plot_points_3d(
    x, y, z, title, outfile,
    *, elev=24, azim=-55, s=6, alpha=0.9, cmap="viridis",
    vmin=None, vmax=None, zlim=None
):
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x, y, z,
        c=z,
        cmap=cmap,
        s=s,
        alpha=alpha,
        linewidths=0.0,
        vmin=vmin, vmax=vmax,
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("Invariant lattice index (x)")
    ax.set_ylabel("Invariant lattice index (y)")
    ax.set_zlabel("Oxidation occupancy [0,1]")

    if zlim is not None:
        ax.set_zlim(*zlim)

    try:
        ax.set_box_aspect((1, 1, 0.45))
    except Exception:
        pass

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Oxidation")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_points_2d(
    x, y, z, title, outfile,
    *, s=6, alpha=0.9, cmap="viridis", vmin=None, vmax=None
):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=z, cmap=cmap, s=s, alpha=alpha, linewidths=0.0, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Invariant lattice index (x)")
    ax.set_ylabel("Invariant lattice index (y)")
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label("Oxidation")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Construct Oxi-Shapes discrete lattice visualisations.")
    p.add_argument("--csv", required=True, help="Input CSV path containing site-wise oxidation columns.")
    p.add_argument("--young_col", required=True, help="Column name for young oxidation (% or fraction).")
    p.add_argument("--old_col", required=True, help="Column name for old oxidation (% or fraction).")
    p.add_argument("--out_prefix", default="oxishapes", help="Output prefix for figures.")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    args = p.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    if not ({args.young_col, args.old_col} <= set(df.columns)):
        raise ValueError(f"Required columns not found: {args.young_col}, {args.old_col}")

    df["_y"] = to_unit(df[args.young_col], name=args.young_col)
    df["_o"] = to_unit(df[args.old_col], name=args.old_col)

    mask = df["_y"].notna() & df["_o"].notna()
    shared = df.loc[mask].copy()

    shared["_site_id"] = choose_site_id(shared)

    # Assert uniqueness (critical for invariance)
    if shared["_site_id"].duplicated().any():
        dup = shared["_site_id"][shared["_site_id"].duplicated()].iloc[:10].tolist()
        raise ValueError(
            f"Non-unique site identifiers detected (showing up to 10): {dup}. "
            "Refine choose_site_id() to include additional site-resolving columns."
        )

    order, x, y, _ = assign_grid_xy(shared["_site_id"].to_numpy())
    shared_sorted = shared.iloc[order].reset_index(drop=True)

    yvals = shared_sorted["_y"].to_numpy()
    ovals = shared_sorted["_o"].to_numpy()
    N = yvals.size

    print(f"Sites with measurements in BOTH (Young & Old): {N}")
    print("\n=== Mean oxidation ===")
    print(f"Young mean = {yvals.mean():.6f} ({100*yvals.mean():.3f}%)")
    print(f"Old   mean = {ovals.mean():.6f} ({100*ovals.mean():.3f}%)")
    d = ovals.mean() - yvals.mean()
    print(f"Δ mean (Old−Young) = {d:.6f} ({100*d:.3f}%)")

    zmin = float(np.nanmin([yvals.min(), ovals.min()]))
    zmax = float(np.nanmax([yvals.max(), ovals.max()]))

    out_y_3d = f"{args.out_prefix}_young_points.png"
    out_o_3d = f"{args.out_prefix}_old_points.png"
    out_y_2d = f"{args.out_prefix}_young_topdown.png"
    out_o_2d = f"{args.out_prefix}_old_topdown.png"

    plot_points_3d(x, y, yvals, f"Oxi-Shapes (discrete points) — Young (N={N})", out_y_3d,
                   cmap=args.cmap, vmin=zmin, vmax=zmax, zlim=(zmin, zmax))
    plot_points_3d(x, y, ovals, f"Oxi-Shapes (discrete points) — Old (N={N})", out_o_3d,
                   cmap=args.cmap, vmin=zmin, vmax=zmax, zlim=(zmin, zmax))

    plot_points_2d(x, y, yvals, f"Oxi-Shapes (top-down) — Young (N={N})", out_y_2d,
                   cmap=args.cmap, vmin=zmin, vmax=zmax)
    plot_points_2d(x, y, ovals, f"Oxi-Shapes (top-down) — Old (N={N})", out_o_2d,
                   cmap=args.cmap, vmin=zmin, vmax=zmax)


if __name__ == "__main__":
    main()
