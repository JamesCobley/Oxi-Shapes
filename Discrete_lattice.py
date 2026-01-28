import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, sqrt

# -----------------------
# User params
# -----------------------
CSV_PATH = "/content/site_all.csv"
Y_COL    = "oxi_percent_brain.young"
O_COL    = "oxi_percent_brain.old"

# Output
DPI       = 300
CMAP      = "viridis"
OUT_YOUNG = "/content/oxishapes_young_points.png"
OUT_OLD   = "/content/oxishapes_old_points.png"

# -----------------------
# Helpers
# -----------------------
def to_unit(x: pd.Series) -> pd.Series:
    """Convert % values to [0,1] if needed and clip to [0,1]."""
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().size and x.dropna().max() > 1.5:
        x = x / 100.0
    return x.clip(0, 1)

def choose_site_id(df: pd.DataFrame) -> pd.Series:
    """
    Pick a stable site identifier if available; otherwise fall back to row index.
    This is used to keep (x,y) invariant across Young/Old and across runs.

    TIP: If you have a known unique ID column (e.g., 'site_id'), add it to the
    top of the candidates list to avoid collisions.
    """
    candidates = [
        "site_id", "SiteID", "SITE_ID",
        "site", "Site", "SITE",
        "cys_site", "CysSite", "cysteine_site",
        "protein_site", "ProteinSite",
        "uniprot", "UniProt", "protein", "Protein",
        "gene", "Gene",
        "position", "Position", "residue", "Residue",
        "peptide", "Peptide",
    ]
    present = [c for c in candidates if c in df.columns]
    if present:
        cols = present[:3]  # concatenate a small stable set
        sid = df[cols].astype(str).agg("|".join, axis=1)
        return sid
    return df.index.astype(str)

def assign_grid_xy(site_ids: np.ndarray):
    """
    Deterministic invariant (x,y) from sorted site IDs.
    Places points on a near-square grid (lattice), indexed by sorted site id.

    NOTE: This is an invariant embedding for visualization only.
    It does NOT assert biological adjacency or continuity.
    """
    order = np.argsort(site_ids)
    site_ids_sorted = site_ids[order]

    N = site_ids_sorted.size
    ncols = int(ceil(sqrt(N)))

    x = (np.arange(N) % ncols).astype(float)
    y = (np.arange(N) // ncols).astype(float)

    # Normalize to [0,1] for stable aesthetics across N
    if x.max() > 0:
        x = x / x.max()
    if y.max() > 0:
        y = y / y.max()

    return order, x, y, site_ids_sorted

def plot_points_3d(x, y, z, title, outfile, elev=24, azim=-55, s=6, alpha=0.9):
    """
    Discrete Oxi-Shapes visualization: points only (no triangulation / no interpolation).
    This avoids implicitly imposing neighborhood structure in early figures.
    """
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x, y, z,
        c=z,
        cmap=CMAP,
        s=s,
        alpha=alpha,
        linewidths=0.0
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("Invariant x (site index lattice)")
    ax.set_ylabel("Invariant y (site index lattice)")
    ax.set_zlabel("Oxidation (unit interval)")

    # Optional: consistent aspect so z doesn't look exaggerated/flattened across figures
    try:
        ax.set_box_aspect((1, 1, 0.45))
    except Exception:
        pass

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Oxidation")

    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.show()

def plot_points_2d(x, y, z, title, outfile, s=6, alpha=0.9):
    """
    Optional top-down companion view (often helpful for papers).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=z, cmap=CMAP, s=s, alpha=alpha, linewidths=0.0)
    ax.set_title(title)
    ax.set_xlabel("Invariant x (site index lattice)")
    ax.set_ylabel("Invariant y (site index lattice)")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label("Oxidation")
    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.show()

# -----------------------
# Load and filter
# -----------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
if not ({Y_COL, O_COL} <= set(df.columns)):
    raise ValueError(f"Required columns not found: {Y_COL}, {O_COL}")

# Convert to [0,1]
df["_y"] = to_unit(df[Y_COL])
df["_o"] = to_unit(df[O_COL])

# Keep shared sites (measured in both)
mask = df["_y"].notna() & df["_o"].notna()
shared = df.loc[mask].copy()

# Stable site IDs
shared["_site_id"] = choose_site_id(shared)

# Sort sites deterministically and assign invariant x,y
order, x, y, site_ids_sorted = assign_grid_xy(shared["_site_id"].to_numpy())
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

# After yvals/ovals are defined
zmin = float(np.nanmin([yvals.min(), ovals.min()]))
zmax = float(np.nanmax([yvals.max(), ovals.max()]))

def plot_points_3d(x, y, z, title, outfile, elev=24, azim=-55, s=6, alpha=0.9, vmin=None, vmax=None, zlim=None):
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x, y, z,
        c=z,
        cmap=CMAP,
        s=s,
        alpha=alpha,
        linewidths=0.0,
        vmin=vmin, vmax=vmax
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("Invariant x (site index lattice)")
    ax.set_ylabel("Invariant y (site index lattice)")
    ax.set_zlabel("Oxidation (unit interval)")

    if zlim is not None:
        ax.set_zlim(*zlim)

    try:
        ax.set_box_aspect((1, 1, 0.45))
    except Exception:
        pass

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Oxidation")

    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.show()

def plot_points_2d(x, y, z, title, outfile, s=6, alpha=0.9, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=z, cmap=CMAP, s=s, alpha=alpha, linewidths=0.0, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Invariant x (site index lattice)")
    ax.set_ylabel("Invariant y (site index lattice)")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label("Oxidation")
    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.show()

# Use shared scaling for BOTH figures
plot_points_3d(x, y, yvals, f"Oxi-Shapes (discrete points) — Young (N={N})", OUT_YOUNG,
              vmin=zmin, vmax=zmax, zlim=(zmin, zmax))
plot_points_3d(x, y, ovals, f"Oxi-Shapes (discrete points) — Old (N={N})", OUT_OLD,
              vmin=zmin, vmax=zmax, zlim=(zmin, zmax))

plot_points_2d(x, y, yvals, f"Oxi-Shapes (top-down) — Young (N={N})", OUT_YOUNG.replace(".png", "_topdown.png"),
              vmin=zmin, vmax=zmax)
plot_points_2d(x, y, ovals, f"Oxi-Shapes (top-down) — Old (N={N})", OUT_OLD.replace(".png", "_topdown.png"),
              vmin=zmin, vmax=zmax)
