"""
Oxi-Shapes — curvature + Dirichlet energy + Morse energy on a measurement-native lattice
Standalone script (Colab-ready)

Outputs (all at 600 DPI):
- 3D scatter for Young and Old: curvature (graph Laplacian), Dirichlet energy density, Morse energy density
- Top-down delta plots (Old − Young) for each of the above
- Printed summaries: N sites, means, totals, deltas, sanity checks

NOTES (austere + explicit):
- The base object is a bounded scalar field φ ∈ [0,1] on a discrete index lattice.
- Relational structure is introduced ONLY here, via 4-neighbour adjacency on the index grid.
  This adjacency is mathematical (index-based) and does NOT imply biological interaction/proximity.
- Curvature is defined as the combinatorial graph Laplacian: κ = Lφ, with Lφ_i = Σ_{j~i} (φ_i − φ_j).
- Dirichlet energy is the quadratic variation of φ across edges.
- Morse energy here is a monotone Morse potential with explicit ground state at φ=0:
    V(φ) = (1 − exp(−a φ))^2  (minimum at 0, increasing for φ>0).
"""

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

DPI  = 600
CMAP = "viridis"

# Morse potential parameter (steepness). Larger = sharper rise away from ground.
MORSE_A = 6.0

# Output paths
OUT_CURV_Y  = "/content/curvature_young_3d.png"
OUT_CURV_O  = "/content/curvature_old_3d.png"
OUT_DIR_Y   = "/content/dirichlet_young_3d.png"
OUT_DIR_O   = "/content/dirichlet_old_3d.png"
OUT_MOR_Y   = "/content/morse_young_3d.png"
OUT_MOR_O   = "/content/morse_old_3d.png"

OUT_DCURV   = "/content/delta_curvature_topdown.png"
OUT_DDIR    = "/content/delta_dirichlet_topdown.png"
OUT_DMOR    = "/content/delta_morse_topdown.png"


# -----------------------
# Helpers: unit interval + stable IDs + lattice coords
# -----------------------
def to_unit(x: pd.Series) -> pd.Series:
    """Convert % values to [0,1] if needed. No clipping is applied by default."""
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().size and x.dropna().max() > 1.5:
        x = x / 100.0
    return x

def choose_site_id(df: pd.DataFrame) -> pd.Series:
    """
    Stable deterministic site identifier for invariant molecular identities.
    Prefer true unique IDs if present; else concatenate a minimal stable set.
    """
    candidates = [
        "site_id", "SiteID", "SITE_ID",
        "protein_site", "ProteinSite",
        "uniprot", "UniProt", "protein", "Protein",
        "position", "Position", "residue", "Residue",
        "peptide", "Peptide",
        "gene", "Gene",
        "site", "Site", "SITE",
    ]
    present = [c for c in candidates if c in df.columns]
    if present:
        cols = present[:3]
        return df[cols].astype(str).agg("|".join, axis=1)
    return df.index.astype(str)

def assign_grid(site_ids: np.ndarray):
    """
    Deterministic near-square grid assignment based on sorted site IDs.
    Returns:
      order (indices into original array),
      row, col (integer grid coords),
      x, y (normalised to [0,1] for plotting),
      nrows, ncols
    """
    order = np.argsort(site_ids)
    s = site_ids[order]
    N = s.size
    ncols = int(ceil(sqrt(N)))
    nrows = int(ceil(N / ncols))

    idx = np.arange(N)
    col = (idx % ncols).astype(int)
    row = (idx // ncols).astype(int)

    # Normalised for consistent aesthetics (not biology)
    x = col.astype(float)
    y = row.astype(float)
    if x.max() > 0: x = x / x.max()
    if y.max() > 0: y = y / y.max()

    return order, row, col, x, y, nrows, ncols

def sanity_unit_interval(name: str, v: np.ndarray, tol=1e-9):
    """Hard check: oxidation occupancies should be in [0,1] up to numerical tolerance."""
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    ok = (vmin >= -tol) and (vmax <= 1.0 + tol)
    if not ok:
        raise ValueError(f"[{name}] values outside [0,1] beyond tol={tol}: min={vmin}, max={vmax}")
    return vmin, vmax


# -----------------------
# Graph / adjacency (index grid, 4-neighbour)
# -----------------------
def build_edges(row: np.ndarray, col: np.ndarray, nrows: int, ncols: int):
    """
    Build undirected edge list for 4-neighbour adjacency on the index grid.
    Uses integer (row,col). Sites only exist for indices 0..N-1; last row may be partially filled.

    Returns list of (i,j) with i<j.
    """
    N = row.size
    # Map grid cell -> node index
    grid_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}

    edges = []
    for i in range(N):
        r, c = int(row[i]), int(col[i])
        # Right and Down only (avoid duplicates), then enforce i<j anyway
        for dr, dc in [(0, 1), (1, 0)]:
            rr, cc = r + dr, c + dc
            j = grid_to_idx.get((rr, cc), None)
            if j is not None:
                a, b = (i, j) if i < j else (j, i)
                edges.append((a, b))
    return edges


# -----------------------
# Curvature + Energies
# -----------------------
def graph_laplacian_curvature(phi: np.ndarray, edges):
    """
    Combinatorial graph Laplacian curvature: κ = Lφ, with
      (Lφ)_i = Σ_{j~i} (φ_i − φ_j)
    Returns κ and degree array.
    """
    N = phi.size
    deg = np.zeros(N, dtype=float)
    kappa = np.zeros(N, dtype=float)

    for i, j in edges:
        # undirected
        deg[i] += 1.0
        deg[j] += 1.0

        d = phi[i] - phi[j]
        kappa[i] += d
        kappa[j] -= d  # because (φ_j − φ_i) = -(φ_i − φ_j)

    return kappa, deg

def dirichlet_energy(phi: np.ndarray, edges):
    """
    Dirichlet energy on an undirected graph:
      E_D = (1/2) Σ_{(i,j)∈E} (φ_i − φ_j)^2

    Also returns a per-node density:
      e_i = (1/2) Σ_{j~i} (φ_i − φ_j)^2
    so that Σ_i e_i = 2 E_D.
    """
    N = phi.size
    e_node = np.zeros(N, dtype=float)
    e_total = 0.0

    for i, j in edges:
        d = phi[i] - phi[j]
        w = d * d
        e_total += 0.5 * w
        # distribute half to each node (still sums to 2E_D across nodes by this definition)
        e_node[i] += 0.5 * w
        e_node[j] += 0.5 * w

    return e_total, e_node

def morse_potential(phi: np.ndarray, a: float = 6.0):
    """
    Monotone Morse potential with explicit ground state at 0:
      V(φ) = (1 − exp(−a φ))^2,  φ∈[0,1]
    Returns per-node Morse energy density and total (mean and sum are both useful).
    """
    v = (1.0 - np.exp(-a * phi)) ** 2
    return v


# -----------------------
# Plotting
# -----------------------
def plot_3d_scatter(x, y, z, title, outfile, vmin=None, vmax=None, zlabel="Value",
                    elev=24, azim=-55, s=6, alpha=0.95):
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x, y, z,
        c=z, cmap=CMAP,
        s=s, alpha=alpha, linewidths=0.0,
        vmin=vmin, vmax=vmax
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("Invariant x (index lattice)")
    ax.set_ylabel("Invariant y (index lattice)")
    ax.set_zlabel(zlabel)

    # stable aspect for readability
    try:
        ax.set_box_aspect((1, 1, 0.45))
    except Exception:
        pass

    cbar = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.10)
    cbar.set_label(zlabel)

    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.show()

def plot_topdown_delta(x, y, dz, title, outfile, vlim=None, cbar_label="Δ (Old − Young)",
                       s=6, alpha=0.95):
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    if vlim is None:
        vmax = float(np.nanmax(np.abs(dz)))
        vlim = (-vmax, vmax)
    vmin, vmax = vlim

    sc = ax.scatter(
        x, y,
        c=dz, cmap=CMAP, s=s, alpha=alpha, linewidths=0.0,
        vmin=vmin, vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel("Invariant x (index lattice)")
    ax.set_ylabel("Invariant y (index lattice)")
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.show()


# -----------------------
# Main
# -----------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
if not ({Y_COL, O_COL} <= set(df.columns)):
    raise ValueError(f"Required columns not found: {Y_COL}, {O_COL}")

# Convert to [0,1] (no clipping; we assert boundedness as a sanity check)
df["_y"] = to_unit(df[Y_COL])
df["_o"] = to_unit(df[O_COL])

# Shared sites only
mask = df["_y"].notna() & df["_o"].notna()
shared = df.loc[mask].copy()
shared["_site_id"] = choose_site_id(shared)

# Deterministic lattice coordinates
order, row, col, x, y, nrows, ncols = assign_grid(shared["_site_id"].to_numpy())
shared = shared.iloc[order].reset_index(drop=True)

phi_y = shared["_y"].to_numpy(dtype=float)
phi_o = shared["_o"].to_numpy(dtype=float)

N = phi_y.size
print(f"(f) Sites used (shared Young & Old): {N}")

# Hard sanity check: bounded
ymin, ymax = sanity_unit_interval("Young", phi_y)
omin, omax = sanity_unit_interval("Old",   phi_o)
print(f"Sanity (unit interval): Young[min,max]=({ymin:.6f},{ymax:.6f})  Old[min,max]=({omin:.6f},{omax:.6f})")

# Internal redox entropy (volume ≡ mean oxidation)
Hy = float(np.mean(phi_y))
Ho = float(np.mean(phi_o))
dH = Ho - Hy
print("\n(a) Internal redox entropy (volume ≡ mean oxidation)")
print(f"Young: {Hy:.12f}  ({100*Hy:.6f}%)")
print(f"Old  : {Ho:.12f}  ({100*Ho:.6f}%)")

# (c) sanity: internal redox entropy equals arithmetic mean (trivial by definition, but explicit)
tol = 1e-12
print("\n(c) Sanity check: internal redox entropy equals arithmetic mean")
print(f"Young |H-mean| = {abs(Hy - np.mean(phi_y)):.3e} (tol={tol})")
print(f"Old   |H-mean| = {abs(Ho - np.mean(phi_o)):.3e} (tol={tol})")

# (d) delta
print("\n(d) Δ mean (Old − Young)")
print(f"Δ = {dH:.12f}  ({100*dH:.6f}%)")
print("\nHeadroom (reducing direction):")
print(f"Young reducing headroom to 0 = {Hy:.12f}")
print(f"Old   reducing headroom to 0 = {Ho:.12f}")
print(f"Fraction of Young headroom represented by |Δ| = {abs(dH)/Hy if Hy>0 else np.nan:.6f}")

# Build adjacency (index grid, 4-neighbour)
edges = build_edges(row, col, nrows, ncols)
print(f"\nAdjacency: 4-neighbour index grid")
print(f"Nodes N={N}, edges |E|={len(edges)}")

# Curvature: κ = Lφ
kappa_y, deg = graph_laplacian_curvature(phi_y, edges)
kappa_o, _   = graph_laplacian_curvature(phi_o, edges)
dkappa = kappa_o - kappa_y

# Dirichlet
ED_y, eD_y = dirichlet_energy(phi_y, edges)
ED_o, eD_o = dirichlet_energy(phi_o, edges)
dED_total  = ED_o - ED_y
deD        = eD_o - eD_y

# Morse (ground state at 0)
eM_y = morse_potential(phi_y, a=MORSE_A)
eM_o = morse_potential(phi_o, a=MORSE_A)
dEM_mean = float(np.mean(eM_o) - np.mean(eM_y))
dEM_sum  = float(np.sum(eM_o) - np.sum(eM_y))
deM      = eM_o - eM_y

# Print summaries
print("\n=== Curvature summary (κ = Lφ) ===")
print(f"Young: mean={np.mean(kappa_y):.6e},  sd={np.std(kappa_y):.6e}")
print(f"Old  : mean={np.mean(kappa_o):.6e},  sd={np.std(kappa_o):.6e}")
print(f"Δκ   : mean={np.mean(dkappa):.6e}, sd={np.std(dkappa):.6e}")

print("\n=== Dirichlet energy (quadratic variation) ===")
print(f"E_D Young (total) = {ED_y:.6e}")
print(f"E_D Old   (total) = {ED_o:.6e}")
print(f"ΔE_D (Old−Young)  = {dED_total:.6e}")
print(f"Dirichlet density (per-node): Young mean={np.mean(eD_y):.6e}, Old mean={np.mean(eD_o):.6e}")

print("\n=== Morse energy (ground-state at 0; V(φ)=(1-exp(-aφ))^2) ===")
print(f"a = {MORSE_A}")
print(f"Morse mean Young = {np.mean(eM_y):.6e}   (sum={np.sum(eM_y):.6e})")
print(f"Morse mean Old   = {np.mean(eM_o):.6e}   (sum={np.sum(eM_o):.6e})")
print(f"Δ Morse mean     = {dEM_mean:.6e}")
print(f"Δ Morse sum      = {dEM_sum:.6e}")

# -----------------------
# Plot scaling (shared Young/Old per metric)
# -----------------------
# Curvature scaling
curv_min = float(np.nanmin([kappa_y.min(), kappa_o.min()]))
curv_max = float(np.nanmax([kappa_y.max(), kappa_o.max()]))

# Dirichlet density scaling (non-negative)
dir_min  = float(np.nanmin([eD_y.min(), eD_o.min()]))
dir_max  = float(np.nanmax([eD_y.max(), eD_o.max()]))

# Morse density scaling (non-negative, bounded)
mor_min  = float(np.nanmin([eM_y.min(), eM_o.min()]))
mor_max  = float(np.nanmax([eM_y.max(), eM_o.max()]))

# -----------------------
# (e) Plots: 3D Young/Old + top-down deltas
# -----------------------
# Curvature 3D
plot_3d_scatter(x, y, kappa_y, f"Curvature (graph Laplacian) — Young (N={N})", OUT_CURV_Y,
                vmin=curv_min, vmax=curv_max, zlabel="κ = Lφ")
plot_3d_scatter(x, y, kappa_o, f"Curvature (graph Laplacian) — Old (N={N})",   OUT_CURV_O,
                vmin=curv_min, vmax=curv_max, zlabel="κ = Lφ")

# Dirichlet density 3D
plot_3d_scatter(x, y, eD_y, f"Dirichlet energy density — Young (N={N})", OUT_DIR_Y,
                vmin=dir_min, vmax=dir_max, zlabel="e_D (node)")
plot_3d_scatter(x, y, eD_o, f"Dirichlet energy density — Old (N={N})",   OUT_DIR_O,
                vmin=dir_min, vmax=dir_max, zlabel="e_D (node)")

# Morse density 3D
plot_3d_scatter(x, y, eM_y, f"Morse energy density (ground at 0) — Young (N={N})", OUT_MOR_Y,
                vmin=mor_min, vmax=mor_max, zlabel="V(φ)")
plot_3d_scatter(x, y, eM_o, f"Morse energy density (ground at 0) — Old (N={N})",   OUT_MOR_O,
                vmin=mor_min, vmax=mor_max, zlabel="V(φ)")

# Delta top-down (symmetric)
plot_topdown_delta(x, y, dkappa, f"Δ Curvature (Old − Young) — top-down (N={N})", OUT_DCURV,
                   cbar_label="Δκ (Old − Young)")
plot_topdown_delta(x, y, deD,    f"Δ Dirichlet density (Old − Young) — top-down (N={N})", OUT_DDIR,
                   cbar_label="Δe_D (Old − Young)")
plot_topdown_delta(x, y, deM,    f"Δ Morse density (Old − Young) — top-down (N={N})", OUT_DMOR,
                   cbar_label="ΔV(φ) (Old − Young)")

print("\nSaved outputs:")
for p in [OUT_CURV_Y, OUT_CURV_O, OUT_DIR_Y, OUT_DIR_O, OUT_MOR_Y, OUT_MOR_O, OUT_DCURV, OUT_DDIR, OUT_DMOR]:
    print(" -", p)
