import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, sqrt
from math import lgamma

# ============================================================
# USER PARAMS
# ============================================================
CSV_PATH = "/content/site_all.csv"
Y_COL    = "oxi_percent_brain.young"
O_COL    = "oxi_percent_brain.old"

DPI  = 600
CMAP = "viridis"

# outputs (keeps standalone panels + adds degeneracy arc + delta lattice)
OUT_DEG   = "/content/degeneracy_arc_with_arrows.png"
OUT_D3D   = "/content/oxishapes_delta_3d.png"
OUT_D2D   = "/content/oxishapes_delta_topdown.png"

# ============================================================
# HELPERS
# ============================================================
def to_unit_no_clip(x: pd.Series) -> pd.Series:
    """
    Convert % values to [0,1] if needed.
    NO clipping. Values outside [0,1] are categorically invalid, so we error later.
    """
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().size and x.dropna().max() > 1.5:
        x = x / 100.0
    return x

def choose_site_id(df: pd.DataFrame) -> pd.Series:
    """
    Stable identifier for invariant (x,y) assignment.
    Prefers explicit IDs if present; otherwise concatenates a small stable set;
    final fallback is row index.
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
        cols = present[:3]
        return df[cols].astype(str).agg("|".join, axis=1)
    return df.index.astype(str)

def assign_grid_xy(site_ids: np.ndarray):
    """
    Deterministic invariant (x,y) from sorted site IDs.
    Near-square grid, normalized to [0,1] purely for plotting.
    """
    order = np.argsort(site_ids)
    ids_sorted = site_ids[order]
    N = ids_sorted.size
    ncols = int(ceil(sqrt(N)))

    x = (np.arange(N) % ncols).astype(float)
    y = (np.arange(N) // ncols).astype(float)

    if x.max() > 0: x = x / x.max()
    if y.max() > 0: y = y / y.max()
    return order, x, y

def log10_choose(n: int, k: int) -> float:
    """log10(C(n,k)) using lgamma for numerical stability."""
    return (lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)) / np.log(10)

def sci_from_log10(log10v: float) -> str:
    """Format huge numbers as a × 10^b from log10."""
    b = int(np.floor(log10v))
    a = 10**(log10v - b)
    return f"{a:.6f} × 10^{b}"

def plot_delta_3d(x, y, dz, title, outfile, elev=24, azim=-55, s=6, alpha=0.9):
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, dz, c=dz, cmap=CMAP, s=s, alpha=alpha, linewidths=0.0)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("Invariant x (site index lattice)")
    ax.set_ylabel("Invariant y (site index lattice)")
    ax.set_zlabel("Δ oxidation (Old − Young)")

    try:
        ax.set_box_aspect((1, 1, 0.45))
    except Exception:
        pass

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Δ oxidation")

    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.show()

def plot_delta_topdown(x, y, dz, title, outfile, s=6, alpha=0.9):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=dz, cmap=CMAP, s=s, alpha=alpha, linewidths=0.0)

    ax.set_title(title)
    ax.set_xlabel("Invariant x (site index lattice)")
    ax.set_ylabel("Invariant y (site index lattice)")
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label("Δ oxidation (Old − Young)")

    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.show()

# ============================================================
# LOAD + FILTER
# ============================================================
df = pd.read_csv(CSV_PATH, low_memory=False)
req = {Y_COL, O_COL}
missing = sorted(req - set(df.columns))
if missing:
    raise ValueError(f"Required columns not found: {missing}")

df["_y"] = to_unit_no_clip(df[Y_COL])
df["_o"] = to_unit_no_clip(df[O_COL])

# shared sites
shared = df.loc[df["_y"].notna() & df["_o"].notna()].copy()
shared["_site_id"] = choose_site_id(shared)

# enforce categorical bounds (no clipping)
bad_y = shared[(shared["_y"] < 0) | (shared["_y"] > 1)]
bad_o = shared[(shared["_o"] < 0) | (shared["_o"] > 1)]
if len(bad_y) or len(bad_o):
    raise ValueError(
        f"Found values outside [0,1] after conversion. "
        f"Young bad: {len(bad_y)}, Old bad: {len(bad_o)}. "
        f"Fix upstream rather than clipping."
    )

# invariant lattice coordinates
order, x, y = assign_grid_xy(shared["_site_id"].to_numpy())
shared = shared.iloc[order].reset_index(drop=True)

yvals = shared["_y"].to_numpy()
ovals = shared["_o"].to_numpy()
dz    = ovals - yvals
N = yvals.size

# (f) sanity check N
print(f"(f) Sites used (shared Young & Old): {N}")

# ============================================================
# (a) internal redox entropy = arithmetic mean (by construction)
# ============================================================
H_y = float(np.mean(yvals))
H_o = float(np.mean(ovals))

print("\n(a) Internal redox entropy (defined as volume ≡ mean oxidation)")
print(f"Young: {H_y:.12f}  ({H_y*100:.6f}%)")
print(f"Old  : {H_o:.12f}  ({H_o*100:.6f}%)")

# ============================================================
# (c) sanity check: entropy equals arithmetic mean (numerical identity)
# ============================================================
tol = 1e-12
print("\n(c) Sanity check: internal redox entropy equals arithmetic mean")
print(f"Young |H-mean| = {abs(H_y - float(np.mean(yvals))):.3e} (tol={tol})")
print(f"Old   |H-mean| = {abs(H_o - float(np.mean(ovals))):.3e} (tol={tol})")

# ============================================================
# (d) delta between means
# ============================================================
dH = H_o - H_y
print("\n(d) Δ mean (Old − Young)")
print(f"Δ = {dH:.12f}  ({dH*100:.6f}%)")

# ============================================================
# (b) degeneracy arc for N sites + arrows for Young/Old means
# ============================================================
# Convert mean to nearest admissible k (integer oxidized count)
k_y = int(np.rint(H_y * N))
k_o = int(np.rint(H_o * N))
k_y = max(0, min(N, k_y))
k_o = max(0, min(N, k_o))

mean_grid = np.arange(N+1) / N
log10_deg = np.array([log10_choose(N, k) for k in range(N+1)], dtype=float)

log10_y = float(log10_choose(N, k_y))
log10_o = float(log10_choose(N, k_o))

print("\n(b) Configurational degeneracy at rounded mean (Ω = C(N,k))")
print(f"Young: mean≈{k_y}/{N}={k_y/N:.6f}, log10Ω={log10_y:.6f}, Ω≈{sci_from_log10(log10_y)}")
print(f"Old  : mean≈{k_o}/{N}={k_o/N:.6f}, log10Ω={log10_o:.6f}, Ω≈{sci_from_log10(log10_o)}")

plt.figure(figsize=(9, 5.5))
plt.plot(mean_grid, log10_deg, linewidth=1)
plt.xlabel("Mean oxidation (k/N)")
plt.ylabel("log10 configurational degeneracy  log10(C(N,k))")
plt.title(f"Degeneracy in bounded binary state space (N={N} sites)")

def add_arrow(mu, yv, label):
    plt.annotate(
        label,
        xy=(mu, yv),
        xytext=(mu, yv + 0.5),
        arrowprops=dict(arrowstyle="->"),
        ha="center"
    )

add_arrow(k_y/N, log10_y, "Young mean")
add_arrow(k_o/N, log10_o, "Old mean")

plt.tight_layout()
plt.savefig(OUT_DEG, dpi=DPI)
plt.show()

# ============================================================
# (e) delta lattice (3D + top-down)
# ============================================================
plot_delta_3d(x, y, dz, f"Oxi-Shapes Δ lattice (Old − Young), N={N}", OUT_D3D)
plot_delta_topdown(x, y, dz, f"Oxi-Shapes Δ lattice (Old − Young), N={N}", OUT_D2D)

print("\nSaved:")
print(" -", OUT_DEG)
print(" -", OUT_D3D)
print(" -", OUT_D2D)
