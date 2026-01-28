import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# User params
# -----------------------
CSV_PATH = "/content/site_all.csv"
Y_COL    = "oxi_percent_brain.young"
O_COL    = "oxi_percent_brain.old"

# Outputs
DPI = 600
OUT = "/content/bounded_delta_rgb.png"

# Classification tolerance (absolute oxidation units; 1e-3 = 0.1%)
eps0 = 1e-3

# -----------------------
# Helpers
# -----------------------
def to_unit(x: pd.Series) -> pd.Series:
    """
    Convert percent to unit interval if needed.
    No clipping is applied (boundedness is categorical), but we sanity-check.
    """
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().size and x.dropna().max() > 1.5:
        x = x / 100.0
    return x

def choose_site_id(df: pd.DataFrame) -> pd.Series:
    """
    Deterministic site identifier used only to ensure invariant identity across conditions.
    Prefer a true unique ID if present; otherwise concatenate stable columns; else row index.
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

def assert_unit_interval(name: str, arr: np.ndarray, tol: float = 1e-9):
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    print(f"Sanity (unit interval): {name}[min,max]=({mn:.6f},{mx:.6f})")
    if mn < -tol or mx > 1 + tol:
        raise ValueError(f"{name} has values outside [0,1] beyond tol={tol}: min={mn}, max={mx}")

# -----------------------
# Load and filter
# -----------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
req = {Y_COL, O_COL}
if not req.issubset(df.columns):
    raise ValueError(f"Required columns not found: {sorted(req - set(df.columns))}")

df["_y"] = to_unit(df[Y_COL])
df["_o"] = to_unit(df[O_COL])

# Shared measured sites only
mask = df["_y"].notna() & df["_o"].notna()
shared = df.loc[mask].copy()
shared["_site_id"] = choose_site_id(shared)

# Deterministic ordering (not a biological ordering)
shared = shared.sort_values("_site_id", kind="mergesort").reset_index(drop=True)

# Extract
yvals = shared["_y"].to_numpy(dtype=float)
ovals = shared["_o"].to_numpy(dtype=float)
N = yvals.size

print(f"(f) Sites used (shared Young & Old): {N}")
assert_unit_interval("Young", yvals)
assert_unit_interval("Old",   ovals)

# -----------------------
# Site-wise bounded change object
# -----------------------
x = yvals
d = ovals - yvals  # Δ = Old - Young

# Feasibility check: -x <= d <= 1-x
viol_lo = int(np.sum(d < (-x - 1e-12)))
viol_hi = int(np.sum(d > ((1 - x) + 1e-12)))
print(f"Feasibility violations (tol=1e-12): lower={viol_lo}, upper={viol_hi}")

# ---- classify with tolerance ----
m_zero = np.abs(d) <= eps0
m_up   = d >  eps0
m_down = d < -eps0

print("\nClass counts:")
print(f"Identity (|Δ|≤{eps0:g}): {int(m_zero.sum())}")
print(f"Up (Δ>{eps0:g}):         {int(m_up.sum())}")
print(f"Down (Δ<-{eps0:g}):      {int(m_down.sum())}")

print("\nΔ summary:")
print(f"mean Δ = {float(np.mean(d)):.6e}")
print(f"sd   Δ = {float(np.std(d, ddof=0)):.6e}")
print(f"min  Δ = {float(np.min(d)):.6f}")
print(f"max  Δ = {float(np.max(d)):.6f}")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(7.2, 5.6))

# feasible region + bounds
xline = np.linspace(0, 1, 600)
plt.fill_between(xline, -xline, 1-xline, alpha=0.10)
plt.plot(xline, 1-xline, linewidth=1.2)
plt.plot(xline, -xline, linewidth=1.2)
plt.axhline(0, linewidth=1.0)

# scatter by class
plt.scatter(x[m_down], d[m_down], s=10, alpha=0.6, linewidths=0,
            c="blue", label="Δ < 0 (reduction)")
plt.scatter(x[m_up], d[m_up], s=10, alpha=0.6, linewidths=0,
            c="red", label="Δ > 0 (oxidation)")
plt.scatter(x[m_zero], d[m_zero], s=10, alpha=0.9, linewidths=0,
            c="black", label=f"|Δ| ≤ {eps0:g} (identity)", zorder=5)

plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.xlabel("Start oxidation $x_A$ (Young) in [0,1]")
plt.ylabel("Δ oxidation (Old − Young)")
plt.title(f"Bounded site-wise change space (N={N})")

plt.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(OUT, dpi=DPI)
plt.show()

print(f"\nSaved: {OUT}")
