# Oxi-Shapes: Tropical Geometry for Bounded Redox Proteomics

This repository contains the reference implementation and analysis scripts for **Oxi-Shapes**, a measurement-native tropical geometric framework for analysing bounded biochemical state spaces. The code accompanies a paired theoretical and applied submission to *Patterns* and provides a fully reproducible instantiation of the framework on empirical redox proteomics data.

---

## Conceptual overview

Many biochemical measurements—such as cysteine oxidation occupancy—are **categorically bounded** to the unit interval [0,1]. Oxi-Shapes provides an algebraically correct geometric formalism for analysing such data without rescaling, transformation, or loss of physical meaning.

The framework operates at two complementary levels:

- **Global lattice geometry**  
  The redox proteome is represented as a discrete lattice with a bounded scalar field, enabling computation of:
  - Internal redox entropy (measurement-native mean occupancy)
  - Configurational degeneracy
  - Graph-based curvature
  - Dirichlet (variation) energy
  - Morse-type ground-state potential

- **Site-wise bounded change geometry**  
  Redox state changes between conditions are represented exactly as points in a constrained (x, Δ) space, revealing:
  - Hard geometric bounds on admissible changes
  - Symmetric vs asymmetric site decomposition
  - Normalised signed redox change (fraction of available freedom used)
  - Rank-based salience of site-wise action

All constructions are **representation-only** and do not assume kinetics, dynamics, or biological coupling.

---

## Repository contents

├── data/
│ └── site_all.csv # Input site-level redox data (Young / Old)
│
├── analysis/
│ ├── bounded_delta_analysis.py # Site-wise bounded Δ geometry
│ ├── normalised_action.py # Normalised signed redox change (a ∈ [-1,1])
│ ├── lattice_entropy_energy.py # Global lattice entropy & energy functionals
│ └── violin_symmetry.py # Symmetric vs asymmetric site distributions
│
├── figures/
│ ├── bounded_delta_rgb.png
│ ├── bounded_delta_abs_companion.png
│ ├── normalised_signed_redox_change.png
│ └── symmetric_asymmetric_violin.png
│
└── README.md


---

## Data requirements

Input data must contain **site-resolved oxidation measurements** for two conditions (e.g. Young / Old), expressed either as percentages or fractions. The code automatically converts percentages to the unit interval.

Only sites measured in **both conditions** are retained to ensure invariant identity across comparisons.

---

## Reproducibility

- All analyses are deterministic and require no random seeds.
- No statistical hypothesis testing is performed.
- All figures are generated directly from the measured data.
- Output figures are saved at publication-ready resolution (300–600 dpi).

---

## Scope and generality

Although demonstrated here on cysteine redox proteomics, the framework is **mathematically agnostic to chemistry** and can be applied to any bounded biochemical measurement, including:
- Methionine oxidation
- Phosphorylation occupancy
- Stoichiometric PTM datasets
- Other bounded omics state spaces
