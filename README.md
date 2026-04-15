# Superconductivity Lab — Analysis Scripts

Python analysis scripts for characterising a sintered polycrystalline YBa₂Cu₃O₇₋σ (YBCO) pellet.
Accompanies the lab report by Vladimir Sheydorov, March 2026.

## Scripts

**calibration.py** — Thermometer calibration and critical temperature Tc
- Loads voltage–time data from Excel (with Parquet caching for speed)
- Performs ODR linear calibration from three fixed points: liquid nitrogen (77.34 K), dry ice (194.7 K), and room temperature
- Converts thermometer voltage Uₐ to temperature with full covariance-aware uncertainty propagation
- Identifies Tc on both the cooling and warming branches from the Ub threshold crossing
- Computes resistivity ρ(T) with Gaussian error propagation on sample geometry

**bean_model.py** — Magnetic field modelling and Bean model
- Fits measured axial field B(z) to a disc magnet model (Derby et al.) using `scipy.optimize.curve_fit`
- Computes the analytical field gradient ∂B/∂z
- Derives the magnetic moment m(z) = F(z) / (∂B/∂z) from balance force measurements
- Constructs the magnetisation curve M(Hₐ) and estimates the critical current density jc via the Bean model

## Requirements

pip install numpy scipy matplotlib pandas pyarrow

Python 3.8+ recommended.

## Data and folder structure

project/
├── data/
│   ├── Day_2.xlsx          # Liquid nitrogen run (cooling + warming)
│   ├── Day_3_dry_ice.xlsx  # Dry ice calibration run
│   └── Day_4_full.xlsx     # Bean model force and field data
├── Plots/
│   ├── Plots_calibration/
│   └── Plots_bean/
├── calibration.py
└── bean_model.py


Excel files are cached as Parquet on first load; subsequent runs read from cache.

## Key results

| Quantity | Value |
|---|---|
| Tc (cooling) | 80.2 ± 0.2 K |
| Tc (warming) | 83.6 ± 0.2 K |
| jc | (4.6 ± 0.4) × 10⁵ A m⁻² |

Both Tc values fall below the literature optimum of 93 K, consistent with oxygen deficiency σ ≈ 0.2.
The jc value lies within the range reported by Babić et al. for sintered polycrystalline YBCO.
