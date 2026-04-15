import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit








##########################################################################################################################
########################################## Data Extraction from Excel ####################################################
##########################################################################################################################
file_path_bean_model = "data/bean_model_data.xlsx"
cache_path_bean      = "data/cache_bean.parquet"

if os.path.exists(cache_path_bean):
    df_bean = pd.read_parquet(cache_path_bean)
    print()
    print(f"Loaded from {cache_path_bean}")
else:
    df_bean = pd.read_excel(
        file_path_bean_model,
        usecols="E:W",
        skiprows=1,
        nrows=24,
        decimal=","
    )
    df_bean.to_parquet(cache_path_bean, index=False)
    print("Day bean model data read from Excel and saved cache.")

z_decreasing_0              = (df_bean.iloc[:, 0].to_numpy().flatten() + 4.3) * 1e-3    # m
m_decreasing_0              = df_bean.iloc[:, 1].to_numpy().flatten()                   # g

z_increasing_0              = (df_bean.iloc[:, 3].to_numpy().flatten() + 4.3) * 1e-3    # m
m_increasing_0              = df_bean.iloc[:, 4].to_numpy().flatten()                   # m
m_increasing_corrected_0    = df_bean.iloc[:, 5].to_numpy().flatten()                   # g

z_decreasing_1              = (df_bean.iloc[:, 8].to_numpy().flatten()  + 4.3) * 1e-3   # m
m_decreasing_1              = df_bean.iloc[:, 10].to_numpy().flatten()                  # g

z_increasing_1              = (df_bean.iloc[:, 12].to_numpy().flatten() + 4.3) * 1e-3   # m
m_increasing_1              = df_bean.iloc[:, 14].to_numpy().flatten()                  # g

z_field     = df_bean.iloc[:, 16].to_numpy().flatten() * 1e-2                          # m  (raw: cm)
B_field     = df_bean.iloc[:, 17].to_numpy().flatten() * 1e-3                          # T  (raw: mT)
B_field_err = df_bean.iloc[:, 18].to_numpy().flatten() * 1e-3                          # T  (raw: mT)

######################################################################################################################
####################################### End of Data Extraction #######################################################
######################################################################################################################



















##########################################################################################################################
########################################## Bean Model ###################################################################
##########################################################################################################################

mu0 = 4 * np.pi * 1e-7
g   = 9.81

R_magnet     = 0.0075               # m
R_magnet_err = 0.025e-3             # m
L_magnet     = 0.003                # m
L_magnet_err = 0.05e-3             # m

b_T     = 0.3e-3                    # T,  background B-field
b_T_err = 0.001e-3                  # T

R_superconductor     = 11.3e-3       # m
R_superconductor_err = 0.025e-3      # m
L_superconductor     = 3.5e-3        # m
L_superconducotr_err = 0.025e-3      # m
volume_superconductor = (R_superconductor**2) * np.pi * L_superconductor    # m^3


# --- 1. Disk magnet B-field model (returns T) ---
def disk_magnet_field(z_m, Br, z0_m):
    z_rel = z_m - z0_m                                  # m
    z_rel = np.where(z_rel < 1e-4, 1e-4, z_rel)
    Bz = (Br / 2.0) * (
        (z_rel + L_magnet) / np.sqrt((z_rel + L_magnet)**2 + R_magnet**2)
        - z_rel            / np.sqrt(z_rel**2              + R_magnet**2)
    ) + b_T
    return Bz    # T

# --- 2. Disk magnet dB/dz (returns T/m) model ---
def grad_disk_magnet_field(z_m, Br, z0_m):
    z_rel = z_m - z0_m                                  # m
    z_rel = np.where(z_rel < 1e-4, 1e-4, z_rel)
    dBz_dz = (Br * R_magnet**2 / 2.0) * (
        1.0 / ((z_rel + L_magnet)**2 + R_magnet**2)**1.5
        - 1.0 / (z_rel**2            + R_magnet**2)**1.5
    )
    return dBz_dz    # T/m


# --- 3. Fit ---
mask_f  = np.isfinite(z_field) & np.isfinite(B_field)
z_f     = z_field[mask_f]       # m
B_f     = B_field[mask_f]       # T
B_f_err = B_field_err[mask_f]   # T
sidx    = np.argsort(z_f)
z_f, B_f, B_f_err = z_f[sidx], B_f[sidx], B_f_err[sidx]
# z_f, B_f, B_f_err = z_f[:-5], B_f[:-5], B_f_err[:-5]     # exclude last 5 points (largest z)
# z_f, B_f, B_f_err = z_f[2:], B_f[2:], B_f_err[2:]     # exclude last 5 points (largest z)


p0     = [1.2,  0.0]
bounds = ([0.5, -5e-3],
          [1.5,  5e-3])

popt, pcov = curve_fit(disk_magnet_field, z_f, B_f,
                       sigma=B_f_err, absolute_sigma=True,
                       p0=p0, bounds=bounds, maxfev=40000)

print(f"Fit → Br={popt[0]:.3f}±{np.sqrt(pcov[0,0]):.2e} T  "
      f"(fixed)R={R_magnet*1e3:.1f}±{R_magnet_err*1e3:.3f} mm  "
      f"(fixed)L={L_magnet*1e3:.1f}±{L_magnet_err*1e3:.3f} mm  "
      f"z0={popt[1]*1e3:.2f}±{np.sqrt(pcov[1,1])*1e3:.2e} mm  "
      f"(fixed)b={b_T*1e3:.3f}±{b_T_err*1e3:.3f} mT")


# --- Reduced chi-squared of B-field fit ---
B_fit_at_data = disk_magnet_field(z_f, *popt)          # model evaluated at data points
residuals     = B_f - B_fit_at_data                     # T
chi2          = np.sum((residuals / B_f_err)**2)        # dimensionless
n_params      = 2                                       # fitted: Br, z0
dof           = len(z_f) - n_params                    # degrees of freedom
chi2_reduced  = chi2 / dof

print(f"Chi2         = {chi2:.4f}")
print(f"DOF          = {dof}")
print(f"Chi2_reduced = {chi2_reduced:.4f}")



# --- 4. Fine grid ---
z_fine = np.linspace(z_f.min(), z_f.max(), 5000)    # m
B_fine = disk_magnet_field(z_fine, *popt)           # T


def B_at(z_m):
    return disk_magnet_field(z_m, *popt)            # T

def dBdz_at(z_m):
    return grad_disk_magnet_field(z_m, *popt)       # T/m


# --- 5. Partial derivatives for uncertainty propagation ---
def dB_dBr(z_m, Br, z0_m):
    """∂B/∂Br — B with Br=1, returns T"""
    z_rel = z_m - z0_m
    z_rel = np.where(z_rel < 1e-4, 1e-4, z_rel)
    return 0.5 * (
        (z_rel + L_magnet) / np.sqrt((z_rel + L_magnet)**2 + R_magnet**2)
        - z_rel            / np.sqrt(z_rel**2              + R_magnet**2)
    )    # T (dimensionless × Br=1 T)

def dB_dz0(z_m, Br, z0_m):
    """∂B/∂z0 = −∂B/∂z, returns T/m"""
    return -grad_disk_magnet_field(z_m, Br, z0_m)    # T/m

def dB_dR(z_m, Br, z0_m):
    """∂B/∂R_magnet, returns T/m"""
    z_rel = z_m - z0_m
    z_rel = np.where(z_rel < 1e-4, 1e-4, z_rel)
    return (Br * R_magnet / 2.0) * (
        -(z_rel + L_magnet) / ((z_rel + L_magnet)**2 + R_magnet**2)**1.5
        + z_rel             / (z_rel**2              + R_magnet**2)**1.5
    )    # T/m  (σ_R is in m so units match)

def dB_dL(z_m, Br, z0_m):
    """∂B/∂L_magnet, returns T/m"""
    z_rel = z_m - z0_m
    z_rel = np.where(z_rel < 1e-4, 1e-4, z_rel)
    return (Br / 2.0) * (
        1.0 / np.sqrt((z_rel + L_magnet)**2 + R_magnet**2)
        - (z_rel + L_magnet)**2 / ((z_rel + L_magnet)**2 + R_magnet**2)**1.5
    )    # T/m

def dB_db(z_m):
    """∂B/∂b = 1 everywhere (b additive), returns dimensionless"""
    return np.ones_like(z_m)    # T/T = 1

def B_uncertainty(z_m, popt, pcov, z_err_m=2e-3):
    """Returns σ_B in T"""
    Br = popt[0]
    z0 = popt[1]
    var_B = (
        (dB_dBr(z_m, Br, z0) * np.sqrt(pcov[0, 0]))**2 +    # σ_Br contribution
        (dB_dz0(z_m, Br, z0) * np.sqrt(pcov[1, 1]))**2 +    # σ_z0 contribution
        (dB_dR( z_m, Br, z0) * R_magnet_err        )**2 +    # σ_R contribution
        (dB_dL( z_m, Br, z0) * L_magnet_err        )**2 +    # σ_L contribution
        (dB_db( z_m)         * b_T_err             )**2 +    # σ_b contribution
        (grad_disk_magnet_field(z_m, Br, z0) * z_err_m)**2 + # σ_z (position) contribution
        2 * dB_dBr(z_m, Br, z0) * dB_dz0(z_m, Br, z0) * pcov[0, 1]  # Br–z0 covariance
    )
    return np.sqrt(var_B)    # T

eps = 1e-7

def sigma_gradient(z_m, popt, pcov):
    """Uncertainty in dBz/dz from fitted Br, z0 (and fixed R, L)."""
    Br, z0 = popt

    # Numerical partials of grad w.r.t. Br and z0
    dg_dBr = (grad_disk_magnet_field(z_m, Br + eps, z0) -
              grad_disk_magnet_field(z_m, Br - eps, z0)) / (2 * eps)
    dg_dz0 = (grad_disk_magnet_field(z_m, Br, z0 + eps) -
              grad_disk_magnet_field(z_m, Br, z0 - eps)) / (2 * eps)

    return np.sqrt(
        (dg_dBr * np.sqrt(pcov[0, 0]))**2 +
        (dg_dz0 * np.sqrt(pcov[1, 1]))**2 +
        2 * dg_dBr * dg_dz0 * pcov[0, 1]   # Br–z0 covariance
    )


# --- 6. mass → force → moment ---
def compute_m_of_Ha(z_m, mass_g, popt, pcov, label="", z_err_m=2e-3):
    mask   = np.isfinite(z_m) & np.isfinite(mass_g)
    z_m    = z_m[mask]
    mass_g = mass_g[mask]

    F_N     = mass_g * 1e-3 * g            # N
    sigma_F = g * 0.02e-3                  # N  (constant)

    B_vals  = B_at(z_m)                    # T   (x-axis)
    B_err   = B_uncertainty(z_m, popt, pcov, z_err_m)  # T
    dB_vals = dBdz_at(z_m)                 # T/m
    sigma_dB = sigma_gradient(z_m, popt, pcov)         # T/m

    safe   = np.abs(dB_vals) > 0.01 * np.max(np.abs(dB_vals))
    m_Am2  = np.full_like(F_N, np.nan)
    m_err  = np.full_like(F_N, np.nan)

    m_Am2[safe] = F_N[safe] / dB_vals[safe]    # A·m²

    m_err[safe] = np.abs(m_Am2[safe]) * np.sqrt(
        (sigma_F          / F_N[safe]       )**2 +
        (sigma_dB[safe]   / dB_vals[safe]   )**2
    )

    return B_vals, B_err, m_Am2, m_err

B_dec, B_err_dec, m_dec, m_err_dec = compute_m_of_Ha(z_decreasing_1, m_decreasing_1, popt, pcov, "Decreasing z")
B_inc, B_err_inc, m_inc, m_err_inc = compute_m_of_Ha(z_increasing_1, m_increasing_1, popt, pcov, "Increasing z")

M_dec = m_dec / volume_superconductor    # A/m
M_inc = m_inc / volume_superconductor    # A/m

# Volume uncertainty (scalar)
sigma_V = volume_superconductor * np.sqrt(
    (2 * R_superconductor_err / R_superconductor)**2 +
    (L_superconducotr_err     / L_superconductor)**2
)

# σ_M = M * sqrt((σ_m/m)^2 + (σ_V/V)^2)
M_err_dec = np.abs(M_dec) * np.sqrt(
    (m_err_dec / np.where(np.isfinite(m_dec) & (m_dec != 0), m_dec, np.nan))**2 +
    (sigma_V   / volume_superconductor)**2
)
M_err_inc = np.abs(M_inc) * np.sqrt(
    (m_err_inc / np.where(np.isfinite(m_inc) & (m_inc != 0), m_inc, np.nan))**2 +
    (sigma_V   / volume_superconductor)**2
)


# --- Jc from Bean model for disk geometry: M = -(1/3) * jc * R ---
# jc = -3 * M / R, using minimum of M_dec (most negative point)

# Find minimum M_dec (ignoring NaNs)
valid_dec = np.isfinite(M_dec) & np.isfinite(M_err_dec)
idx_min   = np.nanargmin(M_dec[valid_dec])

M_min     = M_dec[valid_dec][idx_min]        # A/m  (most negative)
M_min_err = M_err_dec[valid_dec][idx_min]    # A/m

# jc = -3 * M_min / R
jc     = -3 * M_min / R_superconductor       # A/m^2
sigma_jc = 3 * np.sqrt(
    (M_min_err          / R_superconductor          )**2 +
    (M_min * R_superconductor_err / R_superconductor**2)**2
)    # A/m^2  (propagating R uncertainty too)

print()
print("--- Critical current density from Bean model ---")
print(f"M_min      = {M_min:.4f} ± {M_min_err:.4f} A/m")
print(f"mu0*M_min  = {mu0*M_min*1e3:.4f} ± {mu0*M_min_err*1e3:.4f} mT")
print(f"j_c        = {jc:.4e} ± {sigma_jc:.4e} A/m^2")
print()


# --- 7. Mass vs Distance plot ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(z_decreasing_1 * 1e3, m_decreasing_1, xerr=2, yerr=0.02,
            fmt='s', color="darkorange", ecolor="orange",
            elinewidth=1.2, capsize=3, markersize=5,
            markeredgecolor="darkred", markeredgewidth=1.2, zorder=3,
            label="Decreasing distance")
ax.errorbar(z_increasing_1[:-4] * 1e3, m_increasing_1[:-4], xerr=2, yerr=0.02,
            fmt='^', color="royalblue", ecolor="royalblue",
            elinewidth=1.2, capsize=3, markersize=5,
            markeredgecolor="darkblue", markeredgewidth=1.2, zorder=3,
            label="Increasing distance")
ax.set_title("Mass vs Distance")
ax.set_xlabel("Distance (mm)")
ax.set_ylabel("Mass (g)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_bean/mass_vs_distance.pdf", bbox_inches="tight")
plt.show()


# --- 8. B-field fit plot ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(
    z_f * 1e3, B_f * 1e3,
    xerr=2, yerr=B_f_err * 1e3,
    fmt='^', color="grey", ecolor="grey",
    elinewidth=1.2, capsize=4, markersize=5, zorder=3,
    markeredgecolor="black", markeredgewidth=1.2,
    label="Measured B (mT)"
)
ax.plot(z_fine * 1e3, B_fine * 1e3, color="red", lw=2, label="Disk magnet fit")
ax.set_xlabel("Distance z (mm)")
ax.set_ylabel("B (mT)")
ax.set_title(r"$B(z)$ – disk magnet fit")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_bean/B_field_fit.pdf", bbox_inches="tight")
plt.show()


# --- 9. Plot residuals for B fit ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(z_f * 1e3, residuals * 1e3,
            yerr=B_f_err * 1e3,
            fmt='o', color="darkblue", ecolor="royalblue",
            elinewidth=1.2, capsize=4, markersize=5,
            markeredgecolor="darkblue", markeredgewidth=1.2)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Distance z (mm)")
ax.set_ylabel("Residual B (mT)")
ax.set_title("B-field fit residuals")
plt.tight_layout()
fig.savefig("Plots/Plots_bean/B_field_residuals.pdf", bbox_inches="tight")
plt.show()

# --- 10. m vs B plot ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(B_dec * 1e3, m_dec, xerr=B_err_dec * 1e3, yerr=m_err_dec,
            fmt='s', color="darkorange", ecolor="orange",
            elinewidth=1.2, capsize=3, markersize=5,
            markeredgecolor="darkred", markeredgewidth=1.2, zorder=3,
            label=r"Decreasing $z$ (approaching)")
ax.errorbar(B_inc * 1e3, m_inc, xerr=B_err_inc * 1e3, yerr=m_err_inc,
            fmt='^', color="royalblue", ecolor="royalblue",
            elinewidth=1.2, capsize=3, markersize=5,
            markeredgecolor="darkblue", markeredgewidth=1.2, zorder=3,
            label=r"Increasing $z$ (retreating)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$\mu_0 H_a$ (mT)")
ax.set_ylabel(r"Magnetic moment $m$ (A·m²)")
ax.set_title(r"Bean model: $m(H_a)$")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_bean/m_of_Ha.pdf", bbox_inches="tight")
plt.show()


# --- 11. mu_0*M(H_a) vs B plot---
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(B_dec * 1e3, mu0 * M_dec * 1e3, xerr=B_err_dec * 1e3, yerr=mu0 * M_err_dec * 1e3,
            fmt='s', color="darkorange", ecolor="orange",
            elinewidth=1.2, capsize=3, markersize=5,
            markeredgecolor="darkred", markeredgewidth=1.2, zorder=3,
            label=r"Decreasing $z$ (approaching)")
ax.errorbar(B_inc * 1e3, mu0 * M_inc * 1e3, xerr=B_err_inc * 1e3, yerr=mu0 * M_err_inc * 1e3,
            fmt='^', color="royalblue", ecolor="royalblue",
            elinewidth=1.2, capsize=3, markersize=5,
            markeredgecolor="darkblue", markeredgewidth=1.2, zorder=3,
            label=r"Increasing $z$ (retreating)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$\mu_0 H_a$ (mT)")
ax.set_ylabel(r"$\mu_0 M$ (mT)")
ax.set_title(r"Bean model: M(H_a)")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_bean/mu0_M_of_Ha.pdf", dpi=150, bbox_inches="tight")
plt.show()

