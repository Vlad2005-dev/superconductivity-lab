import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt

import pandas as pd
import os

##########################################################################################################################
########################################## Data Extraction from Excel ####################################################
##########################################################################################################################
file_path_bean_model = "data/bean_model_data.xlsx"                                      # Original day 2 slow path
cache_path_bean = "data/cache_bean.parquet"                                # Day 2 Fast path

# Day 2 data
if os.path.exists(cache_path_bean):
    # Fast path: load from cache
    df_bean = pd.read_parquet(cache_path_bean)
    print()
    print(f"Loaded from {cache_path_bean}")
else:
    # Slow path: read Excel and save cache
    df_bean = pd.read_excel(
        file_path_bean_model,
        usecols="E:W",
        skiprows=1,                     # 1 higher than the needed row
        nrows=24,                    # n of data rows, not counting the title row
        decimal=","
    )
    df_bean.to_parquet(cache_path_bean, index=False)
    print("Day bean model data read from Excel and saved cache.")



z_decreasing_0      = df_bean.iloc[:, 0].to_numpy().flatten() + 4.3             # mm
m_decreasing_0      = df_bean.iloc[:, 1].to_numpy().flatten()                   # g 

z_increasing_0      = df_bean.iloc[:, 3].to_numpy().flatten() + 4.3             # mm
m_increasing_0      = df_bean.iloc[:, 4].to_numpy().flatten()                   # mm
m_increasing_corrected_0      = df_bean.iloc[:, 5].to_numpy().flatten()         # g


z_decreasing_1      = df_bean.iloc[:, 8].to_numpy().flatten() + 4.3             # mm
m_decreasing_1      = df_bean.iloc[:, 10].to_numpy().flatten()                  # g

z_increasing_1      = df_bean.iloc[:, 12].to_numpy().flatten() + 4.3            # mm
m_increasing_1      = df_bean.iloc[:, 14].to_numpy().flatten()                  # g  


z_field      = df_bean.iloc[:, 16].to_numpy().flatten()                         # cm
B_field      = df_bean.iloc[:, 17].to_numpy().flatten()                         # mT
B_field_err  = df_bean.iloc[:, 18].to_numpy().flatten()                         # mT

######################################################################################################################
####################################### End of Data Extraction #######################################################
######################################################################################################################




# Mass vs Distance plot
plt.scatter(z_decreasing_1, m_decreasing_1, label="Decreasing distance", color="orange", 
            marker="s", s=40, edgecolors="darkorange", linewidths=1.2)        # squares

plt.scatter(z_increasing_1[:-4], m_increasing_1[:-4], label="Increasing distance", 
            color="blue", marker="^", s=40, edgecolors="darkblue", linewidths=1.2)   # triangles

plt.title("Mass vs Distance")
plt.xlabel("Distance (mm)")
plt.ylabel("Mass (g)")
plt.legend()
plt.show()






##########################################################################################################################
########################################## Bean Model: m(H_a) Plot - FIXED ###############################################
##########################################################################################################################

from scipy.optimize import curve_fit

mu0 = 4 * np.pi * 1e-7
g   = 9.81

R_magnet  = 0.0075   # m, radius of permanent magnet
R_magnet_err = 0.025 * 1e-3   #m
L_magnet  = 0.003    # m  thickness of permanent magnet
L_magnet_err = 0.025 * 1e-3   #m

b_mT = 0.3           # mT, background B-field
b_T = b_mT * 1e-3    # T,  background B-field
b_T_err = 0.001 * 1e-3 #T

R_superconductor = 11.3 * 1e-3      # m, radius of superconductor
L_superconductor = 3.5 * 1e-3       # m, thickness of superconductor
volume_superconductor = (R_superconductor**2) * np.pi * L_magnet        # m^3, volume of superconductor


# --- 1. Correct disk magnet B-field model (returns mT) and dB/dz function (returns T/m) ---
def disk_magnet_field(z_mm, Br, z0_mm):
    z_m = (z_mm - z0_mm) * 1e-3             # mm → m
    z_m = np.where(z_m < 1e-4, 1e-4, z_m)
    Bz = (Br / 2.0) * (
        (z_m + L_magnet) / np.sqrt((z_m + L_magnet)**2 + R_magnet**2)
        - z_m       / np.sqrt(z_m**2          + R_magnet**2)
    ) + b_T
    return Bz * 1e3    # T → mT

def grad_disk_magnet_field(z_mm, Br, z0_mm):
    z_m = (z_mm - z0_mm) * 1e-3             # mm → m
    z_m = np.where(z_m < 1e-4, 1e-4, z_m)
    
    dBz_dz = (Br * R_magnet**2 / 2.0) * (
        1.0 / ((z_m + L_magnet)**2 + R_magnet**2)**1.5
        - 1.0 / (z_m**2          + R_magnet**2)**1.5
    )
    
    return dBz_dz     # T/m 


# --- 2. Fit ---            
mask_f = np.isfinite(z_field) & np.isfinite(B_field)
z_f    = z_field[mask_f] * 10.0    # cm → mm
B_f    = B_field[mask_f]           # mT
B_f_err = B_field_err[mask_f]      # mT
sidx   = np.argsort(z_f)
z_f, B_f = z_f[sidx], B_f[sidx]

p0     = [1.2,  0.0]
bounds = ([0.5, -5],
          [1.5,  5])

popt, pcov = curve_fit(disk_magnet_field, z_f, B_f,
                       sigma=B_f_err, absolute_sigma=True,
                       p0=p0, bounds=bounds, maxfev=40000)


print(f"Fit → Br={popt[0]:.3f}±{pcov[0][0]:.9f} T  (fixed)R={R_magnet*1e3:.1f}±{R_magnet_err*1e3} mm  "
      f"(fixed)L={L_magnet*1e3:.1f}±{L_magnet_err*1e3} mm  z0={popt[1]:.2f}±{pcov[1][1]:.9f} mm  (fixed)b_mT={b_mT}±{b_T_err*1e3} mT ")

# --- 3. Fine grid  ---         DOUBLE CHECK THE UNITS OF DERIVATIVE ARE RIGHT IN NEXT CALCULATIONS
z_fine = np.linspace(z_f.min(), z_f.max(), 5000)           # mm
B_fine = disk_magnet_field(z_fine, *popt)                  # mT



def B_at(z_mm):
    return disk_magnet_field(z_mm, *popt)                  # mT

def dBdz_at(z_mm):                                         # ACTUAL DERIVATIVE FUNCTION
    return grad_disk_magnet_field(z_mm, *popt)             # T/m



def dB_dBr(z_mm, Br, z0_mm):
    """∂B/∂Br — just B with Br=1"""
    z_m = (z_mm - z0_mm) * 1e-3
    z_m = np.where(z_m < 1e-4, 1e-4, z_m)
    return 0.5 * (
        (z_m + L_magnet) / np.sqrt((z_m + L_magnet)**2 + R_magnet**2)
        - z_m            / np.sqrt(z_m**2              + R_magnet**2)
    ) * 1e3   # mT

def dB_dz0(z_mm, Br, z0_mm):
    """∂B/∂z0 = -∂B/∂z (shifting z0 is same as shifting z in opposite direction)"""
    return -grad_disk_magnet_field(z_mm, Br, z0_mm)   # T/m 

def dB_dR(z_mm, Br, z0_mm):
    """∂B/∂R_magnet"""
    z_m = (z_mm - z0_mm) * 1e-3
    z_m = np.where(z_m < 1e-4, 1e-4, z_m)
    return (Br * R_magnet / 2.0) * (
        -( z_m + L_magnet) / ((z_m + L_magnet)**2 + R_magnet**2)**1.5
        + z_m              / (z_m**2              + R_magnet**2)**1.5
    ) * 1e3   # mT/m  (sigma_R is in metres so units match)

def dB_dL(z_mm, Br, z0_mm):
    """∂B/∂L_magnet"""
    z_m = (z_mm - z0_mm) * 1e-3
    z_m = np.where(z_m < 1e-4, 1e-4, z_m)
    return (Br / 2.0) * (
        1.0 / np.sqrt((z_m + L_magnet)**2 + R_magnet**2)
        - (z_m + L_magnet)**2 / ((z_m + L_magnet)**2 + R_magnet**2)**1.5
    ) * 1e3   # mT/m

def dB_db(z_mm):
    """∂B/∂b = 1 everywhere (b is additive), converting T → mT"""
    return np.ones_like(z_mm) * 1e3  # mT/T

def B_uncertainty(z_mm, popt, pcov, z_err_mm=2.0):
    Br = popt[0]
    z0 = popt[1]

    var_B = (
        (dB_dBr(z_mm, Br, z0) * np.sqrt(pcov[0,0]))**2 +
        (dB_dz0(z_mm, Br, z0) * np.sqrt(pcov[1,1]))**2 +
        (dB_dR( z_mm, Br, z0) * R_magnet_err        )**2 +
        (dB_dL( z_mm, Br, z0) * L_magnet_err        )**2 +
        (dB_db( z_mm)         * b_T_err             )**2 +
        (grad_disk_magnet_field(z_mm, Br, z0) * 1e-3 * 1e3 * z_err_mm)**2 +
        2 * dB_dBr(z_mm, Br, z0) * dB_dz0(z_mm, Br, z0) * pcov[0,1]
    )
    return np.sqrt(var_B)

# --- 4. mass → force → moment ---
def compute_m_of_Ha(z_mm, mass_g, popt, pcov, label="", z_err_mm=2.0) :
    mask   = np.isfinite(z_mm) & np.isfinite(mass_g)
    z_mm   = z_mm[mask]
    mass_g = mass_g[mask]

    F_N      = (mass_g) * 1e-3 * g           # Newtons
    B_vals   = B_at(z_mm)                                  # mT (x-axis)
    B_err    = B_uncertainty(z_mm, popt, pcov, z_err_mm)       # mT  ← new
    dB_vals  = dBdz_at(z_mm)                               # T/m

    # Filter points where derivative is too small (far field)
    safe     = np.abs(dB_vals) > 0.01 * np.max(np.abs(dB_vals))
    m_Am2    = np.full_like(F_N, np.nan)
    m_Am2[safe] = F_N[safe] / dB_vals[safe]                # m = F / dBdz

    return B_vals, B_err, m_Am2

B_dec, B_err_dec, m_dec = compute_m_of_Ha(z_decreasing_1,      m_decreasing_1, popt, pcov,    "Decreasing z")
B_inc, B_err_inc, m_inc = compute_m_of_Ha(z_increasing_1,      m_increasing_1, popt, pcov,    "Increasing z")

M_dec = m_dec / volume_superconductor
M_inc = m_inc / volume_superconductor





# --- 5. B-Field fit quality ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(
    z_f, B_f,
    xerr = 2, yerr=B_f_err,          
    fmt='o',                # circle markers, no connecting line
    color="orange",
    ecolor="orange",        # error bar colour
    elinewidth=1.2,
    capsize=4,
    markersize=5,
    zorder=3,
    label="Measured B (mT)"
)
ax.plot(z_fine, B_fine, color="red", lw=2, label="Disk magnet fit")
ax.set_xlabel("Distance z (mm)"); ax.set_ylabel("B (mT)")
ax.set_title(r"$B(z)$ – disk magnet fit"); ax.legend()
plt.tight_layout(); plt.show()

# --- 6. m(H_a) ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(B_dec, m_dec, color="orange", marker="s", s=50,
           edgecolors="darkorange", linewidths=1.2, zorder=3,
           label=r"Decreasing $z$ (approaching)")
ax.scatter(B_inc, m_inc, color="royalblue", marker="^", s=50,
           edgecolors="darkblue", linewidths=1.2, zorder=3,
           label=r"Increasing $z$ (retreating)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$\mu_0 H_a$ (mT)")
ax.set_ylabel(r"Magnetic moment $m$ (A·m²)")
ax.set_title(r"Bean model: $m(H_a)$")
ax.legend(); plt.tight_layout(); plt.show()

# --- 7. M(H_a) ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(B_dec, M_dec, color="orange", marker="s", s=50,
           edgecolors="darkorange", linewidths=1.2, zorder=3,
           label=r"Decreasing $z$ (approaching)")
ax.scatter(B_inc, M_inc, color="royalblue", marker="^", s=50,
           edgecolors="darkblue", linewidths=1.2, zorder=3,
           label=r"Increasing $z$ (retreating)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$\mu_0 H_a$ (mT)")
ax.set_ylabel(r"M (A/m)")
ax.set_title(r"Bean model: $M(H_a)$")
ax.legend(); plt.tight_layout(); plt.show()

# --- 8. mu_0 * M(H_a) ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(B_dec, mu0 * M_dec * 1e3, xerr=B_err_dec, fmt='s', color="orange",
            ecolor="darkorange", elinewidth=1.2, capsize=3, markersize=5,
            markeredgecolor="darkorange", markeredgewidth=1.2, zorder=3,
            label=r"Decreasing $z$ (approaching)")
ax.errorbar(B_inc, mu0 * M_inc * 1e3, xerr=B_err_inc, fmt='^', color="royalblue",
            ecolor="darkblue", elinewidth=1.2, capsize=3, markersize=5,
            markeredgecolor="darkblue", markeredgewidth=1.2, zorder=3,
            label=r"Increasing $z$ (retreating)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$\mu_0 H_a$ (mT)")
ax.set_ylabel(r"$\mu_0$M (mT)")
ax.set_title(r"Bean model: $M(H_a)$")
ax.legend(); plt.tight_layout(); plt.show()


