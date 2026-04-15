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
        usecols="E:V",
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

# --- 1. Correct disk magnet B-field model (returns T) ---
def disk_magnet_field(z_m, Br, R_m, L_m, z0_m):
    z_rel = (z_m - z0_m)              # m
    z_rel = np.where(z_rel < 1e-4, 1e-4, z_rel)
    Bz = (Br / 2.0) * (
        (z_rel + L_m) / np.sqrt((z_rel + L_m)**2 + R_m**2)
        - z_rel       / np.sqrt(z_rel**2          + R_m**2)
    )
    return Bz    # T

# dB/dz function (returns T/m)
def grad_disk_magnet_field(z_m, Br, R_m, L_m, z0_m):
    z_rel = z_m - z0_m                # m
    z_rel = np.where(z_rel < 1e-4, 1e-4, z_rel)

    dBz_dz = (Br * R_m**2 / 2.0) * (
        1.0 / ((z_rel + L_m)**2 + R_m**2)**1.5
        - 1.0 / (z_rel**2          + R_m**2)**1.5
    )
    
    return dBz_dz     # T/m 


# --- 2. Fit ---            
mask_f = np.isfinite(z_field) & np.isfinite(B_field)
z_f    = z_field[mask_f] * 0.01     # cm → m
B_f    = B_field[mask_f] * 1e-3    # mT → T
sidx   = np.argsort(z_f)
z_f, B_f = z_f[sidx], B_f[sidx]

p0     = [1.2,  0.015, 0.005, 0.0]
bounds = ([0.1, 1e-3, 5e-4, -0.005],
          [2.0, 0.1,  0.05,  0.005])

popt, _ = curve_fit(disk_magnet_field, z_f, B_f, p0=p0, bounds=bounds, maxfev=40000)   # z_f (mm), B_f (T) 
print(f"Fit → Br={popt[0]:.3f} T  R={popt[1]*1e3:.1f} mm  "
      f"L={popt[2]*1e3:.1f} mm  z0={popt[3]*1000:.2f} mm")

# --- 3. Fine grid  ---         DOUBLE CHECK THE UNITS OF DERIVATIVE ARE RIGHT IN NEXT CALCULATIONS
z_fine = np.linspace(z_f.min(), z_f.max(), 5000)           # m
B_fine = disk_magnet_field(z_fine, *popt)                  # T

def B_at(z_m):
    return disk_magnet_field(z_m, *popt)                  # T

def dBdz_at(z_m):                                         # ACTUAL DERIVATIVE FUNCTION
    return grad_disk_magnet_field(z_m, *popt)      # T/m

# --- 4. mass → force → moment ---
def compute_m_of_Ha(z_m, mass_g, label=""):
    mask   = np.isfinite(z_m) & np.isfinite(mass_g)
    z_m   = z_m[mask]
    mass_g = mass_g[mask]

    # Baseline from 5 points at largest z
    idx_far    = np.argsort(z_m)[-5:]
    baseline_g = np.median(mass_g[idx_far])
    print(f"  {label}: baseline={baseline_g:.5f} g  (z >= {z_m[idx_far].min():.1f} mm)")

    F_N      = (mass_g ) * 1e-3 * g           # Newtons
    B_vals   = B_at(z_m)                                  # T (x-axis)
    dB_vals  = dBdz_at(z_m)                               # T/m

    # Filter points where derivative is too small (far field)
    safe     = np.abs(dB_vals) > 0.001 * np.max(np.abs(dB_vals))
    m_Am2    = np.full_like(F_N, np.nan)
    m_Am2[safe] = F_N[safe] / dB_vals[safe]                    # m = F / dBdz
    # m_Am2 = F_N / dB_vals                                    # m = F / dBdz


    return B_vals, m_Am2

B_dec, m_dec = compute_m_of_Ha(z_decreasing_1 * 1e-3,      m_decreasing_1,      "Decreasing z")
B_inc, m_inc = compute_m_of_Ha(z_increasing_1 * 1e-3,      m_increasing_1,      "Increasing z")

# --- 5. Field fit quality ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(z_f, B_f, color="orange", s=30, zorder=3, label="Measured B (T)")
ax.plot(z_fine, B_fine, color="red", lw=2, label="Disk magnet fit")
ax.set_xlabel("Distance z (m)"); ax.set_ylabel("B (T)")
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
ax.set_xlabel(r"Applied field $\mu_0 H_a$ (T)")
ax.set_ylabel(r"Magnetic moment $m$ (A·m²)")
ax.set_title(r"Bean model: $m(H_a)$")
ax.legend(); plt.tight_layout(); plt.show()