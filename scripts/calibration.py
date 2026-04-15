import numpy as np
import scipy.optimize as scopt
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt

import pandas as pd
import os

##########################################################################################################################
########################################## Data Extraction from Excel ####################################################
##########################################################################################################################
file_path_day_2 = "data/Day_2.xlsx"                                      # Original day 2 slow path
cache_path_2 = "data/Day_2_cache.parquet"                                # Day 2 Fast path

file_path_day_3 = "data/Day_3_dry_ice.xlsx"                              # Original day 3 slow path
cache_path_3 = "data/Day_3_dry_ice_cache.parquet"                        # Day 3 Fast path

file_path_day_4 = "data/Day_4_full.xlsx"                                 # Original day 4 slow path
cache_path_4 = "data/Day_4_full_cache.parquet"                           # Day 4 Fast path

print()

# Day 2 data
if os.path.exists(cache_path_2):
    df_day_2 = pd.read_parquet(cache_path_2)
    print(f"Loaded from {cache_path_2}")
else:
    df_day_2 = pd.read_excel(
        file_path_day_2,
        usecols="M:P",
        skiprows=0,
        nrows=58476,
        decimal=","
    )
    df_day_2.to_parquet(cache_path_2, index=False)
    print("Day 2 data read from Excel and saved cache.")

# Day 3 data
if os.path.exists(cache_path_3):
    df_day_3 = pd.read_parquet(cache_path_3)
    print(f"Loaded from {cache_path_3}")
else:
    df_day_3 = pd.read_excel(
        file_path_day_3,
        usecols="A:C",
        skiprows=0,
        nrows=32868,
        decimal=","
    )
    df_day_3.to_parquet(cache_path_3, index=False)
    print("Day 3 data read from Excel and saved cache.")

# Day 4 data
if os.path.exists(cache_path_4):
    df_day_4 = pd.read_parquet(cache_path_4)
    print(f"Loaded from {cache_path_3}")
else:
    df_day_4 = pd.read_excel(
        file_path_day_4,
        usecols="D:F",
        skiprows=0,
        nrows=70660,
        decimal=","
    )
    df_day_4.to_parquet(cache_path_4, index=False)
    print("Day 4 data read from Excel and saved cache.")

room_temp_day_2 = 21.2
t_day_2      = df_day_2.iloc[:, 0].to_numpy().flatten()
U_a_day_2    = df_day_2.iloc[:, 1].to_numpy().flatten()     # warming starts at 14805
U_b_day_2    = df_day_2.iloc[:, 2].to_numpy().flatten()

t_day_3      = df_day_3.iloc[:, 0].to_numpy().flatten()
U_a_day_3    = df_day_3.iloc[:, 1].to_numpy().flatten()
U_b_day_3    = df_day_3.iloc[:, 2].to_numpy().flatten()

room_temp_day_4 = 19.2
t_day_4      = df_day_4.iloc[:, 0].to_numpy().flatten()
U_a_day_4    = df_day_4.iloc[:, 1].to_numpy().flatten()     # warming starts at 19233
U_b_day_4    = df_day_4.iloc[:, 2].to_numpy().flatten()

######################################################################################################################
####################################### End of Data Extraction #######################################################
######################################################################################################################


# Voltage vs Time plot for dry ice
fig, ax = plt.subplots()
ax.scatter(t_day_3/1000, U_a_day_3, label="$U_a$, Temperature voltage", s=3)
ax.scatter(t_day_3/1000, U_b_day_3, label="$U_b$, Superconductor voltage", s=3)
ax.set_title("Voltage vs Time (Dry ice)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_calibration/voltage_vs_time_dry_ice.pdf", bbox_inches="tight")
plt.show()

# Voltage vs Time plot for liquid nitrogen (day 2)
fig, ax = plt.subplots()
ax.scatter(t_day_2, U_a_day_2, label="$U_a$, Temperature voltage", s=3)
ax.scatter(t_day_2, U_b_day_2, label="$U_b$, Superconductor voltage", s=3)
ax.set_title("Voltage vs Time (Liquid Nitrogen)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_calibration/voltage_vs_time_LN2.pdf", bbox_inches="tight")
plt.show()

print()
# Given values
nitrogen_U_a = np.max(U_a_day_2)
print(f"Liquid nitrogen boiling point voltage: {nitrogen_U_a}")
nitrogen_temp_C = -195.8
nitrogen_temp_K = 77.34

dry_ice_U_a = np.max(U_a_day_3)
print(f"Dry ice sublimation point voltage: {dry_ice_U_a}")
dry_ice_temp_C = -78.5
dry_ice_temp_K = 194.7

room_temp_U_a = -0.0209
room_temp_C = 21.2
room_temp_K = room_temp_C + 273.15

# Three data points
U = np.array([nitrogen_U_a, dry_ice_U_a, room_temp_U_a])
T = np.array([nitrogen_temp_K, dry_ice_temp_K, room_temp_K])

# Define linear model for ODR
def linear(params, x):
    k, c = params
    return k * x + c

# Uncertainties
U_err = np.array([0.0001, 0.0001, 0.0001])
T_err = np.array([0.16, 0.0001, 0.0001])

# Set up ODR
linear_model = Model(linear)
data = RealData(U, T, sx=U_err, sy=T_err)
odr = ODR(data, linear_model, beta0=[np.polyfit(U, T, 1)[0], np.polyfit(U, T, 1)[1]])

output = odr.run()
k, c = output.beta
k_err, c_err = output.sd_beta

print()
print("Linear fit calibration parameters from ODR: ")
print(f"k = {k:.4f} ± {k_err:.4f} K/V")
print(f"c = {c:.4f} ± {c_err:.4f} K")

# ODR calibration plot
U_plot = np.linspace(min(U), max(U), 100)
T_plot = k * U_plot + c

fig, ax = plt.subplots()
ax.errorbar(U, T, yerr=T_err, xerr=U_err, fmt='o', capsize=3, alpha=0.5, label='Data points')
ax.plot(U_plot, T_plot, label='ODR fit', color='orange')
ax.set_xlabel("Voltage $U_a$ (V)")
ax.set_ylabel("Temperature T (K)")
ax.set_title("ODR Linear Calibration: T = k$U_a$ + c")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_calibration/ODR_calibration.pdf", bbox_inches="tight")
plt.show()

# Full covariance matrix from ODR
cov = output.cov_beta
var_k = cov[0, 0]
var_c = cov[1, 1]
cov_kc = cov[0, 1]
rho = cov_kc / (np.sqrt(var_k) * np.sqrt(var_c))
print(f"Correlation between k and c: {rho:.4f}")

def T_uncertainty(U_val, sigma_U=0.0001):
    return np.sqrt(
        U_val**2 * var_k +
        var_c +
        2 * U_val * cov_kc +
        k**2 * sigma_U**2
    )

index = 0
sigma_T_no_cov   = np.sqrt(U_a_day_2[index]**2 * var_k + var_c + k**2 * 0.0001**2)
sigma_T_with_cov = np.sqrt(U_a_day_2[index]**2 * var_k + var_c + 2 * U_a_day_2[index] * cov_kc + k**2 * 0.0001**2)

print()
print("Seeing impact of covariance on the uncertainty of the first voltage value:")
print(f"Without cov: {U_a_day_2[index]*k+c} ± {sigma_T_no_cov:.4f} K")
print(f"With cov: {U_a_day_2[index]*k+c} ± {sigma_T_with_cov:.4f} K")

# Finding Tc
threshold = 0.0001
target = -0.0009

indices = np.where(np.abs(U_b_day_2 - target) <= threshold)[0]
print()
print("T_c is defined as the point at which voltage crosses -0.0009 V.")
if len(indices) > 0:
    T_c_index_cooling = indices[0]
    T_c_cooling = (U_a_day_2[T_c_index_cooling] * k) + c
    sigma_Tc = T_uncertainty(U_a_day_2[T_c_index_cooling])
    print(f"T_c = {T_c_cooling:.3f} ± {sigma_Tc:.3f} K (cooling)")

    T_c_index_warming = indices[-1]
    T_c_warming = (U_a_day_2[T_c_index_warming] * k) + c
    sigma_Tc = T_uncertainty(U_a_day_2[T_c_index_warming])
    print(f"T_c = {T_c_warming:.3f} ± {sigma_Tc:.3f} K (warming)")
else:
    print("No point found within threshold of target voltage")

# Ub Voltage vs Temperature plot
split = 14805
temp_day_2_cooling     = (U_a_day_2 * k + c)[:split]
U_b_day_2_cooling      = U_b_day_2[:split]
temp_err_day_2_cooling = T_uncertainty(U_a_day_2[:split])

temp_day_2_warming     = (U_a_day_2 * k + c)[split:]
U_b_day_2_warming      = U_b_day_2[split:]
temp_err_day_2_warming = T_uncertainty(U_a_day_2[split:])

fig, ax = plt.subplots()
ax.scatter((U_a_day_2 * k + c)[:split], U_b_day_2[:split],
           label="$U_b$ (cooling)", marker="o", s=0.5, color='blue')
ax.scatter((U_a_day_2 * k + c)[split:], U_b_day_2[split:],
           label="$U_b$ (warming)", marker="o", s=0.5, color='orange')

ax.annotate("Cooling",
    xy=(0.3, 0.7), xytext=(0.45, 0.8),
    xycoords='axes fraction', textcoords='axes fraction',
    arrowprops=dict(arrowstyle="->", color='blue', lw=2),
    fontsize=10, color='blue')

ax.annotate("Warming",
    xy=(0.55, 0.7), xytext=(0.32, 0.59),
    xycoords='axes fraction', textcoords='axes fraction',
    arrowprops=dict(arrowstyle="->", color='orange', lw=2),
    fontsize=10, color='orange')

ax.set_title("$U_b$ vs Temperature")
ax.set_xlabel("Temperature T (K)")
ax.set_ylabel("Voltage $U_b$ (V)")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_calibration/Ub_vs_temperature.pdf", bbox_inches="tight")
plt.show()


def rho_uncertainty(U_b):
    w = 3.0e-3;    sigma_w = 0.1e-3
    h = 1.9e-3;    sigma_h = 0.3e-3
    L = 9.6e-3;    sigma_L = 0.3e-3
    I = 140e-3
    sigma_V = 0.0001

    A = w * h
    sigma_A = A * np.sqrt((sigma_w/w)**2 + (sigma_h/h)**2)

    rho = np.where(
        U_b > 0.001,
        (U_b * A) / (L * I),
        np.nan
    )

    sigma_rho = np.where(
        np.abs(U_b) > 0.001,
        np.abs(rho) * np.sqrt((sigma_V/U_b)**2 + (sigma_A/A)**2 + (sigma_L/L)**2),
        np.nan
    )

    return rho, sigma_rho

with np.errstate(divide='ignore', invalid='ignore'):
    rho_day_2, sigma_rho_day_2 = rho_uncertainty(U_b_day_2)

sigma_V = 0.0001
w, sigma_w = 3.0e-3, 0.1e-3
h, sigma_h = 1.9e-3, 0.3e-3
L, sigma_L = 9.6e-3, 0.3e-3
A = w * h

U_b_typical = 0.16
print()
print(f"Seeing contribution of errors on U_b = {U_b_typical}")
print(f"sigma_V/U_b  = {sigma_V/abs(U_b_typical):.4f}")
print(f"sigma_A/A    = {np.sqrt((sigma_w/w)**2 + (sigma_h/h)**2):.4f}")
print(f"sigma_L/L    = {sigma_L/L:.4f}")
print()

rho_day_2_cooling      = rho_day_2[:split]
sigma_rho_day_2_cooling = sigma_rho_day_2[:split]

rho_day_2_warming      = rho_day_2[split:]
sigma_rho_day_2_warming = sigma_rho_day_2[split:]

# Resistivity vs Temperature (cooling only)
skip = 50

fig, ax = plt.subplots()
plot1 = ax.errorbar(temp_day_2_cooling[::skip], rho_day_2_cooling[::skip],
                    yerr=sigma_rho_day_2_cooling[::skip], xerr=temp_err_day_2_cooling[::skip],
                    fmt='o', markersize=3, capsize=2, alpha=1,
                    label="$U_b$ (cooling)", color='blue')
plot1[0].set_alpha(0.3)
ax.set_title("Resistivity vs Temperature")
ax.set_xlabel("Temperature T (K)")
ax.set_ylabel("Resistivity (Ωm)")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_calibration/resistivity_cooling_only.pdf", bbox_inches="tight")
plt.show()

# Resistivity vs Temperature (cooling + warming)
skip = 120
end_index = 4000

fig, ax = plt.subplots()
plot1 = ax.errorbar(temp_day_2_cooling[::skip], rho_day_2_cooling[::skip],
                    yerr=sigma_rho_day_2_cooling[::skip], xerr=temp_err_day_2_cooling[::skip],
                    fmt='o', markersize=3, capsize=2, alpha=1,
                    label="$U_b$ (cooling)", color='blue')
plot2 = ax.errorbar(temp_day_2_warming[:end_index:skip], rho_day_2_warming[:end_index:skip],
                    yerr=sigma_rho_day_2_warming[:end_index:skip], xerr=temp_err_day_2_warming[:end_index:skip],
                    fmt='o', markersize=3, capsize=2, alpha=1,
                    label="$U_b$ (warming)", color='orange')
plot1[0].set_alpha(0.3)
ax.set_title("Resistivity vs Temperature")
ax.set_xlabel("Temperature T (K)")
ax.set_ylabel("Resistivity (Ωm)")
ax.legend()
plt.tight_layout()
fig.savefig("Plots/Plots_calibration/resistivity_cooling_warming.pdf", bbox_inches="tight")
plt.show()