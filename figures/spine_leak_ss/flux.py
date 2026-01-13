import matplotlib.pyplot as plt
import numpy as np
import os
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from src.custom_colormap import custom_plasma
from scipy.optimize import root_scalar
from matplotlib.gridspec import GridSpec

plt.rcParams.update(get_font_parameters())

set_matplotlib_defaults()

formatted_R = [0.001, 0.002, 0.003, 0.005, 0.007,
                0.011, 0.016, 0.024, 0.036, 0.053,
                0.079, 0.117, 0.174, 0.259, 0.386,
                0.574, 0.853, 1.269, 1.887, 2.807,
                4.175, 6.21, 9.237, 13.738, 20.434,
                30.392, 45.204, 67.234, 100.0]

R_vals =[0.001, 0.002, 0.003, 0.005, 0.007,
                0.011, 0.016, 0.024, 0.036, 0.053,
                0.079, 0.117, 0.174, 0.259, 0.386,
                0.574, 0.853, 1.269, 1.887, 2.807,
                4.175, 6.21, 9.237, 13.738, 20.434,
                30.392, 45.204, 67.234, 100.0]

formatted_R_custom = [
    "1.0em03", "2.0em03", "3.0em03", "5.0em03", "7.0em03",
    "1.1em02", "1.6em02", "2.4em02", "3.6em02", "5.3em02",
    "7.9em02", "1.2em01", "1.7em01", "2.6em01", "3.9em01",
    "5.7em01", "8.5em01", "1.3ep00", "1.9ep00", "2.8ep00",
    "4.2ep00", "6.2ep00", "9.2ep00", "1.4ep01", "2.0ep01",
    "3.0ep01", "4.5ep01", "6.7ep01", "1.0ep02"
]

final = []
valid_indices = []
base_path = '/home/s2611897/Desktop/FEniCS/Dispersion-2D3D/output/casestudies/spine_leak/steady_state/'

# Final outputs
final_array_fb = np.array(final)

final_AG_pct = []  # AG drainage percentage
final_LHS_flux = []  # LHS flux

# Loop over R values
for i, R in enumerate(formatted_R):
    ag_values = []
    lhs_values = []

    file_path = os.path.join(base_path, f"case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_fluxes.csv")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=1)  # load all 6 values

        if np.isscalar(data):
            data = np.array([data])

        ag_pct = data[1,] / (2*data[4]) if data[4,] != 0 else np.nan
        ag_values.append(abs(ag_pct))

        lhs_flux = data[2]
        lhs_values.append(lhs_flux)

    except Exception as e:
        print(f"File error at {file_path}: {e}")
        ag_values.append(np.nan)
        lhs_values.append(np.nan)

    final_AG_pct.append(ag_values)
    final_LHS_flux.append(lhs_values)

# Convert to arrays
AG_array_fb = np.array(final_AG_pct)

final_AG_pct = []  # AG drainage percentage
final_LHS_flux = []  # LHS flux

# Loop over R values
for i, R in enumerate(formatted_R_custom):
    ag_values = []
    lhs_values = []

    file_path = os.path.join(
    base_path,
    f"prod_drain/case_R_{R}_dt_0.001_mesh_200_200/"
    f"case_R_{R}_dt_0.001_mesh_200_200_fluxes.csv"
    )
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=1)  # load all 6 values

        if np.isscalar(data):
            data = np.array([data])

        ag_pct = (data[2,]) / (2*data[9]) if data[9,] != 0 else np.nan
        if ag_pct >1:
            ag_pct = 1
            print('correcting error - check pd full brain fluxes')
        ag_values.append(abs(ag_pct))

        lhs_flux = data[2]
        lhs_values.append(lhs_flux)

    except Exception as e:
        print(f"File error at {file_path}: {e}")
        ag_values.append(np.nan)
        lhs_values.append(np.nan)

    final_AG_pct.append(ag_values)
    final_LHS_flux.append(lhs_values)

# Convert to arrays
AG_array_fb_pd = np.array(final_AG_pct) 
print(AG_array_fb_pd)

# === AG Drainage (PD) ===
fig = plt.figure(figsize=(6, 4))  # Change width and height here
ax = fig.add_subplot(1, 1, 1)

ax.plot(R_vals[:len(AG_array_fb)], AG_array_fb * 100)
ax.plot(R_vals[:len(AG_array_fb_pd)], AG_array_fb_pd * 100)

log_ticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
log_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$']

ax.set_xscale('log')
ax.set_xticks(log_ticks)
ax.set_xticklabels(log_labels)
ax.set_xlabel("$Pe_{osc}$")
ax.set_ylabel("AG Drainage (\%)")
ax.grid(True)

os.makedirs("outputs/concentration", exist_ok=True)
plt.savefig("outputs/concentration/spine_flux.pdf", dpi=300)
plt.show()