import matplotlib.pyplot as plt
import numpy as np
import os
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from src.custom_colormap import custom_plasma

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

final = []
valid_indices = []

threshold = 1e-2  # Set your concentration threshold here
x_at_threshold_pd = []

# Loop over R values
for i, R in enumerate(formatted_R_custom):
    file_path = os.path.join(base_path, f"prod_drain/case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_profile_y_-1.00.csv")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=0)
        print(f"Loaded R={R}:", data.shape)

        x_vals = data[:, 0]
        c_vals = data[:, 1]

        # Find first x where concentration exceeds threshold
        above_thresh = np.where(c_vals > threshold)[0]
        if len(above_thresh) > 0:
            x_thresh = x_vals[above_thresh[0]]
        else:
            x_thresh = np.nan  # or set to 1.0 if you want to cap

        x_at_threshold_pd.append(x_thresh)

    except Exception as e:
        print(f"File not found or unreadable at {file_path}: {e}")
        x_at_threshold_pd.append(np.nan)

final = []
valid_indices = []

threshold = 1e-2  # Set your concentration threshold here
x_at_threshold = []

# Loop over R values
for i, R in enumerate(formatted_R):
    x_vals=[]
    file_path = os.path.join(base_path, f"case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_profile_y_-1.00.csv")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        print(f"Loaded R={R}:", data.shape)
        print(data)
        x_vals = data[:, 0]
        c_vals = data[:, 1]

        # Find first x where concentration exceeds threshold
        above_thresh = np.where(c_vals > threshold)[0]
        
        if len(above_thresh) > 0:
            x_thresh = x_vals[above_thresh[0]]
        else:
            x_thresh = np.nan  # or set to 1.0 if you want to cap

        x_at_threshold.append(x_thresh)

    except Exception as e:
        print(f"File not found or unreadable at {file_path}: {e}")
        x_at_threshold.append(np.nan)

# === Set figure parameters ===
from src.figure_format import get_font_parameters, set_matplotlib_defaults
plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

# === Dummy data loading ===
# (Assume you've already created these variables in your script)
# R_vals, x_at_threshold, x_at_threshold_pd
# AG_array_fb, AG_array_fb_pd

# === Plot Setup ===
fig = plt.figure(figsize=(7, 3.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

# === Subplot (a) – X-location ===
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(R_vals, x_at_threshold, label='No PD', color='blue')
ax1.plot(R_vals, x_at_threshold_pd, label='With PD', color='blue', linestyle='--')
ax1.set_xlabel('$Pe_s$')
ax1.set_ylabel(f'$x$ ($\eta=-1$) where $c_0 > {threshold}$')
ax1.set_xscale('log')
ax1.grid(True)
#ax1.legend(fontsize=8)
ax1.text(-0.15, 1.05, "(a)", transform=ax1.transAxes,
         fontsize=12, fontweight='bold', va='top')

# === Subplot (b) – AG Drainage % ===
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(R_vals[:len(AG_array_fb)], AG_array_fb * 100, label='No PD', color = 'blue')
ax2.plot(R_vals[:len(AG_array_fb_pd)], AG_array_fb_pd * 100, label='With PD', color = 'blue', linestyle= '--')
ax2.set_xscale('log')
ax2.set_xlabel('$Pe_s$')
ax2.set_ylabel('AG Drainage ($\%$)')
ax2.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
ax2.set_xticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$'])
ax2.grid(True)
#ax2.legend(fontsize=8)
ax2.text(-0.15, 1.05, "(b)", transform=ax2.transAxes,
         fontsize=12, fontweight='bold', va='top')

# === Final Layout and Save ===
plt.tight_layout()
os.makedirs("outputs/concentration", exist_ok=True)
plt.savefig("outputs/concentration/spine_combined.pdf", dpi=300)
plt.show()
