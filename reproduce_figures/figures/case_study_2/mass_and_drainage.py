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

# formatted_R = [0.001,
#                 0.011, 0.117, 1.269, 13.738, ]

# R_vals =[0.001, 
#                 0.011, 0.117, 1.269, 13.738]

x0_values = [0.6, 0.7, 0.8]
final = []
valid_indices = []
base_path = '/home/s2611897/Desktop/FEniCS/Dispersion-2D3D/output/casestudies/point_leak/steady_state/'

# Loop over R values
for i, R in enumerate(formatted_R):
    values_per_x0 = []
    found = False
    for x0 in x0_values:
        #x_str = f"{x0:.3f}"
        file_path = os.path.join(base_path, f"pe_{R}/x_{x0}/case_R_{R}_dt_0.001_mesh_100_200/case_R_{R}_dt_0.001_mesh_100_200_mass.csv")
        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            print("Loaded:", data, "Shape:", np.shape(data))

            # Promote scalar to array
            if np.isscalar(data):
                data = np.array([data])

            values_per_x0.append(data)
        except Exception as e:
            print(f"File not found or unreadable at {file_path}: {e}")
            values_per_x0.append(np.nan)
        
    final.append(values_per_x0)
# Final outputs
final_array = np.array(final)

final_pd = []
valid_indices = []

# Loop over R values
for i, R in enumerate(formatted_R):
    values_per_x0 = []
    found = False
    for x0 in x0_values:
        #x_str = f"{x0:.3f}"
        # file_path = os.path.join(base_path, f"prod_drain_lower/pe_{R}/xi_gauss_{x0}/case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_mass.csv")
        # try:
        file_path = os.path.join(base_path, f"prod_drain/pe_{R}/xi_gauss_{x0}/case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_mass.csv")
        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            print("Loaded:", data, "Shape:", np.shape(data))

            # Promote scalar to array
            if np.isscalar(data):
                data = np.array([data/2])

            values_per_x0.append(data/2)
        except Exception as e:
            print(f"File not found or unreadable at {file_path}: {e}")
            values_per_x0.append(np.nan)
        
    final_pd.append(values_per_x0)

# Final outputs
final_array_pd = np.array(final_pd)

final_AG_pct_pd = []  # AG drainage percentage
final_LHS_flux_pd = []  # LHS flux


# Loop over R values
for i, R in enumerate(formatted_R):
    ag_values = []
    lhs_values = []

    for x0 in x0_values:
        file_path = os.path.join(
            base_path,
            # f"prod_drain_lower/pe_{R}/xi_gauss_{x0}/case_R_{R}_dt_0.001_mesh_200_200/"
            # f"case_R_{R}_dt_0.001_mesh_200_200_fluxes.csv"
            f"prod_drain/pe_{R}/xi_gauss_{x0}/case_R_{R}_dt_0.001_mesh_200_200/"
            f"case_R_{R}_dt_0.001_mesh_200_200_fluxes.csv"
        )
        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=1)  # load all 6 values
            print(np.shape(data))
            if np.isscalar(data):
                data = np.array([data])

            ag_pct = -data[6,] / data[7] if data[0,] != 0 else np.nan
            print(data[0], data[1], data[1]/data[0])
            ag_values.append(ag_pct)

            lhs_flux = data[2]
            lhs_values.append(lhs_flux)

        except Exception as e:
            print(f"File error at {file_path}: {e}")
            ag_values.append(np.nan)
            lhs_values.append(np.nan)

    final_AG_pct_pd.append(ag_values)
    final_LHS_flux_pd.append(lhs_values)

# Convert to arrays
AG_array_pd = np.array(final_AG_pct_pd)       # shape (n_R, n_x0)

final_AG_pct = []  # AG drainage percentage
final_LHS_flux = []  # LHS flux

# Loop over R values
for i, R in enumerate(formatted_R):
    ag_values = []
    lhs_values = []

    for x0 in x0_values:
        file_path = os.path.join(
            base_path,
            f"pe_{R}/x_{x0}/case_R_{R}_dt_0.001_mesh_100_200/"
            f"case_R_{R}_dt_0.001_mesh_100_200_fluxes.csv"
        )
        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=1)  # load all 6 values

            if np.isscalar(data):
                data = np.array([data])

            ag_pct = data[1,] / data[0] if data[0,] != 0 else np.nan
            ag_values.append(ag_pct)

            lhs_flux = data[2]
            lhs_values.append(lhs_flux)

        except Exception as e:
            print(f"File error at {file_path}: {e}")
            ag_values.append(np.nan)
            lhs_values.append(np.nan)

    final_AG_pct.append(ag_values)
    final_LHS_flux.append(lhs_values)

# Convert to arrays
AG_array = np.array(final_AG_pct)  


base_path = '/home/s2611897/Desktop/FEniCS/Dispersion-2D3D/output/casestudies/full_brain/steady_state/'
print(formatted_R)

final = []
valid_indices = []
# Loop over R values
for i, R in enumerate(formatted_R):
    print(R)
    #x_str = f"{x0:.3f}"
    file_path = os.path.join(base_path, f"pe_{R}/case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_mass.csv")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        print("Loaded:", data, "Shape:", np.shape(data))
        final.append(data)

        # Promote scalar to array
        if np.isscalar(data):
            data = np.array([data])

            values_per_x0.append(data)

    except Exception as e:
        print(f"File not found or unreadable at {file_path}: {e}")
        values_per_x0.append(np.nan)     

# Final outputs
final_array_fb = np.array(final)

final_AG_pct = []  # AG drainage percentage
final_LHS_flux = []  # LHS flux

# Loop over R values
for i, R in enumerate(formatted_R):
    ag_values = []
    lhs_values = []

    file_path = os.path.join(base_path, f"pe_{R}/case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_fluxes.csv")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=1)  # load all 6 values

        if np.isscalar(data):
            data = np.array([data])

        ag_pct = data[1,] / data[0] if data[0,] != 0 else np.nan
        ag_values.append(ag_pct)

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
print(AG_array_fb)

final = []
valid_indices = []
base_path = '/home/s2611897/Desktop/FEniCS/Dispersion-2D3D/output/casestudies/full_brain/steady_state/prod_drain/'

# Loop over R values
for i, R in enumerate(formatted_R):
    #x_str = f"{x0:.3f}"
    file_path = os.path.join(base_path, f"pe_{R}/case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_mass.csv")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        print("Loaded:", data, "Shape:", np.shape(data))
        final.append(data)

    except Exception as e:
        print(f"File not found or unreadable at {file_path}: {e}")

# Final outputs
final_array_fb_pd = np.array(final)

final_AG_pct = []  # AG drainage percentage
final_LHS_flux = []  # LHS flux

# Loop over R values
for i, R in enumerate(formatted_R):
    ag_values = []
    lhs_values = []

    file_path = os.path.join(
    base_path,
    f"pe_{R}/case_R_{R}_dt_0.001_mesh_200_200/"
    f"case_R_{R}_dt_0.001_mesh_200_200_fluxes.csv"
    )
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=1)  # load all 6 values

        if np.isscalar(data):
            data = np.array([data])

        ag_pct = data[6,] / data[0] if data[0,] != 0 else np.nan
        if ag_pct >1:
            ag_pct = 1
            print('correcting error - check pd full brain fluxes')
        ag_values.append(ag_pct)

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

fig = plt.figure(figsize=(6.75, 5.25))  # increase height to make room
gs = GridSpec(3, 2, height_ratios=[1, 3, 3], hspace=0.4)

# === Top schematic ===
ax_top = fig.add_axes([0.25, 0.8, 0.5, 0.075]) 
x_vals = np.linspace(0.5, 1, 500)
def q(x, q0=1, center=0.6, steepness=30):
    return  1*(1 - 1 / (1 + np.exp(-steepness * (x - center))) + ( - 1 / (1 + np.exp(-steepness * (1 - x - center)))))
y_vals = 1*q(x_vals)

import matplotlib as mpl

# Get Matplotlib's default color cycle
prop_cycle = mpl.rcParams['axes.prop_cycle']
default_colors = prop_cycle.by_key()['color']


ax_top.plot(x_vals, y_vals, color='black')

#ax_top.axhline(y=-0.2, color='black', linewidth=1, linestyle='--')
ax_top.axhline(y=0, color='grey', linewidth=1, linestyle='-')
leak_points = [0.6, 0.7, 0.8]
colors = default_colors[:3]

# Leak points
for x, c, lab in zip(leak_points, colors, ['$x_0=0.6$', '$x_0=0.7$', '$x_0=0.8$']):
    ax_top.axvline(x, color=c, linestyle='--', linewidth=0.8)
    ax_top.plot(x, -0.2, 'o', color=c, label=lab, zorder=2)

ax_top.axhline(y=-0.2, color='magenta', linewidth=1, linestyle='-', label = 'Full brain', zorder=1)

# Legend on the right
ax_top.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

ax_top.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax_top.set_yticks([0])
ax_top.set_xlim(0.5, 1)
ax_top.set_ylim(-0.35, 1.2)
ax_top.set_title("Schematic: Leak points", fontsize=10)
ax_top.set_xlabel('$x$')
ax_top.set_ylabel(r'$d(x)$')


# === Bottom plots ===
axs = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
       fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

# === Colors ===
cmap = plt.colormaps["tab10"]
colors = [cmap(i % cmap.N) for i in range(3)]

# === Ticks ===
log_ticks = [1e-2, 1e-1, 1e0, 1e1, 1e2]
log_labels = [r'$10^{-2}$', r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$']

# === Mass (no PD) ===
ax = axs[0]
for j, x0 in enumerate(x0_values):
    ax.plot(R_vals, final_array[:, j], label=f"$x_0 = {x0}$", color=colors[j])
ax.plot(R_vals, final_array_fb, color='magenta', label='Full brain')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(log_ticks)
ax.set_xticklabels([])  # Hide top x-tick labels
ax.set_ylabel(r"$u_{pd} = 0$")
ax.set_title("Solute Mass")
ax.grid(True)

# === AG Drainage (no PD) ===
ax = axs[1]
for j, x0 in enumerate(x0_values):
    ax.plot(R_vals, AG_array[:, j] * 100, label=f"$x_0 = {x0}$", color=colors[j])
ax.plot(R_vals, AG_array_fb * 100, color='magenta', label='Full brain')

ax.set_xscale('log')
ax.set_xticks(log_ticks)
ax.set_xticklabels([])  # Hide top x-tick labels
ax.set_ylabel("")
ax.set_title("AG Drainage (\%)")
ax.grid(True)
#ax.legend(fontsize=8)

# === Mass (PD) ===
ax = axs[2]
for j, x0 in enumerate(x0_values):
    ax.plot(R_vals, final_array_pd[:, j], linestyle='--', color=colors[j])
ax.plot(R_vals[:len(final_array_fb_pd)], final_array_fb_pd, linestyle='--', color='magenta')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(log_ticks)
ax.set_xticklabels(log_labels)
ax.set_xlabel("$Pe_{s}$")
ax.set_ylabel(r"$u_{pd} \neq 0$")
ax.grid(True)

# === AG Drainage (PD) ===
ax = axs[3]
for j, x0 in enumerate(x0_values):
    ax.plot(R_vals, AG_array_pd[:, j] * 100, linestyle='--', color=colors[j])
ax.plot(R_vals[:len(AG_array_fb_pd)], AG_array_fb_pd * 100, linestyle='--', color='magenta')

ax.set_xscale('log')
ax.set_xticks(log_ticks)
ax.set_xticklabels(log_labels)
ax.set_xlabel("$Pe_{s}$")
ax.set_ylabel("")
ax.grid(True)

# === Optional: Annotations or subplot labels ===
axs[0].text(-0.2, 0.95, '(a)', transform=axs[0].transAxes, fontweight='bold')
axs[1].text(-0.19, 0.95, '(b)', transform=axs[1].transAxes, fontweight='bold')
axs[2].text(-0.2, 0.95, '(c)', transform=axs[2].transAxes, fontweight='bold')
axs[3].text(-0.19, 0.95, '(d)', transform=axs[3].transAxes, fontweight='bold')

plt.tight_layout()
# === Save and Show ===
os.makedirs("outputs/concentration/check_pepd", exist_ok=True)
# plt.savefig("outputs/concentration/check_pepd/mass_vs_AG_2x2.pdf", dpi=300)
plt.savefig("outputs/concentration/mass_vs_AG_2x2.png", dpi=1500)
plt.show()