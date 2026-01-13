import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from velocities import u0, u1, v0, v1, us, vs, uL, vL, upd, vpd
import argparse
from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from src.custom_colormap import custom_plasma

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

plt.figure(figsize=(5, 3))

# Get parameters for regime
def get_parameters(species):
    """Retrieve parameters for the given species."""
    try:
        return PARAMETERS[species]  # Ensure PARAMETERS[species] exists
    except KeyError:
        raise ValueError(f"Invalid species '{species}'")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Select CSF transport parameters.")
parser.add_argument("species", choices=["human", "mouse"], help="Choose species: human or mouse")

args = parser.parse_args()
params = get_parameters(args.species)

# Set pars
#A = params[args.regime]["A"]
eps = params["eps"]
S = 4096

cardiac_params = params["cardiac"]
A_c = cardiac_params["A"]  # Accessing 'A' from the cardiac regime
alpha_c = cardiac_params["alpha"]
A_c_min = cardiac_params["A_min"]
A_c_max = cardiac_params["A_max"]
alpha_c_min = cardiac_params["alpha_min"]
alpha_c_max = cardiac_params["alpha_max"]

# Access respiratory regime
respiratory_params = params["respiratory"]
A_r = respiratory_params["A"]
alpha_r = respiratory_params["alpha"]
A_r_min = respiratory_params["A_min"]
A_r_max = respiratory_params["A_max"]
alpha_r_min = respiratory_params["alpha_min"]
alpha_r_max = respiratory_params["alpha_max"]

# Access sleep regime
sleep_params = params["sleep"]
A_s = sleep_params["A"]
alpha_s = sleep_params["alpha"]
A_s_min = sleep_params["A_min"]
A_s_max = sleep_params["A_max"]
alpha_s_min = sleep_params["alpha_min"]
alpha_s_max = sleep_params["alpha_max"]

print(f"Baby we got em! alpha_c = {alpha_c}, A_c = {A_c}, alpha_r = {alpha_r}, A_r = {A_r},alpha_s = {alpha_s}, A_s = {A_s}")

A_values = np.logspace(-4, -1/4, 50)  # Values of A
alpha_values = np.logspace(-2, 1, 50)  # Values of alpha
from matplotlib.colors import LogNorm
# Prepare arrays to store results
Pe_eta_values = np.zeros((len(alpha_values), len(A_values)))
max_uL_values = np.zeros((len(alpha_values), len(A_values)))
Pe_x_values = np.zeros((len(alpha_values), len(A_values)))
max_vL_values = np.zeros((len(alpha_values), len(A_values)))

# Define the domain
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(-1, 0, 100)
X, Y = np.meshgrid(x_values, y_values)

# Iterate over A and omega values to compute Pe_eff
for i, alpha in enumerate(alpha_values):
    for j, A in enumerate(A_values):
        u1_values = uL(X, Y, alpha, A, eps)
        v1_values = vL(X, Y, alpha, A, eps)
        max_uL_values[i,j] = np.max(np.abs(u1_values))
        max_vL_values[i,j] = np.max(np.abs(v1_values))
        Pe_eta_values[i, j] = alpha**2 * A * S




# Create a contour plot of Pe_eff
A_grid, omega_grid = np.meshgrid(A_values, alpha_values)

contour = plt.contourf(A_grid, omega_grid, Pe_eta_values, levels=np.logspace(np.log10(Pe_eta_values.min()), np.log10(Pe_eta_values.max()), 100), cmap=custom_plasma(),norm=LogNorm(vmin=Pe_eta_values.min(), vmax=Pe_eta_values.max()) )
cbar = plt.colorbar(contour, label=r'$Pe$')

# Set the colorbar ticks and labels to show powers of 10
cbar.set_ticks([10**(i) for i in range(int(np.log10(Pe_eta_values.min())), int(np.log10(Pe_eta_values.max()))+1)])
cbar.set_ticklabels([f"$10^{{{i}}}$" for i in range(int(np.log10(Pe_eta_values.min())), int(np.log10(Pe_eta_values.max()))+1)])


# Define marker positions for three different regimes
#markers = [
#    {'alpha': np.sqrt(2 * np.pi), 'A': 0.05},
#    {'alpha': np.sqrt(2 * np.pi * 10**(1/2)), 'A': 0.05},
#    {'alpha': np.sqrt(2 * np.pi* 10**(-1/2)), 'A': 0.05},
#]

# markers = [
#     {'regime': "Cardiac", 'alpha': alpha_c, 'A': A_c},
#     {'regime': "Resp.",'alpha': alpha_r, 'A': A_r},
#     {'regime': "Sleep",'alpha': alpha_s, 'A': A_s},
#     {'regime': "(a)",'alpha': np.sqrt(2 * np.pi* 10**(-1/2)), 'A': 0.05},
#     {'regime': "(b)",'alpha': np.sqrt(2 * np.pi), 'A': 0.05},
#     {'regime': "(c)",'alpha': np.sqrt(2 * np.pi * 10**(1/2)), 'A': 0.05},
#     
# ]

markers = [
    {'alpha': np.sqrt(2 * np.pi)* 10**(-1/4), 'A': 0.05},
    {'alpha': alpha_c, 'A': A_c},
    {'alpha': np.sqrt(2 * np.pi)* 10**(1/4), 'A': 0.05},
]
print(markers)

# Normalize the Pe_eta value to match the colormap
norm = LogNorm(vmin=Pe_eta_values.min(), vmax=Pe_eta_values.max())
cmap = custom_plasma()

# Define regimes in a list
regimes = [
    {
        "alpha_min": alpha_c_min, "alpha_max": alpha_c_max,
        "A_min": A_c_min, "A_max": A_c_max,
        "label": "Cardiac", "color": "red"
    },
    {
        "alpha_min": alpha_r_min, "alpha_max": alpha_r_max,
        "A_min": A_r_min, "A_max": A_r_max,
        "label": "Respiratory", "color": "cyan"
    },
    {
        "alpha_min": alpha_s_min, "alpha_max": alpha_s_max,
        "A_min": A_s_min, "A_max": A_s_max,
        "label": "Sleep", "color": "green"
    }
]

# Plot each regime
for regime in regimes:
    plt.plot(
        [regime["A_min"], regime["A_max"], regime["A_max"], regime["A_min"], regime["A_min"]],
        [regime["alpha_min"], regime["alpha_min"], regime["alpha_max"], regime["alpha_max"], regime["alpha_min"]],
        color=regime["color"],
        linestyle='--',
        linewidth=2,  # Thinner lines
        #alpha=0.3,  # More transparency
        label=regime["label"]
    )


legend_entries = {}

# Plot regimes in boxes


# Add specific contour levels
target_levels = [ eps, 1/(eps)]  # Specify desired levels

#target_levels = [eps**(2), eps, 1, eps**(-1), eps**(-2)]  # Specify desired levels
contour_lines = plt.contour(A_grid, omega_grid, Pe_eta_values, levels=target_levels, colors='white', linewidths=1.5)
#plt.clabel(contour_lines, fmt={target_levels[1]: r'$1/\epsilon$'}, inline=True, fontsize=1)
#plt.clabel(contour_lines, fmt={target_levels[0]: r'$\epsilon$'}, inline=True, fontsize=1)
label_alpha = alpha_values[len(alpha_values)//2 +1]
label_A_inv_eps = 1/(eps * label_alpha**2 * S)

label_alpha2 = alpha_values[len(alpha_values)//20]
print(label_alpha2)
label_A_eps = eps / (label_alpha2**2 * S)
print(label_A_eps)

plt.text(label_A_eps+0.0025, label_alpha2, r'$\varepsilon$', color='white', fontsize=12,
         ha='right', va='bottom', rotation=0)
plt.text(label_A_inv_eps+0.15, label_alpha-0.1, r'$1/\varepsilon$', color='white', fontsize=12,
         ha='left', va='bottom', rotation=0)

# (iii): Below Pe = eps
plt.text(2e-4, 0.02, r'(iii)', color='black', fontsize=12)

# (iv): Between Pe = eps and Pe = 1/eps
plt.text(0.002, 0.11, r'(iv)', color='black', fontsize=12)

# (v): Above Pe = 1/eps
plt.text(0.2, 5, r'(v)', color='black', fontsize=12)

# Plot the marker
#plt.scatter(A_marker, omega_marker, color=color_marker, edgecolor='white', s=100, marker='s', label=r'Sample $Pe_{osc}$', zorder=5)
plt.legend(loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$A$')
plt.ylabel(r'$\alpha$')
#plt.title(r'Human transport regimes')
plt.xlim(np.min(A_values), np.max(A_values))  # Limit x-axis to between 0 and 0.1
plt.ylim(np.min(alpha_values), np.max(alpha_values))

# Save the plot to the 'outputs' folder
plt.savefig('outputs/pe_A_vs_alpha.png')