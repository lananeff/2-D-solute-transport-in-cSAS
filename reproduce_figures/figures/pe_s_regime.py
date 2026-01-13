"""
Figure 6: Steady streaming transport regime map (Pe_s) in (A, alpha) space.

This script generates the regime-map figure used in the paper by computing a steady streaming
 Péclet number over a grid of amplitude A and frequency parameter alpha:

    Pe_s = alpha^4 * A^2 * S * u_scale

where S and eps are taken from `src/parameters.PARAMETERS` for a chosen species.
u_scale=0.006 is required to correctly scale steady streaming longitudinal velocity component. 
Analytical velocity fields u_L and v_L are evaluated on a 2-D grid to compute
max|u_L| and max|v_L| (used for scaling/diagnostics).

The script overlays:
- boxed parameter ranges for cardiac, respiratory, and sleep regimes (for the selected species),
- the boundary A = 1/(alpha^2 S eps),
- the balance curve Pe_s = Pe_pd (using a fixed pdscale),
- coloured square markers at selected (A, alpha) points, with legend entries showing Pe_s.

Usage:
Default runs for human:
    python pe_osc_regime_logscale_u.py
Or for mouse:
    python pe_osc_regime_logscale_u.py --species mouse

Output:
- outputs/pe_osc_regime_logscale_u.png
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from velocities import u0, u1, v0, v1, us, vs, uL, vL, upd, vpd
import argparse
from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from src.custom_colormap import custom_plasma
from scipy.interpolate import RegularGridInterpolator

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

plt.figure(figsize=(6.75, 3.5))

# Get parameters for regime
def get_parameters(species):
    """Retrieve parameters for the given species."""
    try:
        return PARAMETERS[species]  # Ensure PARAMETERS[species] exists
    except KeyError:
        raise ValueError(f"Invalid species '{species}'")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Select CSF transport parameters.")
parser.add_argument("--species", choices=["human", "mouse"], default = "human", help="Choose species: human or mouse")

args = parser.parse_args()
params = get_parameters(args.species)

# Set pars
#A = params[args.regime]["A"]
eps = params["eps"]
S = params["S"]

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

A_values = np.linspace(0.001, 0.3, 75)  # Values of A
alpha_values = np.linspace(0.25, 6, 75)  # Values of omega
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

uscale= 0.006 # vscale=0.002
pdscale = 0.7604517994874807

# Iterate over A and omega values to compute Pe_eff
for i, alpha in enumerate(alpha_values):
    for j, A in enumerate(A_values):
        u1_values = uL(X, Y, alpha, A, eps)
        v1_values = vL(X, Y, alpha, A, eps)
        max_uL_values[i,j] = np.max(np.abs(u1_values))
        max_vL_values[i,j] = np.max(np.abs(v1_values))
        Pe_eta_values[i, j] = alpha**4 * A**2 * S * uscale




# Create a contour plot of Pe_eff
A_grid, omega_grid = np.meshgrid(A_values, alpha_values)

# Set colour bar scale from Pe > 1/eps condition
# Calculate A using the equation A = 1 / (alpha^2 * S * epsilon)
As = 1 / (alpha_values**2 * S * eps)
# Find the minimum Pe_eta along this line
Pe_min = As[0]**2 * alpha_values[0]**4 * S *uscale

# A_balance = np.sqrt(eps * pdscale / ( alpha_values**4 * S * uscale)) # 0.00055 comes from vlmax as varied with alpha A=0.0025
# plt.plot(A_balance, alpha_values, color='white', linestyle='--', linewidth=1.5, label=r'$Pe_{s} = \epsilon Pe_{pd}$')

A_big = np.sqrt(pdscale / (alpha_values**4 * S * uscale))
plt.plot(A_big, alpha_values, color='lightgrey', linestyle='--', linewidth=1.5, label=r'$Pe_{s} = Pe_{pd}$')


contour = plt.contourf(A_grid, omega_grid, Pe_eta_values, levels=np.logspace(np.log10(Pe_eta_values.min()), np.log10(Pe_eta_values.max()), 100), cmap=custom_plasma(),norm=LogNorm(vmin=Pe_min, vmax=Pe_eta_values.max()) )
#contour = plt.contourf(A_grid, omega_grid, Pe_eta_values,
                    #    levels=100,
                    #    cmap=custom_plasma())
cbar = plt.colorbar(contour, label=r'$Pe_{s}$')

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
    #{'alpha': 2.5, 'A': 0.0025/6},
    #{'alpha': 2.5, 'A': 0.0025*np.sqrt(10)/6},
    {'alpha': 2.5, 'A': 0.0025*10/np.sqrt(6)},
    {'alpha': 2.5, 'A': (0.0025*np.sqrt(10)**3)/np.sqrt(6)},
    {'alpha': 2.5, 'A': (0.0025*10**2)/np.sqrt(6)}
]
print(markers)

# Normalize the Pe_eta value to match the colormap
norm = LogNorm(vmin=Pe_min, vmax=Pe_eta_values.max())
cmap = custom_plasma()

# Define regimes in a list
regimes = [
    {
        "alpha_min": alpha_c_min, "alpha_max": alpha_c_max,
        "A_min": A_c_min, "A_max": A_c_max,
        "label": "-- Cardiac", "color": "#E34234"
    },
    {
        "alpha_min": alpha_r_min, "alpha_max": alpha_r_max,
        "A_min": A_r_min, "A_max": A_r_max,
        "label": "-- Respiratory", "color": "cyan"
    },
    {
        "alpha_min": alpha_s_min, "alpha_max": alpha_s_max,
        "A_min": A_s_min, "A_max": A_s_max,
        "label": "-- Sleep", "color": "#A6FF4D"
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
        label=regime["label"]
    )


legend_entries = {}

def round_sig_custom(x, sig=1):
    if x == 0:
        return 0
    rounded = round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1))
    
    # Snap close-to-integer values to their nearest round number
    for target in [1, 10, 100, 1000, 0.1, 0.01, 0.001]:
        if abs(rounded - target) / target < 0.25:  # within 5%
            return target
    return rounded

# Plot each marker
for marker in markers:
    omega_marker = marker['alpha']
    A_marker = marker['A']
    regime_name = marker.get('regime', None) 
    interp_func = RegularGridInterpolator((alpha_values, A_values), Pe_eta_values, bounds_error=False, fill_value=None)
    Pe_eff_marker = interp_func([[omega_marker, A_marker]])[0]
    color_marker = cmap(norm(Pe_eff_marker))
    plt.scatter(A_marker, omega_marker, color=color_marker, edgecolor='white', s=100, marker='s', zorder=5)

    # Add line showing where else is this colour?
    Pe_eff_rounded = round_sig_custom(Pe_eff_marker, sig=1)

    if regime_name:
        label = label = rf"{regime_name}: $Pe_{{osc}} = {Pe_eff_rounded}$"
    else:
        label = label = rf"$Pe_{{osc}} = {Pe_eff_rounded}$" # No regime, just Pe value

    
    # Only add unique legend entries
    if label not in legend_entries:
        legend_entries[label] = plt.scatter(A_marker, omega_marker, color=color_marker, 
                                            edgecolor='white', s=100, marker='s', zorder=5, 
                                            label=label)

# Plot regimes in boxes
print(f"eps = {eps}")

# Plot the line where A alpha^2 S = 1/eps which is the lower boundary for our transport regime
# Plot the line where A = 1 / (alpha^2 * S * epsilon)
plt.plot(As, alpha_values, color='black')


# Plot the marker
#plt.scatter(A_marker, omega_marker, color=color_marker, edgecolor='white', s=100, marker='s', label=r'Sample $Pe_{s}$', zorder=5)
legend = plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), labelspacing=1)
# legend.get_frame().set_facecolor('lightgray')  # or 'black', 'gray', etc.
# legend.get_frame().set_edgecolor('none')  
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('$A$')
plt.ylabel(r'$\alpha$')
#plt.title(r'Effective Peclet Number $Pe_{s}$')
plt.xlim(np.min(A_values), np.max(A_values))  # Limit x-axis to between 0 and 0.1
plt.ylim(np.min(alpha_values), np.max(alpha_values))

# Save the plot to the 'outputs' folder
plt.savefig('outputs/pe_osc_regime_logscale_u.png')