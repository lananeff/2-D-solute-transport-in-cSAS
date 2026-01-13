"""
Figure 2: Regime map in (A, alpha) space.

This script reproduces Fig. 2 of the paper by plotting the dimensionless quantity
Pe = alpha^2 * A * S over a log-spaced grid in (A, alpha), and overlaying the
parameter ranges corresponding to the cardiac, respiratory, and sleep regimes for
human and mouse (from `src/parameters.py`).

Output:
- outputs/pe_A_vs_alpha_human_mouse.png

Run:
    python pe_A_vs_alpha_human_mouse.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from velocities import uL, vL
from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from src.custom_colormap import custom_plasma
from matplotlib.colors import LogNorm

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

plt.figure(figsize=(6.75, 3.5))

# Shared grid
A_values = np.logspace(-4, -1/4, 50)
alpha_values = np.logspace(-2, 1, 50)
A_grid, alpha_grid = np.meshgrid(A_values, alpha_values)

# Precompute Pe_eta (human S for normalization)
S = PARAMETERS["human"]["S"]
eps = PARAMETERS["human"]["eps"]

Pe_eta_values = np.zeros((len(alpha_values), len(A_values)))
for i, alpha in enumerate(alpha_values):
    for j, A in enumerate(A_values):
        Pe_eta_values[i, j] = alpha**2 * A * S

# Background contour
contour = plt.contourf(
    A_grid, alpha_grid, Pe_eta_values,
    levels=np.logspace(np.log10(Pe_eta_values.min()), np.log10(Pe_eta_values.max()), 100),
    cmap=custom_plasma(),
    norm=LogNorm(vmin=Pe_eta_values.min(), vmax=Pe_eta_values.max())
)
cbar = plt.colorbar(contour, label=r"$Pe$")
cbar.set_ticks([10**i for i in range(-2, 5)])
cbar.set_ticklabels([f"$10^{{{i}}}$" for i in range(-2, 5)])

# Fixed regime colors
regime_colors = {"cardiac": "#E34234", "respiratory": "cyan", "sleep": "#A6FF4D"}
species_styles = {"human": "--", "mouse": "-."}

for species in ["human", "mouse"]:
    params = PARAMETERS[species]
    eps = params["eps"]

    for regime_name in ["cardiac", "respiratory", "sleep"]:
        if regime_name not in params or not params[regime_name]:
            continue
        regime = params[regime_name]

        A_min, A_max = regime["A_min"], regime["A_max"]
        alpha_min, alpha_max = regime["alpha_min"], regime["alpha_max"]

        # Explicitly label with species
        label = f"--  {regime_name.capitalize()} "

        plt.plot(
            [A_min, A_max, A_max, A_min, A_min],
            [alpha_min, alpha_min, alpha_max, alpha_max, alpha_min],
            color=regime_colors[regime_name],
            linestyle=species_styles[species],
            linewidth=2,
            label=label,
        )

# Asymptotic contours
target_levels = [eps, 1/eps]
plt.contour(A_grid, alpha_grid, Pe_eta_values,
            levels=target_levels, colors="white", linewidths=1.5)

# Region labels
plt.text(2e-4, 0.02, r"(iii)", color="black", fontsize=12)
plt.text(0.002, 0.11, r"(iv)", color="black", fontsize=12)
plt.text(0.2, 5, r"(v)", color="black", fontsize=12)


# Full legend outside on the right
plt.legend(loc="center left", bbox_to_anchor=(1.3, 0.5), fontsize=10)

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$A$")
plt.ylabel(r"$\alpha$")
plt.xlim(np.min(A_values), np.max(A_values))
plt.ylim(np.min(alpha_values), np.max(alpha_values))

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pe_A_vs_alpha_human_mouse.png", dpi=300, bbox_inches="tight")

