import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from velocities import u0, u1, v0, v1, us, vs, uL, vL, upd, vpd
import argparse
from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Get parameters for regime
def get_parameters(species, regime):
    """Retrieve parameters for the given species and regime."""
    try:
        species_params = PARAMETERS[species]
        if regime:
            regime_params = species_params.get(regime, {})
            params = {**species_params, **regime_params}  # Merge species and regime parameters
        else:
            params = species_params.copy()
        return params
    except KeyError:
        raise ValueError(f"Invalid species '{species}' or regime '{regime}'")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Select CSF transport parameters.")
parser.add_argument("species", choices=["human", "mouse"], help="Choose species: human or mouse")
parser.add_argument("--regime", choices=["cardiac", "respiratory", "sleep"], help="Choose flow regime")

args = parser.parse_args()
params = get_parameters(args.species, args.regime)

print(f"Selected parameters for {args.species}, {args.regime if args.regime else 'all'}:")
for key, value in params.items():
    print(f"{key}: {value}")

# Set pars
alpha = params["alpha"]
eps = params["eps"]
omega = params["omega"]

nu = params["nu"]
h = params["h"]

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

golden_ratio = (5**.5 - 1) / 2

# Figure width in inches
fig_width_in = 6
# Figure height in inches
fig_height_in = fig_width_in * golden_ratio
fig_height_in

# Create figure

# Replace varying alpha with varying A
As = np.logspace(-3, -1, 50)  # e.g. A from 0.01 to 100
alpha = params["alpha"]  # Set a fixed alpha (you may also hardcode e.g., alpha = 1)
xs = np.linspace(0, 1, 100)
ys = np.linspace(-1, 0, 100)
t0 = 0

max_u1 = np.zeros(len(As))
max_us = np.zeros(len(As))
max_v1 = np.zeros(len(As))
max_vs = np.zeros(len(As))
max_ul = np.zeros(len(As))
max_vl = np.zeros(len(As))
max_upd = np.zeros(len(As))
max_vpd = np.zeros(len(As))

# Iterate over A
for k, A in enumerate(As):
    udata1 = np.zeros((100, 100))
    usdata = np.zeros((100, 100))
    vdata1 = np.zeros((100, 100))
    vsdata = np.zeros((100, 100))
    uldata = np.zeros((100, 100))
    vldata = np.zeros((100, 100))
    upddata = np.zeros((100, 100))
    vpddata = np.zeros((100, 100))

    omega = (alpha**2 * nu) / h**2

    for i in range(len(ys)):
        for j in range(len(xs)):
            udata1[i, j] = A * omega * h * u1(xs[j], ys[i], alpha, A, eps)
            usdata[i, j] = A * h * omega * us(xs[j], ys[i], alpha, A, eps)
            vdata1[i, j] = eps * A * h * omega * v1(xs[j], ys[i], alpha, A, eps)
            vsdata[i, j] = eps * A * h * omega * vs(xs[j], ys[i], alpha, A, eps)
            uldata[i, j] = A * h * omega * (uL(xs[j], ys[i], alpha, A, eps) + upd(xs[j], ys[i], alpha, A, eps, omega, h))
            vldata[i, j] = eps * A * h * omega * (vL(xs[j], ys[i], alpha, A, eps) + vpd(xs[j], ys[i], alpha, A, eps, omega, h))
            upddata[i, j] = A * h * omega * upd(xs[j], ys[i], alpha, A, eps, omega, h)
            vpddata[i, j] = eps * A * h * omega * vpd(xs[j], ys[i], alpha, A, eps, omega, h)

    max_u1[k] = np.max(udata1)
    max_us[k] = np.max(usdata)
    max_v1[k] = np.max(vdata1)
    max_vs[k] = np.max(vsdata)
    max_ul[k] = np.max(uldata)
    max_vl[k] = np.max(vldata)
    max_upd[k] = np.max(upddata)
    max_vpd[k] = np.max(vpddata)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(5, 2))
ax1, ax2 = axs

# u velocities
ax1.plot(As, max_u1, label=r'$\tilde{u}_s$', color='black', linewidth=1)
ax1.plot(As, max_us, label=r'$\tilde{u}_{\omega}$', linestyle='--', color='black', linewidth=2)
ax1.plot(As, max_ul, label=r'$\tilde{u}_L$', linestyle=':', color='black', linewidth=1)
ax1.plot(As, max_upd, label=r'$\tilde{u}_{pd}$', color='black', linestyle='dashdot', linewidth=1)
ax1.set_xlabel(r'$A$')
ax1.set_ylabel('Max Value')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()

# v velocities
ax2.plot(As, max_v1, label=r'$\tilde{v}_s$', color='black', linewidth=1)
ax2.plot(As, max_vs, label=r'$\tilde{v}_{\omega}$', linestyle='--', color='black', linewidth=2)
ax2.plot(As, max_vl, label=r'$\tilde{v}_L$', linestyle=':', color='black', linewidth=1)
ax2.plot(As, max_vpd, label=r'$\tilde{v}_{pd}$', color='black', linestyle='dashdot', linewidth=1)
ax2.set_xlabel(r'$A$')
#ax2.set_ylabel('Max Value')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()

# Subplot labels
labels = ['(a)', '(b)']
for ax, label in zip(axs, labels):
    ax.text(-0.25, 1.1, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('outputs/comparison_u_vs_v_velocities_varying_A.png')
