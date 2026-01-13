"""
Figure 5: Maximum velocity magnitude vs amplitude A, with stacked flow profiles.

This script reproduces the figure used in the paper that:
1) computes the maximum (over x,y) magnitude of several analytical velocity fields
   as a function of oscillation amplitude A, and
2) shows three representative spatial profiles (stacked panels) at selected A values.

Model/inputs:
- Analytical velocity fields are imported from `velocities/`.
- Parameter values (alpha, eps, omega, h) are taken from `src/parameters.py`
  for the chosen species (currently hard-coded as `species = "human"`).

Outputs:
- Saves: outputs/max_vs_A_and_stacked_profiles.png

Usage:
- Run from the repository root (after environment setup):
    python max_vs_A_and_stacked_profiles.py

Notes:
- The amplitude sweep is log-spaced (As = 1e-3 ... 1e-1).
- The right-hand panels plot the magnitude of the (scaled) Lagrangian + production–drainage
  velocity field, with vectors overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
from velocities import u0, u1, v0, v1, us, vs, uL, vL, upd, vpd
from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d

# Set plotting defaults
plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def sci_notation_2sf(x, _):
    """Format tick labels in scientific notation with ~2 significant figures."""
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exponent
    return r"${:.2f} \times 10^{{{:d}}}$".format(coeff, exponent)

# === Parameters ===
species = "human"
params = PARAMETERS[species]
alpha, eps, omega, h = params["alpha"], params["eps"], params["omega"], params["h"]
print(alpha, eps, omega, h)


tikzfont = 8

# Grids
xn = 50
yn = 50
As = np.linspace(1e-3, 1e-1, 50)
xs = np.linspace(0.5, 1, xn)
ys = np.linspace(-1, 0, yn)
X, Y = np.meshgrid(xs, ys)

# Preallocate max arrays
max_u1 = np.zeros(len(As))
max_us = np.zeros(len(As))
max_v1 = np.zeros(len(As))
max_vs = np.zeros(len(As))
max_ul = np.zeros(len(As))
max_vl = np.zeros(len(As))
max_upd = np.zeros(len(As))
max_vpd = np.zeros(len(As))

# === Loop over A values and compute max magnitudes ===
for k, A in enumerate(As):
    udata1 = np.zeros((xn, yn))
    usdata = np.zeros((xn, yn))
    vdata1 = np.zeros((xn, yn))
    vsdata = np.zeros((xn, yn))
    uldata = np.zeros((xn, yn))
    vldata = np.zeros((xn, yn))
    upddata = np.zeros((xn, yn))
    vpddata = np.zeros((xn, yn))

    for i in range(len(ys)):
        for j in range(len(xs)):
            udata1[i, j] = A*h*omega*u1(xs[j], ys[i], alpha, A, eps)
            usdata[i, j] = A*h*omega*us(xs[j], ys[i], alpha, A, eps)
            uldata[i, j] = A*h*omega*uL(xs[j], ys[i], alpha, A, eps) +  A*h*omega*upd(xs[j], ys[i], alpha, A, eps, omega, h) 
            upddata[i, j] = A*h*omega*upd(xs[j], ys[i], alpha, A, eps, omega, h)
            plt.imshow(upddata)

            

    max_u1[k] = np.max(abs(udata1))
    max_us[k] = np.max(abs(usdata))
    max_ul[k] = np.max(abs(uldata))
    max_upd[k] = np.max(abs(upddata))
    
print(max_upd[1])
interp_max_ul = interp1d(As, max_ul, kind='linear', fill_value='extrapolate')

# === Create Figure with GridSpec ===
fig_width = 6.75  # inches
fig_height = 2.5  # enough for 3 stacked subplots

fig = plt.figure(figsize=(fig_width, fig_height))
gs = GridSpec(nrows=3, ncols=2, width_ratios=[1, 1.5], height_ratios=[1, 1, 1], figure=fig, wspace=0.4, hspace=0.4)

# === Left: Summary plot spanning all rows ===
ax_summary = fig.add_subplot(gs[:, 0])
ax_summary.plot(As, max_u1, label=r'$ \tilde{u}_s$', color='blue', linewidth=1)
ax_summary.plot(As, max_us, label=r'$\tilde{u}_{\omega}$', linestyle='--', color='blue', linewidth=1)
ax_summary.plot(As, max_ul, label=r'$\tilde{u}_L$', linestyle=':', color='blue', linewidth=2)
ax_summary.plot(As, max_upd, label=r'$\tilde{u}_{pd}$', color='blue', linestyle='dashdot', linewidth=1)
A_vals = [1e-3, 0.025, 1e-1]  # Low, mid, high
labels = ['(i)', '(ii)', '(iii)']

# for A_val, label in zip(A_vals, labels):
#     ax_summary.axvline(A_val, color='black', linestyle=':', linewidth=1)
#     ax_summary.text(A_val, ax_summary.get_ylim()[1], label,
#                     ha='center', va='bottom')

for A_val, label in zip(A_vals, labels):
    ul_val = interp_max_ul(A_val)
    ax_summary.plot(A_val, ul_val, 'o', color='black', markersize=4)
    ax_summary.text(A_val, ul_val * 1.15, label, fontsize=8,
                    ha='center', va='bottom')

ax_summary.set_xscale('log')
ax_summary.set_yscale('log')
ax_summary.set_ylim(1e-9, 1e-3) 
ax_summary.set_xlabel(r'$A$')
ax_summary.set_ylabel('Max value (m/s)')
#ax_summary.set_title('(a)', loc='left', x=-0.5, fontsize=12, fontweight='bold')
ax_summary.legend(loc='lower right')


# === Right: 3 stacked flow magnitude plots ===
titles = [f'$A = {A:.0e}$' for A in A_vals]
scales = [100, 7, 12]
fig.text(0.39, 0.92, '(b)',  fontweight='bold')
fig.text(0, 0.92, '(a)', fontweight='bold')
for i, A in enumerate(A_vals):
    ax = fig.add_subplot(gs[i, 1])

    udata = np.zeros_like(X)
    vdata = np.zeros_like(Y)
    # for j in range(len(ys)):
    #     for k in range(len(xs)):
    #         udata[j, k] = uL(xs[j], ys[i], alpha, A, eps) #+  A*h*omega*upd(xs[j], ys[i], alpha, A, eps, omega, h) 
    #         vdata[j, k] = vL(xs[j], ys[i], alpha, A, eps)  #+ A*h*omega*vpd(xs[k], ys[j], alpha, A, eps, omega, h)
    for j in range(len(ys)):
        for k in range(len(xs)):
            udata[j, k] = uL(xs[k], ys[j], alpha, A, eps) + upd(xs[k], ys[j], alpha, A, eps, omega, h)
            vdata[j, k] = (vL(xs[k], ys[j], alpha, A, eps) + vpd(xs[k], ys[j], alpha, A, eps, omega, h))


    #mag = np.sqrt(udata**2 + (eps*vdata)**2)
    mag = np.sqrt((A*h*omega)**2*(udata**2 + (eps*vdata)**2))

    raw_max = np.max(mag)

    nice_max = float(f"{raw_max:.1g}") 

    # c = ax.pcolormesh(xs, ys, mag, shading='auto', cmap='plasma', vmin=0, vmax=np.max(mag))
    c = ax.pcolormesh(xs, ys, mag, shading='auto', cmap='plasma', vmin = 0, vmax = nice_max)

    n = 3  # Subsampling rate

    ax.quiver(
        X[::2*n, ::n],
        Y[::2*n, ::n],
        udata[::2*n, ::n],
        (1/10)*vdata[::2*n, ::n],  # Anisotropic scaling (if needed)
        width=0.00125 * 3,
        headwidth=3,
        headlength=3,
        headaxislength=3,
        color = 'black',
        scale_units='xy',
        scale = scales[i]
    )

    cb = fig.colorbar(c, ax=ax, orientation='vertical', aspect=10)
    
    tick_vals = [0, nice_max]
    # tick_vals = [0, np.max(mag)]
    cb.set_ticks(tick_vals)

    cb.formatter.set_powerlimits((0,0))
    cb.update_ticks()

    cb.set_label(r'$|\tilde{\boldsymbol{u}}_L|$', labelpad=0)
    cb.ax.yaxis.set_label_position('right')  # Or 'left' if you want it on the left
    cb.ax.yaxis.label.set_verticalalignment('center') 
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-1, 0)

    # Show x-axis ticks and label only on the bottom plot
    if i == 2:
        ax.set_xlabel('$x$')
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(['0.5', '0.75', '1'])
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('')  # Remove x-axis label

    # Y-axis ticks only at -1 and 0
    ax.set_yticks([-1, 0])
    ax.set_yticklabels([r'$-\varepsilon$', '$0$'])

    # Only middle plot gets y-axis label
    
    ax.set_ylabel(r'$\varepsilon\eta$', labelpad=0)
    ax.yaxis.set_label_coords(-0.05, 0.5) 

    ax.set_title(labels[i], loc='left', x=-0.3, fontsize=9)

print(alpha, eps)

# === Save the figure ===
os.makedirs("outputs", exist_ok=True)
plt.savefig('outputs/max_vs_A_and_stacked_profiles.png', bbox_inches='tight')

