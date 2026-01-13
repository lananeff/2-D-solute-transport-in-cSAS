import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from velocities import u0, u1, v0, v1, us, vs, uL, vL, upd, vpd
import argparse
from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults

def get_parameters(species, regime=None, A=None, alpha=None, A_scale=None):
    try:
        species_params = PARAMETERS[species]
        
        if regime is not None:
            regime_params = species_params.get(regime, {})
            params = {**species_params, **regime_params}
        else:
            params = species_params.copy()

        if alpha is not None:
            params["alpha"] = alpha
        
        if A is not None:
            params["A"] = A

        if A_scale is not None:
            if "A" not in params:
                raise ValueError("Cannot scale A: base value is missing.")
            params["A"] *= A_scale

        return params

    except KeyError:
        raise ValueError(f"Invalid species ('{species}') or regime ('{regime}')")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Select CSF transport parameters.")
parser.add_argument("species", choices=["human", "mouse"], help="Choose species: human or mouse")
parser.add_argument("--regime", choices=["cardiac", "respiratory", "sleep"], default=None,
                    help="Optional: choose flow regime (cardiac, respiratory, or sleep)")
parser.add_argument("--A_scale", type=float, default=None,
                    help="Optional: scale factor for A (e.g., 0.1, 1.0, 10)")

args = parser.parse_args()
params = get_parameters(args.species, args.regime)

print(f"Selected parameters for {args.species}, {args.regime if args.regime else 'all'}:")
for key, value in params.items():
    print(f"{key}: {value}")

# Set pars
alpha = params["alpha"]
eps = params["eps"]
omega = params["omega"]
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

As = np.linspace(1e-3,1e-1,50)
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


# Iterate over each alpha value
for k, A in enumerate(As):
    udata1 = np.zeros((100, 100))
    usdata = np.zeros((100, 100))
    vdata1 = np.zeros((100, 100))
    vsdata = np.zeros((100, 100))
    uldata = np.zeros((100,100))
    vldata = np.zeros((100,100))
    upddata = np.zeros((100,100))
    vpddata = np.zeros((100,100))

    # Compute the function values over the grid
    # for i in range(len(ys)):
    #    for j in range(len(xs)):
    #        udata1[i, j] = A*h*omega*u1(xs[j], ys[i], alpha, A, eps)
    #        usdata[i, j] = A*h*omega*us(xs[j], ys[i], alpha, A, eps)
    #        vdata1[i, j] = A*h*omega*v1(xs[j], ys[i], alpha, A, eps) 
    #        vsdata[i, j] = A*h*omega*vs(xs[j], ys[i], alpha, A, eps)
    #        uldata[i,j] = A*h*omega*uL(xs[j], ys[i], alpha, A, eps)
    #        vldata[i,j] = A*h*omega*vL(xs[j], ys[i], alpha, A, eps) 
    #        upddata[i,j] = A*h*omega*upd(xs[j], ys[i], alpha, A, eps, omega, h)
    #        vpddata[i,j] = eps * A*h*omega*vpd(xs[j], ys[i], alpha, A, eps, omega, h)

    for i in range(len(ys)):
        for j in range(len(xs)):
            udata1[i, j] = u1(xs[j], ys[i], alpha, A, eps)
            usdata[i, j] = us(xs[j], ys[i], alpha, A, eps)
            vdata1[i, j] = v1(xs[j], ys[i], alpha, A, eps) 
            vsdata[i, j] = vs(xs[j], ys[i], alpha, A, eps)
            uldata[i,j] = uL(xs[j], ys[i], alpha, A, eps) + upd(xs[j], ys[i], alpha, A, eps, omega, h)
            vldata[i,j] = vL(xs[j], ys[i], alpha, A, eps) + vpd(xs[j], ys[i], alpha, A, eps, omega, h)
            upddata[i,j] = upd(xs[j], ys[i], alpha, A, eps, omega, h)
            vpddata[i,j] = vpd(xs[j], ys[i], alpha, A, eps, omega, h)

    # Find the maximum values for each function
    max_u1[k] = np.max(udata1)
    max_us[k] = np.max(usdata)
    max_v1[k] = np.max(vdata1)
    max_vs[k] = np.max(vsdata)
    max_ul[k] = np.max(uldata)
    max_vl[k] = np.max(vldata)
    max_upd[k] = np.max(upddata)
    max_vpd[k] = np.max(vpddata)

plt.plot(As, max_u1, label='$ u_s$', color='red')
plt.plot(As, max_us, label=r'$u_{\omega}$', linestyle='--', color='red')
plt.plot(As, max_v1, label='$v_s$', color='blue')
plt.plot(As, max_vs, label=r'$v_{\omega}$', linestyle='--', color='blue')
plt.plot(As, max_ul, label='$u_L$', linestyle = ':', color='red', linewidth =4)
plt.plot(As, max_vl, label='$v_L$', linestyle = ':', color='blue', linewidth =4)
plt.plot(As, max_upd, label='$u_{pd}$', color = 'red', linestyle = 'dashdot')
plt.plot(As, max_vpd, label='$v_{pd}$', color = 'blue', linestyle = 'dashdot')

plt.xlabel(r'$A$')
plt.ylabel('Max Value')
plt.yscale('log')
plt.xscale('log')
plt.legend()

# Save the plot to the 'outputs' folder
plt.savefig('outputs/comparison_u1_vs_us_varying_A.png')
