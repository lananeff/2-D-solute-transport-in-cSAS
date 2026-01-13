import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
from velocities import u0, u1, v0, v1, us, vs, uL, vL, upd, vpd
from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults

# Set plotting defaults
plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

# === Parameters ===
species = "human"
params = PARAMETERS[species]
A, eps, omega, h = params["A"], params["eps"], params["omega"], params["h"]

A = 0.01
tikzfont = 8

# Grids
xn = 20
yn = 20
alphas = np.linspace(1e-1, 1e+1, 50)
xs = np.linspace(0.5, 1, xn)
ys = np.linspace(-1, 0, yn)
X, Y = np.meshgrid(xs, ys)

# Preallocate max arrays
max_u1 = np.zeros(len(alphas))
max_us = np.zeros(len(alphas))
max_v1 = np.zeros(len(alphas))
max_vs = np.zeros(len(alphas))
max_ul = np.zeros(len(alphas))
max_vl = np.zeros(len(alphas))
max_upd = np.zeros(len(alphas))
max_vpd = np.zeros(len(alphas))

# === Loop over A values and compute max magnitudes ===
for k, alpha in enumerate(alphas):
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

# === Create Figure with GridSpec ===
fig_width = 6.75  # inches
fig_height = 2.5  # enough for 3 stacked subplots

fig = plt.figure(figsize=(fig_width, fig_height))
gs = GridSpec(nrows=3, ncols=2, width_ratios=[1.2, 1], height_ratios=[1, 1, 1], figure=fig, wspace=0.4, hspace=0.4)

# === Left: Summary plot spanning all rows ===
ax_summary = fig.add_subplot(gs[:, 0])
ax_summary.plot(alphas, max_u1, label=r'$ \tilde{u}_s$', color='blue', linewidth=1)
ax_summary.plot(alphas, max_us, label=r'$\tilde{u}_{\omega}$', linestyle='--', color='blue', linewidth=1)
ax_summary.plot(alphas, max_ul, label=r'$\tilde{u}_L$', linestyle=':', color='blue', linewidth=2)
ax_summary.plot(alphas, max_upd, label=r'$\tilde{u}_{pd}$', color='blue', linestyle='dashdot', linewidth=1)
A_vals = [alphas[0], alphas[len(alphas)//10], alphas[-1]]  # Low, mid, high
labels = ['(i)', '(ii)', '(iii)']

# for A_val, label in zip(A_vals, labels):
#     ax_summary.axvline(A_val, color='black', linestyle=':', linewidth=1)
#     ax_summary.text(A_val, ax_summary.get_ylim()[1], label,
#                     ha='center', va='bottom')

for A_val, label in zip(A_vals, labels):
    idx = np.argmin(np.abs(alphas - A_val))
    ax_summary.plot(A_val, max_ul[idx], 'o', color='black', markersize=4)
    ax_summary.text(A_val, max_ul[idx]*1.15, label, fontsize=8,
                    ha='center', va='bottom')

ax_summary.set_xscale('log')
ax_summary.set_yscale('log')
ax_summary.set_ylim(1e-9, 1e-3) 
ax_summary.set_xlabel(r'$A$')
ax_summary.set_ylabel('Max value (m/s)')
#ax_summary.set_title('(a)', loc='left', x=-0.5, fontsize=12, fontweight='bold')
ax_summary.legend(loc='lower right')


# === Right: 3 stacked flow magnitude plots ===
A_vals = [alphas[0], alphas[len(alphas)//10], alphas[-1]]  # Low, mid, high
titles = [f'$A = {A:.0e}$' for A in A_vals]
fig.text(0.5, 0.92, '(b)',  fontweight='bold')
fig.text(0, 0.92, '(a)', fontweight='bold')
for i, alpha in enumerate(A_vals):
    ax = fig.add_subplot(gs[i, 1])

    udata = np.zeros_like(X)
    vdata = np.zeros_like(Y)
    # for j in range(len(ys)):
    #     for k in range(len(xs)):
    #         udata[j, k] = A*h*omega*uL(xs[j], ys[i], alpha, A, eps) +  A*h*omega*upd(xs[j], ys[i], alpha, A, eps, omega, h) 
    #         vdata[j, k] = eps*A*h*omega*vL(xs[j], ys[i], alpha, A, eps)  + eps*A*h*omega*vpd(xs[k], ys[j], alpha, A, eps, omega, h)
    for j in range(len(ys)):
        for k in range(len(xs)):
            udata[j, k] = uL(xs[k], ys[j], alpha, A, eps) + upd(xs[k], ys[j], alpha, A, eps, omega, h)
            vdata[j, k] = (vL(xs[k], ys[j], alpha, A, eps) + vpd(xs[k], ys[j], alpha, A, eps, omega, h))


    mag = np.sqrt(udata**2 + (eps*vdata)**2)

    c = ax.pcolormesh(xs, ys, mag, shading='auto', cmap='plasma', vmin=0, vmax=np.max(mag))

    n = 2  # Subsampling rate

    ax.quiver(
        X[::2*n, ::n],
        Y[::2*n, ::n],
        udata[::2*n, ::n],
        (1/10)*vdata[::2*n, ::n],  # Anisotropic scaling (if needed)
        width=0.005,
        headwidth=5,
        headlength=5,
        headaxislength=5,
        color = 'white',
        scale=2.5
    )

    cb = fig.colorbar(c, ax=ax, orientation='vertical', aspect=10)
    cb.set_ticks([0, np.max(mag)])
    cb.set_ticklabels(['0', f'{np.max(mag):.2f}'])
    cb.set_label(r'$|u_L|$', labelpad=0)
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
    ax.set_yticklabels(['$-\epsilon$', '$0$'])

    # Only middle plot gets y-axis label
    
    ax.set_ylabel('$\eta$', labelpad=0)
    ax.yaxis.set_label_coords(-0.05, 0.5) 

    ax.set_title(labels[i], loc='left', x=-0.3, fontsize=9)

# === Save the figure ===
os.makedirs("outputs", exist_ok=True)
plt.savefig('outputs/max_vs_alpha_and_stacked_profiles.png', bbox_inches='tight')
