import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from velocities import u0, u1, v0, v1, us, vs, uL, vL, upd, vpd
import argparse
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from src.custom_colormap import custom_plasma

###################################################
# Get parameters for regime
##################################################

def get_parameters(species, regime=None):
    """Retrieve parameters for the given species and regime."""
    try:
        species_params = PARAMETERS[species]
        if regime is not None:
            regime_params = species_params.get(regime, {})
            return {**species_params, **regime_params}
        return species_params
    except KeyError:
        raise ValueError(f"Invalid species '{species}' or regime '{regime}'")

parser = argparse.ArgumentParser(description="Generate and save leading order CSF flow profile animation.")
parser.add_argument("species", choices=["human", "mouse"], help="Choose species: human or mouse")
parser.add_argument("--regime", choices=["cardiac", "respiratory", "sleep"], required=False, help="Choose flow regime")

args = parser.parse_args()

#######################################################
# Set up figure format
#######################################################
params = get_parameters(args.species, args.regime)

plt.rcParams.update(get_font_parameters())

set_matplotlib_defaults()

# Example values for alpha, A, eps (these might come from params)
alpha, A, eps, omega, h , S = params["alpha"], params["A"], params["eps"], params["omega"], params["h"], params["S"]

scale = 5
n = 100 * scale
newxs = np.linspace(0, 1, n)
newys = np.linspace(-1, 0, n)

new_udata = np.zeros((n,n))
new_vdata = np.zeros((n,n))
new_udata0 = np.zeros((n,n))
new_vdata0 = np.zeros((n,n))

X, Y = np.meshgrid(newxs, newys)


t0 = 5*np.pi/3
for i in range(len(newys)):
        for j in range(len(newxs)):
            new_udata0[i,j] = u0(newxs[j], newys[i], t0 , alpha, A, eps)
            new_vdata0[i,j] = v0(newxs[j], newys[i], t0 , alpha, A, eps)

zeros = np.zeros((n,n))
fig, ax= plt.subplots(figsize=(20,3), constrained_layout=True)
magnitude = np.sqrt(new_udata0**2 + eps*new_vdata0**2)
color_mesh = ax.pcolormesh(newxs, newys, magnitude, shading='auto', cmap=custom_plasma(), vmin=0, vmax=1)

desired_n_x_arrows = 33
n_x = 3*scale # Subsampling rate; increase to reduce density
n_y = n_x*2
pic_eps = 3/20 # From subplot ratio

quiver = ax.quiver(
    X[::n_y, ::n_x], Y[::n_y, ::n_x],
    new_udata0[::n_y, ::n_x], pic_eps * new_vdata0[::n_y, ::n_x],
    scale=20,           # scales all arrow lengths (increase to shrink)
    width=0.0025,       # narrower arrow shafts
    headwidth=2,        # smaller head
    headlength=4,
    headaxislength=4
)

ax.set_xticks([0, 1/2, 1])
ax.set_yticks([-1, -1/2, 0])
ax.set_xticklabels(['0', '1/2', '1'], fontsize=30)
ax.set_yticklabels(['-$\epsilon$', r'-$\frac{\epsilon}{2}$', '0'], fontsize=30)



# Increase colorbar tick label size




cbar = fig.colorbar(color_mesh, ax=ax, orientation='vertical', pad=0.02, aspect=10)
 # Adjust the aspect to fit the figure's layout

cbar.set_label(r"$|\overline{u}_0|$", fontsize=30)
cbar.set_ticks([0, 1])                    # Only show ticks at 0 and 1
cbar.set_ticklabels(['0', '1'])           # Optional: ensure labels are formatted cleanly
cbar.ax.tick_params(labelsize=30)   

# Define output folder and save the plot
output_dir = f"outputs/{args.species}/"
os.makedirs(output_dir, exist_ok=True)

# Time values over one full period
n_frames = 120
t_values = np.linspace(0, 2 * np.pi, n_frames)

# Initialize animation
def update(frame):
    t = t_values[frame]
    for i in range(len(newys)):
        for j in range(len(newxs)):
            new_udata[i, j] = u0(newxs[j], newys[i], t, alpha, A, eps)
            new_vdata[i, j] = v0(newxs[j], newys[i], t, alpha, A, eps)

    magnitude = np.sqrt(new_udata**2 + eps * new_vdata**2)
    color_mesh.set_array(magnitude.ravel())  # .ravel() for pcolormesh

    quiver.set_UVC(new_udata[::n_y, ::n_x], pic_eps * new_vdata[::n_y, ::n_x])
    return color_mesh, quiver

anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

# Save animation
anim.save(f"{output_dir}/animation_u0.gif", dpi=100, writer=PillowWriter(fps=15))