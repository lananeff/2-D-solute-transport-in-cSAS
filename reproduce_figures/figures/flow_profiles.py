"""
This script generates and saves flow profiles based on different velocity types for a given species and regime.

The available velocity types are:
    - "leading_order"
    - "steady_streaming"
    - "stokes_drift"
    - "lagrangian_velocity"

To run this script, use the following command line format:

    python3 figures/flow_profiles.py <species> --regime <regime> --velocity_type <velocity_type>

Where:
    <species>      - Choose between 'human' or 'mouse' (e.g., "human").
    <regime>       - Choose a flow regime: 'cardiac', 'respiratory', or 'sleep'.
    <velocity_type> - Choose the type of velocity to plot:
                        "leading_order", "steady_streaming", "stokes_drift", "lagrangian_velocity", or "pd" (production drainage).

Example usage:
    python3 figures/flow_profiles.py human --regime cardiac --velocity_type steady_streaming
    python3 figures/flow_profiles.py mouse --regime respiratory --velocity_type stokes_drift

The script will generate a figure and save it under the 'outputs' directory in a subfolder structure based on species and regime.

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

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


###################################################
# Get parameters for regime
##################################################

def get_parameters(species, regime=None):
    """Retrieve parameters for the given species and regime."""
    try:
        species_params = PARAMETERS[species]
        if regime:
            regime_params = species_params.get(regime, {})
            return {**species_params, **regime_params}
        return species_params
    except KeyError:
        raise ValueError(f"Invalid species '{species}' or regime '{regime}'")

#######################################################
# Set up figure format
#######################################################


plt.rcParams.update(get_font_parameters())

set_matplotlib_defaults()


###################################################################
def compute_velocity_profile(velocity_type, xs, ys, alpha, A, eps, omega, h, t=0, n=100):
    """Compute the flow profile based on velocity type"""
    udata = np.zeros((n,n))
    vdata = np.zeros((n,n))

    for i in range(len(ys)):
        for j in range(len(xs)):
            if velocity_type == "leading_order":
                udata[i,j] = u0(xs[j], ys[i], t, alpha, A, eps)
                vdata[i,j] = v0(xs[j], ys[i], t, alpha, A, eps)
            elif velocity_type == "steady_streaming":
                udata[i,j] = u1(xs[j], ys[i], alpha, A, eps)
                vdata[i,j] = v1(xs[j], ys[i], alpha, A, eps)
            elif velocity_type == "stokes_drift":
                udata[i,j] = us(xs[j], ys[i], alpha, A, eps)
                vdata[i,j] = vs(xs[j], ys[i], alpha, A, eps)
            elif velocity_type == "lagrangian_velocity":
                udata[i,j] = uL(xs[j], ys[i], alpha, A, eps)
                vdata[i,j] = vL(xs[j], ys[i], alpha, A, eps)
            elif velocity_type == "pd":
                udata[i,j] = upd(xs[j], ys[i], alpha, A, eps, omega, h)
                vdata[i,j] = vpd(xs[j], ys[i], alpha, A, eps, omega, h)
            elif velocity_type == "lag_pd":
                udata[i,j] = upd(xs[j], ys[i], alpha, A, eps, omega, h) + uL(xs[j], ys[i], alpha, A, eps)
                vdata[i,j] = vpd(xs[j], ys[i], alpha, A, eps, omega, h) + vL(xs[j], ys[i], alpha, A, eps)
    
    return udata, vdata

def plot_velocity_profile(velocity_type, species, regime, alpha, A, eps, omega, h, t=0, n=100):
    """Generate and save a flow profile based on the velocity type."""

    # Define spatial grid
    xs = np.linspace(0, 1, n)
    ys = np.linspace(-1, 0, n)
    X, Y = np.meshgrid(xs, ys)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(5, 0.5))

    # Compute data
    velocity_data = compute_velocity_profile(velocity_type, xs, ys, alpha, A, eps, omega, h)

    magnitude = np.sqrt(velocity_data[0]**2 + eps*velocity_data[1]**2)
    color_mesh = ax.pcolormesh(xs, ys, magnitude, shading='auto', cmap=custom_plasma())

    if velocity_type == "leading_order":
        n = 9  # Subsampling rate; increase to reduce density
        quiver = plt.quiver(X[::n, ::n], Y[::n, ::n], velocity_data[0][::n, ::n], velocity_data[1][::n, ::n],
                   width=0.0035,  # Arrow shaft width
                   headwidth=3,  # Width of the arrow head
                   headlength=5,  # Length of the arrow head
                   headaxislength=4.5)
    else:
        n=3
        quiver = plt.quiver(X[::n, ::n], Y[::n, ::n], velocity_data[0][::n, ::n], velocity_data[1][::n, ::n],
                   width=0.001,  # Arrow shaft width
                   headwidth=3,  # Width of the arrow head
                   headlength=5,  # Length of the arrow head
                   headaxislength=4.5)

    

    ax.set_xticks([0, 1/2, 1])
    ax.set_yticks([-1, -1/2, 0])
    ax.set_xticklabels(['0', '1/2', '1'])
    ax.set_yticklabels([r'-$\varepsilon$', r'-$\varepsilon/2$', '0'])

    cbar = fig.colorbar(color_mesh, ax=ax, aspect=10, orientation='vertical') # Adjust the aspect to fit the figure's layout

    # Set colorbar label based on velocity type
    if velocity_type == "steady_streaming":
        cbar_label = r"$|\overline{u}_s|$"
    elif velocity_type == "stokes_drift":
        cbar_label = r"$|\overline{u}_{\omega}|$"
    elif velocity_type == "lagrangian_velocity":
        cbar_label = r"$|\overline{u}_L|$"
    elif velocity_type == "pd":
        cbar_label = r"$|\overline{u}_{pd}|$"
    elif velocity_type == "lag_pd":
        cbar_label = r"$|\overline{u}_L|$"
    else:
        cbar_label = "Magnitude"  # Default label

    #cbar.set_ticklabels([r'4 ', r'$2 $', r'$0$'])
    cbar.formatter.set_powerlimits((0,0))
    cbar.update_ticks()


    # Define the subfolder path based on species and regime
    # Create output directory dynamically
    output_dir = f"outputs/{species}/{regime}/flow_profiles"
    os.makedirs(output_dir, exist_ok=True)

    if velocity_type == 'leading_order':
        plt.savefig(f"{output_dir}/{velocity_type}_flow_t={t}.png")
    else:
        plt.savefig(f"{output_dir}/{velocity_type}_flow.png")

def plot_multiple_velocity_profiles(velocity_types, species, regime, alpha, A, eps, omega, h, S, t=0, n=100):
    """Generate and save a figure with multiple flow profiles based on different velocity types."""

    # Define spatial grid
    xs = np.linspace(0, 1, n)
    ys = np.linspace(-1, 0, n)
    X, Y = np.meshgrid(xs, ys)

    # Create a figure and set up subplots
    num_plots = len(velocity_types)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots*1 +1))  # Adjust size based on number of subplots
    plt.tight_layout()

    # If only one subplot, make sure axes is iterable (to avoid errors)
    if num_plots == 1:
        axes = [axes]

    # scale_dict = {
    #     'leading_order': 1.5,
    #     'steady_streaming': 0.25,
    #     'stokes_drift': 0.015,
    #     'lagrangian_velocity': 1.5,
    #     'pd': 10
    # }

    scale_dict = {
        'leading_order': 1.5,
        'steady_streaming': 2,
        'stokes_drift': 0.3,
        'lagrangian_velocity': 1.5,
        'pd': 5
    }

    for idx, velocity_type in enumerate(velocity_types):
        velocity_data = compute_velocity_profile(velocity_type, xs, ys, alpha, A, eps, omega, h)
        magnitude = np.sqrt(velocity_data[0]**2 + (eps * velocity_data[1])**2)

        color_mesh = axes[idx].pcolormesh(xs, ys, magnitude, shading='auto', cmap=custom_plasma())

        n = 3  # Subsampling rate

        axes[idx].quiver(
            X[::2*n, ::n],
            Y[::2*n, ::n],
            velocity_data[0][::2*n, ::n],
            (1/10) * velocity_data[1][::2*n, ::n],  # Anisotropic scaling (if needed)
            width=0.00125,
            headwidth=3,
            headlength=3,
            headaxislength=3,
            color = 'black',
            scale=scale_dict[velocity_type]
        )

        axes[idx].set_xticks([])
        axes[idx].set_yticks([-1, -1/2, 0])
        axes[idx].set_ylabel(r'$\varepsilon\eta$', fontsize=11)
        
        axes[idx].set_yticklabels([r'-$\varepsilon$', r'-$\varepsilon/2$', '0'], fontsize=10)
        #axes[idx].set_title(velocity_type.replace("_", " ").title())

        # Set colorbar label based on velocity type
        if velocity_type == "steady_streaming":
            cbar_label = r"$|\boldsymbol{u}_s|$"
        elif velocity_type == "stokes_drift":
            cbar_label = r"$|\boldsymbol{u}_{\omega}|$"
        elif velocity_type == "lagrangian_velocity":
            cbar_label = r"$|\boldsymbol{u}_s + \boldsymbol{u}_{\omega}|$"
        elif velocity_type == "pd":
            cbar_label = r"$|\boldsymbol{u}_{pd}|$"
        elif velocity_type == "lag_pd":
            cbar_label = r"$|\boldsymbol{u}_L|$"
        else:
            cbar_label = "Magnitude"  # Default label

        # Add colorbar
        cbar = fig.colorbar(color_mesh, ax=axes[idx], aspect=10, orientation='vertical')
        cbar.set_label(cbar_label, fontsize=11)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        cbar.ax.yaxis.get_offset_text().set_fontsize(10)

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    for ax, label in zip(axes, labels):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')


    axes[-1].set_xticks([0, 1/2, 1])
    axes[-1].set_xticklabels(['0', '1/2', '1'], fontsize = 10)
    axes[-1].set_xlabel(r'$x$', fontsize=11)

    # Define output folder and save the plot
    output_dir = f"outputs/{species}/{regime}/flow_profiles"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()  # To ensure proper spacing between subplots
    plt.savefig(f"{output_dir}/multiple_flow_profiles.png", dpi=300)

    print(A**2 * alpha**4 * S)

######################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save CSF flow profiles.")
    parser.add_argument("species", choices=["human", "mouse"], help="Choose species: human or mouse")
    parser.add_argument("--regime", choices=["cardiac", "respiratory", "sleep"], required=True, help="Choose flow regime")
    parser.add_argument("--velocity_type", choices=["leading_order", "steady_streaming", "stokes_drift", "lagrangian_velocity", "pd", "lag_pd"], required=True, help="Select velocity type")
    parser.add_argument("--leading_order_snapshots", type=int, default=0,
                    help="Number of time snapshots to plot for leading order velocity (between 0 and 2pi)")

    args = parser.parse_args()
    
    
    # Retrieve parameters for the given species and regime
    params = get_parameters(args.species)
    
    # Example values for alpha, A, eps (these might come from params)
    alpha, A, eps, omega, h , S = params["alpha"], params["A"]*10, params["eps"], params["omega"], params["h"], params["S"]

    # If multiple velocity types are provided, use the multiple plotting function
    velocity_types = args.velocity_type.split(',')

    if len(velocity_types) == 1:
        # Call the single velocity type plot function
        plot_velocity_profile(velocity_types[0], args.species, args.regime, alpha, A, eps, omega, h)
    else:
        # Call the multiple velocity types plot function
        plot_multiple_velocity_profiles(velocity_types, args.species, args.regime, alpha, A, eps, omega, h)

    # Example usage:
    plot_multiple_velocity_profiles(velocity_types=["steady_streaming", "stokes_drift", "pd"], species=args.species, regime=None, alpha=alpha, A=A, eps=eps, omega=omega, h=h, S =S)
    print(A, alpha, eps)

    plot_velocity_profile(velocity_types[0], args.species, args.regime, alpha, A*10, eps, omega, h)

   