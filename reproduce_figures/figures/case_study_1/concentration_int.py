"""
Figure 8 (pre-processing): Extract 1-D concentration profiles from simulation output.

This script reads concentration data exported from FEniCS simulations for a Gaussian
bolus case and generates 1-D concentration profiles at selected time steps.
Each profile is saved as a transparent PNG image for later stacking and combination
into the final Fig. 8 composite.

Inputs:
- Concentration data files:
  <base_path>/default_case_100_200_<Pe>/default_case_100_200_<Pe>_concentration.txt

Outputs:
- One PNG per (Pe, time index), saved to:
  figures/paraview_figs/fig_8/concentration_int/

Notes:
- Time steps are selected via row indices in the concentration file.
- Images are saved without axes for clean compositing in later scripts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Parameters for gaussian
sigma_gauss = 0.005
xi_gauss = 0.75
# Base path and file pattern
pe_vals = [0.1,1.0, 10.0]

# Base path where simulation is saved.
base_path = '/home/s2611897/Desktop/FEniCS/Dispersion-2D3D/output/casestudies/gaussian'
file_pattern = '{base_path}/default_case_100_200_{pe_vals}/default_case_100_200_{pe_vals}_concentration.txt'

# Save image location
image_folder = "./figures/paraview figs/fig 8/concentration_int/"

# Define a list of indices (locations) to loop through
locations = [0, 11, 21, 31]  # These indices correspond to the rows in the data pe bigger
#locations = [0, 189, 380, -1] # For pe = 43.86

# Loop through each `pe` value and each location
for pe in pe_vals:
    for loc in locations:
        # Format the file path for the current pe value
        file_path = file_pattern.format(base_path=base_path, pe_vals=str(pe))

        # Load the data from the file
        data = np.loadtxt(file_path, delimiter=",", skiprows=2)


        # Create figure with specified size
        fig, ax = plt.subplots(figsize=(4, 1))  # Adjust (width, height) as needed

        # Plot Gaussian in white (background)
        #ax.plot(x_vals, gaussian, color="red", zorder=0)

        # Plot the specific row based on the current location
        ax.plot(data[loc, 1:], label=f"Row {loc}")

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)  # Hides the entire frame

        # Optional: Hide axes labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(0, len(data[0, 1:]))
        ax.set_ylim(-0.1, 1)

        # Save figure with filename based on the rounded `pe` value and timepoint (location)
        rounded_pe = round(pe, 0)  # Round `pe` value for cleaner file name
        output_image = image_folder + f"output_pe{rounded_pe}_step{loc}.png"
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_image, bbox_inches="tight", dpi=1500, transparent=True)

        print(f"Saved image: {output_image}")

