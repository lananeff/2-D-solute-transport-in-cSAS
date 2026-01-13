import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base path and file pattern

pe_vals = [0.1,10.000000000000002, 1000.0000000000003]
base_path = '/home/s2611897/Desktop/FEniCS/Dispersion-2D3D/output/1007/casestudies/point_leak_proddrain'
file_pattern = '{base_path}/default_case_100_200_{pe_vals}/default_case_100_200_{pe_vals}_concentration.txt'

# Save image location
image_folder = "./figures/paraview figs/fig 8/pd_point_leak/concentration_int/"

# Define a list of indices (locations) to loop through
locations = [2, 11, 21, -1]  # These indices correspond to the rows in the data
#locations = [6, 190, 380, -1]
#locations = [6, 191, 378, -1]

# Loop through each `pe` value and each location
for pe in pe_vals:
    for loc in locations:
        # Format the file path for the current pe value
        file_path = file_pattern.format(base_path=base_path, pe_vals=str(pe))

        # Load the data from the file
        data = np.loadtxt(file_path, delimiter=",", skiprows=2)

        # Create figure with specified size
        fig, ax = plt.subplots(figsize=(4, 1))  # Adjust (width, height) as needed

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
        ax.set_ylim(-0.1, 10)

        # Save figure with filename based on the rounded `pe` value and timepoint (location)
        rounded_pe = round(pe, 0)  # Round `pe` value for cleaner file name
        output_image = image_folder + f"output_pe{rounded_pe}_step{loc}.png"
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_image, bbox_inches="tight", dpi=1500, transparent=True)

        print(f"Saved image: {output_image}")

