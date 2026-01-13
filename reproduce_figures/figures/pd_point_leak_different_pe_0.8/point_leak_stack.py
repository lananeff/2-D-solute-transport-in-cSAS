from PIL import Image
import os
import numpy as np
import argparse
from src import stack_images_vertically

# Parse command-line argument for mode
parser = argparse.ArgumentParser(description="Combine or copy image pairs.")
parser.add_argument("--mode", choices=["stack", "single"], default="stack",
                    help="Choose whether to 'stack' images or save just the first image in each pair.")
args = parser.parse_args()

# Define folders
image_folder = "./figures/paraview figs/fig 8/pd_point_leak_0.8/"
output_folder = os.path.join(image_folder, "combos")
os.makedirs(output_folder, exist_ok=True)

# Define parameters
pe_vals = [0.1, 1, 10]
location_set = [2, 11, 21, -1]
ts = [0.05, 0.5, 1, "ss"]

# Generate image pairs
image_pairs = []
for idx, pe in enumerate(pe_vals):
    loc_set = location_set
    for t, step in zip(ts, loc_set):
        img1 = os.path.join(image_folder, f"pe_{pe}_t{t}.png")
        img2 = os.path.join(image_folder, f"concentration_int/output_pe{pe}.0_step{step}.png")

        if os.path.exists(img1) and os.path.exists(img2):
            image_pairs.append([img1, img2])
            print(f"Paired {pe}: {t}")

# Process and save output
for idx, (img1, img2) in enumerate(image_pairs):
    output_image = os.path.join(output_folder, f"combined_images_{idx + 1}.png")
    print(f"Processing pair {idx + 1}...")

    if args.mode == "stack":
        stack_images_vertically([img1, img2], output_image)
        print(f"Stacked image saved at: {output_image}")
    else:  # mode == "single"
        Image.open(img1).save(output_image)
        print(f"Single image saved at: {output_image}")
