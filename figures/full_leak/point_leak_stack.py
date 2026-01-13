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
image_folder = "./figures/paraview figs/fig 8/full_leak/"
output_folder = os.path.join(image_folder, "combos")
os.makedirs(output_folder, exist_ok=True)

# Define parameters
pe_vals = [0.1, 1, 10]
ts = ["ss"]  # Only steady state
wpd = ['', 'pd_']

# Generate image list
image_list = []
for pd_prefix in wpd:
    for pe in pe_vals:
        for t in ts:
            filename = f"{pd_prefix}pe_{pe}_t{t}.png"
            img_path = os.path.join(image_folder, filename)

            if os.path.exists(img_path):
                image_list.append((img_path, pd_prefix, pe, t))
                print(f"Found: {img_path}")
            else:
                print(f"Missing: {img_path}")

# Rename and save with index
for idx, (img_path, pd_prefix, pe, t) in enumerate(image_list):
    output_filename = f"combined_images_{idx + 1}.png"
    output_image = os.path.join(output_folder, output_filename)

    Image.open(img_path).save(output_image)
    print(f"Saved: {output_image}")
