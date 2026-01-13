from PIL import Image
import os
import numpy as np
import argparse
from src import stack_images_vertically

# Argument parser to choose mode
parser = argparse.ArgumentParser(description="Combine or copy image pairs for Fig 10.")
parser.add_argument("--mode", choices=["stack", "single"], default="stack",
                    help="Choose 'stack' to combine images or 'single' to copy the first.")
args = parser.parse_args()

# Folder setup
image_folder = "./figures/paraview figs/fig 10/"
output_folder = os.path.join(image_folder, "combos")
os.makedirs(output_folder, exist_ok=True)

# Parameters
pe_vals = [1]
locations = [2, 11, 21, -1]  # Time indices / steps
ts = [0.05, 0.5, 1, "ss"]     # Optional if you want human-readable labels

# Generate image pairs
image_pairs = []
for pe in pe_vals:
    print(f"Pairing images for Pe = {pe}")
    for t, step in zip(ts, locations):
        img1 = os.path.join(image_folder, f"pe_{pe}_t{t}.png")
        img2 = os.path.join(image_folder, f"concentration_int/output_pe{pe}.0_step{step}.png")

        if os.path.exists(img1) and os.path.exists(img2):
            image_pairs.append((img1, img2))
            print(f"Paired Pe {pe}, step {step}")
        else:
            if not os.path.exists(img1):
                print(f"Missing: {img1}")
            if not os.path.exists(img2):
                print(f"Missing: {img2}")

# Combine or copy images
for idx, (img1, img2) in enumerate(image_pairs):
    output_path = os.path.join(output_folder, f"combined_images_{idx + 1}.png")
    print(f"Processing pair {idx + 1}...")

    if args.mode == "stack":
        stack_images_vertically([img1, img2], output_path)
        print(f"Stacked and saved: {output_path}")
    else:
        Image.open(img1).save(output_path)
        print(f"Copied first image: {output_path}")
