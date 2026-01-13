from PIL import Image
import os
import numpy as np
from src import stack_images_vertically

# Define image folder
image_folder = "./figures/paraview figs/fig 6/"
output_folder = os.path.join(image_folder, "combos")
os.makedirs(output_folder, exist_ok=True)

# Define pe values and locations
pe_vals = [0.1, 1, 10]
#pe_vals = [1, 10, 100]

locations = [0, 11, 21, 31]  # locations for 438 and 4386
alt_locations = [0, 11, 21, 31]  # Alternate set for first pe value
ts = [0, 0.5, 1, 1.5]

# Generate image pairs dynamically
image_pairs = []
for idx, pe in enumerate(pe_vals):
    loc_set = locations 
    for t, step in zip(ts, loc_set):
        img1 = os.path.join(image_folder, f"pe_{pe}_t{t}.png")
        img2 = os.path.join(image_folder, f"concentration_int/output_pe{pe}.0_step{step}.png")
        
        print(f"Checking PE={pe}, t={t}, step={step}")
        print(f"Looking for: {img1}")
        print(f"           + {img2}")
        if os.path.exists(img1) and os.path.exists(img2):
            image_pairs.append([img1, img2])

# Loop over each pair, process and save the combined image
for idx, pair in enumerate(image_pairs):
    output_image = os.path.join(output_folder, f"combined_images_{idx + 1}.png")
    print(f"Combining pair {idx + 1}...")
    
    # Stack images and save
    stack_images_vertically(pair, output_image)
    
    print(f"Elements combined. Output saved at: {output_image}")