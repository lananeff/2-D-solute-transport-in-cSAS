"""
Prepare steady-state concentration images for Figure 11.

This script copies pre-rendered steady-state ParaView images for different
Péclet numbers and renames them into a consistent sequence for figure assembly.

Specifically:
- Reads images of the form: `pe_<Pe>_tss.png`
- Copies each image into `ss/combos/`
- Renames them as: `combined_images_<index>.png`

No image processing or stacking is performed; this script only standardizes
file names and directory structure for downstream plotting.

Expected input directory:
- figures/paraview figs/fig 11/

Output directory:
- figures/paraview figs/fig 11/ss/combos/
"""
from PIL import Image
import os

# Folder setup
image_folder = "./figures/paraview figs/fig 11/"
output_folder = os.path.join(image_folder, "ss/combos")
os.makedirs(output_folder, exist_ok=True)

# Parameters
pe_vals = [0.1, 1, 10]
ts = ["ss"]  # Only using 'ss' for steady state

# Copy and rename each image
for idx, pe in enumerate(pe_vals):
    t = ts[0]
    img_path = os.path.join(image_folder, f"pe_{pe}_t{t}.png")

    if os.path.exists(img_path):
        output_path = os.path.join(output_folder, f"combined_images_{idx + 1}.png")
        Image.open(img_path).save(output_path)
        print(f"Copied and renamed: {img_path} -> {output_path}")
    else:
        print(f"Missing: {img_path}")
