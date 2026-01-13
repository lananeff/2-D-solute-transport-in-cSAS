"""
Figure 11 (pre-processing): Rename concentration profiles with production--drainage.

This script copies pre-rendered steady-state ParaView images corresponding to
production–drainage simulations at different Péclet numbers and renames them
into a consistent sequence for downstream figure assembly.

Specifically:
- Reads images of the form: `pd_pe_<Pe>_tss.png`
- Copies each image into `ss/pd/combos/`
- Renames them as: `combined_images_<index>.png`

No image manipulation or stacking is performed; the script only standardizes
file names and directory structure for later plotting.

Expected input directory:
- figures/paraview figs/fig 10/

Output directory:
- figures/paraview figs/fig 10/ss/pd/combos/
"""
from PIL import Image
import os

# Folder setup
image_folder = "./figures/paraview figs/fig 11/"
output_folder = os.path.join(image_folder, "ss/pd/combos")
os.makedirs(output_folder, exist_ok=True)

# Parameters
pe_vals = [0.1, 1, 10]
ts = ["ss"]  # Only using 'ss' for steady state

# Copy and rename each image
for idx, pe in enumerate(pe_vals):
    t = ts[0]
    img_path = os.path.join(image_folder, f"pd_pe_{pe}_t{t}.png")

    if os.path.exists(img_path):
        output_path = os.path.join(output_folder, f"combined_images_{idx + 1}.png")
        Image.open(img_path).save(output_path)
        print(f"Copied and renamed: {img_path} -> {output_path}")
    else:
        print(f"Missing: {img_path}")
