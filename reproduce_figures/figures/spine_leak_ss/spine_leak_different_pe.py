import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# === Formatting ===
plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

# === Paths to image folders ===
image_folder_standard = "./figures/paraview figs/fig 10/ss/combos/"
image_folder_pd = "./figures/paraview figs/fig 10/ss/pd/combos/"
image_files_standard = [os.path.join(image_folder_standard, f"combined_images_{i+1}.png") for i in range(3)]
image_files_pd = [os.path.join(image_folder_pd, f"combined_images_{i+1}.png") for i in range(3)]

titles = [r"$Pe_{s} = 0.1$", r"$Pe_{s} = 1$", r"$Pe_{s} =10$"]

def crop_transparency(img):
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_array = np.array(img)
    alpha_channel = img_array[:, :, 3]
    non_transparent = np.where(alpha_channel > 0)
    if non_transparent[0].size == 0:
        return img
    y_min, y_max = np.min(non_transparent[0]), np.max(non_transparent[0])
    x_min, x_max = np.min(non_transparent[1]), np.max(non_transparent[1])
    return img.crop((x_min, y_min, x_max, y_max))

# === Set up figure and subplots ===
fig, axes = plt.subplots(2, 3, figsize=(6.75, 16/3))  # slightly taller to fit two rows
axes = axes.flatten()

# === Add images ===
for row, image_files in enumerate([image_files_standard, image_files_pd]):
    for col, img_file in enumerate(image_files):
        ax = axes[row * 3 + col]
        if os.path.exists(img_file):
            img = Image.open(img_file)
            img = crop_transparency(img)
            ax.imshow(img)
            width, height = img.size

            ax.set_xticks([width * 0, width * 0.5, width])
            ax.set_xticklabels([0.5, 0.75, 1])
            ax.set_xlabel('$x$')

            ax.set_yticks([height * 0, height * 0.5, height * 1])
            ax.set_yticklabels([0, -0.5, -1])
            ax.set_ylabel('$\eta$')

            if col != 0:
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel('')


            if row == 0:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel('')
                ax.set_title(titles[col], fontsize=12)

# === Add shared colorbar at bottom ===
cbar_ax = fig.add_axes([0.3, 0.03, 0.5, 0.025])  # [left, bottom, width, height]
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Concentration', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# === Add row labels ===
fig.text(0.1, 0.95, "(a)", fontsize=12, fontweight='bold', va='center')
fig.text(0.1, 0.5, "(b)", fontsize=12, fontweight='bold', va='center')

# === Save figure ===
plt.tight_layout(rect=[0.05, 0.06, 1, 1])  # leave space for colorbar
plt.savefig('outputs/spine_leak_combined_ss.png', dpi=300)

