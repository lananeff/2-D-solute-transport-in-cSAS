import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

image_folder = "./figures/paraview figs/fig 8/pd_point_leak_0.8/combos/"  # Go up one level and into the correct folder

# Dynamically list the combined images
image_files = [os.path.join(image_folder, f"combined_images_{i+1}.png") for i in range(12)]



fig = plt.figure(figsize=(6.75, 8))
gs = plt.GridSpec(3, 6, width_ratios=[1, 1, 1, 0.05, 1, 0.05], wspace=0.3)

axes = []
for i in range(3):         # rows
    for j in [0, 1, 2, 4]: # skip 3 (first colorbar) and 5 (second colorbar)
        ax = fig.add_subplot(gs[i, j])
        axes.append(ax)

# Titles and labels
titles = [r"$\tau$ = 0.05", r"$\tau$ = 0.5", r"$\tau$ = 1", "Steady state"]
row_labels = [r"$Pe = 0.1$", r"$Pe = 10$", r"$Pe = 1000$"]
panel_labels = ['(a)', '(b)', '(c)']

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

for i, img_file in enumerate(image_files):
    if os.path.exists(img_file):
        img = Image.open(img_file)
        img = crop_transparency(img)
        axes[i].imshow(img)
        width, height = img.size

    ax = axes[i]
    ax.axis("on")
    ax.set_xticks([width * 0, width * 0.5, width])
    ax.set_xticklabels([0.5, 0.75, 1])
    ax.set_xlabel('$x$')
    ax.set_yticks([height * 0, height * 0.5, height * 1])
    ax.set_yticklabels([0, -0.5, -1])
    ax.set_ylabel('$\eta$')

    if i < 4:
        ax.set_title(titles[i], fontsize=12)
    if i < 8:
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_xlabel('')
    if i % 4 != 0:
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_ylabel('')

for k in range(3):
    y_position =  0.9 - k * 0.27
    fig.text(0.1, y_position, panel_labels[k], fontsize=12, ha='right', va='center', fontweight='bold')

norm_ranges = [
    (0, 0.8),  # Row 1, col 3
    (0, 0.8),  # Row 1, col 4
    (0, 3), # Row 2, col 3
    (0, 3.5), # Row 2, col 4
    (0, 3),  # Row 3, col 3
    (0, 3)   # Row 3, col 4
]

norms = [Normalize(vmin=v[0], vmax=v[1]) for v in norm_ranges]
cmaps = ['coolwarm'] * 3

for row in range(3):
    ax = axes[row * 4 + 3]  # 4th column cbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = ScalarMappable(norm=norms[row * 2 + 1], cmap=cmaps[row])
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label("Concentration", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

for row in range(3):
    ax = axes[row * 4 + 2]  # 3rd column cbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = ScalarMappable(norm=norms[row * 2], cmap=cmaps[row])
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=8)

# Adjust layout
plt.tight_layout()

plt.savefig('outputs/pd_point_leak_different_pe_0.8.png')