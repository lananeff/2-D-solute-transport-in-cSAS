import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

image_folder = "./figures/paraview figs/fig 10/combos/"
image_files = [os.path.join(image_folder, f"combined_images_{i+1}.png") for i in range(4)]

fig = plt.figure(figsize=(6.75, 2.75))
#gs = plt.GridSpec(1, 6, width_ratios=[1, 1, 1, 0.05, 1, 0.05], wspace=0.3)
gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)

axes = []
# Add figures at columns 0, 1, 2, 4 (skip 3 and 5 for colorbars)
# for j in [0, 1, 2, 4]:
#     ax = fig.add_subplot(gs[0, j])
#     axes.append(ax)

for j in [0, 1, 2, 3]:
    ax = fig.add_subplot(gs[0, j])
    axes.append(ax)

titles = [r"$\tau$ = 0.05", r"$\tau$ = 0.5", r"$\tau$ = 1", "Steady state"]
norm_ranges = [(0, 1), (0, 1), (0, 2), (0, 9)]  # Custom vmin/vmax for each image
norms = [Normalize(vmin=v[0], vmax=v[1]) for v in norm_ranges]
cmaps = ['coolwarm'] * 4  # Color maps for each image

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

# Add images
for i, img_file in enumerate(image_files):
    if os.path.exists(img_file):
        img = Image.open(img_file)
        img = crop_transparency(img)
        axes[i].imshow(img)
        width, height = img.size

        ax = axes[i]
        ax.set_title(titles[i], fontsize=12)
        ax.set_xticks([width * 0, width * 0.5, width])
        ax.set_xticklabels([0.5, 0.75, 1])
        ax.set_yticks([height * 0, height * 0.5, height * 1])
        ax.set_yticklabels([0, -0.5, -1])
        ax.set_xlabel('$x$')
        ax.set_ylabel('$\eta$')

        if i != 0:
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_ylabel('')

# Add colorbars to columns 3 and 5
# for i, col in enumerate([3, 5]):
#     ax = axes[2 + i]  # colorbars for fig 3 and 4
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     sm = ScalarMappable(norm=norms[2 + i], cmap=cmaps[2 + i])
#     sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    
#     cbar.ax.tick_params(labelsize=8)

#     if col ==5:
#         cbar.set_label("Concentration", fontsize=8)

# === Add colorbar at the bottom center ===
cbar_ax = fig.add_axes([0.25, -0.125, 0.5, 0.05])  # [left, bottom, width, height]
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])  # Needed for matplotlib < 3.1

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Concentration', fontsize=10)
cbar.ax.tick_params(labelsize=9)


plt.tight_layout()
plt.savefig('outputs/spine_leak_different_pe.png')
