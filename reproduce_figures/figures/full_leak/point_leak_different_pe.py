import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable  # <-- added

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

# === Setup ===
image_folder = "./figures/paraview figs/fig 8/full_leak/combos/"
image_files = [os.path.join(image_folder, f"combined_images_{i+1}.png") for i in range(6)]

titles = [r"$Pe_{osc} = 0.1$", r"$Pe_{osc} = 1$", r"$Pe_{osc} = 10$"]

def crop_transparency(img):
    """Remove transparent background from an RGBA image"""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')  # Ensure image has alpha channel
    
    img_array = np.array(img)  # Convert to NumPy array
    alpha_channel = img_array[:, :, 3]  # Extract alpha channel
    
    # Find bounding box of non-transparent pixels
    non_transparent = np.where(alpha_channel > 0)  
    if non_transparent[0].size == 0:  
        return img  # Return original if fully transparent
    
    y_min, y_max = np.min(non_transparent[0]), np.max(non_transparent[0])
    x_min, x_max = np.min(non_transparent[1]), np.max(non_transparent[1])
    
    # Crop and return image
    return img.crop((x_min, y_min, x_max, y_max))

# === Set up figure and subplots ===
fig, axes = plt.subplots(2, 3, figsize=(6.75, 16/3))
gs = plt.GridSpec(2, 6, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05], wspace=0.3)  # (kept as-is even if unused)
axes = axes.flatten()

# === Add images ===
for idx, img_file in enumerate(image_files):
    row, col = divmod(idx, 3)   # 0–2 → row 0, 3–5 → row 1
    ax = axes[idx]
    if os.path.exists(img_file):
        img = Image.open(img_file)
        img = crop_transparency(img)
        ax.imshow(img)  # not synced with colorbar (as you wanted)
        width, height = img.size

        ax.set_xticks([0, width * 0.5, width])
        ax.set_xticklabels([0.5, 0.75, 1])
        ax.set_xlabel('$x$')

        ax.set_yticks([0, height * 0.5, height * 1])
        ax.set_yticklabels([0, -0.5, -1])
        ax.set_ylabel('$\eta$')

        if col != 0:  # only first column has y-axis labels
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel('')

        if row == 0:  # top row has titles, hides x-labels
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.set_title(titles[col], fontsize=12)

# Per-panel (independent) colorbar ranges (6 panels)
norm_ranges = [
    (0, 5),   # panel 1
    (0, 0.6),  # panel 2
    (0, 0.4),   # panel 3
    (0, 0.1),   # panel 4
    (0, 0.3),   # panel 5
    (0, 0.4)    # panel 6
]
norms = [Normalize(vmin=a, vmax=b) for (a, b) in norm_ranges]
cmaps = ['coolwarm'] * 6  # one entry per panel

# === Add ONE colorbar per subplot (unsynced with imshow) ===
for idx, ax in enumerate(axes):
    row, col = divmod(idx, 3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = ScalarMappable(norm=norms[idx], cmap=cmaps[idx])
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    if col == 2:  # only rightmost column
        cbar.set_label("Concentration", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

# === Add row labels ===
fig.text(0.1, 0.95, "(a)", fontsize=12, fontweight='bold', va='center')
fig.text(0.1, 0.5, "(b)", fontsize=12, fontweight='bold', va='center')

# === Save figure ===
plt.tight_layout(rect=[0.05, 0.06, 1, 1])  # leave space for colorbars
# (your save calls here if needed)


# === Save ===
plt.savefig('outputs/full_steady_states.png', dpi=300)
plt.savefig('outputs/full_steady_states.pdf', dpi=300, bbox_inches='tight')


