"""_summary_
    Create custom plasma colormap.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
import numpy as np

def custom_plasma():
    # Define the original plasma colormap
    plasma = plt.colormaps['plasma']

    # Create an array of colors from the plasma colormap, then blend with white
    colors = to_rgba_array(plasma(np.linspace(0, 1, 256)))
    white = np.array([1, 1, 1, 1])  # RGBA for white
    blend_factor = 0.25  # Factor to blend by; 0 is full plasma, 1 is full white

    # Create the blended colors array
    blended_colors = (1 - blend_factor) * colors + blend_factor * white

    # Create a custom colormap from the blended colors
    custom_plasma = LinearSegmentedColormap.from_list("custom_plasma", blended_colors)
    return custom_plasma