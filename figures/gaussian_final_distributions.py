import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from src.custom_colormap import custom_plasma

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

#pe_values = [4.386490844928603, 10.245620840467005, 23.930916561235975,
#             55.895955587083215, 130.5573835000582,
#            304.9456835893186, 712.2681800659341, 1663.660079929753, 
#            3885.8465659601125, 9076.255249703154, 21199.604245673596,
#            49516.37077283392, 115656.45028553471]

pe_values = [55.895955587083215, 130.5573835000582,
            304.9456835893186, 712.2681800659341, 1663.660079929753, 
            3885.8465659601125, 9076.255249703154, 21199.604245673596]

# Base path and file pattern
base_path = '/home/s2611897/Desktop/FEniCS/Dispersion-2D3D/output/averaged_default/iterations/'
file_pattern = '{base_path}/default_case_150_300_{pe}/default_case_150_300_{pe}_concentration.txt'

final = []
valid_indices = []

for i, pe in enumerate(pe_values):
    try:
        file_path = file_pattern.format(base_path=base_path, pe=str(pe))
        print(f'Loading data for pe={pe}')
        data = np.loadtxt(file_path, delimiter=",", skiprows=2)
        finrow = data[-1, 1:]
        final.append(finrow)
        valid_indices.append(i)
    except FileNotFoundError:
        print(f"File for pe={pe} not found. Skipping.")
# Get corresponding pe labels
rounded_labels = [round(pe_values[i], 1) for i in valid_indices]
# 150 evenly spaced x-values between 0 and 1
xs = np.linspace(1/2, 1, 150)

# Choose a manageable number of ticks (e.g., 10)
num_ticks = 10
x_tick_locs = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
x_tick_labels = [round(xs[i], 2) for i in x_tick_locs]

# Plot
plt.figure(figsize=(6, 4))
plt.imshow(final, aspect='auto', cmap = custom_plasma())
plt.yticks(ticks=np.arange(len(rounded_labels)), labels=rounded_labels)
plt.xticks(ticks=x_tick_locs, labels=x_tick_labels)
plt.xlabel(r"$x$")
plt.ylabel(r"$Pe_{osc}$ (rounded)")
plt.colorbar(label="Concentration")
plt.gca().invert_yaxis()
plt.tight_layout()

# Save the plot to the 'outputs' folder
plt.savefig('outputs/gaussian_final_distributions.png')