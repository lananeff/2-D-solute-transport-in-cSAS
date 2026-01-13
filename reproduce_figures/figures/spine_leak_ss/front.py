import matplotlib.pyplot as plt
import numpy as np
import os
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from src.custom_colormap import custom_plasma
from scipy.optimize import root_scalar
from matplotlib.gridspec import GridSpec

plt.rcParams.update(get_font_parameters())

set_matplotlib_defaults()

final = []
valid_indices = []
base_path = '/home/s2611897/Desktop/FEniCS/Dispersion-2D3D/output/casestudies/spine_leak/steady_state/'

formatted_R = [0.001, 0.002, 0.003, 0.005, 0.007,
                0.011, 0.016, 0.024, 0.036, 0.053,
                0.079, 0.117, 0.174, 0.259, 0.386,
                0.574, 0.853, 1.269, 1.887, 2.807,
                4.175, 6.21, 9.237, 13.738, 20.434,
                30.392, 45.204, 67.234, 100.0]

formatted_R_custom = [
    "1.0em03", "2.0em03", "3.0em03", "5.0em03", "7.0em03",
    "1.1em02", "1.6em02", "2.4em02", "3.6em02", "5.3em02",
    "7.9em02", "1.2em01", "1.7em01", "2.6em01", "3.9em01",
    "5.7em01", "8.5em01", "1.3ep00", "1.9ep00", "2.8ep00",
    "4.2ep00", "6.2ep00", "9.2ep00", "1.4ep01", "2.0ep01",
    "3.0ep01", "4.5ep01", "6.7ep01", "1.0ep02"
]

R_vals =[0.001, 0.002, 0.003, 0.005, 0.007,
                0.011, 0.016, 0.024, 0.036, 0.053,
                0.079, 0.117, 0.174, 0.259, 0.386,
                0.574, 0.853, 1.269, 1.887, 2.807,
                4.175, 6.21, 9.237, 13.738, 20.434,
                30.392, 45.204, 67.234, 100.0]

final = []
valid_indices = []

threshold = 1e-2  # Set your concentration threshold here
x_at_threshold_pd = []

# Loop over R values
for i, R in enumerate(formatted_R_custom):
    file_path = os.path.join(base_path, f"prod_drain/case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_profile_y_-1.00.csv")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=0)
        print(f"Loaded R={R}:", data.shape)

        x_vals = data[:, 0]
        c_vals = data[:, 1]

        # Find first x where concentration exceeds threshold
        above_thresh = np.where(c_vals > threshold)[0]
        if len(above_thresh) > 0:
            x_thresh = x_vals[above_thresh[0]]
        else:
            x_thresh = np.nan  # or set to 1.0 if you want to cap

        x_at_threshold_pd.append(x_thresh)

    except Exception as e:
        print(f"File not found or unreadable at {file_path}: {e}")
        x_at_threshold_pd.append(np.nan)

final = []
valid_indices = []

threshold = 1e-2  # Set your concentration threshold here
x_at_threshold = []

# Loop over R values
for i, R in enumerate(formatted_R):
    x_vals=[]
    file_path = os.path.join(base_path, f"case_R_{R}_dt_0.001_mesh_200_200/case_R_{R}_dt_0.001_mesh_200_200_profile_y_-1.00.csv")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        print(f"Loaded R={R}:", data.shape)
        print(data)
        x_vals = data[:, 0]
        c_vals = data[:, 1]

        # Find first x where concentration exceeds threshold
        above_thresh = np.where(c_vals > threshold)[0]
        
        if len(above_thresh) > 0:
            x_thresh = x_vals[above_thresh[0]]
        else:
            x_thresh = np.nan  # or set to 1.0 if you want to cap

        x_at_threshold.append(x_thresh)

    except Exception as e:
        print(f"File not found or unreadable at {file_path}: {e}")
        x_at_threshold.append(np.nan)

# Plot x position where concentration > threshold vs R
plt.figure(figsize=(6, 3))
plt.plot(R_vals, x_at_threshold)
plt.plot(R_vals, x_at_threshold_pd)
plt.xlabel('$Pe_{osc}$')
plt.ylabel(f'$(x, -1)$ where $c_0$ $>$ {threshold}')
#plt.title('Threshold Front Position vs R')
plt.grid(True)
plt.tight_layout()
plt.xscale('log')
plt.legend()

os.makedirs("outputs/concentration", exist_ok=True)
plt.savefig("outputs/concentration/spine_xloc.pdf", dpi=300)
plt.show()