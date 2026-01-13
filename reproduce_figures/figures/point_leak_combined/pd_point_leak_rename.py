"""
Figure 9 (pre-processing): Rename/copy ParaView snapshots into a numbered sequence.

This script collects ParaView-rendered concentration snapshots for selected
(Pe, t) values and copies them into a `combos/` directory using a consistent
filename pattern:

    combined_images_1.png, combined_images_2.png, ...

These images are then used directly by the final figure assembly script.

Inputs:
- figures/paraview_figs/fig_9/pd_point_leak_0.8/pe_<Pe>_t<t>.png

Output:
- figures/paraview_figs/fig_9/pd_point_leak_0.8/combos/combined_images_<index>.png
"""

from pathlib import Path
from PIL import Image

# --- Configuration ---
image_folder = Path("./figures/paraview figs/fig 9/pd_point_leak_0.8/")
output_folder = image_folder / "combos"
output_folder.mkdir(parents=True, exist_ok=True)

pe_vals = [0.1, 1, 10]
ts = [0.05, 0.5, 1, "ss"]  # time labels as used in filenames

def snapshot_path(pe, t) -> Path:
    """Return the expected filepath for a ParaView snapshot."""
    return image_folder / f"pe_{pe}_t{t}.png"

# --- Collect snapshots in deterministic order ---
snapshots = []
for pe in pe_vals:
    for t in ts:
        p = snapshot_path(pe, t)
        if p.exists():
            snapshots.append((pe, t, p))
        else:
            print(f"[missing] {p}")

# --- Copy/rename into combined_images_*.png ---
for idx, (pe, t, src) in enumerate(snapshots, start=1):
    dst = output_folder / f"combined_images_{idx}.png"
    Image.open(src).save(dst)
    print(f"[saved] {dst}  (from Pe={pe}, t={t})")

