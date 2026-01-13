# Figure 11 - Steady states for spinal leak for varying $A$
This figure shows the steady states reached by a solute emitted from the spinal canal
for varying $Pe_s$ with and without production--drainage flow.

## Prerequisites
- Simulations run for different $A$ using fenics code.
- The concentration profiles should be saved from paraview at the required timesteps to
```bash
figures/paraview figs/fig 11/
```
with naming convention **pe_{pe_s}_t{time}.png** for steady streaming only and 
**pd_pe_{pe_s}_t{time}.png** for production--drainage flow included. 

## Reproducing the figure

From the reproduce_figures/figures/case_study_2/ directory, run the following scripts in *order*:
```bash
python3 spine_leak_rename.py
python3 pd_spine_leak_rename.py
python3 spine_leak_combined.py
```
## Description of the scripts
- **spine_leak_rename.py**
  Copies and renames steady-state ParaView concentration images for Figure 11.
  The script standardizes file names (`combined_images_*.png`) and places them
  in a single directory for subsequent figure assembly. No image processing
  or stacking is performed.
    Prerequisite:
    - Steady-state ParaView images must already exist as
    `pe_<Pe>_tss.png` in `figures/paraview figs/fig 11/`.

- **pd_spine_leak_rename.py**
  Copies and renames steady-state production–drainage ParaView images for
  Figure 11. The script standardizes file names (`combined_images_*.png`) and
  places them in a common directory for subsequent figure assembly. No image
  processing or stacking is performed.
  Prerequisite:
    - Steady-state ParaView images must already exist as
    `pd_pe_<Pe>_tss.png` in `figures/paraview figs/fig 11/`.

- **spine_leak_combined.py**
  Assembles the final multi-panel figure for Fig. 8 by arranging the vertically
  stacked images produced by `pre_stack.py` into a grid, adding axis labels,
  panel labels, and a shared colour bar.

## Output
The final figure is saved to
```bash
outputs/spine_leak_combined_ss.png
``````