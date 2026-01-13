# Figure 8 - Time snapshots of bolus transport for varying $A$
This figure shows time snapshots of a bolus concentration field for different oscillation amplitudes 
$A$, based on numerical simulations visualised in ParaView.

## Prerequisites
- Simulations run for different $A$ using fenics code.
- The concentration profiles should be saved from paraview at the reuired timesteps to
```bash
figures/paraview figs/fig 8/
```
with naming convention pe_{pe_s}_t{time}.png

## Reproducing the figure

From the reproduce_figures/figures/ directory, run the following scripts in *order*:
```bash
python3 concentration_int.py
python3 pre_stack.py
python3 combined_fig.py
```
## Description of the scripts
- **concentration_init.py**
  Reads concentration data files from FEniCS simulations and exports 1-D
  concentration profiles at selected time indices as transparent PNG images
  for use in Fig. 8.

- **pre_stack.py**
  Combines ParaView-rendered concentration fields and 1-D concentration profiles
  (from `concentration_int.py`) into vertically stacked images that serve as
  intermediate panels for Fig. 8. Note: This script assumes a fixed file-naming convention for 
  ParaView snapshot images and concentration profile outputs.

- **combined_fig.py**
  Assembles the final multi-panel figure for Fig. 8 by arranging the vertically
  stacked images produced by `pre_stack.py` into a grid, adding axis labels,
  panel labels, and a shared colour bar.

## Output
The final figure is saved to
```bash
outputs/case_study_1.png
``````