# Figure 9 - Time snapshotsand steady states of point leak for varying $A$ 
This figure shows time snapshots and steady states of a point leak concentration field for 
different oscillation amplitudes $A$, based on numerical simulations visualised in ParaView. 
Both steady streaming a=only and steady streaming plus production--drainage flow are plotted.

## Prerequisites
- Simulations run for different $A$ using fenics code.
- The concentration profiles should be saved from paraview at the required timesteps for
steady streaming only
```bash
figures/paraview figs/fig 9/point_leak_0.8/
```
or production--drainage included
```bash
figures/paraview figs/fig 9/pd_point_leak_0.8/
```
with naming convention pe_{pe_s}_t{time}.png.

## Reproducing the figure

From the reproduce_figures/figures/point_leak_combined/ directory, run the following scripts in *order*:
```bash
python3 point_leak_rename.py
python3 pd_point_leak_rename.py
python3 point_leak_combined.py
```
## Description of the scripts
- **point_leak_rename.py**
  Copies and renames ParaView snapshot images for the point-leak case without
  production–drainage flow into a sequential format used to assemble Fig. 9.

- **pd_point_leak_rename.py**
  Copies and renames ParaView snapshot images for the point-leak case with
  production–drainage flow into a sequential format used to assemble Fig. 9.

- **point_leak_combined.py**
  Assembles the final multi-panel figure for Fig. 9, comparing solute transport in
  a point-leak geometry with zero production–drainage flow ($u_{pd}=0$) and with
  non-zero production–drainage flow ($u_{pd}\neq 0$). The script arranges
  ParaView-rendered concentration snapshots into a grid organised by oscillatory
  Péclet number (rows) and time or steady state (columns), and adds row-wise
  colour bars to indicate concentration magnitude.

## Output
The final figure is saved to
```bash
outputs/point_leak_combined_0.8.png
``````