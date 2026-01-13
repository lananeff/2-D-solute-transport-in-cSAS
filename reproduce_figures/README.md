# Reproducing Figures

This directory contains the scripts used to reproduce the figures as they appear in the manuscript  
“How brain pulsations drive solute transport in the cranial subarachnoid space: insights from a toy model”.

The scripts reproduce the final figures and document the modelling, parameter choices, and plotting decisions used in each case.

---

## Repository structure
```
.
├── figures/        # Scripts to generate the final figures in the paper  
├── src/            # Shared parameters, helper functions, and plotting utilities  
├── velocities/     # Analytical velocity profiles derived in the Supplementary Material  
└── outputs/        # Location where generated figures are saved  
```
---

## Figures generated directly from Python
Figures produced in python are summarised here.
Note that figures 8, 9 and 11 require simulation outputs generated from the fenics depository.

### Figure 2 — Transport regime map

Figure 2 shows the Péclet number Pe as the oscillation amplitude A and frequency ω are varied, highlighting distinct transport regimes.

Command:
```bash
python3 figures/pe_A_vs_alpha.py
```


Output:
**outputs/pe_A_vs_alpha_human_mouse.png**

Notes:
- Human and mouse icons are added later in the LaTeX manuscript.

---

### Figure 4 — Velocity profiles

Figure 4 shows spatial profiles of the steady streaming, Stokes drift, and production–drainage velocity fields.

Command:
```bash
python3 figures/flow_profiles.py
```


Output:
**outputs/multiple_flow_profiles.png**

---

### Figure 5 — Velocity magnitudes and dimensional profiles

Figure 5 shows the maximum magnitudes of the velocity components u_i as functions of A, together with representative dimensional Lagrangian velocity profiles.

Command:
```bash
python3 figures/comp_A_and_profiles.py
```

Output:
**outputs/max_vs_A_and_stacked_profiles.png**

Notes:
- This script may take a few minutes to run due to high spatial resolution.

---

### Figure 6 — Steady streaming transport regime

Figure 6 shows the steady-streaming Péclet number as A and α are varied, identifying oscillatory transport regimes.

Command:
```bash
python3 figures/pe_s_regime.py
```

Output:
**outputs/pe_osc_regime_logscale_u.png**

Notes:
- Runtime is a few minutes due to grid resolution.
- Human and mouse annotations are added later in LaTeX.

---

## Figures requiring FEniCS simulations and ParaView

Figures 8, 9, 10, and 11 rely on numerical simulations run using the FEniCS code provided elsewhere in this repository.

General workflow:
1. Run simulations using the FEniCS code.
2. Open .pvd output files in ParaView.
3. Export concentration fields at the required time steps with transparent backgrounds.
4. Save exported images into the appropriate subdirectory under:
   figures/paraview figs/
5. Run the Python post-processing scripts below to assemble the final figures.

---

## Figure 8 — Bolus transport for varying amplitude

Figure 8 shows time snapshots of a bolus transport simulation as the oscillation amplitude A is varied.

Location:
**figures/case_study_1/**

Commands (run sequentially):
```bash
python3 figures/case_study_1/concentration_int.py  
python3 figures/case_study_1/pre_stack.py  
python3 figures/case_study_1/combined_fig.py  
```
Output:
**outputs/case_study_1.png**

Notes:
- Requires pre-saved ParaView images from FEniCS simulations.
- Image paths are assumed to follow the directory structure used in the repository.

---

## Figure 9 — Point leak with and without production–drainage

Figure 9 shows time snapshots and steady states for a point leak, comparing advection by steady streaming alone with advection including production–drainage flow.

Location:
**figures/case_study_2/point_leak_combined/**

Commands (run sequentially):
```bash
python3 figures/case_study_2/point_leak_combined/point_leak_rename.py  
python3 figures/case_study_2/point_leak_combined/pd_point_leak_rename.py  
python3 figures/case_study_2/point_leak_combined/point_leak_combined.py 
``` 

Output:
**outputs/point_leak_combined_0.8.png**

Notes:
- Requires ParaView-exported images from FEniCS simulations.

---

## Figure 10 — Steady-state mass and drainage summary

Figure 10 summarises steady-state solute mass accumulation and arachnoid granulation (AG) drainage across a range of oscillatory Péclet numbers. Results are shown for multiple point-leak locations and compared to a full-brain reference case, with and without production–drainage flow.

Command:
```bash
python3 figures/case_study_2/mass_and_drainage.py
```

Output:
**outputs/concentration/mass_vs_AG_2x2.png**

Notes:
- This script reads simulation outputs from a hard-coded local directory.
- You may need to update base_path to match your file system.

---

## Figure 11 — Steady-state spinal leak comparison

Figure 11 shows steady-state concentration fields for solute leaking from the spinal canal at different Pe_s, comparing cases with and without production–drainage flow.

Location:
**figures/case_study_3/**

Commands (run sequentially):
```bash
python3 figures/case_study_3/spine_leak_rename.py  
python3 figures/case_study_3/pd_spine_leak_rename.py  
python3 figures/case_study_3/spine_leak_combined.py  
```

Output:
**outputs/spine_leak_combined.png**

Notes:
- Requires ParaView-exported steady-state images from FEniCS simulations.
