The scripts are intended to reproduce the figures as they appear in the manuscript and to document the modelling and plotting choices used in each case.

---

## Repository structure
```
.
├── figures/ # Code to generate the final figures included in the paper
├── src/ # Parameters, helper functions, and custom plotting utilities
├── velocities/ # Analytical velocity profiles derived in the Supplementary Material,
│ # used to generate figures
└── outputs/ # Location where generated figures are saved 
```

Figure 2 shows P\'eclet number Pe as amplitude A and frequency \omega are varied. Generate fig 2 from pe_A_vs_alpha.py. Output saves to "outputs/pe_A_vs_alpha_human_mouse.png"

Figure 4 shows plots of the steady streaming, Stokes drift and production--drainge velocity profiles. Generate fig 4 from flow_profiles.py. Flows included for choices of steady streaming, Stokes drift, production--drainage flow and Lagrangian mean velocity can be varied in the script.