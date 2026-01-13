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

Figure 2 shows P\'eclet number Pe as amplitude A and frequency \omega are varied. Generate fig 2 from python3 figures/pe_A_vs_alpha.py. Output saves to "outputs/pe_A_vs_alpha_human_mouse.png". Note that the little humans and mice are added in latex doc. 

Figure 4 shows plots of the steady streaming, Stokes drift and production--drainge velocity profiles. Generate fig 4 from python3 figures/flow_profiles.py. Output saves to "outputs/multiple_flow_profiles.png". 

Figure 5 shows the maximum magnitudes of u_i and the dimensional profiles associated with Lagrangian velocities for different A. Generate fig 5 from python3 figures/comp_A_and_profiles.py. Output saves to "outputs/max_vs_A_and_stacked_profiles.png". Note: takes a few minutes to run due to resolution.

Figure 6 shows the steady streaming P\'eclet number as A and alpha are varied. Generate fig 6 from python3 figures/pe_s_regime.py. Output saves to "outputs/pe_osc_regime_logscale_u.png". Note: takes a couple of minutes to run due to resolution. Little humans and mice added in latex doc.

Figure 8 



