# 2-D-solute-transport-in-cSAS
# Figures for: “How brain pulsations drive solute
# transport in the cranial subarachnoid space: 
# insights from a toy model”

This repository contains the code used to generate the figures in the paper

> How brain pulsations drive solute transport in the cranial subarachnoid space: insights from a toy model 
> Alannah Neff, Alexandra Vallet, Mariia Dvoriashyna
> <Journal / 2026>

The scripts are intended to run the simulations and reproduce the figures as they appear in the manuscript, and to document the modelling and plotting choices used in each case.

---

## Repository structure
```
.
├── fenics/ # Code to simulate solute transport in the simplified 2-D channel cSAS geometry
├── reproduce_figures/ # Code files to reproduce figures from paper
```
---

## Getting started

From the top-level directory, navigate to either:

```bash
cd fenics
```
or 
```bash
cd reproduce_figures
```

Once in the desired directory, run:
```bash 
source setup.rc
```
You are then ready to run simulations or generate figures. 
