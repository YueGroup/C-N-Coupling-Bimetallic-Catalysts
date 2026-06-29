# Electrochemical C–N Coupling on Bimetallic Catalysts

This repository contains the machine-learning and density functional theory (DFT) workflows used to investigate electrochemical C–N coupling on A3B L12 bimetallic catalysts. The workflow combines high-throughput thermodynamic screening, transition-state calculations, vibrational analyses, and DFT validation to identify activity trends governing C–N bond formation.

## Overview

The computational workflow consists of:

1. **High-throughput thermodynamic screening** of 80 A3B L1₂ bimetallic alloys using the UMA-m-1p1 machine-learning interatomic potential.
2. **DFT calculations** for lattice constant determination, adsorption-energy validation, and reaction energetics.
3. **Transition-state searches** for electrochemical C–N coupling pathways.
4. **Vibrational frequency analyses** for transition-state validation and free-energy corrections.

## Repository Structure

```text
.
├── A3B.npy
├── README.md
├── scripts/
│   ├── adsorbate.py
│   ├── transition_state/
│   └── frequency/
├── inputs/
│   ├── adsorption/
│   ├── neb/
│   └── frequency/
└── results/
```

## Main Files

### A3B.npy

Database of all A3B L1₂ alloy candidates screened in this study. The file contains alloy compositions and DFT-relaxed lattice constants used for alloy structure generation, adsorption calculations, and reaction-energy evaluations.

### adsorbate.py

Generates A3B(111) alloy surfaces, enumerates adsorption sites using CatKit, performs structural relaxations with the FAIRChem UMA-m-1p1 potential, and exports optimized structures for thermodynamic screening.

### transition_state/

Representative workflows and input files for locating C–N coupling transition states, including initial-state preparation, reaction-path optimization, and DFT single-point energy refinement.

### frequency/

Representative workflows and input files for vibrational frequency calculations used to verify transition states and obtain zero-point energy and entropic corrections.

## Software Requirements

### Python Environment

* Python 3.11+
* NumPy
* ASE
* CatKit
* PyTorch
* FAIRChem (version ≥ 2.0.0)

### Electronic Structure Software

* VASP 6.2.1

VASP was used for:

* DFT lattice-constant calculations of A3B alloys
* Validation of machine-learning adsorption and reaction energetics
* Geometry optimization of initial and final states along C–N coupling pathways
* Single-point energy corrections on transition-state


