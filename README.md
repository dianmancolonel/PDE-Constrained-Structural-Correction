# PDE-Constrained Structural Correction

Official code and data for the paper **"PDE-Constrained Structural Correction: A Physics-Informed Mechanism for Mismatched ODEs in Coupled Systems"**.

## Overview

This repository contains the implementation used to study whether accurate PDE constraints can structurally correct an intentionally mismatched ODE component in a coupled PDE--ODE system using physics-informed neural networks (PINNs).

The main implementation uses two neural networks:

- **UNet** for approximating the PDE state (u(x,t));
- **WNet** for reconstructing the ODE state (w(t)).

The ODE parameter used during training is deliberately mismatched, while the PDE constraint provides an additional optimization signal through the coupling coefficient (alpha(w)).

## Theory and Numerical Implementation

The theoretical analysis in the manuscript adopts the linear coupling

[
alpha(w)=alpha_{min}+(alpha_{max}-alpha_{min})w,
]

which provides a transparent analytical form for studying the correction mechanism and its coupling sensitivity.

The numerical implementation released in this repository uses

[
alpha(w)=alpha_{min}+(alpha_{max}-alpha_{min})w^2.
]

This nonlinear mapping was used in the numerical experiments associated with the reported results and is therefore retained in the released code to support reproducibility. It preserves the essential one-way coupling from the ODE state (w(t)) to the PDE coefficient while introducing a state-dependent coupling sensitivity.

Accordingly, the linear form in the theoretical analysis should be viewed as an analytically convenient prototype of the coupling mechanism, whereas the repository preserves the numerical implementation used for the experiments.

## Main File

- `mem_pinn.py`: main PINN implementation for the coupled PDE--ODE mismatch-correction experiment.

## Requirements

The implementation uses Python with the following main dependencies:

- PyTorch
- NumPy
- SciPy
- Matplotlib

CUDA is used automatically when an available GPU is detected.

## Reproducibility Note

The repository is intended to preserve the numerical implementation used for the reported experiments. Experimental parameters, mismatch settings, network architectures, loss weights, and sampling settings can be modified through `MemristorPINNConfig` and the `main()` function.

## Citation

Citation information will be added after publication.
