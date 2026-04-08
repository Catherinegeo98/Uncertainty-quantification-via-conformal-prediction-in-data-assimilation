# Uncertainty-quantification-via-conformal-prediction-in-data-assimilation
This repository contains the code and experiments for our study on applying Conformal Prediction (CP) to uncertainty quantification in a controlled, idealized data assimilation setting. The goal is to assess whether CP, a novel machine learning (ML) framework, can provide reliable uncertainty estimates for nonlinear dynamical systems.
We conduct our experiments using a 1D Modified shallow water toy model that mimics convective processes, allowing for a clear and interpretable evaluation of uncertainty behavior. 

We implement and compare three variants of CP:
Standard Conformal Prediction (SCP),
Normalized Conformal Prediction (NCP) and
Conformalized Quantile Regression (CQR).

These methods are evaluated using key uncertainty metrics, including:
Empirical coverage,
Interval width,
Average Interval Score Loss (AISL) and
Ensemble mean snapshots at different timesteps.

In addition to CP-based approaches, we benchmark against traditional ensemble-based uncertainty measures, such as standard deviation intervals and ensemble spread, to understand the relative strengths of each method.
A central component of this work is the integration of CP-derived uncertainty into the data assimilation (DA) cycle. We investigate how introducing CP-based perturbations influences the evolution of uncertainty during forecast–analysis updates.

The repository provides:
Implementation of CP methods (SCP, NCP, CQR),
Tools for computing uncertainty metrics (coverage, AISL, interval width, snapshots).


