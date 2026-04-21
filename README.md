# Uncertainty-quantification-via-conformal-prediction-in-data-assimilation
This repository contains the code for our study on applying Conformal Prediction (CP) to uncertainty quantification in an idealized data assimilation setting using a 1D Modified Shallow Water model. We implement and compare three CP variants - Split Conformal Prediction (SCP), Normalised Conformal Prediction (NCP), and Conformalized Quantile Regression (CQR) and evaluate them using empirical coverage, interval width, Average Interval Score Loss (AISL), and ensemble mean snapshots. We also benchmark against traditional ensemble-based uncertainty measures like standard deviation interval and ensemble spread and investigate how CP-based perturbations influence the DA cycle in two configurations.

├── CQR/                                      # CQR training, inference, and DA scripts
├── DA_CP_CQR/                                # DA cycling with CQR-based perturbations with both configurtaions
├── DA_CP_NCP/                                # DA cycling with NCP-based perturbations with both configurtaions
├── DA_CP_SCP/                                # DA cycling with SCP-based perturbations with both configurtaions
├── AISL_evaluation.ipynb                     # AISL, coverage, and width for all CP methods
├── Coverage_and_Interval_size.ipynb          # Coverage and interval size across all CP methods
├── Plotting_snapshots_CPvsTraditional.ipynb  # Snapshot comparison: CP vs traditional approaches
├── Snapshots_CP.ipynb                        # Snapshots for all CP methods
└── RMSE_CP_NoCP.py                           # RMSE: CP-perturbed DA vs no-CP baseline for both configurtaions

