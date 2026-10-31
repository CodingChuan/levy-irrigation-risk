# Lévy-driven soil-moisture risk and irrigation

Code snapshots and synthetic numerical results supporting a study of first-crossing risk and risk-constrained irrigation under Brownian, mean-drift-matched diffusion, and finite-activity Lévy dynamics. All data in this repository are synthetic model outputs; field measurements are not included.

## Contents

- `data/`: reported results, validation summaries, risk–water curves, and jump-intensity sensitivity results.
- `implementation/bellman/`: finite-chain Bellman and constrained optimization source snapshots.
- `implementation/scalar_pinn/`: scalar PINN and safeguarded optimization source snapshots.
- `implementation/fixed_feature/`: fixed-feature solver and independent evaluation source snapshots.
- `implementation/mean_matched_extension/`, `implementation/risk_water_frontier/`, and `implementation/intensity_sensitivity/`: related comparison and sensitivity source snapshots.

The intensity sensitivity CSV contains all 16 reported model/grid cells, including 8 reused cells, as recorded by the completed analysis. Costs are synthetic model units.

## Scope and reuse

These files are research source snapshots and result summaries, not an independently tested standalone software release. Some scripts rely on project-specific inputs or directory layouts and may require adaptation; no end-to-end reproduction is claimed. Internal provenance, run logs, local path records, and raw field data are not included. Some provenance metadata were omitted from the accompanying archive; the scientific data files were not altered.

The paper is in preparation. No reuse license has yet been specified.
