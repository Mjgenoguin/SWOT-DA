# How the Upper Nan runs used the SWOT observations

This page documents the companion RRI particle-filter software. The two Python modules in this directory implement the observation side of the update. Running the full experiment also requires the RRI executable, the external RRI-PF driver and its dependencies, a configured model domain, rainfall, processed cell observations, fitted operator tables and model states. Reproducing Figures 6–9 from the summary tables requires only the [figure instructions](../README.md#reproduce-the-figures).

## From an overpass to the next generation of particles

| Component in the archived run | Responsibility |
| --- | --- |
| `InitialConfig.py` and `StateSpace.py` | Initialize the 36 realizations and apply the three controls: a multiplier on initial river depth and longitudinal and latitudinal shifts of the rainfall grid. The protected open-loop realization uses a multiplier of one and zero displacement. |
| `Simulation.py` and `main.py` | Advance the simulation through daily 07:00 ICT analysis checkpoints. For an eligible overpass, the driver selects the hourly RRI output nearest the acquisition and supplies the processed SWOT cells and particle outputs to the observation operator. |
| [`swot_rri_transfer_operator.py`](swot_rri_transfer_operator.py) and [`swot_obs_operator.py`](swot_obs_operator.py) | Apply the fitted cell-specific elevation-to-depth offset, assemble the observation uncertainty, and sample model river depth at the same grid cells for each member. The resulting cells stay in the same order in the target vector, uncertainty vector and model matrix. |
| `ParticleFilter.py` | Assign each complete particle a single weight from its agreement with the accepted cells. Use those weights to compute the ensemble mean and allocate parents for the following cycle. |
| `main.py`, the state-handling helpers and the RRI executable | Carry each selected parent's hydraulic state forward, apply the perturbed controls, and propagate the resulting realizations. Gauge measurements at NAN001–NAN003 enter the subsequent evaluation. They were excluded from the SWOT weighting and had informed the earlier RRI calibration. |

For the full-profile experiment, let `r[i,j]` be particle `i`'s modeled depth minus the transformed SWOT target at cell `j`, and `sigma[j]` the total uncertainty in metres. The driver used one diagonal variance per accepted cell. With `m` accepted cells, the filter evaluated `J[i] = sum((r[i,j] / sigma[j])**2 for j in cells) / m` and normalized `exp(-J[i]/2)` across the 36 members. The weighted ensemble mean is the reported estimate. The D’Hondt rule reserved one offspring slot for the original open-loop member and allocated the remaining slots by weight and existing offspring counts. Recalculating all 61 principal update weights from the archived arrays gave a maximum absolute difference of 8.33 × 10⁻¹⁷ from the saved weights and reproduced the offspring counts at every update.

## Which configuration corresponds to which result?

| Experiment | Operator fitting period | SWOT information used for each update |
| --- | --- | --- |
| Principal 2024–2026 results, including Figures 6–8 | Available 2023–2026 pairs | Full accepted profile (`GLOBAL`). These evaluations include fitting-period overlap. |
| Paired 2026 comparison, Figure 9 | Same 2023–2026 fit in both runs | One run used the full profile. The other used only the SWOT target at the exact NAN003 model cell (`LOCAL`). Their run configurations differed only in the weighting domain. The runs developed separate particle histories. |
| Separate 2026 temporal test | Available 2023–2025 pairs only | Full accepted profile using separately fitted offsets and uncertainties. The 2026 observations were withheld from fitting this operator. |

The working configuration inspected on 12 November 2026 selected `LOCAL` at NAN002. The same Python modules were used across these configurations. Reproducing a particular result requires its archived settings and fitted tables as well as the source code.
