# SWOT water-surface elevation to RRI river depth

These two source files prepare the observation and model inputs for one particle-filter update. They are identical to the copies archived with the 2026 Upper Nan full-profile and NAN003-only runs. The principal 2024–2026 experiment used cell offsets fitted to pooled 2023–2026 data. The historical column name `offset_2023wet` and the year in some source comments predate that fit. The parameter tables passed by the caller determine which fit the operator applies. See the [runtime and experiment configurations](workflow.md) for the filtering and resampling steps.

- [swot_rri_transfer_operator.py](swot_rri_transfer_operator.py) converts cell-level SWOT water-surface elevation (WSE) to the RRI river-depth coordinate, constructs the uncertainty for each target, and assembles the inputs to the likelihood.
- [swot_obs_operator.py](swot_obs_operator.py) supplies the shared quality-control settings, particle-file layout and grid sampling. The principal experiment calls the transfer function in the first module. The earlier anomaly functions in the second module are retained in the archived source.

## Inputs for one overpass

| Input | Required contents |
| --- | --- |
| Processed SWOT cells, Parquet | `row`, `col` (zero-based RRI indices), `wse_median` (WSE, m), and `wse_u_median` (WSE uncertainty, m). Include `n_nodes` to apply the minimum-node screen used in the study. If supplied, `time_max_ict` or `cycle_start_ict` populates the cell time in the output. |
| Fitted offsets, CSV | One row per supported cell with `row0`, `col0`, and `offset_2023wet` (m). |
| Fitted cell uncertainties, CSV | One row per supported cell with `row0`, `col0`, and `sigma_idio_regularized_m` (m). The pass-common uncertainty is supplied separately in metres. |
| Model particles | One whitespace-delimited 2-D river-depth grid per particle at the selected model time. For example, index 21 reads `particles/Particle00001/RRI/out/hr_000021.out` and the corresponding file for each other particle. Index 0 reads `RRI/init/hr_init.out` within each particle. |

The calling program chooses the saved model time nearest the overpass and supplies its `hr_index`. Retained cells are ordered by row, then column. For each cell the observation target is the larger of zero and WSE plus the fitted offset. The standard deviation combines measurement, representation, cell-transfer and pass-common components in quadrature. The likelihood receives one standard deviation per cell. Cells without a fitted offset and uncertainty are excluded.

## Call the operator

Install NumPy, pandas and a Parquet reader such as PyArrow (`python -m pip install numpy pandas pyarrow`). Run Python from this directory so the two modules can import one another. The example expects the external input files described above in the current directory. The common standard deviation shown is from the pooled Upper Nan fit.

```python
from pathlib import Path
from swot_rri_transfer_operator import (
    RRITransferConfig,
    make_pf_inputs_from_swot_rri_transfer,
)

config = RRITransferConfig(
    parameter_csv=Path("offsets.csv"),
    cell_sigma_csv=Path("cell_uncertainties.csv"),
    common_sigma_m=0.744587260225238,
)
targets, times, model, sigmas, cells, particle_ids = make_pf_inputs_from_swot_rri_transfer(
    cycle_path=Path("overpass_cells.parquet"),
    particles_root=Path("particles"),
    hr_index=21,  # Example index; supply the one matched to this overpass.
    transfer_cfg=config,
)
```

`targets` and `sigmas` each contain one value in metres per retained cell. In `model`, rows follow `cells` and columns follow `particle_ids`. These arrays are the interface with the external RRI particle-filter driver. Running the full assimilation also requires that driver, an RRI installation, the study inputs, and the fitted tables. The package contains the two operator modules and documentation. The raw agency records and model runs are held separately.

On 12 November 2026, a synthetic test with three particles checked duplicate-cell aggregation, missing fit support, node count screening, the river mask, and clipping at zero depth. For the two retained cells the operator returned the expected depth targets (4.0 and 0.0 m), total standard deviations (0.648074 and 0.700000 m), and all six member depths in the expected order.
