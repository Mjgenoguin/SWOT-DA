# Two-stage NSGA-II calibration of the Upper Nan RRI model

Published source snapshots, 25 November 2026. Both Python drivers are preserved byte for byte from the retained research files. Their SHA-256 hashes and evidence records are listed in [provenance.json](provenance.json).

| Stage | Driver | Fitted quantities | Observations |
| --- | --- | --- | --- |
| 1 | [MultiCoreTrueNSGA2.py](stage1/MultiCoreTrueNSGA2.py) | River roughness and land-use-dependent hydrological parameters, with geometry fixed | NAN001 |
| 2 | [MultiCoreTrueNSGA2_ThreeStations_DepthProfile_Dedup_184d.py](stage2/MultiCoreTrueNSGA2_ThreeStations_DepthProfile_Dedup_184d.py) | Three longitudinal channel-profile coefficients, with hydrological parameters and station-cell depths fixed | NAN001, NAN002 and NAN003 |

## Provenance and scope

**Stage 1 is a recovered source snapshot, not a verified copy of the exact submitted historical source.** Its scoring dates and log basename match the retained `GA_run_log_5LULC.jsonl` archive. However, its defaults specify 20 generations, while the retained execution contains 1,000 records through generation 50. The selected parameter vector first appears at generation 47, individual 14, timestamp `20251205_132611_625682`, with 184 daily pairs. The exact submitted source revision and complete historical optimizer configuration have not been recovered. The bounds, population size, mutation settings and seed in this snapshot must not be presented as independently verified settings of that historical run. No defaults were changed to guess the missing configuration.

**Stage 2 is the retained depth-profile driver associated with the completed generation-100 search.** Job `Nan5LULC_Depth_GA_17923.out` records population 10, 100 generations, 10 Python workers and one OpenMP thread per RRI process. The job resumed an existing checkpoint. The remaining optimizer settings below are source defaults; the job log does not independently establish every default. The selected geometry is identified in `Final_Best_DepthProfile.json` by coefficients `2.2102429623522943`, `1.795562502565869`, and `-1.2755716197777291` in the order shown below.

These scripts document the algorithms. Running them requires separately supplied observations, RRI software and model inputs. This publication does not provide a complete reproduction of the historical simulations, independently validate the fitted geometry, or establish global optimality of the chosen solutions.

## Objectives and selection

Both drivers implement nondominated sorting, crowding distance, tournament selection, uniform crossover, Gaussian mutation and elitist parent-offspring selection directly in Python/NumPy. No external optimization package is required. The general NSGA-II method is described by [Deb et al. (2002)](https://doi.org/10.1109/4235.996017); the RRI inputs, objectives and geometry parameterization are application-specific.

Stage 1 minimizes `(-NSE, RMSE, abs(mean bias))`. Its reported representative is selected lexicographically within the first Pareto front, prioritizing NSE, followed by RMSE and absolute bias. The parser creates genes for eligible values across all land-use columns in `RRI_Input.txt`; the old "One-Land-Use" header and `TARGET_LU` name do not describe that parser's full behavior. The retained five-class input has one river-roughness value and 13 families of five land-use values.

Stage 2 minimizes `(-mean NSE, mean normalized RMSE, worst normalized absolute bias)` over the three equally weighted stations. Error normalization uses each station's observed population standard deviation, with range, mean absolute value and unity as fallback scales. It requires at least 30 matched pairs per station. The final representative minimizes distance to the ideal point after scaling each objective by its range within the first Pareto front.

The geometry genes are:

- `upper_blend`, bounds `[-1, 2.5]`: blends the NAN001–NAN002 seed bed profile toward its endpoint-to-endpoint straight profile.
- `middle_blend`, bounds `[-1, 2.5]`: applies the same construction between NAN002 and NAN003.
- `downstream_slope_factor`, bounds `[-3, 1.5]`: scales the seed bed drop below NAN003, with a downstream taper.

For the two blend coefficients, 0 retains the seed and 1 produces the endpoint-to-endpoint straight profile; values above 1 extrapolate beyond that profile. A negative downstream factor reverses the sign of the seed's bed-elevation difference from NAN003 within the downstream core.

The seed vector is `[0, 0, 1]`. Station-cell depths are retained, and corrections at affected tributary confluences taper upstream. This search starts from the separately prepared `depth_S50_U50_D50_AdaptiveConfluence.txt` seed; the driver does not reconstruct that seed's earlier preparation. Bounds are numerical search limits, not surveyed uncertainty intervals, and permitted candidates are not necessarily monotonic bed profiles.

## Settings in the published files

| Setting | Stage 1 snapshot | Stage 2 snapshot |
| --- | --- | --- |
| Simulation time origin | 1 May 2023, 00:00 | 1 May 2023, 00:00 |
| Scoring dates | 1 May–31 October 2023 | 1 May–31 October 2023 |
| Daily target / tolerance | 00:00 / 120 minutes | 01:00 / 120 minutes |
| Population | 10 | 10 |
| Generations | 20; differs from retained run | 100 |
| Crossover probability | 0.6 | 0.6 |
| Per-coordinate mutation probability | 0.2 | 0.5 |
| Mutation spread, fraction of coordinate range | 0.05 | 0.08 |
| Random seed | 42 | 42 |
| Parallel RRI workers | 3 | 10 |
| Seeded initialization | Enabled | Enabled |
| Local-seed fraction / spread | 0.7 / 0.05 | 0.75 / 0.08 |

The dates above label the simulation hydrograph and scoring window; they do not generate an April warm-up. Supply the appropriate initial states separately. Daily comparisons use within-date matching. Stage 1 sorts timestamps and keeps the first value within the target tolerance on each date; despite the helper's "closest" description, it does not explicitly minimize the time difference when several values qualify. Stage 2 explicitly selects the nearest within-date value. Do not interpret the difference between the two stages' historical scores as a controlled geometry-effect estimate: their daily target hours and matching implementations differ.

## Required inputs

Prepare a dedicated working copy with this layout, then edit each driver's `CONFIG` section to point to it:

```text
experiment/
  Observed/
    2023_Nan_N1_H.txt                 # stage 1
    Nan_WL_HII_2023-2025_RID.txt      # stage 2
  Nan_5LULC/
    RRI_Input.txt
    RRI_Run                         # compatible executable, separately obtained
    location.txt
    topo/adem_2s.asc
    topo/adir_2s.asc
    riv/width.txt
    riv/depth_S50_U50_D50_AdaptiveConfluence.txt
    ... remaining grids, rainfall and initial states referenced by RRI_Input.txt
```

Stage 1 expects a tab-separated, header-bearing observation file with timestamp and value in columns 0 and 1 and timestamp format `%m/%d/%Y %H:%M:%S`. The default filename is historical; verify that its series is the intended NAN001 record and that the first simulated hydrograph station column corresponds to NAN001. SWOT and ICESat inputs are disabled by default.

Stage 2 expects a tab-separated table with `datetime`, `NAN001`, `NAN002` and `NAN003` columns. Its `hydro_hr.txt` mapping is elapsed seconds in column 0, then NAN001, NAN002 and NAN003 in columns 1–3. Station raster locations are one-based: `(88,331)`, `(117,332)`, `(131,326)`. An alternative configured executable name is `0_rri_1_4_2_6_NonOMP_dt600s`.

The gauge series, executable, seed and other model rasters, forcing and initial states are external inputs and are not distributed here. Before reuse, verify station mapping, time origin, units and vertical reference consistently between observations and simulation output. Stage 2 needs the adopted stage-1 hydrological parameters already installed in its model input.

## Configure and run

The drivers have no command-line argument parser. Their retained root paths refer to the original WSL/HPC environments and must be edited for a new machine. Update `ROOT_DIR`, `PARTICLE_DIR`, observation paths, executable names and experiment-specific settings. Use distinct worker, archive and checkpoint paths for separate experiments. Stage 2 accepts the `GA_*` environment overrides visible in its configuration; stage 1's core population/generation/mutation settings are Python constants.

From the repository root, install the versions used for the publication checks in your chosen Python environment:

```sh
python -m pip install -r calibration/requirements.txt
```

After configuring the scripts and external model inputs, a Linux/WSL run can be launched with:

```sh
python calibration/stage1/MultiCoreTrueNSGA2.py
GA_PREFLIGHT_ONLY=1 python calibration/stage2/MultiCoreTrueNSGA2_ThreeStations_DepthProfile_Dedup_184d.py
GA_PREFLIGHT_ONLY=0 python calibration/stage2/MultiCoreTrueNSGA2_ThreeStations_DepthProfile_Dedup_184d.py
```

Each invocation can be computationally expensive except the stage-2 geometry preflight. On a scheduler, allocate enough cores for the configured Python workers and any RRI OpenMP threads; keep NumPy/BLAS thread counts consistent with that allocation. The archived stage-2 job used a non-OpenMP executable and one thread per RRI process. Site-specific Slurm launchers are not included.

Run against a dedicated model copy. Stage 2 may rewrite `location.txt` after creating a backup, refresh worker folders and write the final best-depth raster into the base model's `riv` directory. Its preflight avoids RRI simulation but still prepares directories/location information and temporarily writes a seed test raster. Both drivers automatically resume their configured pickle checkpoint when one exists; use only your trusted, experiment-matched checkpoint. A fresh experiment needs a separate checkpoint path.

## Outputs, verification and citation

The drivers write evaluation logs, archived hydrographs/pairs and checkpoints. Stage 2 also writes a Pareto table, `Final_Best_DepthProfile.json` and `depth_NSGA2_DepthProfile_Best.txt`. Their output paths are configured in the source files.

Publication checks used Python 3.12.3, NumPy 1.26.4 and pandas 2.2.2. They verified unchanged source hashes, compilation/import and selected numerical helpers on synthetic inputs. No RRI calibration or geometry preflight was run as part of publication. This environment is not a reconstructed historical environment.

See [CITATION.md](CITATION.md) for a software reference and use the immutable commit URL in the paper. Cite Deb et al. separately for NSGA-II, and retain the stage-1 provenance qualification when describing replication.
