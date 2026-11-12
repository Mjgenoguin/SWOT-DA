# Persistence and spatial consistency of river water-level improvements from intermittent SWOT assimilation

This repository contains the SWOT observation-operator code and the analysis and plotting code for Figures 6–9 of the Upper Nan River Basin manuscript. The figures evaluate water levels at three main-stem gauges (NAN001, NAN002 and NAN003) after SWOT RiverSP node water-surface elevations were assimilated into the Rainfall–Runoff–Inundation (RRI) model. The open loop is the comparison simulation run without assimilation updates.

## Data assimilation

The [SWOT observation modules](assimilation/README.md) transform processed SWOT water-surface elevations to model-comparable river depths and sample the corresponding cells in each particle. The [RRI particle-filter workflow](assimilation/workflow.md) describes how the external driver uses those values, which configuration produced the principal results, and how the NAN003-only experiment and the 2026 temporal test differ.

## Reproduce the figures

From the repository root, install the dependencies and run:

```sh
python -m pip install -r requirements.txt
python between_updates/plot_gap_day_accuracy.py --out figures
python persistence/plot_persistence.py --out figures
```

These commands read the included summary tables and write the figures to `figures/`. Python 3.12 was used for verification. Times New Roman reproduces the manuscript typography when installed. Both plotting scripts can use DejaVu Serif on systems without that font.

| Figure | Scientific question | Comparison |
| --- | --- | --- |
| 6 | How accurate were daily estimates on dates between successive accepted SWOT updates? | At 07:00 ICT, compare the full-profile estimate and raw open loop with the same gauge observation. Group dates by time since the previous update within each 2024–2026 wet season. |
| 7 | How long did a trajectory initialized on an update day retain an advantage over raw open loop? | Compare water-level RMSE at issuance and at each daily lead through day nine. Available station–year pairs can change with lead. |
| 8 | Does a prior correction to mean water level change the apparent persistence of the assimilation benefit? | Compare raw and offset-corrected series on the same 12 complete July–September 2026 update trajectories. The offsets were fitted in May–June and held fixed. |
| 9 | Does the observation strategy with the lower initial error keep its advantage at later leads? | Compare full-profile and NAN003-only sequential assimilation with open loop on 13 complete shared update trajectories in 2026. |

Figure 6 uses each date's current daily estimate. Figures 7–9 follow trajectories from fixed initializations, holding particle weights fixed even when a later SWOT acquisition informs a newly issued trajectory. The analysis evaluates persistence under the prescribed rainfall used in the experiments. The three gauges supplied evaluation observations outside the SWOT particle weighting and had informed RRI calibration.

## Files and data access

- [Between-update figure](between_updates/README.md) explains the 45 summary rows, paired observations and contributing intervals.
- [Fixed-origin figures](persistence/README.md) explains the three different sets of trajectories and the raw and corrected comparators.
- [Recompute the summary tables](metrics/README.md) documents the input columns and commands for users with matched gauge and simulation time series.

The repository distributes derived statistics and scripts. The upstream RRI model, full particle archives, and station-level observation time series are separate inputs. HII describes how to obtain its hydrological information through [ThaiWater/HII](https://www.hii.or.th/en/faq/how-can-general-public-access-hiis-information/). The [RID station portal](https://bigdata-swoc.rid.go.th/pier/all) is the source for the additional gauges discussed elsewhere in the manuscript. SWOT RiverSP products are available through [NASA PO.DAAC](https://podaac.jpl.nasa.gov/SWOT), and the GFS forecast archive is catalogued by [NSF NCAR](https://doi.org/10.5065/D65D8PWK).

The figure numbering follows the manuscript dated 12 November 2026. Record the Git commit when citing or reproducing this version.
