# Derive the scores for Figures 6–9

The plotting scripts accept the aggregate CSVs supplied with this repository. To recalculate those CSVs, use separately obtained or authorized station water levels and the matching archived model trajectories. The following commands assume the two standardized CSVs are in a local directory called `inputs` and are run from the repository root.

```sh
python metrics/derive.py --full-profile inputs/full_profile_fixed_origin.csv --paired-strategies inputs/paired_2026_strategies.csv --out recalculated
python between_updates/plot_gap_day_accuracy.py --scores recalculated/Actual_gap_day_accuracy_values.csv --out figures
python persistence/plot_persistence.py --data recalculated --out figures
```

Either input can also be provided separately. The full-profile input generates Figures 6–8 source tables. The paired-strategies input generates the Figure 9 source table. The metric script writes only aggregate values. Use a new output directory to preserve an earlier set of calculations.

## Standardized matchup tables

Provide one row per gauge, issuance time and lead. All timestamps are 07:00 local time in Thailand (UTC+7), written as `YYYY-MM-DD HH:MM:SS` without a timezone suffix. Retain rows for missing observations or model values using an empty numeric field. A forecast issued before a subsequent SWOT acquisition retains its original trajectory at all later leads.

| Field | Definition |
| --- | --- |
| `station` | Gauge identifier. The figures use NAN001, NAN002 and NAN003. |
| `origin_ict` | Local issue time of the fixed-origin trajectory. The full-profile table covers daily origins from 3 May through 30 September in 2024, 2025 and 2026, including every T+0 checkpoint. |
| `lead_day` | Integer from 0 to 9, inclusive. At zero lead the row represents that day’s analysis checkpoint. |
| `valid_ict` | Verification time, equal to `origin_ict` plus `lead_day` calendar days. |
| `update` | Whether the full-profile run accepted a SWOT update at issuance. Use true/false or 1/0. Repeat this origin-level flag across leads and stations. |
| `observed_m` | Gauge observation at the verification time in the station-relative reference used for evaluation, in meters. |
| `open_loop_m` | Matching raw open-loop water level in meters. |
| `assimilation_m` | Matching water level in meters from the full-profile assimilation weighted ensemble mean. The manuscript uses the 36-member configuration with the operator fitted on 2023–2026 pairs. |
| `local_update` | Paired-strategies input only. Whether the single-location run accepted a SWOT update at the same origin. |
| `local_m` | Paired-strategies input only. Weighted ensemble mean water level in meters from the sequential NAN003-only assimilation run. |

In the paired-strategies table, first match the two runs on station, origin, lead and verification time. Confirm that their reference observations and open-loop control agree, then supply a single observed and open-loop column. Preserve the update flag for each strategy separately. A NAN003-only run supplies levels at all three evaluation gauges.

## Populations and scoring

- **Figure 6, daily estimates between observations.** Use only zero-lead estimates on dates strictly between consecutive accepted full-profile updates within a wet season. The clock starts on the preceding update date. Summarize days 1–3, 4–6, 7–9 and 10–12 separately, and pool all matched intervening days for the aggregate column. The three manuscript seasons contain 369 candidate dates in 58 within-season intervals before station-specific finite-value filtering. Each row reports matched dates and distinct preceding updates represented.
- **Figure 7, lead-wise persistence.** For each station, year and lead, select accepted update origins with a finite gauge observation and both model estimates. The number of available origins can change with lead.
- **Figure 8, fixed-offset sensitivity.** At each station, fit a separate additive offset for each model series as the average of observation minus simulation across matched T+0 pairs from 3 May through 30 June 2026. Add these offsets to the corresponding model values at every lead. Evaluate July–September 2026 update origins with observations and model values available at all ten leads and with every verification date before 1 October. In the manuscript data there are 59 offset-fitting pairs per station and 12 complete update origins per station. Corrected values are kept at their calculated levels, including any negative values.
- **Figure 9, paired observation strategies.** Use 2026 origins at which both strategies accepted an update. At each station retain only origins with finite values for all three modeled series and the gauge for all ten leads. The manuscript cohort comprises 13 complete shared origins. This population is defined independently of the July–September cohort in Figure 8.

For each model series, RMSE is the square root of the mean squared model–gauge difference on matched rows. The percentage reduction is `100 × (1 − assimilation RMSE / open-loop RMSE)`. The pooled gap-day score uses every paired squared error in the population. A zero open-loop RMSE makes the percentage undefined.

Recalculation on the archived matchup tables reproduced 45 gap-day rows, 90 update-persistence rows, 120 fixed-offset rows and 90 paired-strategy rows. Counts and group labels matched throughout. The largest absolute discrepancy in any published numerical column was below 0.000000000001.
