# Figure 6. Daily water-level accuracy between SWOT updates

This folder reproduces Figure 6 of the Upper Nan manuscript from a table of 45 summary rows. The comparison covers three main-stem gauges (NAN001, NAN002 and NAN003) and the 2024, 2025 and 2026 wet seasons. It evaluates each day's 07:00 ICT (UTC+7) water-level estimate against the gauge at the same checkpoint. Fixed-origin forecasts followed for nine days are evaluated in a separate figure.

## Reproduce the figure

From this directory, run the following with Python, NumPy, pandas and Matplotlib installed.

```bash
python plot_gap_day_accuracy.py --out figures
```

The script reads [the figure table](Actual_gap_day_accuracy_values.csv) from beside the script, checks all 45 percentages and the pooled RMSEs, and writes a PDF and a 400 dpi PNG to the requested output directory. The journal figure uses Times New Roman. If that font is unavailable, the script uses Matplotlib's bundled DejaVu Serif. To select another installed family, add `--font "Family Name"`. Font substitution affects letter shapes and spacing. The data and color scale are identical.

## Sample and calculation

1. **Select dates.** For each wet season, form intervals between consecutive days on which a SWOT update was accepted. Keep only the intervening dates. Both update days and all dates before the first or after the last update of the season are excluded. The 61 accepted checkpoints across three seasons gave 58 within-season intervals and 369 candidate intervening dates.
2. **Match gauge data.** At each station, retain dates with finite observations, full-profile assimilation ensemble means and raw open-loop estimates. Both models are scored on those same observations. The sample size varies by station and year.
3. **Group by age.** Assign each date its elapsed time in calendar days since the preceding update. The four groups cover days 1–3, 4–6, 7–9 and 10–12. A single interval can contribute multiple dates and can appear in several groups. The 10–12-day group compares daily estimates during actual gaps, beyond the horizon of the separate issued-trajectory test.
4. **Calculate the displayed change.** For each model, RMSE is the square root of the sum of squared water-level errors divided by the number of paired dates. Both RMSEs are in meters. The displayed percentage is `100 × (1 − RMSE_assimilation / RMSE_open_loop)`. The denominator is the raw, unassimilated open-loop RMSE over the same station, year and dates. A positive percentage means reduced error. The “All gap days” column pools squared errors over paired dates before calculating the RMSEs and their percentage change.

| CSV field | Definition |
| --- | --- |
| `year`, `station` | Wet season and gauge identifier |
| `age_days` | Elapsed-day group or the pooled “All gap days” group |
| `paired_days` | Number of daily checkpoints with both model estimates and a gauge observation |
| `rmse_open_loop_m` | RMSE of the raw open loop on paired dates, in meters |
| `rmse_assimilation_m` | RMSE of the assimilation ensemble mean on paired dates, in meters |
| `rmse_reduction_pct` | Percentage change computed with the sign convention above |
| `distinct_gaps` | Number of update intervals represented among the paired dates; count distinct intervals again when pooling age groups |

The table contains aggregate statistics. It supports figure reproduction and internal numerical checks. Recalculating individual daily errors requires the source gauge and model time series.
