# Fixed-origin river water-level comparisons

Figures 7–9 follow simulations initialized at a daily 07:00 ICT checkpoint in the Upper Nan River Basin, Thailand. An accepted SWOT update at the starting checkpoint can reflect earlier assimilation history. Once issued, each trajectory retains that initialization and receives prescribed rainfall through day nine, without additional SWOT updates. A nine-day trajectory therefore measures persistence under the supplied forcing. Each panel evaluates a different primary main-stem gauge. NAN001, NAN002 and NAN003 informed the original RRI calibration, and all three were excluded from the SWOT particle weighting.

## Reproduce the figures

Using Python 3.10 or later, install Matplotlib and run from the directory containing this README.

```sh
python -m pip install matplotlib
python plot_persistence.py --out figures
```

The script writes three 500 dpi PNGs and three vector PDFs at 170 mm width. It locates the included CSVs relative to the script, so the command can also be run from elsewhere with the path to the script. It selects Times New Roman when installed and otherwise uses Matplotlib's bundled DejaVu Serif. Font files are not distributed. Use `--font "DejaVu Serif"` to select the fallback explicitly. Matplotlib 3.11.2 with Times New Roman reproduced the reference PNGs byte for byte in the authors' environment. Other font or library versions can change rendering while preserving all plotted values. Matplotlib 3.8.4 was also tested.

## Read the data

| File in `data/` | Quantity plotted | Verification sample |
| --- | --- | --- |
| `fig07_update_persistence.csv` | Percentage reduction in assimilation RMSE relative to unadjusted open loop | Available matched update-origin pairs at each gauge, year and lead during 2024–2026, 90 aggregated values |
| `fig08_fixed_offsets.csv` | Water-level RMSE for four series, in metres | Same 12 complete July–September 2026 update origins at every lead and primary gauge, 120 aggregated values |
| `fig09_paired_strategies.csv` | Water-level RMSE for open loop and two sequential assimilation strategies, in metres | Same 13 complete, shared 2026 update origins for both strategies at every lead and primary gauge, 90 aggregated values |

- `station` identifies the gauge and `lead_day` gives the elapsed whole days from the starting checkpoint. Day zero is the initial analysis. Comparisons at each lead use gauge levels for the corresponding valid time.
- `paired_origins` is the number of starting checkpoints with matched verification data for that row. In Figure 7, each panel header displays the day-zero/day-nine count pair for each of the three years. Missing gauge observations can change the cohort with lead.
- In Figure 7, `rmse_reduction_pct` equals 100 × (open-loop RMSE − assimilation RMSE) / open-loop RMSE. Positive values indicate lower error after assimilation. Curves span the daily leads within the archive. This figure does not use the separate between-update calendar-day sample.
- In Figure 8, `series` identifies the raw open loop, raw full-profile assimilation and their offset-corrected counterparts. A separate gauge-specific mean level offset was estimated for each model series from 59 paired daily observations during 3 May–30 June 2026 and held fixed at every subsequent lead. This allows a model to benefit from a simple earlier level correction. Raw assimilation has lower RMSE than offset-corrected open loop at all three stations through day three. By day nine offset-corrected open loop has lower RMSE at all three. The intermediate crossover depends on the station.
- In Figure 9, `series` distinguishes spatial full-profile assimilation, sequential assimilation restricted to the NAN003 location, and the open loop. Each strategy evolved with its own particle and update history. The shared origins allow paired evaluation, but the observed differences cannot be attributed to spatial coverage alone. Full-profile RMSE is lower than NAN003-only RMSE at each station through day two. NAN003-only RMSE is lower at each station on days three through nine.

Each CSV contains plotted aggregate statistics. The script does not regenerate the RRI ensemble, re-estimate observation operators or calculate RMSE from individual gauge measurements. The released tables contain no hourly or daily observed water levels.

## Sources and credit

The aggregate statistics were derived from the authors' analyses of the Upper Nan RRI simulations. The water-level verification used station records supplied by Thailand's Hydro-Informatics Institute. The assimilated elevations came from the SWOT Level-2 River Single-Pass node product. Credit the original providers when using the underlying observations and the release authors when using the derived figure data or code. Source records and model documentation are available from [HII](https://www.hii.or.th/en/faq/how-can-general-public-access-hiis-information/), [NASA PO.DAAC's SWOT mission and product catalog](https://podaac.jpl.nasa.gov/SWOT), and [Sayama et al. (2012), the RRI model description](https://doi.org/10.1080/02626667.2011.644245).
