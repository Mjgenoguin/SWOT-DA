"""Derive manuscript Figures 6–9 from separately supplied gauge–model matchups."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


STATIONS = ["NAN001", "NAN002", "NAN003"]
LEVELS = ["observed_m", "open_loop_m", "assimilation_m"]
KEY = ["station", "origin_ict", "lead_day"]


def read_pairs(path, local=False):
    flags = ["update", "local_update"] if local else ["update"]
    levels = LEVELS + (["local_m"] if local else [])
    d = pd.read_csv(path, usecols=KEY + ["valid_ict"] + flags + levels)
    d = d[d.station.isin(STATIONS)].copy()
    if set(d.station) != set(STATIONS):
        raise ValueError("Input must contain all three primary gauges.")
    for flag in flags:
        values = d[flag].astype(str).str.strip().str.lower()
        if not values.isin(["true", "false", "1", "0"]).all():
            raise ValueError(f"{flag} must contain true/false or 1/0 for every row.")
        d[flag] = values.isin(["true", "1"])
    for col in ["origin_ict", "valid_ict"]:
        d[col] = pd.to_datetime(d[col], errors="raise")
        if d[col].dt.tz is not None or not d[col].eq(d[col].dt.normalize() + pd.Timedelta(hours=7)).all():
            raise ValueError(f"{col} must contain unambiguous local 07:00 ICT timestamps without an offset.")
    if not d.lead_day.isin(range(10)).all() or d.duplicated(KEY).any():
        raise ValueError("Each station/origin/lead key must be unique with a lead from 0 through 9.")
    if not d.valid_ict.eq(d.origin_ict + pd.to_timedelta(d.lead_day, unit="D")).all():
        raise ValueError("A valid time must be its own origin plus the declared lead.")
    if d.groupby("origin_ict")[flags].nunique().gt(1).any().any():
        raise ValueError("The accepted-update flag for a strategy must agree across all rows from one origin.")
    for col in levels:
        d[col] = pd.to_numeric(d[col], errors="raise")
    d["year"] = d.origin_ict.dt.year
    d = d.sort_values(KEY)
    return d


def matched(d, levels=LEVELS):
    return d[np.isfinite(d[levels]).all(axis=1)].copy()


def complete(d):
    counts = d.groupby(["station", "origin_ict"]).lead_day.transform("size")
    return d[counts.eq(10)].copy()


def rmse(d, values):
    return float(np.sqrt(np.mean((values - d.observed_m)**2)))


def reduction(ol, da):
    return 100 * (1 - da/ol) if ol > 0 else np.nan


def gap_scores(d):
    """Daily estimates strictly bracketed by two accepted updates in the same season."""
    day = d[d.lead_day.eq(0)].copy()
    dates = day[["origin_ict", "year", "update"]].drop_duplicates().sort_values("origin_ict")
    for year in (2024, 2025, 2026):
        season = pd.date_range(f"{year}-05-03 07:00", f"{year}-09-30 07:00", freq="D")
        for station in STATIONS:
            actual = day.loc[day.station.eq(station) & day.year.eq(year), "origin_ict"]
            if not np.array_equal(actual.to_numpy(), season.to_numpy()):
                raise ValueError(f"Retain all daily origins for {station} in {year}, including missing gauge matches.")
    if not d.origin_ict.isin(dates.origin_ict).all() or not d.year.isin([2024, 2025, 2026]).all():
        raise ValueError("Supply only origins in the three evaluated manuscript seasons.")
    dates["previous"] = dates.origin_ict.where(dates["update"]).groupby(dates.year).ffill()
    dates["following"] = dates.origin_ict.where(dates["update"]).groupby(dates.year).bfill()
    gaps = dates[~dates["update"] & dates.previous.notna() & dates.following.notna()].copy()
    gaps["age"] = (gaps.origin_ict - gaps.previous).dt.days
    if gaps.age.gt(12).any():
        raise ValueError("An intervening day exceeds the longest manuscript age bin (12 days).")
    g = matched(day.merge(gaps[["origin_ict", "previous", "age"]], on="origin_ict", validate="many_to_one"))
    rows = []
    for (year, station), h in g.groupby(["year", "station"]):
        for label, lower, upper in [("All gap days", 1, 12), ("1–3", 1, 3), ("4–6", 4, 6),
                                    ("7–9", 7, 9), ("10–12", 10, 12)]:
            group = h[h.age.between(lower, upper)]
            if group.empty:
                continue
            ol, da = rmse(group, group.open_loop_m), rmse(group, group.assimilation_m)
            rows.append([year, station, label, len(group), ol, da, reduction(ol, da), group.previous.nunique()])
    return pd.DataFrame(rows, columns=["year","station","age_days","paired_days","rmse_open_loop_m",
                                      "rmse_assimilation_m","rmse_reduction_pct","distinct_gaps"])


def update_scores(d):
    """Available paired estimates at each lead among full-profile update origins."""
    rows = []
    for (st, lead, year), h in matched(d[d["update"]]).groupby(["station", "lead_day", "year"]):
        ol, da = rmse(h, h.open_loop_m), rmse(h, h.assimilation_m)
        rows.append([st, lead, year, reduction(ol, da), len(h)])
    return pd.DataFrame(rows, columns=["station", "lead_day", "year", "rmse_reduction_pct", "paired_origins"])


def level_scores(d, series):
    rows = []
    for (st, lead), h in d.groupby(["station", "lead_day"]):
        for label,col in series.items():
            rows.append([st, lead, label, rmse(h, h[col]), len(h)])
    return pd.DataFrame(rows, columns=["station","lead_day","series","rmse_m","paired_origins"])


def frozen_scores(d):
    paired = matched(d)
    fitted = paired[paired.lead_day.eq(0) & paired.origin_ict.ge("2026-05-03")
                    & paired.origin_ict.lt("2026-07-01")].copy()
    if set(fitted.station) != set(STATIONS):
        raise ValueError("The earlier-period fit must have matched observations at each primary station.")
    offsets = fitted[["open_loop_m","assimilation_m"]].rsub(fitted.observed_m, axis="index").groupby(fitted.station).mean()
    # Only complete late-season update origins; September-end cut applies to VALID time too.
    evals = complete(paired[paired["update"] & paired.origin_ict.ge("2026-07-01")
                            & paired.valid_ict.lt("2026-10-01")])
    for col in offsets:
        evals["corrected_"+col] = evals[col] + evals.station.map(offsets[col])
    return level_scores(evals, {"Raw open loop":"open_loop_m", "Offset-corrected open loop":"corrected_open_loop_m",
                                "Raw assimilation":"assimilation_m", "Offset-corrected assimilation":"corrected_assimilation_m"})


def strategy_scores(d):
    shared = d[(d.year==2026) & d["update"] & d.local_update]
    common = complete(matched(shared, LEVELS + ["local_m"]))
    return level_scores(common, {"Open loop":"open_loop_m", "Full profile":"assimilation_m", "NAN003 only":"local_m"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-profile", type=Path, help="Fixed-origin, full-profile predictions and gauge matches (Figures 6–8)")
    parser.add_argument("--paired-strategies", type=Path, help="Matched 2026 full-profile and single-location runs (Figure 9)")
    parser.add_argument("--out", type=Path, required=True, help="Directory for aggregated figure-source CSVs")
    args = parser.parse_args()
    if not (args.full_profile or args.paired_strategies):
        parser.error("Supply at least one input table.")
    outputs = {}
    if args.full_profile:
        full = read_pairs(args.full_profile)
        outputs["Actual_gap_day_accuracy_values.csv"] = gap_scores(full)
        outputs["fig07_update_persistence.csv"] = update_scores(full)
        outputs["fig08_fixed_offsets.csv"] = frozen_scores(full)
    if args.paired_strategies:
        outputs["fig09_paired_strategies.csv"] = strategy_scores(read_pairs(args.paired_strategies, local=True))
    if any(table.empty for table in outputs.values()):
        raise ValueError("No eligible finite pairs for one or more requested cohorts. Check the supplied input.")
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, table in outputs.items():
        table.to_csv(args.out/filename, index=False)
        print(f"{filename}: {len(table)} rows")


if __name__ == "__main__":
    main()
