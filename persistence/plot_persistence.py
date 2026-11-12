"""Draw three fixed-origin water-level comparisons from public aggregate statistics."""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt


STATIONS = ("NAN001", "NAN002", "NAN003")
LEADS = tuple(range(10))
MM_PER_INCH = 25.4
WIDTH_MM = 170
DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Comparison:
    filename: str
    height_mm: float
    top_mm: float
    series_column: str
    value_column: str
    y_limits: tuple
    y_ticks: tuple
    paired_origins: int | None
    # Entries give the legend text, stroke colour, line style and marker.
    curves: tuple


FIGURES = (
    Comparison(
        "fig07_update_persistence", 68, 20, "year", "rmse_reduction_pct",
        (-15, 85), (-10, 0, 20, 40, 60, 80), None,
        (("2024", "#3a6897", "--", "o"), ("2025", "#ae526d", "-.", "s"),
         ("2026", "#137647", "-", "^")),
    ),
    Comparison(
        "fig08_fixed_offsets", 64, 14.5, "series", "rmse_m", (0, 3.5), (0, 1, 2, 3), 12,
        (("Raw open loop", "#536878", "--", "s"),
         ("Offset-corrected open loop", "#00838c", "-.", "o"),
         ("Raw assimilation", "#d55e00", "-", "^"),
         ("Offset-corrected assimilation", "#7851a9", ":", "D")),
    ),
    Comparison(
        "fig09_paired_strategies", 59, 13, "series", "rmse_m", (0.4, 2.85),
        (0.5, 1, 1.5, 2, 2.5), 13,
        (("Open loop", "#2377a5", "--", "o"), ("Full profile", "#d2691e", "-", "^"),
         ("NAN003 only", "#8c4ba6", "-.", "s")),
    ),
)


def read_values(directory, comparison):
    path = directory / (comparison.filename + ".csv")
    series_names = [curve[0] for curve in comparison.curves]
    points = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = (row["station"], row[comparison.series_column], int(row["lead_day"]))
            value, count = float(row[comparison.value_column]), int(row["paired_origins"])
            if key in points:
                raise ValueError(f"Duplicate station, series and lead in {path.name}: {key}")
            if not math.isfinite(value) or not comparison.y_limits[0] <= value <= comparison.y_limits[1]:
                raise ValueError(f"Value outside the plotted scale for {key}: {value}")
            if count <= 0 or (comparison.paired_origins is not None and count != comparison.paired_origins):
                raise ValueError(f"Invalid cohort size for {key}: {count}")
            points[key] = (value, count)
    expected = {(s, curve, lead) for s in STATIONS for curve in series_names for lead in LEADS}
    if points.keys() != expected:
        raise ValueError(f"{path.name} does not contain the expected stations, series and leads.")
    return points


def set_style(font):
    if font is None:
        try:
            font_manager.findfont("Times New Roman", fallback_to_default=False)
            font = "Times New Roman"
        except ValueError:
            font = "DejaVu Serif"
    font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": font, "font.size": 8.3, "axes.titlesize": 9, "axes.labelsize": 8.3,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
        "axes.spines.right": False, "axes.spines.top": False, "xtick.major.width": 0.55,
        "ytick.major.width": 0.55, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
    })
    return font


def draw(comparison, points, out):
    relative = comparison.value_column == "rmse_reduction_pct"
    figure, axes = plt.subplots(1, 3, sharey=True, figsize=(
        WIDTH_MM / MM_PER_INCH, comparison.height_mm / MM_PER_INCH))
    figure.subplots_adjust(left=14.5/WIDTH_MM, right=1-3.4/WIDTH_MM,
                           top=1-comparison.top_mm/comparison.height_mm,
                           bottom=13/comparison.height_mm, wspace=0.16)
    for letter, station, ax in zip("abc", STATIONS, axes):
        counts = []
        for name, color, linestyle, marker in comparison.curves:
            values = [points[station, name, day][0] for day in LEADS]
            counts.append(f"{points[station, name, 0][1]}/{points[station, name, 9][1]}")
            ax.plot(LEADS, values, label=name, color=color, linestyle=linestyle, marker=marker,
                    markersize=2.65, linewidth=1.15, markeredgewidth=0.45)
        ax.set_xlim(-0.2, 9.2)
        ax.set_ylim(*comparison.y_limits)
        ax.set_yticks(comparison.y_ticks)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_xticks([1, 2, 4, 5, 7, 8], minor=True)
        ax.tick_params(which="minor", length=1.8, width=0.45)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d9dee2", linewidth=0.5)
        ax.set_title(f"({letter}) {station}", loc="left", fontweight="bold", pad=14 if relative else 3)
        if relative:
            ax.axhline(0, linewidth=0.7, color="#48545c", zorder=1)
            ax.text(0, 1.025, "n: " + "  ·  ".join(counts), transform=ax.transAxes,
                    fontsize=7.1, va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.991),
                  ncol=2 if len(labels) == 4 else 3, borderaxespad=0, frameon=False,
                  fontsize=7.8, handlelength=2, handletextpad=0.40, columnspacing=1.15,
                  labelspacing=0.32)
    ylabel = "RMSE reduction (%)" if relative else "Water-level RMSE (m)"
    figure.text(0.025, 0.48, ylabel, ha="center", va="center", rotation="vertical", fontsize=8.5)
    figure.text(0.53, 0.025, "Lead from daily analysis checkpoint (days)",
                ha="center", va="bottom", fontsize=8.5)
    for extension in ("png", "pdf"):
        figure.savefig(out / f"{comparison.filename}.{extension}", dpi=500)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="Directory containing the three CSV files.")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="Output directory (default: ./figures).")
    parser.add_argument("--font", help="Optional installed font. Defaults to Times New Roman or DejaVu Serif.")
    args = parser.parse_args()
    font = set_style(args.font)
    datasets = [read_values(args.data, comparison) for comparison in FIGURES]
    args.out.mkdir(parents=True, exist_ok=True)
    for comparison, points in zip(FIGURES, datasets):
        draw(comparison, points, args.out)
    print(f"Saved Figures 7-9 as PDF and PNG in {args.out} using {font}.")


if __name__ == "__main__":
    main()
