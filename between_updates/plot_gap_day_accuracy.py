"""Render Figure 6 from the distributed table of station-year summary statistics."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


TABLE = Path(__file__).with_name("Actual_gap_day_accuracy_values.csv")
YEARS = (2024, 2025, 2026)
STATIONS = ("NAN001", "NAN002", "NAN003")
AGES = ("1–3", "4–6", "7–9", "10–12")
ALL = "All gap days"


def read_figure_table(path):
    data = pd.read_csv(path).set_index(["year", "station", "age_days"])
    expected = pd.MultiIndex.from_product([YEARS, STATIONS, (*AGES, ALL)], names=data.index.names)
    if not data.index.is_unique or set(data.index) != set(expected):
        raise ValueError("Expected one row per year, gauge and age group, including each pooled row.")
    data = data.reindex(expected)
    change = 100 * (1 - data.rmse_assimilation_m / data.rmse_open_loop_m)
    np.testing.assert_allclose(change, data.rmse_reduction_pct, rtol=1e-10, atol=1e-10)

    for year in YEARS:
        for station in STATIONS:
            group = data.loc[year, station]
            bins = group.loc[list(AGES)]
            pooled = group.loc[ALL]
            if bins.paired_days.sum() != pooled.paired_days:
                raise ValueError(f"Counts in the age bins do not sum to the pooled count for {station} in {year}.")
            for col in ("rmse_open_loop_m", "rmse_assimilation_m"):
                reconstructed = np.sqrt(np.average(bins[col] ** 2, weights=bins.paired_days))
                np.testing.assert_allclose(reconstructed, pooled[col], rtol=1e-10, atol=1e-10)
    return data


def resolve_font(requested):
    try:
        return FontProperties(fname=findfont(FontProperties(family=requested), fallback_to_default=False)).get_name()
    except ValueError:
        return "DejaVu Serif"


def draw(data, out, font):
    plt.rcParams.update({"font.family": font, "pdf.fonttype": 42, "svg.fonttype": "none"})
    width, height = 170, 110
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    ink = "#202f35"
    muted = "#49575c"
    cmap = LinearSegmentedColormap.from_list(
        "gap_contrast", ["#af542f", "#edcbb2", "#fafaf7", "#acd4cd", "#24796f"], N=257
    )
    bound = max(70.0, np.ceil(data.rmse_reduction_pct.abs().max() / 10) * 10)
    norm = Normalize(vmin=-bound, vmax=bound, clip=False)

    def label(x, y, value, **overrides):
        style = {"fontsize": 8.1, "ha": "center", "va": "center", "color": ink}
        ax.text(x, y, value, **{**style, **overrides})

    cell_width = 24.5
    x_starts = [28, 52.8, 77.6, 102.4, 137.4]
    row_height = 7.35
    top = 85.2
    label(2.4, 106.8, "Water-level RMSE reduction relative to raw open loop (%)", ha="left", fontsize=10.1, weight="bold")
    label(2.4, 101.4, "Positive values indicate lower error after SWOT assimilation.", ha="left", fontsize=8.0)
    label(77.6, 94.6, "Days since the last accepted SWOT update", fontsize=8.35)
    ax.plot([28, 126.9], [91.7, 91.7], color="#a0afad", lw=0.49)
    label(150.0, 90.0, ALL, fontsize=8.45, weight="bold")
    for x, age in zip(x_starts, AGES):
        label(x + cell_width / 2, 89.2, age, fontsize=8.55, weight="bold")

    for year in YEARS:
        label(2.3, top - row_height * 1.5, str(year), fontsize=8.5, weight="bold", ha="left")
        for station in STATIONS:
            y = top - row_height
            label(25.6, y + row_height / 2, station, ha="right", fontsize=7.9)
            for x, age in zip(x_starts, (*AGES, ALL)):
                record = data.loc[year, station, age]
                color = cmap(norm(record.rmse_reduction_pct))
                rgb = np.array(color[:3])
                linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = np.dot(linear, [0.2126, 0.7152, 0.0722])
                foreground = "#152a2a" if luminance > 0.245 else "#ffffff"
                ax.add_patch(Rectangle((x, y + 0.075), cell_width, row_height - 0.18,
                                       facecolor=color, edgecolor="none"))
                value = f"{record.rmse_reduction_pct:+.1f}".replace("-", "−")
                counts = f"{int(record.paired_days)} / {int(record.distinct_gaps)}"
                label(x + cell_width / 2, y + row_height * 0.66, value, fontsize=8.9, weight="bold", color=foreground)
                label(x + cell_width / 2, y + row_height * 0.22, counts, fontsize=7.15, color=foreground)
            top = y
        top -= 1.42
    ax.plot([132.2, 132.2], [y, 92.1], lw=0.55, color="#a1aeab")
    label(2.4, 10.0, "Small numbers give paired daily checkpoints / distinct gaps.", ha="left", fontsize=7.9, color=muted)
    label(2.4, 5.0, "Only dates strictly between successive accepted updates enter the comparison.", ha="left", fontsize=7.9, color=muted)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Actual_gap_day_accuracy.pdf", dpi=400)
    fig.savefig(out / "Actual_gap_day_accuracy.png", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=TABLE, help="Input CSV (defaults to the file beside this script)")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the generated PDF and PNG")
    parser.add_argument("--font", default="Times New Roman", help="Preferred font family (fallback: DejaVu Serif)")
    args = parser.parse_args()
    scores = read_figure_table(args.scores)
    family = resolve_font(args.font)
    draw(scores, args.out, family)
    print(f"Rendered Figure 6 from {len(scores)} summary rows using {family}.")
