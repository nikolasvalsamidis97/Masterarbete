from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_mass_loss_finished_family_plots"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


RESULTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RESULTS_DIR

TITLE_SIZE = 17
LABEL_SIZE = 19
TICK_SIZE = 17

FAMILY_FILES = {
    "solar_system_fixed": "solar_system_fixed.txt",
    "distance_sweep": "distance_sweep.txt",
    "real_reference_systems": "real_reference_systems.txt",
}

SOLAR_ORDER = ["mercury_like", "earth_like", "mars_like", "cold_jupiter"]
REAL_ORDER = ["gj1132_b", "gj1214_b", "gj436_b", "hd209458_b"]


def read_total_rows(family_name: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / FAMILY_FILES[family_name]
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row for row in rows if row["species"] == "TOTAL_INCLUDED_SPECIES"]


def to_float(value: str) -> float:
    return float(value) if value not in {"", "nan"} else 0.0


def science_axis(axis) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    axis.yaxis.set_major_formatter(formatter)
    axis.tick_params(axis="both", labelsize=TICK_SIZE)


def styled_axes(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=TITLE_SIZE, pad=10)
    axis.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    axis.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    axis.grid(axis="y", alpha=0.25)
    science_axis(axis)


def wrap_label(label: str) -> str:
    replacements = {
        "Mercury-like rocky planet": "Mercury-like\nrocky planet",
        "Earth-like rocky planet": "Earth-like\nrocky planet",
        "Mars-like rocky planet": "Mars-like\nrocky planet",
        "Inflated hot Jupiter": "Inflated hot\nJupiter",
        "Super-Earth rocky planet": "Super-Earth\nrocky planet",
    }
    return replacements.get(label, label)


def ordered_rows(rows: list[dict[str, str]], order: list[str]) -> list[dict[str, str]]:
    rank = {key: index for index, key in enumerate(order)}
    return sorted(rows, key=lambda row: rank.get(row["planet"], 999))


def plot_solar_system_fixed(axis, rows: list[dict[str, str]]) -> None:
    rows = ordered_rows(rows, SOLAR_ORDER)
    labels = [wrap_label(row["planet_label"]) for row in rows]
    values = [to_float(row["mass_loss_rate_g_s"]) for row in rows]

    axis.bar(labels, values, color=["#c97c5d", "#4f6d7a", "#c0a080", "#7a9e7e"])
    styled_axes(axis, "Solar System Analogues", "Planet", "Total mass-loss rate [g s$^{-1}$]")
    axis.tick_params(axis="x", rotation=15, labelsize=TICK_SIZE)


def plot_distance_sweep(axis, rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda row: to_float(row["distance_AU"]))
    distances = [to_float(row["distance_AU"]) for row in rows]
    values = [to_float(row["mass_loss_rate_g_s"]) for row in rows]
    label = rows[0]["planet_label"] if rows else "Completed systems"

    axis.plot(distances, values, marker="o", linewidth=2.2, color="#4f6d7a")
    styled_axes(axis, "Distance Sweep", "Orbital distance [AU]", "Total mass-loss rate [g s$^{-1}$]")
    axis.tick_params(axis="x", labelsize=TICK_SIZE)
    axis.text(
        0.03,
        0.95,
        label,
        transform=axis.transAxes,
        fontsize=TICK_SIZE - 1,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )
    if values and max(values) == 0.0:
        axis.text(
            0.5,
            0.55,
            "All completed cases\ncurrently give zero total escape",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=TICK_SIZE - 1,
        )


def plot_real_reference_systems(axis, rows: list[dict[str, str]]) -> None:
    rows = ordered_rows(rows, REAL_ORDER)
    labels = [row["planet_label"] for row in rows]
    values = [to_float(row["mass_loss_rate_g_s"]) for row in rows]

    axis.bar(labels, values, color=["#d17c8f", "#8a7db8", "#5c9ead", "#d9a441"])
    styled_axes(axis, "Real Reference Systems", "System", "Total mass-loss rate [g s$^{-1}$]")
    axis.tick_params(axis="x", rotation=18, labelsize=TICK_SIZE)


def save_family_figure(filename: str, plotter, rows: list[dict[str, str]], figsize=(8.5, 5.8)) -> None:
    figure, axis = plt.subplots(figsize=figsize)
    plotter(axis, rows)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    solar_rows = read_total_rows("solar_system_fixed")
    distance_rows = read_total_rows("distance_sweep")
    real_rows = read_total_rows("real_reference_systems")

    save_family_figure("solar_system_analogues_total_mass_loss.pdf", plot_solar_system_fixed, solar_rows)
    save_family_figure("distance_sweep_total_mass_loss.pdf", plot_distance_sweep, distance_rows)
    save_family_figure("real_reference_systems_total_mass_loss.pdf", plot_real_reference_systems, real_rows)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    plot_solar_system_fixed(axes[0], solar_rows)
    plot_distance_sweep(axes[1], distance_rows)
    plot_real_reference_systems(axes[2], real_rows)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "finished_families_total_mass_loss_summary.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
