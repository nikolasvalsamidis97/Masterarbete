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
TICK_SIZE = 16
LEGEND_SIZE = 13

FAMILY_FILES = {
    "solar_system_fixed": "solar_system_fixed.txt",
    "distance_sweep": "distance_sweep.txt",
    "real_reference_systems": "real_reference_systems.txt",
    "p0_sweep": "p0_sweep.txt",
    "mu_sweep": "mu_sweep.txt",
    "surface_gravity_sweep": "surface_gravity_sweep.txt",
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


def ordered_rows(rows: list[dict[str, str]], order: list[str]) -> list[dict[str, str]]:
    rank = {key: index for index, key in enumerate(order)}
    return sorted(rows, key=lambda row: rank.get(row["planet"], 999))


def wrap_label(label: str) -> str:
    replacements = {
        "Mercury-like rocky planet": "Mercury-like\nrocky planet",
        "Earth-like rocky planet": "Earth-like\nrocky planet",
        "Mars-like rocky planet": "Mars-like\nrocky planet",
        "Inflated hot Jupiter": "Inflated hot\nJupiter",
        "Super-Earth rocky planet": "Super-Earth\nrocky planet",
    }
    return replacements.get(label, label)


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
    axis.tick_params(axis="both", labelsize=TICK_SIZE)


def add_context_text(axis, lines: list[str]) -> None:
    axis.text(
        0.03,
        0.96,
        "\n".join(lines),
        transform=axis.transAxes,
        fontsize=LEGEND_SIZE,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def plot_solar_system_fixed(axis, rows: list[dict[str, str]]) -> None:
    rows = ordered_rows(rows, SOLAR_ORDER)
    labels = [wrap_label(row["planet_label"]) for row in rows]
    values = [to_float(row["mass_loss_rate_g_s"]) for row in rows]

    axis.bar(labels, values, color=["#c97c5d", "#4f6d7a", "#c0a080", "#7a9e7e"])
    styled_axes(axis, "Solar System Analogues", "Planet", "Total mass-loss rate [g s$^{-1}$]")
    science_axis(axis)
    axis.tick_params(axis="x", rotation=15, labelsize=TICK_SIZE)


def plot_real_reference_systems(axis, rows: list[dict[str, str]]) -> None:
    rows = ordered_rows(rows, REAL_ORDER)
    labels = [row["planet_label"] for row in rows]
    values = [to_float(row["mass_loss_rate_g_s"]) for row in rows]

    axis.bar(labels, values, color=["#d17c8f", "#8a7db8", "#5c9ead", "#d9a441"])
    styled_axes(axis, "Real Reference Systems", "System", "Total mass-loss rate [g s$^{-1}$]")
    science_axis(axis)
    axis.tick_params(axis="x", rotation=18, labelsize=TICK_SIZE)


def plot_positive_sweep(
    axis,
    rows: list[dict[str, str]],
    *,
    x_key: str,
    xlabel: str,
    title: str,
    xscale: str = "linear",
    show_zero_floor: bool = False,
    context_lines: list[str] | None = None,
) -> None:
    sorted_rows = sorted(rows, key=lambda row: to_float(row[x_key]))
    x_all = [to_float(row[x_key]) for row in sorted_rows]
    y_all = [to_float(row["mass_loss_rate_g_s"]) for row in sorted_rows]

    positive = [(x, y) for x, y in zip(x_all, y_all) if y > 0.0]
    if positive:
        x_pos = [x for x, _ in positive]
        y_pos = [y for _, y in positive]
        axis.plot(x_pos, y_pos, marker="o", linewidth=2.2, color="#4f6d7a")
        axis.set_yscale("log")
        if xscale == "log":
            axis.set_xscale("log")
        if show_zero_floor and len(positive) != len(sorted_rows):
            floor = min(y_pos) / 100.0
            zero_x = [x for x, y in zip(x_all, y_all) if y == 0.0]
            if zero_x:
                axis.scatter(
                    zero_x,
                    [floor] * len(zero_x),
                    marker="v",
                    s=42,
                    facecolors="white",
                    edgecolors="#b24c63",
                    linewidths=1.4,
                    zorder=4,
                )
                axis.text(
                    0.98,
                    0.04,
                    "Zero values clipped\nat plot floor",
                    transform=axis.transAxes,
                    fontsize=LEGEND_SIZE - 1,
                    ha="right",
                    va="bottom",
                )
    else:
        axis.plot(x_all, y_all, marker="o", linewidth=2.2, color="#4f6d7a")
        science_axis(axis)
        axis.text(
            0.5,
            0.55,
            "All completed cases\ncurrently give zero total escape",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=TICK_SIZE - 1,
        )

    styled_axes(axis, title, xlabel, "Total mass-loss rate [g s$^{-1}$]")
    if positive:
        axis.grid(True, which="major", axis="both", alpha=0.25)
        axis.grid(True, which="minor", axis="y", alpha=0.12)
    else:
        science_axis(axis)
    if context_lines:
        add_context_text(axis, context_lines)


def plot_distance_sweep(axis, rows: list[dict[str, str]]) -> None:
    plot_positive_sweep(
        axis,
        rows,
        x_key="distance_AU",
        xlabel="Orbital distance [AU]",
        title="Distance Sweep",
        xscale="log",
        context_lines=["Inflated hot Jupiter", "F8 star"],
    )


def plot_p0_sweep(axis, rows: list[dict[str, str]]) -> None:
    plot_positive_sweep(
        axis,
        rows,
        x_key="P0_bar",
        xlabel="$P_0$ [bar]",
        title="$P_0$ Sweep",
        xscale="log",
        context_lines=["Inflated hot Jupiter", "F8 star", "0.05 AU"],
    )


def plot_mu_sweep(axis, rows: list[dict[str, str]]) -> None:
    plot_positive_sweep(
        axis,
        rows,
        x_key="mu_amu",
        xlabel="Mean molecular weight [amu]",
        title="$\\mu$ Sweep",
        xscale="log",
        context_lines=["Inflated hot Jupiter", "F8 star", "0.05 AU"],
    )


def plot_surface_gravity_sweep(axis, rows: list[dict[str, str]]) -> None:
    plot_positive_sweep(
        axis,
        rows,
        x_key="surface_gravity_m_s2",
        xlabel="Surface gravity [m s$^{-2}$]",
        title="Surface Gravity Sweep",
        show_zero_floor=True,
        context_lines=["Super-Earth rocky planet", "F8 star", "0.1 AU"],
    )


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
    p0_rows = read_total_rows("p0_sweep")
    mu_rows = read_total_rows("mu_sweep")
    gravity_rows = read_total_rows("surface_gravity_sweep")

    save_family_figure("solar_system_analogues_total_mass_loss.pdf", plot_solar_system_fixed, solar_rows)
    save_family_figure("distance_sweep_total_mass_loss.pdf", plot_distance_sweep, distance_rows)
    save_family_figure("real_reference_systems_total_mass_loss.pdf", plot_real_reference_systems, real_rows)
    save_family_figure("p0_sweep_total_mass_loss.pdf", plot_p0_sweep, p0_rows)
    save_family_figure("mu_sweep_total_mass_loss.pdf", plot_mu_sweep, mu_rows)
    save_family_figure("surface_gravity_sweep_total_mass_loss.pdf", plot_surface_gravity_sweep, gravity_rows)

    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    plot_solar_system_fixed(axes[0, 0], solar_rows)
    plot_distance_sweep(axes[0, 1], distance_rows)
    plot_real_reference_systems(axes[0, 2], real_rows)
    plot_p0_sweep(axes[1, 0], p0_rows)
    plot_mu_sweep(axes[1, 1], mu_rows)
    plot_surface_gravity_sweep(axes[1, 2], gravity_rows)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "finished_families_total_mass_loss_summary.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
