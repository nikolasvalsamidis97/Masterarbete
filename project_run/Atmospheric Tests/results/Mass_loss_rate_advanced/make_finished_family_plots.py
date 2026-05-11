from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_mass_loss_finished_family_plots"),
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter

sys.path.append(str(Path(__file__).resolve().parents[4]))

from project_func.Templates.Systems.real_mass_loss_reference_systems import (
    REAL_MASS_LOSS_REFERENCE_SYSTEMS,
)


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
SOLAR_EFFECTIVELY_ZERO_THRESHOLD_GS = 1.0e-30
SOLAR_PLOT_YMIN = 1.0e-15
SOLAR_EFFECTIVE_ZERO_BAR_TOP = 1.0e-12
REAL_ORDER = [
    "gj1132_b",
    "55cnc_e",
    "gj1214_b",
    "hd56414_b",
    "gj436_b",
    "51peg_b",
    "hd209458_b",
    "wasp174_b",
    "wasp193_b",
    "kelt9_b",
]
REAL_EFFECTIVELY_ZERO_THRESHOLD_GS = 1.0e-8
REAL_PLOT_YMIN = 1.0e-15
REAL_PLOT_YMAX = 1.0e10
REAL_EFFECTIVE_ZERO_BAR_TOP = 1.0e-12
REAL_TYPE_COLORS = {
    "rocky": "#c97c5d",
    "sub_neptune": "#8a7db8",
    "neptune": "#5c9ead",
    "hot_jupiter": "#d9a441",
    "ultra_hot_jupiter": "#d17c8f",
}
REAL_TYPE_LABELS = {
    "rocky": "Rocky",
    "sub_neptune": "Sub-Neptune",
    "neptune": "Neptune",
    "hot_jupiter": "Hot Jupiter",
    "ultra_hot_jupiter": "Ultra-hot Jupiter",
}
SOLAR_TYPE_COLORS = {
    "rocky": REAL_TYPE_COLORS["rocky"],
    "gas_giant": REAL_TYPE_COLORS["hot_jupiter"],
}
SOLAR_TYPE_LABELS = {
    "rocky": "Rocky",
    "gas_giant": "Gas giant",
}


def read_total_rows(family_name: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / FAMILY_FILES[family_name]
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row for row in rows if row["species"] == "TOTAL_INCLUDED_SPECIES"]


def to_float(value: str) -> float:
    return float(value) if value not in {"", "nan"} else 0.0


def plain_tick_text(value: float) -> str:
    text = f"{value:.6g}"
    if "e" in text or "E" in text:
        return text
    if "." in text:
        return text.rstrip("0").rstrip(".")
    return text


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


def add_context_text(axis, lines: list[str], side: str = "left") -> None:
    x = 0.03 if side == "left" else 0.97
    ha = "left" if side == "left" else "right"
    axis.text(
        x,
        0.96,
        "\n".join(lines),
        transform=axis.transAxes,
        fontsize=LEGEND_SIZE,
        va="top",
        ha=ha,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def plot_solar_system_fixed(axis, rows: list[dict[str, str]]) -> None:
    rows = ordered_rows(rows, SOLAR_ORDER)
    labels = [wrap_label(row["planet_label"]) for row in rows]
    values = [to_float(row["mass_loss_rate_g_s"]) for row in rows]
    positions = list(range(len(rows)))
    bar_width = 0.82

    def solar_type(planet_key: str) -> str:
        return "gas_giant" if planet_key == "cold_jupiter" else "rocky"

    colors = [SOLAR_TYPE_COLORS[solar_type(row["planet"])] for row in rows]
    plot_values = [
        value if value > SOLAR_EFFECTIVELY_ZERO_THRESHOLD_GS else SOLAR_EFFECTIVE_ZERO_BAR_TOP
        for value in values
    ]
    effectively_zero_positions = [
        position
        for position, value in zip(positions, values)
        if value <= SOLAR_EFFECTIVELY_ZERO_THRESHOLD_GS
    ]

    positive_values = [value for value in values if value > SOLAR_EFFECTIVELY_ZERO_THRESHOLD_GS]
    ymax = 1.0e-4
    if positive_values:
        ymax = 10 ** (math.ceil(math.log10(max(positive_values))) + 1)

    axis.bar(positions, plot_values, color=colors, width=bar_width, align="center")
    axis.set_yscale("log")
    axis.set_ylim(SOLAR_PLOT_YMIN, ymax)

    if effectively_zero_positions:
        for pos in effectively_zero_positions:
            left = pos - bar_width / 2.0
            right = pos + bar_width / 2.0
            axis.plot(
                [left, right],
                [SOLAR_PLOT_YMIN, SOLAR_EFFECTIVE_ZERO_BAR_TOP],
                color="black",
                linewidth=1.4,
                zorder=5,
            )
            axis.plot(
                [left, right],
                [SOLAR_EFFECTIVE_ZERO_BAR_TOP, SOLAR_PLOT_YMIN],
                color="black",
                linewidth=1.4,
                zorder=5,
            )

    styled_axes(axis, "Solar System Analogues", "", "Mass-loss rate [log(g s$^{-1}$)]")
    axis.grid(True, which="major", axis="y", alpha=0.25)
    axis.grid(True, which="minor", axis="y", alpha=0.12)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=15, ha="right", rotation_mode="anchor")
    axis.set_xlim(-0.5, len(rows) - 0.5)
    axis.tick_params(axis="x", labelsize=TICK_SIZE)
    axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _pos: (
                f"{int(round(math.log10(value)))}"
                if value > 0.0 and math.isclose(math.log10(value), round(math.log10(value)), abs_tol=1.0e-10)
                else ""
            )
        )
    )
    axis.tick_params(axis="y", labelsize=TICK_SIZE)
    present_types = []
    for row in rows:
        kind = solar_type(row["planet"])
        if kind not in present_types:
            present_types.append(kind)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=8,
            markerfacecolor=SOLAR_TYPE_COLORS[kind],
            markeredgecolor=SOLAR_TYPE_COLORS[kind],
            label=SOLAR_TYPE_LABELS[kind],
        )
        for kind in present_types
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markersize=8,
            markeredgewidth=1.6,
            color="black",
            label="No mass-loss",
        )
    )
    axis.legend(
        handles=legend_handles,
        title="Planet type",
        loc="upper right",
        frameon=False,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
        handlelength=0.8,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.2,
    )


def plot_real_reference_systems(axis, rows: list[dict[str, str]]) -> None:
    rows = ordered_rows(rows, REAL_ORDER)
    nonzero_rows = [
        row for row in rows if to_float(row["mass_loss_rate_g_s"]) > REAL_EFFECTIVELY_ZERO_THRESHOLD_GS
    ]
    effectively_zero_rows = [
        row for row in rows if to_float(row["mass_loss_rate_g_s"]) <= REAL_EFFECTIVELY_ZERO_THRESHOLD_GS
    ]
    rows = nonzero_rows + effectively_zero_rows
    labels = [row["planet_label"] for row in rows]
    values = [to_float(row["mass_loss_rate_g_s"]) for row in rows]
    positions = list(range(len(rows)))
    bar_width = 0.82

    def system_type(system_key: str) -> str:
        system_def = REAL_MASS_LOSS_REFERENCE_SYSTEMS[str(system_key)]
        category = str(system_def.get("category", "")).strip().lower()
        exobase_key = str(system_def.get("exobase_template_key", "")).strip().lower()
        if category == "rocky":
            return "rocky"
        if category == "sub_neptune":
            return "sub_neptune"
        if category == "neptune":
            return "neptune"
        if exobase_key == "ultra_hot_jupiter":
            return "ultra_hot_jupiter"
        return "hot_jupiter"

    colors = [REAL_TYPE_COLORS[system_type(row["planet"])] for row in rows]
    plot_values = [
        value if value > REAL_EFFECTIVELY_ZERO_THRESHOLD_GS else REAL_EFFECTIVE_ZERO_BAR_TOP
        for value in values
    ]
    effectively_zero_positions = [
        position
        for position, value in zip(positions, values)
        if value <= REAL_EFFECTIVELY_ZERO_THRESHOLD_GS
    ]

    axis.bar(positions, plot_values, color=colors, width=bar_width, align="center")
    axis.set_yscale("log")
    axis.set_ylim(REAL_PLOT_YMIN, REAL_PLOT_YMAX)

    if effectively_zero_positions:
        for pos in effectively_zero_positions:
            left = pos - bar_width / 2.0
            right = pos + bar_width / 2.0
            axis.plot(
                [left, right],
                [REAL_PLOT_YMIN, REAL_EFFECTIVE_ZERO_BAR_TOP],
                color="black",
                linewidth=1.4,
                zorder=5,
            )
            axis.plot(
                [left, right],
                [REAL_EFFECTIVE_ZERO_BAR_TOP, REAL_PLOT_YMIN],
                color="black",
                linewidth=1.4,
                zorder=5,
            )

    styled_axes(axis, "Real Reference Systems", "", "Mass-loss rate [log(g s$^{-1}$)]")
    axis.grid(True, which="major", axis="y", alpha=0.25)
    axis.grid(True, which="minor", axis="y", alpha=0.12)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=28, ha="right", rotation_mode="anchor")
    axis.set_xlim(-0.5, len(rows) - 0.5)
    axis.tick_params(axis="x", labelsize=TICK_SIZE)
    axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _pos: (
                f"{int(round(math.log10(value)))}"
                if value > 0.0 and math.isclose(math.log10(value), round(math.log10(value)), abs_tol=1.0e-10)
                else ""
            )
        )
    )
    axis.tick_params(axis="y", labelsize=TICK_SIZE)
    present_types = []
    for row in rows:
        kind = system_type(row["planet"])
        if kind not in present_types:
            present_types.append(kind)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=8,
            markerfacecolor=REAL_TYPE_COLORS[kind],
            markeredgecolor=REAL_TYPE_COLORS[kind],
            label=REAL_TYPE_LABELS[kind],
        )
        for kind in present_types
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markersize=8,
            markeredgewidth=1.6,
            color="black",
            label="No mass-loss",
        )
    )
    axis.legend(
        handles=legend_handles,
        title="Planet type",
        loc="center right",
        bbox_to_anchor=(0.98, 0.62),
        borderaxespad=0.0,
        frameon=False,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
        handlelength=0.8,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.2,
    )


def plot_positive_sweep(
    axis,
    rows: list[dict[str, str]],
    *,
    x_key: str,
    xlabel: str,
    title: str,
    xscale: str = "linear",
    yscale: str = "log",
    show_zero_floor: bool = False,
    context_lines: list[str] | None = None,
    x_ticks_at_data: bool = False,
    y_plain_scale_power: int | None = None,
    y_log_exponents_only: bool = False,
    context_side: str = "left",
) -> None:
    sorted_rows = sorted(rows, key=lambda row: to_float(row[x_key]))
    x_all = [to_float(row[x_key]) for row in sorted_rows]
    y_all = [to_float(row["mass_loss_rate_g_s"]) for row in sorted_rows]

    positive = [(x, y) for x, y in zip(x_all, y_all) if y > 0.0]
    if positive:
        x_pos = [x for x, _ in positive]
        y_pos = [y for _, y in positive]
        axis.plot(x_pos, y_pos, marker="o", linewidth=2.2, color="#4f6d7a")
        if yscale == "log":
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

    ylabel = "Mass-loss rate [g s$^{-1}$]"
    if yscale == "log":
        ylabel = "Mass-loss rate [log(g s$^{-1}$)]"
    styled_axes(axis, title, xlabel, ylabel)
    if positive:
        axis.grid(True, which="major", axis="both", alpha=0.25)
        if yscale == "log":
            axis.grid(True, which="minor", axis="y", alpha=0.12)
    else:
        science_axis(axis)
    if x_ticks_at_data:
        axis.set_xticks(x_all)
        axis.set_xticklabels([plain_tick_text(x) for x in x_all])
    if y_log_exponents_only:
        axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        axis.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _pos: (
                    f"{int(round(math.log10(value)))}"
                    if value > 0.0 and math.isclose(math.log10(value), round(math.log10(value)), abs_tol=1.0e-10)
                    else ""
                )
            )
        )
    if y_plain_scale_power is not None:
        y_scale = 10 ** y_plain_scale_power
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: plain_tick_text(value / y_scale)))
        axis.text(
            0.0,
            1.01,
            rf"$10^{{{y_plain_scale_power}}}$",
            transform=axis.transAxes,
            fontsize=TICK_SIZE,
            ha="left",
            va="bottom",
        )
    if context_lines:
        add_context_text(axis, context_lines, side=context_side)


def plot_distance_sweep(axis, rows: list[dict[str, str]]) -> None:
    plot_positive_sweep(
        axis,
        rows,
        x_key="distance_AU",
        xlabel="Orbital distance [AU]",
        title="Distance Sweep",
        xscale="log",
        yscale="log",
        x_ticks_at_data=True,
        y_log_exponents_only=True,
        context_lines=["Inflated hot Jupiter", r"$T_{\rm eff}=6000$ K"],
        context_side="right",
    )


def plot_p0_sweep(axis, rows: list[dict[str, str]]) -> None:
    plot_positive_sweep(
        axis,
        rows,
        x_key="P0_bar",
        xlabel="$P_0$ [bar]",
        title="$P_0$ Sweep",
        xscale="log",
        yscale="linear",
        y_plain_scale_power=5,
        context_lines=["Inflated hot Jupiter", r"$T_{\rm eff}=6000$ K", "0.05 AU"],
        context_side="left",
    )


def plot_mu_sweep(axis, rows: list[dict[str, str]]) -> None:
    plot_positive_sweep(
        axis,
        rows,
        x_key="mu_amu",
        xlabel="Mean molecular weight [amu]",
        title="$\\mu$ Sweep",
        xscale="log",
        yscale="log",
        x_ticks_at_data=True,
        y_log_exponents_only=True,
        context_lines=["Inflated hot Jupiter", r"$T_{\rm eff}=6000$ K", "0.05 AU"],
        context_side="right",
    )


def plot_surface_gravity_sweep(axis, rows: list[dict[str, str]]) -> None:
    plot_positive_sweep(
        axis,
        rows,
        x_key="surface_gravity_m_s2",
        xlabel="Surface gravity [m s$^{-2}$]",
        title="Surface Gravity Sweep",
        yscale="log",
        y_log_exponents_only=True,
        show_zero_floor=True,
        context_lines=["Super-Earth rocky planet", r"$T_{\rm eff}=6000$ K", "0.1 AU"],
        context_side="right",
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
