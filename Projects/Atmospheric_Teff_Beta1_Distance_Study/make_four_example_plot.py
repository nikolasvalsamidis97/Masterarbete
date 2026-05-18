import pathlib
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, LogLocator
import numpy as np
from astropy import constants as const

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import plot_by_txt_file as plot_txt
from project_utils.exobase_table_path import resolve_exobase_table_path
from project_utils.r_beta1_table_sources import (
    discover_rbeta1_table_files,
    find_species_rbeta1_table,
)
from Templates.Planets.planet_templates import PLANET_TEMPLATES


TABLES_BASE_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Teff_study"
EXOBASE_TABLE = (
    resolve_exobase_table_path(pathlib.Path(__file__).resolve().parents[2])
)
OUTPUT_DIR = TABLES_BASE_DIR / "r_at_beta1" / "four_example"
OUTPUT_PDF = OUTPUT_DIR / "four_example_planets.pdf"

PLANET_ORDER = [
    "earth_like",
    "mercury_like",
    "inflated_hot_jupiter",
    "sub_neptune",
]

PLOT_OVERRIDES: Dict[str, Any] = {
    "figsize": (5.3, 5.3),
    "font_sizes": {
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    },
    "x_scale": "log",
    "y_scale": "log",
    "x_divide_by": 1e4,
    "x_label": r"Stellar $T_{\rm eff}$ [$10^4$ K]",
    "y_label": r"$\log_{10}\!\left(r_{\beta=1} / R_{\rm p}\right)$",
    "title_template": "{planet}: {species} at multiple orbital distances",
    "legend_title": "Distance [AU]",
    "line_colormap": "viridis",
    "line_alpha": 0.9,
    "marker": "o",
    "marker_size": 3.5,
    "line_width": 1.6,
    "grid": True,
    "grid_alpha": 0.3,
    "tight_layout": True,
    "left_margin": 0.08,
    "right_margin": 0.74,
    "bottom_margin": 0.09,
    "top_margin": 0.96,
    "panel_wspace": 0.0,
    "panel_hspace": 0.03,
    "title_pad": 10,
    "legend_anchor": (0.765, 0.5),
    "x_ticks": [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5],
    "x_tick_labels": ["0.2", "0.4", "0.6", "0.8", "1", "2", "3", "4", "5"],
}


def safe_name(value: str) -> str:
    return str(value).replace(" ", "").replace("/", "_")


def load_exobase_heights(table_path: pathlib.Path) -> Dict[tuple[str, str], float]:
    if not table_path.exists():
        return {}

    import csv

    heights: Dict[tuple[str, str], float] = {}
    with table_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            planet = row.get("planet", "").strip()
            species = row.get("species", "").strip()
            z_value = row.get("z_exobase_km", "")
            if not planet or not species:
                continue
            try:
                heights[(planet, species)] = float(z_value)
            except (TypeError, ValueError):
                continue
    return heights


def pretty_planet_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def pretty_species_name(species: str) -> str:
    parts = str(species).split()
    if len(parts) != 2:
        return str(species)

    element, stage = parts
    charge_map = {
        "I": rf"$\mathrm{{{element}}}^{{0}}$",
        "II": rf"$\mathrm{{{element}}}^{{+}}$",
        "III": rf"$\mathrm{{{element}}}^{{2+}}$",
        "IV": rf"$\mathrm{{{element}}}^{{3+}}$",
    }
    return charge_map.get(stage, str(species))


def log10_exponent_label(value: float, _position: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""

    exponent = np.log10(value)
    rounded = round(exponent)
    if not np.isclose(exponent, rounded, atol=1e-10):
        return ""
    return f"{int(rounded)}"


def neutral_exobase_species(species: str) -> str:
    parts = str(species).split()
    if len(parts) != 2:
        return str(species)

    element, stage = parts
    if stage in {"II", "III"}:
        return f"{element} I"
    return str(species)


def exobase_height_km(metadata: Dict[str, str], exobase_heights: Dict[tuple[str, str], float]) -> float | None:
    planet = metadata.get("planet", "")
    species = metadata.get("species", "")
    if not planet or not species:
        return None

    z_exobase_km = exobase_heights.get((planet, species))
    if z_exobase_km is None:
        z_exobase_km = exobase_heights.get((planet, neutral_exobase_species(species)))
    return z_exobase_km


def planet_radius_km(metadata: Dict[str, str]) -> float | None:
    try:
        radius_rjup = float(metadata.get("planet_radius_Rjup", ""))
    except (TypeError, ValueError):
        return None
    return radius_rjup * const.R_jup.to_value("km")


def most_common_available_species(planet_key: str) -> str:
    planet_case = PLANET_TEMPLATES[planet_key]
    discovered_tables = discover_rbeta1_table_files(TABLES_BASE_DIR)
    if not discovered_tables:
        raise FileNotFoundError(f"No r_beta1 txt tables found under: {TABLES_BASE_DIR}")

    ranked_species = sorted(
        planet_case["composition"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    for species, _ in ranked_species:
        table_path = find_species_rbeta1_table(TABLES_BASE_DIR, planet_key, species)
        if table_path is not None:
            return species

    raise FileNotFoundError(
        f"No saved r_beta1 txt file found for any composition species of {planet_key}"
    )


def selected_table_paths() -> list[pathlib.Path]:
    table_paths: list[pathlib.Path] = []
    for planet_key in PLANET_ORDER:
        species = most_common_available_species(planet_key)
        table_path = find_species_rbeta1_table(TABLES_BASE_DIR, planet_key, species)
        if table_path is None:
            raise FileNotFoundError(f"Missing selected example table: {table_path}")
        table_paths.append(table_path)
    return table_paths


def plot_example_panel(
    ax: plt.Axes,
    table_path: pathlib.Path,
    exobase_heights: Dict[tuple[str, str], float],
) -> tuple[list, list, tuple[float, float] | None, tuple[float, float] | None]:
    metadata, columns, data = plot_txt.parse_header_and_table(table_path)
    series_info = plot_txt.extract_series(columns)

    x_values = data[:, 0]
    y_matrix = data[:, 1:]

    cmap = getattr(plt.cm, PLOT_OVERRIDES["line_colormap"])
    color_values = np.linspace(0.15, 0.9, len(series_info))
    x_plot = x_values / PLOT_OVERRIDES["x_divide_by"]
    handles = []
    labels = []
    plotted_x_values = []
    x_limits = None
    y_limits = None
    for idx, ((series_value, _), color_value) in enumerate(zip(series_info, color_values)):
        y_values = y_matrix[:, idx]
        valid = np.isfinite(x_plot) & np.isfinite(y_values) & (x_plot > 0) & (y_values > 0)
        if not np.any(valid):
            continue
        x_series = x_plot[valid]
        y_series = y_values[valid]
        label = str(series_value)
        (line,) = ax.plot(
            x_series,
            y_series,
            marker=PLOT_OVERRIDES["marker"],
            markersize=PLOT_OVERRIDES["marker_size"],
            linewidth=PLOT_OVERRIDES["line_width"],
            color=cmap(color_value),
            alpha=PLOT_OVERRIDES["line_alpha"],
            label=label,
        )
        handles.append(line)
        labels.append(label)
        plotted_x_values.append(x_series)

    ax.set_xscale(PLOT_OVERRIDES["x_scale"])
    ax.set_yscale(PLOT_OVERRIDES["y_scale"])
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
    if plotted_x_values:
        all_x = np.concatenate(plotted_x_values)
        x_min = float(np.nanmin(all_x))
        x_max = float(np.nanmax(all_x))
        x_limits = (x_min, x_max)

    positive_y = y_matrix[np.isfinite(y_matrix) & (y_matrix > 0)]
    if positive_y.size > 0:
        y_min = float(np.nanmin(positive_y))
        y_max = float(np.nanmax(positive_y))
        y_limits = (y_min / 1.15, y_max * 1.15)

    planet = metadata.get("planet", table_path.parent.name.replace("_r_beta1", ""))
    species = metadata.get("species", table_path.stem.replace("_r_beta1", ""))
    title_parts = [
        pretty_planet_name(planet),
        pretty_species_name(species),
    ]
    ax.set_title(" | ".join(title_parts), pad=PLOT_OVERRIDES["title_pad"])
    if PLOT_OVERRIDES.get("grid", True):
        ax.grid(True, alpha=PLOT_OVERRIDES["grid_alpha"])
    ax.set_box_aspect(1)

    return handles, labels, x_limits, y_limits


def apply_shared_figure_layout(
    fig: plt.Figure,
    axes: np.ndarray,
    legend_handles: list,
    legend_labels: list,
    x_limits_list: list[tuple[float, float]],
    y_limits_list: list[tuple[float, float]],
) -> None:
    axes_flat = np.atleast_1d(axes).ravel()
    tick_label_size = PLOT_OVERRIDES["font_sizes"]["xtick.labelsize"]
    tick_pairs = []

    if x_limits_list:
        x_min = min(limit[0] for limit in x_limits_list)
        x_max = max(limit[1] for limit in x_limits_list)
        tick_pairs = [
            (tick, label)
            for tick, label in zip(PLOT_OVERRIDES["x_ticks"], PLOT_OVERRIDES["x_tick_labels"])
            if x_min <= float(tick) <= x_max
        ]
        for ax in axes_flat:
            ax.set_xlim(x_min, x_max)
            if tick_pairs:
                ax.xaxis.set_major_locator(FixedLocator([tick for tick, _ in tick_pairs]))
                ax.xaxis.set_major_formatter(FixedFormatter([label for _, label in tick_pairs]))

    if y_limits_list:
        y_min = min(limit[0] for limit in y_limits_list)
        y_max = max(limit[1] for limit in y_limits_list)
        for ax in axes_flat:
            ax.set_ylim(y_min, y_max)

    if axes.ndim == 2 and tick_pairs:
        left_column_labels = [label for _, label in tick_pairs]
        if left_column_labels:
            left_column_labels[-1] = ""
        for ax in axes[:, 0]:
            ax.xaxis.set_major_locator(FixedLocator([tick for tick, _ in tick_pairs]))
            ax.xaxis.set_major_formatter(FixedFormatter(left_column_labels))

    for ax in axes_flat:
        ax.label_outer()

    if legend_handles:
        legend_fontsize = PLOT_OVERRIDES["font_sizes"]["legend.fontsize"]
        fig.legend(
            legend_handles,
            legend_labels,
            title=PLOT_OVERRIDES["legend_title"],
            loc="center left",
            bbox_to_anchor=PLOT_OVERRIDES["legend_anchor"],
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )

    label_center_x = 0.5 * (PLOT_OVERRIDES["left_margin"] + PLOT_OVERRIDES["right_margin"])
    fig.supxlabel(PLOT_OVERRIDES["x_label"], fontsize=tick_label_size, x=label_center_x, y=0.045)
    fig.supylabel(PLOT_OVERRIDES["y_label"], fontsize=tick_label_size, x=0.006)
    fig.subplots_adjust(
        left=PLOT_OVERRIDES["left_margin"],
        right=PLOT_OVERRIDES["right_margin"],
        bottom=PLOT_OVERRIDES["bottom_margin"],
        top=PLOT_OVERRIDES["top_margin"],
        wspace=PLOT_OVERRIDES["panel_wspace"],
        hspace=PLOT_OVERRIDES["panel_hspace"],
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exobase_heights = load_exobase_heights(EXOBASE_TABLE)
    table_paths = selected_table_paths()

    plt.rcParams.update(PLOT_OVERRIDES["font_sizes"])

    panel_width, panel_height = PLOT_OVERRIDES["figsize"]
    fig, axes = plt.subplots(2, 2, figsize=(2 * panel_width, 2 * panel_height), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    legend_handles = []
    legend_labels = []
    x_limits_list = []
    y_limits_list = []

    for ax, table_path in zip(axes_flat, table_paths):
        handles, labels, x_limits, y_limits = plot_example_panel(ax, table_path, exobase_heights)
        if handles and not legend_handles:
            legend_handles = handles
            legend_labels = labels
        if x_limits is not None:
            x_limits_list.append(x_limits)
        if y_limits is not None:
            y_limits_list.append(y_limits)
    if PLOT_OVERRIDES.get("tight_layout", True):
        apply_shared_figure_layout(
            fig,
            axes,
            legend_handles,
            legend_labels,
            x_limits_list,
            y_limits_list,
        )
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("Selected example species:")
    for table_path in table_paths:
        metadata, _, _ = plot_txt.parse_header_and_table(table_path)
        print(f"  {metadata.get('planet', table_path.parent.name)} -> {metadata.get('species', table_path.stem)}")
    print(f"Saved plot: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
