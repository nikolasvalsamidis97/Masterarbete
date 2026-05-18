import pathlib
import sys
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
from astropy import constants as const

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import plot_by_txt_file as plot_txt
from project_utils.exobase_table_path import resolve_exobase_table_path
from project_utils.r_beta1_table_sources import (
    discover_rbeta1_table_files,
    existing_rbeta1_roots,
    resolve_rbeta1_table_file,
)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
# Parent folder containing one or more r_beta1 output roots.
TABLES_BASE_DIR = (
    pathlib.Path(__file__).resolve().parent
    / "results"
    / "Plots"
    / "Teff_study"
)
EXOBASE_TABLE = (
    resolve_exobase_table_path(pathlib.Path(__file__).resolve().parents[2])
)

# Optional single-file selection. Set to a relative path under any discovered root,
# for example "hot_jupiter_r_beta1/NaI_r_beta1.txt", to plot only one file.
# Leave as None to plot all txt files in all planet subfolders.

SELECTED_TABLE_FILE = None

# These are generic plotting defaults. Change anything here if you want a
# different visual style without touching the saved data table.
PLOT_OVERRIDES: Dict[str, Any] = {
    "figsize": (8.5, 5.5),
    "font_sizes": {
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    },
    "x_scale": "log",
    "y_scale": "log",
    "x_divide_by": 1e4,
    "x_label": r"Stellar $T_{\rm eff}$ [$10^4$ K]",
    "y_label": r"$r_{\beta=1} / R_{\rm p}$",
    "title_template": "{planet}: {species} at multiple orbital distances",
    "legend_title": "Distance",
    "line_colormap": "viridis",
    "line_alpha": 0.9,
    "marker": "o",
    "marker_size": 3.5,
    "line_width": 1.6,
    "grid": True,
    "grid_alpha": 0.3,
    "tight_layout": True,
    "right_margin": 0.78,
    "x_ticks": [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5],
    "x_tick_labels": ["0.2", "0.4", "0.6", "0.8", "1", "2", "3", "4", "5"],
    "xlim": (0.26, 5.0),
}

SAVE_PDF = True
SHOW_PLOT = False

plot_txt.SAVE_PDF = SAVE_PDF
plot_txt.SHOW_PLOT = SHOW_PLOT


def load_exobase_heights(table_path: pathlib.Path) -> Dict[tuple[str, str], float]:
    if not table_path.exists():
        print(f"Critical-height table not found, skipping overlays: {table_path}")
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
        "I": element,
        "II": rf"{element}$^+$",
        "III": rf"{element}$^{{2+}}$",
        "IV": rf"{element}$^{{3+}}$",
    }
    return charge_map.get(stage, str(species))


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


def table_has_any_positive_finite_y(table_file: pathlib.Path) -> bool:
    with table_file.open("r", encoding="utf-8") as f:
        data_started = False
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if not data_started:
                data_started = True
                continue

            parts = [part.strip() for part in line.split("\t")]
            for value in parts[1:]:
                if value.lower() == "nan":
                    continue
                try:
                    y = float(value)
                except ValueError:
                    continue
                if np.isfinite(y) and y > 0:
                    return True
    return False


def plot_table_with_exobase_title(
    table_file: pathlib.Path,
    overrides: Dict[str, Any],
    exobase_heights: Dict[tuple[str, str], float],
) -> pathlib.Path:
    metadata, columns, data = plot_txt.parse_header_and_table(table_file)
    plot_config = plot_txt.build_plot_config(overrides)
    series_info = plot_txt.extract_series(columns)

    x_values = data[:, 0]
    y_matrix = data[:, 1:]

    plt.rcParams.update(plot_config["font_sizes"])
    fig, ax = plt.subplots(figsize=plot_config["figsize"])

    cmap = getattr(plt.cm, plot_config["line_colormap"])
    color_values = np.linspace(0.15, 0.9, len(series_info))

    x_divide_by = plot_config.get("x_divide_by", 1.0)
    x_plot = x_values / x_divide_by
    series_unit = metadata.get("series_unit", "")
    legend_title = plot_config.get("legend_title") or metadata.get("series_label", "Series")

    for idx, ((series_value, _), color_value) in enumerate(zip(series_info, color_values)):
        y_values = y_matrix[:, idx]
        label = f"{series_value} {series_unit}" if series_unit else str(series_value)
        ax.plot(
            x_plot,
            y_values,
            marker=plot_config["marker"],
            markersize=plot_config["marker_size"],
            linewidth=plot_config["line_width"],
            color=cmap(color_value),
            alpha=plot_config["line_alpha"],
            label=label,
        )

    ax.set_xscale(plot_config["x_scale"])
    ax.set_yscale(plot_config["y_scale"])

    if plot_config.get("x_ticks") is not None:
        ax.xaxis.set_major_locator(FixedLocator(plot_config["x_ticks"]))
    if plot_config.get("x_tick_labels") is not None:
        ax.xaxis.set_major_formatter(FixedFormatter(plot_config["x_tick_labels"]))
    if plot_config.get("xlim") is not None:
        ax.set_xlim(*plot_config["xlim"])
    if plot_config.get("ylim") is None:
        positive_y = y_matrix[np.isfinite(y_matrix) & (y_matrix > 0)]
        if positive_y.size > 0:
            y_min = float(np.nanmin(positive_y))
            y_max = float(np.nanmax(positive_y))
            if plot_config["y_scale"] == "log" and y_min > 0:
                ax.set_ylim(y_min / 1.15, y_max * 1.15)
            else:
                pad = 0.05 * max(y_max - y_min, 1.0)
                ax.set_ylim(y_min - pad, y_max + pad)
    if plot_config.get("ylim") is not None:
        ax.set_ylim(*plot_config["ylim"])

    ax.set_xlabel(plot_config.get("x_label", metadata.get("x_label", columns[0])))
    ax.set_ylabel(plot_config.get("y_label", metadata.get("y_label", "y")))

    planet = metadata.get("planet", table_file.parent.name)
    species = metadata.get("species", table_file.stem)
    z_exobase_km = exobase_height_km(metadata, exobase_heights)
    planet_title = pretty_planet_name(planet)
    species_title = pretty_species_name(species)
    radius_km = planet_radius_km(metadata)
    title_parts = [planet_title, species_title]
    if radius_km is not None and np.isfinite(radius_km):
        title_parts.append(rf"$R_{{\rm p}} = {radius_km:.0f}\ \mathrm{{km}}$")
    if z_exobase_km is not None and np.isfinite(z_exobase_km):
        title_parts.append(rf"$r_{{\rm exo}} = {float(z_exobase_km):.0f}\ \mathrm{{km}}$")
    ax.set_title(" | ".join(title_parts))

    if plot_config.get("grid", True):
        ax.grid(True, alpha=plot_config.get("grid_alpha", 0.3))

    ax.legend(title=legend_title)

    right_margin = plot_config.get("right_margin")
    if right_margin is not None:
        fig.subplots_adjust(right=right_margin)

    if plot_config.get("tight_layout", True):
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

    pdf_path = table_file.with_suffix(".pdf")
    if SAVE_PDF:
        fig.savefig(pdf_path)

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    return pdf_path


def main() -> None:
    roots = existing_rbeta1_roots(TABLES_BASE_DIR)
    if not roots:
        raise FileNotFoundError(f"No r_beta1 source roots found under: {TABLES_BASE_DIR}")

    exobase_heights = load_exobase_heights(EXOBASE_TABLE)

    if SELECTED_TABLE_FILE is not None:
        selected_path = resolve_rbeta1_table_file(TABLES_BASE_DIR, SELECTED_TABLE_FILE)
        if selected_path is None:
            raise FileNotFoundError(
                f"Selected table file does not exist under discovered roots: {SELECTED_TABLE_FILE}"
            )
        table_files = [selected_path]
        print(f"Using selected table file: {selected_path}")
    else:
        table_files = discover_rbeta1_table_files(TABLES_BASE_DIR)
        if not table_files:
            raise FileNotFoundError(f"No txt files found under discovered roots in: {TABLES_BASE_DIR}")
        print(f"Found {len(table_files)} unique txt files across discovered roots under: {TABLES_BASE_DIR}")

    for table_file in table_files:
        if not table_has_any_positive_finite_y(table_file):
            print(f"Skipping table with only non-positive/NaN y-values: {table_file}")
            continue

        pdf_path = plot_table_with_exobase_title(table_file, PLOT_OVERRIDES, exobase_heights)
        print(f"Read table: {table_file}")
        if SAVE_PDF:
            print(f"Saved plot: {pdf_path}")


if __name__ == "__main__":
    main()
