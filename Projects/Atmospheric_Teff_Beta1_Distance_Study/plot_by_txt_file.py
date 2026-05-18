import pathlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
# Give one or more saved text tables. With default settings, the script can run
# by only changing TABLE_FILES.
TABLE_FILES = [
    # Example:
    # pathlib.Path(__file__).resolve().parents[1]
    # / "results" / "Plots" / "Teff_study" / "r_at_beta1"
    # / "earth_like_r_beta1" / "NI_r_beta1.txt",
]

SAVE_PDF = True
SHOW_PLOT = True

# Default plotting parameters.
# These act as defaults and can be changed globally here.
DEFAULT_PLOT_CONFIG: Dict[str, Any] = {
    "figsize": (8.5, 5.5),
    "font_sizes": {
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    },
    "x_scale": "log",
    "y_scale": "log",
    "x_divide_by": 1e4,
    "x_label": r"Stellar $T_{\rm eff}$ [$10^4$ K]",
    "y_label": r"$r_{\beta=1} / R_{\rm p}$",
    "title_template": "{y_label_plain} | {planet_pretty} | {species}",
    "legend_title": "Distance",
    "line_colormap": "viridis",
    "line_alpha": 0.9,
    "marker": "o",
    "marker_size": 3.5,
    "line_width": 1.6,
    "grid": True,
    "grid_alpha": 0.3,
    "tight_layout": True,
    "right_margin": None,
    "show_metadata_box": True,
    "metadata_box_keys": [
        "planet_radius_Rjup",
        "planet_mass_Mjup",
        "planet_temperature_K",
        "planet_mu",
    ],
    "x_ticks": [0.26, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5],
    "x_tick_labels": ["0.26", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1", "2", "3", "4", "5"],
    "xlim": (0.26, 5.0),
    # Optional manual overrides:
    # "title": "Custom title",
    # "output_pdf_path": pathlib.Path("custom_name.pdf"),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_header_and_table(table_path: pathlib.Path) -> Tuple[Dict[str, str], List[str], np.ndarray]:
    """Read one saved text table.

    Returns
    -------
    metadata : dict
        Parsed '# key: value' header lines.
    columns : list[str]
        Column names from the first non-comment line.
    data : ndarray
        Numeric table data with NaN support.
    """
    metadata: Dict[str, str] = {}
    columns: List[str] = []
    data_rows: List[List[float]] = []

    with open(table_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                content = line[1:].strip()
                if ":" in content:
                    key, value = content.split(":", 1)
                    metadata[key.strip()] = value.strip()
                continue

            if not columns:
                columns = line.split()
                continue

            parts = line.split()
            row: List[float] = []
            for value in parts:
                if value.lower() == "nan":
                    row.append(np.nan)
                else:
                    row.append(float(value))
            data_rows.append(row)

    if not columns:
        raise ValueError(f"No table columns found in {table_path}")
    if not data_rows:
        raise ValueError(f"No table data found in {table_path}")

    data = np.asarray(data_rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(
            f"Malformed table in {table_path}: expected {len(columns)} columns, got shape {data.shape}"
        )

    return metadata, columns, data



def extract_series(columns: Sequence[str]) -> List[Tuple[str, str]]:
    """Extract series identifiers from columns.

    Returns a list of tuples: (series_value, column_name)
    for all columns after the first x column.
    """
    series_info: List[Tuple[str, str]] = []
    for col in columns[1:]:
        if "__" in col:
            series_value = col.split("__", 1)[1]
        else:
            series_value = col
        series_info.append((series_value, col))
    return series_info



def output_pdf_path_from_table(table_path: pathlib.Path, plot_config: Mapping[str, Any]) -> pathlib.Path:
    custom_path = plot_config.get("output_pdf_path")
    if custom_path is not None:
        return pathlib.Path(custom_path)
    return table_path.with_suffix(".pdf")



def build_plot_config(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    config = dict(DEFAULT_PLOT_CONFIG)
    config["font_sizes"] = dict(DEFAULT_PLOT_CONFIG["font_sizes"])

    if overrides:
        for key, value in overrides.items():
            if key == "font_sizes" and value is not None:
                config["font_sizes"].update(dict(value))
            else:
                config[key] = value
    return config



def _metadata_or_default(metadata: Mapping[str, str], key: str, default: str) -> str:
    value = metadata.get(key)
    if value is None or value == "":
        return default
    return value


def _pretty_planet_name(name: str) -> str:
    parts = str(name).replace("_", " ").split()
    return " ".join(part.capitalize() for part in parts)


def _plain_label(text: str) -> str:
    text = str(text)
    replacements = {
        r"$r_{\beta=1} / R_{\rm p}$": "Critical height",
        r"$r_{\beta=1}/R_{\rm p}$": "Critical height",
        "r_beta1 / R_p": "Critical height",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text



def _format_float_string(value: str, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _build_default_title(metadata: Mapping[str, str], table_path: pathlib.Path, y_label_text: str) -> str:
    planet = metadata.get("planet", table_path.parent.name)
    species = metadata.get("species", table_path.stem)
    planet_pretty = _pretty_planet_name(planet)
    y_label_plain = _plain_label(y_label_text)
    return DEFAULT_PLOT_CONFIG["title_template"].format(
        planet=planet,
        planet_pretty=planet_pretty,
        species=species,
        dataset_name=metadata.get("dataset_name", table_path.stem),
        y_label_plain=y_label_plain,
    )



def _metadata_box_label(key: str, value: str) -> str:
    if key == "planet_radius_Rjup":
        return rf"$R = {_format_float_string(value)}\ R_{{\rm jup}}$"
    if key == "planet_mass_Mjup":
        return rf"$M = {_format_float_string(value)}\ M_{{\rm jup}}$"
    if key == "planet_temperature_K":
        return rf"$T = {_format_float_string(value, decimals=0)}\ \mathrm{{K}}$"
    if key == "planet_mu":
        return rf"$\mu = {_format_float_string(value)}$"
    return f"{key} = {value}"



def _build_metadata_box_lines(metadata: Mapping[str, str], plot_config: Mapping[str, Any]) -> List[str]:
    if not plot_config.get("show_metadata_box", True):
        return []

    keys = plot_config.get("metadata_box_keys", [])
    lines: List[str] = []
    for key in keys:
        value = metadata.get(key)
        if value is not None and value != "":
            lines.append(_metadata_box_label(str(key), str(value)))
    return lines



def plot_table(table_path: pathlib.Path, overrides: Mapping[str, Any] | None = None) -> pathlib.Path:
    metadata, columns, data = parse_header_and_table(table_path)
    plot_config = build_plot_config(overrides)
    series_info = extract_series(columns)

    x_values = data[:, 0]
    y_matrix = data[:, 1:]

    if y_matrix.shape[1] != len(series_info):
        raise ValueError(
            f"Series columns mismatch in {table_path}: matrix has {y_matrix.shape[1]} columns, "
            f"but parsed {len(series_info)} series labels"
        )

    plt.rcParams.update(plot_config["font_sizes"])
    plt.figure(figsize=plot_config["figsize"])

    cmap = getattr(plt.cm, plot_config["line_colormap"])
    color_values = np.linspace(0.15, 0.9, len(series_info))

    x_divide_by = plot_config.get("x_divide_by", 1.0)
    x_plot = x_values / x_divide_by

    series_unit = metadata.get("series_unit", "")
    legend_title = plot_config.get("legend_title")
    if legend_title is None:
        legend_title = metadata.get("series_label", "Series")

    for idx, ((series_value, _), color_value) in enumerate(zip(series_info, color_values)):
        curve_color = cmap(color_value)
        y_values = y_matrix[:, idx]
        if series_unit:
            label = f"{series_value} {series_unit}"
        else:
            label = str(series_value)

        plt.plot(
            x_plot,
            y_values,
            marker=plot_config["marker"],
            markersize=plot_config["marker_size"],
            linewidth=plot_config["line_width"],
            color=curve_color,
            alpha=plot_config["line_alpha"],
            label=label,
        )

    ax = plt.gca()
    ax.set_xscale(plot_config["x_scale"])
    ax.set_yscale(plot_config["y_scale"])

    if plot_config.get("x_ticks") is not None:
        ax.xaxis.set_major_locator(FixedLocator(plot_config["x_ticks"]))
    if plot_config.get("x_tick_labels") is not None:
        ax.xaxis.set_major_formatter(FixedFormatter(plot_config["x_tick_labels"]))
    if plot_config.get("xlim") is not None:
        ax.set_xlim(*plot_config["xlim"])
    if plot_config.get("ylim") is not None:
        ax.set_ylim(*plot_config["ylim"])

    default_x_label = _metadata_or_default(metadata, "x_label", columns[0])
    default_y_label = _metadata_or_default(metadata, "y_label", "y")
    x_label = plot_config.get("x_label_override", plot_config.get("x_label", default_x_label))
    y_label = plot_config.get("y_label_override", plot_config.get("y_label", default_y_label))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    title = plot_config.get("title")
    if title is None:
        title = _build_default_title(metadata, table_path, default_y_label)
    ax.set_title(title)

    if plot_config.get("grid", True):
        ax.grid(True, alpha=plot_config.get("grid_alpha", 0.3))

    legend = None
    if plot_config.get("legend", True):
        legend = ax.legend(title=legend_title)

    metadata_box_lines = _build_metadata_box_lines(metadata, plot_config)

    right_margin = plot_config.get("right_margin")
    if right_margin is not None:
        plt.subplots_adjust(right=right_margin)

    if metadata_box_lines and legend is not None:
        ax.text(
            0.73,
            0.96,
            "\n".join(metadata_box_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            multialignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
    elif metadata_box_lines:
        ax.text(
            0.98,
            0.96,
            "\n".join(metadata_box_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            multialignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )

    if plot_config.get("tight_layout", True):
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

    pdf_path = output_pdf_path_from_table(table_path, plot_config)
    if SAVE_PDF:
        plt.savefig(pdf_path)

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()

    return pdf_path



def main() -> None:
    if not TABLE_FILES:
        raise ValueError(
            "TABLE_FILES is empty. Add one or more saved .txt tables to plot."
        )

    for table_file in TABLE_FILES:
        table_path = pathlib.Path(table_file)
        if not table_path.exists():
            raise FileNotFoundError(f"Table file does not exist: {table_path}")

        pdf_path = plot_table(table_path)
        print(f"Read table: {table_path}")
        if SAVE_PDF:
            print(f"Saved plot: {pdf_path}")


if __name__ == "__main__":
    main()
