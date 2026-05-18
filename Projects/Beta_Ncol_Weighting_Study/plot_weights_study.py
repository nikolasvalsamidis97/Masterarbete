import math
import pathlib
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from Templates.Stars.stars_templates import infer_teff_from_star_template


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
TABLES_ROOT = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Beta_vs_ncol"
FILE_GLOB = "*_beta_vs_ncol_weights.txt"
FIGSIZE = (8, 5)
TITLE_SIZE = 17
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 13
LINEWIDTH = 1.0
SAVE_FIGURE = True
SHOW_FIGURE = True
OUTPUT_NAME = "Fe_boltzmann_weighting_beta_vs_ncol.pdf"
X_MAX = 1e20
Y_MIN = 1e-7
Y_MAX = 10**1.1
PANEL_BORDER_WIDTH = 1.8

TEMP_COLORS = {
    3000: "tab:blue",
    6000: "tab:orange",
    10000: "tab:red",
    20000: "tab:green",
    30000: "tab:purple",
    40000: "tab:brown",
    50000: "tab:pink",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def try_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def pretty_species_name(species: str) -> str:
    parts = str(species).split()
    if len(parts) != 2:
        return str(species)

    element, stage = parts
    charge_map = {
        "I": rf"$\mathrm{{{element}}}^{{0}}$",
        "II": rf"$\mathrm{{{element}}}^{{+}}$",
        "III": rf"$\mathrm{{{element}}}^{{2+}}$",
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



def parse_series_values(value: Any) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if not isinstance(value, str):
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    result: List[float] = []
    for part in parts:
        try:
            result.append(float(part))
        except ValueError:
            continue
    return result



def read_plotdata_txt(path: pathlib.Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    metadata: Dict[str, Any] = {}
    header: List[str] | None = None
    rows: List[List[str]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                content = line[1:].strip()
                if not content:
                    continue
                if ":" in content:
                    key, value = content.split(":", 1)
                    metadata[key.strip()] = try_float(value.strip())
                continue

            if header is None:
                header = [part.strip() for part in line.split("\t")]
            else:
                rows.append([part.strip() for part in line.split("\t")])

    if header is None:
        raise ValueError(f"No table header found in {path}")
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    x_values = np.array([float(row[0]) for row in rows], dtype=float)
    n_y = len(header) - 1
    if n_y < 1:
        raise ValueError(f"Expected at least one y column in {path}")

    y_matrix = np.empty((len(rows), n_y), dtype=float)
    for i, row in enumerate(rows):
        for j in range(n_y):
            value = row[j + 1]
            y_matrix[i, j] = np.nan if value.lower() == "nan" else float(value)

    return metadata, x_values, y_matrix


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    table_files = sorted(TABLES_ROOT.glob(FILE_GLOB), reverse=True)
    if not table_files:
        raise FileNotFoundError(f"No txt files found in {TABLES_ROOT} with glob {FILE_GLOB!r}")

    datasets: List[Tuple[str, Dict[str, Any], np.ndarray, np.ndarray, List[float]]] = []
    for table_file in table_files:
        metadata, x_values, y_matrix = read_plotdata_txt(table_file)
        species = str(metadata.get("species", table_file.stem.replace("_beta_vs_ncol_weights", ""))).replace("_", " ")
        series_values = parse_series_values(metadata.get("series_values", ""))
        if len(series_values) != y_matrix.shape[1]:
            series_values = [float(i) for i in range(y_matrix.shape[1])]
        datasets.append((species, metadata, x_values, y_matrix, series_values))

    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels, figsize=FIGSIZE, sharex=True, sharey=False, gridspec_kw={"wspace": 0.0})
    if n_panels == 1:
        axes = [axes]

    global_x_arrays: List[np.ndarray] = []

    for ax, (species, metadata, x_values, y_matrix, series_values) in zip(axes, datasets):
        local_y_arrays: List[np.ndarray] = []
        plotted_any = False

        for j, temp_value in enumerate(series_values):
            y_values = y_matrix[:, j]
            valid = np.isfinite(y_values) & (y_values > 0)
            if not np.any(valid):
                continue

            x_plot = x_values[valid]
            y_plot = y_values[valid]
            x_valid_limit = x_plot <= X_MAX
            if not np.any(x_valid_limit):
                continue
            x_plot = x_plot[x_valid_limit]
            y_plot = y_plot[x_valid_limit]
            temp_key = int(round(temp_value))

            ax.plot(
                x_plot,
                y_plot,
                linewidth=LINEWIDTH,
                color=TEMP_COLORS.get(temp_key, None),
                label=rf"$T={temp_value:.0f}$ K",
            )
            plotted_any = True
            global_x_arrays.append(x_plot)
            local_y_arrays.append(y_plot)

        ax.axhline(1.0, linestyle="--", linewidth=1.2, color="0.4")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
        ax.yaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
        ax.set_title(pretty_species_name(species), fontsize=TITLE_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", which="both", left=(ax is axes[0]), labelleft=(ax is axes[0]))
        ax.tick_params(axis="x", which="major", labelbottom=True)
        ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 1)
        ax.grid(True, which="major", alpha=0.35)

        if not plotted_any:
            ax.text(
                0.5,
                0.5,
                "No valid curves",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=LEGEND_SIZE,
            )

    if not global_x_arrays:
        raise ValueError("No plottable curves were found in the saved txt files.")

    x_min = min(np.nanmin(arr) for arr in global_x_arrays)
    x_max = min(X_MAX, max(np.nanmax(arr) for arr in global_x_arrays))
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(Y_MIN, Y_MAX)

    axes[0].set_ylabel(r"$\log_{10}\beta$", fontsize=AXIS_LABEL_SIZE)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    for ax in axes[:-1]:
        ax.spines["right"].set_linewidth(PANEL_BORDER_WIDTH)
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)

    for ax in axes:
        ax.set_xlabel(r"$\log_{10}\!\left(N_{\rm col}\,[\mathrm{cm}^{-2}]\right)$", fontsize=AXIS_LABEL_SIZE)

    first_metadata = datasets[0][1]
    star_key = str(first_metadata.get("star", ""))
    try:
        stellar_teff = infer_teff_from_star_template(star_key)
    except Exception:
        stellar_teff = first_metadata.get("stellar_teff_K", None)

    distance_au = first_metadata.get("distance_AU", None)
    if isinstance(stellar_teff, (int, float)):
        title = rf"$\beta$ vs column density for Fe ionization stages | $T_{{\rm eff}}={float(stellar_teff):.0f}$ K"
    else:
        title = r"$\beta$ vs column density for Fe ionization stages"

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.88, 0.5),
            framealpha=0.9,
            fontsize=LEGEND_SIZE,
            title="temperature",
            title_fontsize=LEGEND_SIZE,
        )

    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.98), w_pad=0.0)

    if SAVE_FIGURE:
        output_path = TABLES_ROOT / OUTPUT_NAME
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
