

import math
import pathlib
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from Templates.Stars.stars_templates import infer_teff_from_star_template


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
STUDY_ROOT = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "gravity_vs_beta1"
PLANET_FOLDER_GLOB = "*_r_beta1"
FILE_GLOB = "*_r_beta1.txt"
OUTPUT_PDF = None  # None -> save in each planet folder as <planet>_gravity_vs_beta1.pdf
FIGSIZE = (10, 6)
AXES_LABEL_SIZE = 14
TITLE_SIZE = 16
LEGEND_SIZE = 12
TICK_LABEL_SIZE = 12
X_MIN = None
X_MAX = None
AUTO_X_LIMITS = True
AUTO_X_MARGIN_FRACTION = 0.08
Y_MIN = None
Y_MAX = None
SHOW_GRID = True
SHOW_LEGEND = True
SHOW_METADATA_BOX = True
LEGEND_NCOL = 2
LEGEND_LOC = "upper left"
LEGEND_BBOX_TO_ANCHOR = (0.02, 0.98)
METADATA_BOX_AXES_POS = (0.02, 0.72)
SKIP_ALL_NAN_FILES = True
AUTO_LEGEND_IF_CROWDED = True
LEGEND_OUTSIDE_IF_CROWDED = False
CROWDED_SERIES_THRESHOLD = 6

COLOR_BY_FAMILY = {
    "H": "tab:blue",
    "Na": "tab:cyan",
    "Fe": "tab:red",
    "NO": "tab:purple",
}
LINESTYLE_BY_TYPE = {
    "neutral": "-",
    "ion": "--",
    "double_ion": ":",
    "molecule": "-",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def try_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value



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

    if len(header) < 2:
        raise ValueError(f"Expected at least one y column in {path}")

    y_values = []
    for row in rows:
        value = row[1]
        if value.lower() == "nan":
            y_values.append(np.nan)
        else:
            y_values.append(float(value))

    return metadata, x_values, np.array(y_values, dtype=float)



def safe_output_name(name: str) -> str:
    return name.replace(" ", "_")



def infer_planet_name(entries: List[Tuple[pathlib.Path, Dict[str, Any], np.ndarray, np.ndarray]]) -> str:
    for _, metadata, _, _ in entries:
        planet = metadata.get("planet")
        if isinstance(planet, str) and planet:
            return planet.replace("_", " ")
    return "unknown planet"



def make_metadata_box_text(metadata: Dict[str, Any]) -> str:
    lines = []

    star = metadata.get("star")
    distance = metadata.get("distance_AU")
    radius = metadata.get("planet_radius_Rjup")
    temp = metadata.get("planet_temperature_K")
    mu = metadata.get("planet_mu")

    if isinstance(star, str):
        try:
            teff_value = infer_teff_from_star_template(star)
        except Exception:
            teff_value = None

        if teff_value is not None:
            lines.append(rf"$T_{{\rm eff}} = {float(teff_value):.0f}\ \mathrm{{K}}$")
        else:
            lines.append(rf"star = {star}")
    if isinstance(distance, (int, float)):
        lines.append(rf"$d = {distance:.2f}\ \mathrm{{AU}}$")
    if isinstance(radius, (int, float)):
        lines.append(rf"$R = {radius:.2f}\ R_{{\rm jup}}$")
    if isinstance(temp, (int, float)):
        lines.append(rf"$T_{{\rm atm}} = {temp:.0f}\ \mathrm{{K}}$")
    if isinstance(mu, (int, float)):
        lines.append(rf"$\mu = {mu:.2f}$")

    return "\n".join(lines)



def get_style(metadata: Dict[str, Any], species_label: str) -> Dict[str, Any]:
    family = metadata.get("species_color_family")
    if not isinstance(family, str) or not family:
        family = species_label

    if "III" in species_label:
        marker_type = "double_ion"
    elif "II" in species_label:
        marker_type = "ion"
    elif species_label == "NO":
        marker_type = "molecule"
    else:
        marker_type = "neutral"

    color = COLOR_BY_FAMILY.get(family, None)
    linestyle = LINESTYLE_BY_TYPE.get(marker_type, "-")
    return {
        "color": color,
        "linestyle": linestyle,
    }



def choose_output_path(planet_folder: pathlib.Path, planet_name: str, output_pdf):
    if output_pdf is not None:
        return pathlib.Path(output_pdf)
    return planet_folder / f"{safe_output_name(planet_name)}_gravity_vs_beta1.pdf"



def plot_one_planet_folder(planet_folder: pathlib.Path) -> None:
    table_files = sorted(
        path for path in planet_folder.glob(FILE_GLOB)
        if path.name != pathlib.Path(__file__).name
    )
    if not table_files:
        print(f"No txt files found in {planet_folder} with glob {FILE_GLOB!r}; skipping")
        return

    entries: List[Tuple[pathlib.Path, Dict[str, Any], np.ndarray, np.ndarray]] = []
    for table_file in table_files:
        metadata, x_values, y_values = read_plotdata_txt(table_file)

        if SKIP_ALL_NAN_FILES:
            good = y_values[np.isfinite(y_values) & (y_values > 0)]
            if good.size == 0:
                print(f"Skipping all-NaN/non-positive file: {table_file}")
                continue

        entries.append((table_file, metadata, x_values, y_values))

    if not entries:
        print(f"No plottable txt files found in {planet_folder} after filtering; skipping")
        return

    planet_name = infer_planet_name(entries)
    output_path = choose_output_path(planet_folder, planet_name, OUTPUT_PDF)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    n_series = len(entries)
    finite_y_blocks: List[np.ndarray] = []
    finite_x_blocks: List[np.ndarray] = []

    for table_file, metadata, x_values, y_values in entries:
        species = str(metadata.get("species", table_file.stem.replace("_r_beta1", ""))).replace("_", " ")
        style = get_style(metadata, species)

        ax.plot(
            x_values,
            y_values,
            label=species,
            linewidth=2.0,
            **style,
        )

        good_y = y_values[np.isfinite(y_values) & (y_values > 0)]
        good_x = x_values[np.isfinite(x_values) & (x_values > 0)]
        if good_y.size:
            finite_y_blocks.append(good_y)
        if good_x.size:
            finite_x_blocks.append(good_x)

    ax.set_xlabel(r"Surface gravity, $g$ [m s$^{-2}$]", fontsize=AXES_LABEL_SIZE)
    ax.set_ylabel(r"Critical height, $r_{\beta=1}/R_p$", fontsize=AXES_LABEL_SIZE)
    ax.set_title(rf"Critical height vs gravity | {planet_name}", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 1)

    if SHOW_GRID:
        ax.grid(True, which="major", alpha=0.35)

    if finite_x_blocks:
        all_x = np.concatenate(finite_x_blocks)
        xmin_auto = float(np.nanmin(all_x))
        xmax_auto = float(np.nanmax(all_x))

        if X_MIN is not None or X_MAX is not None:
            xmin = X_MIN if X_MIN is not None else xmin_auto
            xmax = X_MAX if X_MAX is not None else xmax_auto
            ax.set_xlim(xmin, xmax)
        elif AUTO_X_LIMITS:
            ax.relim()
            ax.autoscale(enable=True, axis="x", tight=False)
            ax.margins(x=AUTO_X_MARGIN_FRACTION)

    if finite_y_blocks:
        all_y = np.concatenate(finite_y_blocks)
        ymin_auto = 10 ** math.floor(np.log10(np.nanmin(all_y)))
        ymax_auto = 10 ** math.ceil(np.log10(np.nanmax(all_y)))
        ymin = Y_MIN if Y_MIN is not None else ymin_auto
        ymax = Y_MAX if Y_MAX is not None else ymax_auto
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    if SHOW_LEGEND:
        legend_loc = LEGEND_LOC
        legend_bbox = LEGEND_BBOX_TO_ANCHOR
        legend_ncol = LEGEND_NCOL

        if AUTO_LEGEND_IF_CROWDED and n_series >= CROWDED_SERIES_THRESHOLD:
            legend_ncol = 1
            if LEGEND_OUTSIDE_IF_CROWDED:
                legend_loc = "upper left"
                legend_bbox = (1.02, 1.0)

        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=legend_ncol,
            framealpha=0.90,
            fontsize=LEGEND_SIZE,
            title="species",
            title_fontsize=LEGEND_SIZE,
        )

    if SHOW_METADATA_BOX:
        metadata_text = make_metadata_box_text(entries[0][1])
        if metadata_text:
            ax.text(
                METADATA_BOX_AXES_POS[0],
                METADATA_BOX_AXES_POS[1],
                metadata_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=LEGEND_SIZE,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.90, edgecolor="0.7"),
            )

    if AUTO_LEGEND_IF_CROWDED and LEGEND_OUTSIDE_IF_CROWDED and n_series >= CROWDED_SERIES_THRESHOLD:
        fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")



def main() -> None:
    planet_folders = sorted(
        path for path in STUDY_ROOT.glob(PLANET_FOLDER_GLOB)
        if path.is_dir()
    )
    if not planet_folders:
        raise FileNotFoundError(f"No planet folders found in {STUDY_ROOT} with glob {PLANET_FOLDER_GLOB!r}")

    for planet_folder in planet_folders:
        print(f"Processing planet folder: {planet_folder}")
        plot_one_planet_folder(planet_folder)


if __name__ == "__main__":
    main()
