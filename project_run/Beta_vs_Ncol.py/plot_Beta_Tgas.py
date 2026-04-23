import pathlib
import sys
import math
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
TABLES_ROOT = pathlib.Path(__file__).resolve().parent
TXT_NAME_TAU1 = "beta_vs_Texc_atoms.txt"

# Leave empty to show all species. Otherwise list a few species or element
# roots to restrict the heatmap.
SELECTED_SPECIES = [
    "H I",
    "He I",
    "Li I",
    "Be I",
    "B I",
    "C I",
    "N I",
    "O I",
    "F I",
    "Ne I",
    "Na I",
    "Mg I",
    "Al I",
    "Si I",
    "P I",
    "S I",
    "Cl I",
    "Ar I",
    "K I",
    "Ca I",
    "Sc I",
    "Ti I",
    "V I",
    "Cr I",
    "Mn I",
    "Fe I",
]

FIGWIDTH = 12.0
MIN_FIGHEIGHT = 4.8
ROW_HEIGHT = 0.30
TITLE_SIZE = 15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 13
CELL_TEXT_SIZE = 8
ANNOTATE_CELLS = False
SAVE_FIGURE = True
SHOW_FIGURE = True
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Beta_vs_tgas"
OUTPUT_NAME_TAU1 = "beta_vs_Texc_atoms_heatmap.pdf"
OUTPUT_NAME_SMALL_MULTIPLES = "beta_vs_Texc_atoms_small_multiples.pdf"

# Representative element roots for a second, more thesis-style figure.
PANEL_ROOTS = ["H", "C", "O", "Na", "Mg", "Si", "Ca", "Fe"]
PANEL_COLUMNS = 4
PANEL_FIGSIZE = (13.0, 7.5)
STAGE_COLORS = {
    0: "#1f4e79",
    1: "#c75b12",
    2: "#4f772d",
}
STAGE_LABELS = {
    0: "I",
    1: "II",
    2: "III",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def try_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value



def parse_series_values(value: Any) -> List[str]:
    if isinstance(value, (int, float)):
        return [str(value)]
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]



def restore_species_label(label: str) -> str:
    return str(label).replace("_", " ")



def read_plotdata_txt(path: pathlib.Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[str], List[str]]:
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

    column_names = header[1:]
    series_values = parse_series_values(metadata.get("series_values", ""))
    if len(series_values) != n_y:
        series_values = [restore_species_label(name) for name in column_names]
    else:
        series_values = [restore_species_label(name) for name in series_values]

    return metadata, x_values, y_matrix, series_values, column_names



def species_stage(species: str) -> int | None:
    if species.endswith(" I"):
        return 0
    if species.endswith(" II"):
        return 1
    if species.endswith(" III"):
        return 2
    return None



def element_root(species: str) -> str:
    parts = str(species).split()
    if not parts:
        return str(species)
    return parts[0]



def expand_selected_species(all_species: List[str], selected_species: List[str]) -> List[str]:
    if not selected_species:
        return list(all_species)

    selected_labels = set(selected_species)
    selected_roots = {element_root(label) for label in selected_species}

    expanded = []
    for species in all_species:
        if species in selected_labels or element_root(species) in selected_roots:
            expanded.append(species)
    return expanded



def build_stage_groups(species_labels: List[str], selected_species: List[str]) -> List[List[str]]:
    eligible = expand_selected_species(species_labels, selected_species)
    groups = [[], [], []]
    for species in species_labels:
        if species not in eligible:
            continue
        stage = species_stage(species)
        if stage is None:
            continue
        if 0 <= stage <= 2:
            groups[stage].append(species)
    return groups


def build_species_lookup(species_labels: List[str]) -> Dict[str, int]:
    return {species: idx for idx, species in enumerate(species_labels)}




# Inserted after build_stage_groups:

def plot_dataset(
    metadata: Dict[str, Any],
    x_values: np.ndarray,
    y_matrix: np.ndarray,
    species_labels: List[str],
    output_name: str,
    mode_label: str,
) -> None:
    stage_groups = build_stage_groups(species_labels, SELECTED_SPECIES)
    if not any(stage_groups):
        raise ValueError(f"No selected species could be assigned to neutral/singly/doubly ionized panels for {mode_label}.")

    valid_beta = y_matrix[np.isfinite(y_matrix) & (y_matrix > 0)]
    if valid_beta.size == 0:
        raise ValueError(f"No plottable beta values found for {mode_label}.")

    log_beta = np.log10(valid_beta)
    vmin = float(np.floor(np.nanmin(log_beta)))
    vmax = float(np.ceil(np.nanmax(log_beta)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"Could not determine color scale for {mode_label}.")
    if vmin == vmax:
        vmax = vmin + 1.0

    max_rows = max(len(group) for group in stage_groups if group)
    fig_height = max(MIN_FIGHEIGHT, 1.8 + ROW_HEIGHT * max_rows)
    fig, axes = plt.subplots(1, 3, figsize=(FIGWIDTH, fig_height), sharex=True)
    panel_titles = ["Neutral", "Singly ionized", "Doubly ionized"]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")
    image = None

    for ax, panel_species, panel_title in zip(axes, stage_groups, panel_titles):
        if not panel_species:
            ax.set_axis_off()
            continue

        column_indices = [species_labels.index(species) for species in panel_species]
        panel_matrix = y_matrix[:, column_indices].T
        panel_log = np.where(np.isfinite(panel_matrix) & (panel_matrix > 0), np.log10(panel_matrix), np.nan)
        masked_panel = np.ma.masked_invalid(panel_log)

        image = ax.imshow(
            masked_panel,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(panel_title, fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel(r"$T_{\rm exc}$ [K]", fontsize=AXIS_LABEL_SIZE)
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([f"{value:.0f}" for value in x_values], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(panel_species)))
        ax.set_yticklabels(panel_species)
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
        ax.set_ylabel("Species", fontsize=AXIS_LABEL_SIZE)

        if ANNOTATE_CELLS and panel_matrix.shape[0] <= 8:
            for row_idx, species in enumerate(panel_species):
                for col_idx, beta_value in enumerate(panel_matrix[row_idx]):
                    if np.isfinite(beta_value) and beta_value > 0:
                        ax.text(
                            col_idx,
                            row_idx,
                            f"{np.log10(beta_value):.1f}",
                            ha="center",
                            va="center",
                            fontsize=CELL_TEXT_SIZE,
                            color="white",
                        )

        for species, beta_values in zip(panel_species, panel_matrix):
            valid = np.isfinite(beta_values) & (beta_values > 0)
            if np.any(valid):
                print(f"{mode_label} | {species}: min_beta={np.nanmin(beta_values[valid]):.6g}, max_beta={np.nanmax(beta_values[valid]):.6g}")

    star_key = metadata.get("star_key", "")
    b_km_s = metadata.get("b_km_s", None)
    distance_au = metadata.get("distance_AU", None)

    title_parts = [r"$\log_{10}(\beta)$ heatmap vs $T_{\rm exc}$", mode_label]
    if star_key:
        title_parts.append(str(star_key))
    if isinstance(b_km_s, (int, float)):
        title_parts.append(rf"$b={float(b_km_s):g}$ km s$^{{-1}}$")
    if isinstance(distance_au, (int, float)):
        title_parts.append(rf"$d={float(distance_au):g}$ AU")

    fig.suptitle(" | ".join(title_parts), fontsize=TITLE_SIZE)
    fig.subplots_adjust(right=0.9, top=0.84, bottom=0.2, wspace=0.25)
    if image is None:
        raise ValueError(f"No heatmap image was created for {mode_label}.")
    cbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\log_{10}(\beta)$", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / output_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def plot_small_multiples(
    metadata: Dict[str, Any],
    x_values: np.ndarray,
    y_matrix: np.ndarray,
    species_labels: List[str],
    output_name: str,
    element_roots: List[str],
) -> None:
    species_lookup = build_species_lookup(species_labels)
    available_roots = []
    for root in element_roots:
        if any(f"{root} {roman}" in species_lookup for roman in ("I", "II", "III")):
            available_roots.append(root)

    if not available_roots:
        raise ValueError("None of the requested element roots were found in the dataset.")

    plotted_series = []
    for root in available_roots:
        for stage in (0, 1, 2):
            species = f"{root} {STAGE_LABELS[stage]}"
            idx = species_lookup.get(species)
            if idx is None:
                continue
            values = y_matrix[:, idx]
            valid = np.isfinite(values) & (values > 0)
            if np.any(valid):
                plotted_series.append(values[valid])

    if not plotted_series:
        raise ValueError("No valid beta values found for the small-multiples plot.")

    y_min = min(np.nanmin(values) for values in plotted_series)
    y_max = max(np.nanmax(values) for values in plotted_series)

    n_panels = len(available_roots)
    ncols = min(PANEL_COLUMNS, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=PANEL_FIGSIZE, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, root in zip(axes, available_roots):
        plotted_any = False
        for stage in (0, 1, 2):
            species = f"{root} {STAGE_LABELS[stage]}"
            idx = species_lookup.get(species)
            if idx is None:
                continue

            values = y_matrix[:, idx]
            valid = np.isfinite(values) & (values > 0)
            if not np.any(valid):
                continue

            ax.plot(
                x_values[valid],
                values[valid],
                linewidth=1.8,
                color=STAGE_COLORS[stage],
                label=species,
            )
            plotted_any = True

        ax.set_title(root, fontsize=AXIS_LABEL_SIZE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE - 1)
        ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 2)
        if plotted_any:
            ax.set_ylim(y_min, y_max)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=TICK_LABEL_SIZE,
            )

    for ax in axes[n_panels:]:
        ax.set_axis_off()

    for row in range(nrows):
        axes[row * ncols].set_ylabel(r"$\beta$", fontsize=AXIS_LABEL_SIZE)
    for ax in axes[max(0, (nrows - 1) * ncols): nrows * ncols]:
        if ax.axison:
            ax.set_xlabel(r"$T_{\rm exc}$ [K]", fontsize=AXIS_LABEL_SIZE)

    legend_handles = [
        plt.Line2D([0], [0], color=STAGE_COLORS[stage], lw=2, label=f"Stage {STAGE_LABELS[stage]}")
        for stage in (0, 1, 2)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=TICK_LABEL_SIZE,
        bbox_to_anchor=(0.5, 0.98),
    )

    star_key = metadata.get("star_key", "")
    b_km_s = metadata.get("b_km_s", None)
    distance_au = metadata.get("distance_AU", None)
    title_parts = [r"$\beta$ vs $T_{\rm exc}$", r"species-wise fixed $N_{\tau=1}$"]
    if star_key:
        title_parts.append(str(star_key))
    if isinstance(b_km_s, (int, float)):
        title_parts.append(rf"$b={float(b_km_s):g}$ km s$^{{-1}}$")
    if isinstance(distance_au, (int, float)):
        title_parts.append(rf"$d={float(distance_au):g}$ AU")

    fig.suptitle(" | ".join(title_parts), fontsize=TITLE_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / output_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    table_path_tau1 = TABLES_ROOT / TXT_NAME_TAU1

    if not table_path_tau1.exists():
        raise FileNotFoundError(f"Could not find txt file: {table_path_tau1}")

    metadata_tau1, x_tau1, y_tau1, species_tau1, _ = read_plotdata_txt(table_path_tau1)

    plot_dataset(
        metadata_tau1,
        x_tau1,
        y_tau1,
        species_tau1,
        output_name=OUTPUT_NAME_TAU1,
        mode_label=r"species-wise fixed $N_{\tau=1}$",
    )

    plot_small_multiples(
        metadata_tau1,
        x_tau1,
        y_tau1,
        species_tau1,
        output_name=OUTPUT_NAME_SMALL_MULTIPLES,
        element_roots=PANEL_ROOTS,
    )


if __name__ == "__main__":
    main()
