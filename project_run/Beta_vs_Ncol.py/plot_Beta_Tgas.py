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
PLOT_ATOMS = False
PLOT_MOLECULES = True
TXT_NAME_ATOMS = "beta_vs_Texc_atoms.txt"
TXT_NAME_MOLECULES = "beta_vs_Texc_molecules.txt"

# Leave empty to show all species. Otherwise list a few species or element
# roots to restrict the heatmap.
SELECTED_SPECIES = []

FIGWIDTH = 12.0
MIN_FIGHEIGHT = 4.8
ROW_HEIGHT = 0.30
TITLE_SIZE = 15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 13
YTICK_LABEL_SIZE = 11
CELL_TEXT_SIZE = 8
ANNOTATE_CELLS = False
SAVE_FIGURE = True
SHOW_FIGURE = True
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Beta_vs_tgas"
OUTPUT_NAME_ATOMS = "beta_vs_Texc_atoms_heatmap.pdf"
OUTPUT_NAME_MOLECULES = "beta_vs_Texc_molecules_heatmap.pdf"
CATEGORY_ORDER = [
    "Non-metals",
    "Noble gases",
    "Semi-metals",
    "Alkali metals",
    "Alkaline earth metals",
    "Metals",
]

ELEMENT_CATEGORY = {
    "H": "Non-metals",
    "C": "Non-metals",
    "N": "Non-metals",
    "O": "Non-metals",
    "F": "Non-metals",
    "P": "Non-metals",
    "S": "Non-metals",
    "Cl": "Non-metals",
    "He": "Noble gases",
    "Ne": "Noble gases",
    "Ar": "Noble gases",
    "B": "Semi-metals",
    "Si": "Semi-metals",
    "Li": "Alkali metals",
    "Na": "Alkali metals",
    "K": "Alkali metals",
    "Be": "Alkaline earth metals",
    "Mg": "Alkaline earth metals",
    "Ca": "Alkaline earth metals",
    "Sc": "Metals",
    "Ti": "Metals",
    "V": "Metals",
    "Cr": "Metals",
    "Mn": "Metals",
    "Fe": "Metals",
    "Al": "Metals",
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





def build_category_groups(species_labels: List[str], selected_species: List[str]) -> List[Tuple[str, List[str]]]:
    eligible = set(expand_selected_species(species_labels, selected_species))
    grouped: Dict[str, List[str]] = {category: [] for category in CATEGORY_ORDER}

    def species_sort_key(species: str) -> Tuple[str, int, str]:
        stage = species_stage(species)
        stage_sort = 99 if stage is None else stage
        return (element_root(species), stage_sort, species)

    for species in species_labels:
        if species not in eligible:
            continue
        root = element_root(species)
        category = ELEMENT_CATEGORY.get(root)
        if category is None:
            continue
        grouped[category].append(species)

    for category in CATEGORY_ORDER:
        grouped[category] = sorted(grouped[category], key=species_sort_key)

    return [(category, grouped[category]) for category in CATEGORY_ORDER]


def build_species_lookup(species_labels: List[str]) -> Dict[str, int]:
    return {species: idx for idx, species in enumerate(species_labels)}


def selected_molecule_species(species_labels: List[str]) -> List[str]:
    if not SELECTED_SPECIES:
        return list(species_labels)
    selected = set(SELECTED_SPECIES)
    return [species for species in species_labels if species in selected]


def plot_molecule_dataset(
    x_values: np.ndarray,
    y_matrix: np.ndarray,
    species_labels: List[str],
    output_name: str,
    mode_label: str,
) -> None:
    panel_species = selected_molecule_species(species_labels)
    if not panel_species:
        raise ValueError(f"No selected molecular species available for {mode_label}.")

    column_indices = [species_labels.index(species) for species in panel_species]
    panel_matrix = y_matrix[:, column_indices].T
    valid_beta = panel_matrix[np.isfinite(panel_matrix) & (panel_matrix > 0)]
    if valid_beta.size == 0:
        raise ValueError(f"No plottable molecular beta values found for {mode_label}.")

    low = 0.5
    high = 1.5
    log_beta = np.log10(valid_beta)
    vmin = float(np.floor(low * np.nanmin(log_beta)))
    vmax = float(np.ceil(high * np.nanmax(log_beta)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"Could not determine color scale for {mode_label}.")
    if vmin == vmax:
        vmax = vmin + 1.0

    panel_log = np.where(np.isfinite(panel_matrix) & (panel_matrix > 0), np.log10(panel_matrix), np.nan)
    masked_panel = np.ma.masked_invalid(panel_log)

    fig_height = max(MIN_FIGHEIGHT, 2.2 + ROW_HEIGHT * len(panel_species))
    fig, ax = plt.subplots(figsize=(FIGWIDTH, fig_height))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")
    image = ax.imshow(
        masked_panel,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(r"$T_{\rm exc}$ [$10^3$ K]", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{(value / 1e3):g}" for value in x_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(panel_species)))
    ax.set_yticklabels(panel_species)
    ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", which="major", labelsize=YTICK_LABEL_SIZE)

    if ANNOTATE_CELLS and len(panel_species) <= 12:
        for row_idx, _species in enumerate(panel_species):
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

    fig.suptitle(" | ".join([r"$\log_{10}(\beta)$ heatmap vs $T_{\rm exc}$", mode_label]), fontsize=TITLE_SIZE)
    fig.subplots_adjust(right=0.88, top=0.90, bottom=0.18)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
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


def plot_dataset(
    metadata: Dict[str, Any],
    x_values: np.ndarray,
    y_matrix: np.ndarray,
    species_labels: List[str],
    output_name: str,
    mode_label: str,
) -> None:
    category = str(metadata.get("category", "")).strip().lower()
    if category == "molecule":
        plot_molecule_dataset(
            x_values,
            y_matrix,
            species_labels,
            output_name=output_name,
            mode_label=mode_label,
        )
        return

    category_groups = build_category_groups(species_labels, SELECTED_SPECIES)
    if not any(group for _, group in category_groups):
        raise ValueError(f"No selected species could be assigned to periodic-table categories for {mode_label}.")

    valid_beta = y_matrix[np.isfinite(y_matrix) & (y_matrix > 0)]
    if valid_beta.size == 0:
        raise ValueError(f"No plottable beta values found for {mode_label}.")

    low = 0.5
    high = 1.5
    log_beta = np.log10(valid_beta)
    vmin = float(np.floor(low * np.nanmin(log_beta)))
    vmax = float(np.ceil(high * np.nanmax(log_beta)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"Could not determine color scale for {mode_label}.")
    if vmin == vmax:
        vmax = vmin + 1.0

    nonempty_groups = [(title, group) for title, group in category_groups if group]
    max_rows = max(len(group) for _, group in nonempty_groups)
    fig_height = max(MIN_FIGHEIGHT * 1.6, 2.6 + ROW_HEIGHT * max_rows)
    fig, axes = plt.subplots(2, 3, figsize=(FIGWIDTH * 1.5, fig_height), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    ncols = 3
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")
    image = None

    for ax, (panel_title, panel_species) in zip(axes, category_groups):
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
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(panel_title, fontsize=AXIS_LABEL_SIZE)
        panel_index = category_groups.index((panel_title, panel_species))
        row_index = panel_index // ncols
        if row_index == 1:
            ax.set_xlabel(r"$T_{\rm exc}$ [$10^3$ K]", fontsize=AXIS_LABEL_SIZE)
        else:
            ax.set_xlabel("")
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([f"{(value / 1e3):g}" for value in x_values], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(panel_species)))
        ax.set_yticklabels(panel_species)
        ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", which="major", labelsize=YTICK_LABEL_SIZE)

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

    for ax in axes[len(category_groups):]:
        ax.set_axis_off()

    title_parts = [r"$\log_{10}(\beta)$ heatmap vs $T_{\rm exc}$", mode_label]

    fig.suptitle(" | ".join(title_parts), fontsize=TITLE_SIZE)
    fig.subplots_adjust(right=0.92, top=0.9, bottom=0.14, wspace=0.22, hspace=0.22)
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




# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    dataset_specs = []
    if PLOT_ATOMS:
        dataset_specs.append((TXT_NAME_ATOMS, OUTPUT_NAME_ATOMS))
    if PLOT_MOLECULES:
        dataset_specs.append((TXT_NAME_MOLECULES, OUTPUT_NAME_MOLECULES))

    plotted_any = False
    for txt_name, output_name in dataset_specs:
        table_path_tau1 = TABLES_ROOT / txt_name
        if not table_path_tau1.exists():
            print(f"Skipping missing txt file: {table_path_tau1}")
            continue

        metadata_tau1, x_tau1, y_tau1, species_tau1, _ = read_plotdata_txt(table_path_tau1)
        plot_dataset(
            metadata_tau1,
            x_tau1,
            y_tau1,
            species_tau1,
            output_name=output_name,
            mode_label=r"species-wise fixed $N_{\tau=1}$",
        )
        plotted_any = True

    if not plotted_any:
        raise FileNotFoundError("No enabled beta(T_exc) txt files were found to plot.")



if __name__ == "__main__":
    main()
