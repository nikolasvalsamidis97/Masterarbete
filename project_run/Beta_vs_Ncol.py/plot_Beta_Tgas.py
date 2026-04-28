import pathlib
import sys
import math
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
TABLES_ROOT = pathlib.Path(__file__).resolve().parent
PLOT_ATOMS = True
PLOT_MOLECULES = False
TXT_NAME_ATOMS = "beta_vs_Texc_atoms.txt"
TXT_NAME_MOLECULES = "beta_vs_Texc_molecules.txt"
ATOM_FILE_GLOB = "beta_vs_Texc_atoms*.txt"
ATOM_FILE_EXCLUDE_SUBSTRINGS = ["fixedN"]
ATOM_TARGETS = [
    ("lowest", 2600.0),
    ("10000K", 10000.0),
    ("50000K", 50000.0),
]

# Leave empty to show all species. Otherwise list a few species or element
# roots to restrict the heatmap.
SELECTED_SPECIES = []

FIGWIDTH = 12.0
MIN_FIGHEIGHT = 4
ROW_HEIGHT = 0.2
TITLE_SIZE = 15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 13
YTICK_LABEL_SIZE = 10
CELL_TEXT_SIZE = 8
ANNOTATE_CELLS = False
SAVE_FIGURE = True
SHOW_FIGURE = False
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Beta_vs_tgas"
OUTPUT_NAME_ATOMS = "beta_vs_Texc_atoms_heatmap.pdf"
OUTPUT_NAME_MOLECULES = "beta_vs_Texc_molecules_heatmap.pdf"
ATOM_OUTPUT_STEM = "beta_vs_Texc_atoms_heatmap"
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
ELEMENT_ORDER = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
]
ELEMENT_ORDER_INDEX = {element: idx for idx, element in enumerate(ELEMENT_ORDER)}
ATOM_BETA_LOW = 0.5
ATOM_BETA_HIGH = 1.5
ATOM_BETA_GREEN_LOW = 0.95
ATOM_BETA_GREEN_HIGH = 1.05
ATOM_FIXED_NCOL_CM2 = 1e-20


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


def discover_atom_table_files() -> List[pathlib.Path]:
    files = sorted(TABLES_ROOT.glob(ATOM_FILE_GLOB))
    return [
        path
        for path in files
        if not any(token in path.name for token in ATOM_FILE_EXCLUDE_SUBSTRINGS)
    ]



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


def dataset_stellar_teff(metadata: Dict[str, Any]) -> float | None:
    value = metadata.get("stellar_teff_K")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_atom_target_datasets(
    datasets: List[Tuple[pathlib.Path, Dict[str, Any], np.ndarray, np.ndarray, List[str], List[str]]]
) -> List[Tuple[str, pathlib.Path, Dict[str, Any], np.ndarray, np.ndarray, List[str], List[str]]]:
    if not datasets:
        return []

    selected = []
    used_paths: set[pathlib.Path] = set()

    for target_label, target_teff in ATOM_TARGETS:
        candidates = [item for item in datasets if item[0] not in used_paths]
        if not candidates:
            break

        chosen = min(
            candidates,
            key=lambda item: (
                float("inf") if dataset_stellar_teff(item[1]) is None else abs(dataset_stellar_teff(item[1]) - target_teff),
                item[0].name,
            ),
        )
        selected.append((target_label, *chosen))
        used_paths.add(chosen[0])

    return selected



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


def format_atom_species_label(species: str) -> str:
    root = element_root(species)
    stage = species_stage(species)
    if stage == 1:
        return rf"{root}$^+$"
    if stage == 2:
        return rf"{root}$^{{++}}$"
    return root



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


def atom_species_sort_key(species: str) -> Tuple[str, int, str]:
    stage = species_stage(species)
    stage_sort = 99 if stage is None else stage
    root = element_root(species)
    element_sort = ELEMENT_ORDER_INDEX.get(root, len(ELEMENT_ORDER_INDEX))
    return (element_sort, stage_sort, species)


def ordered_atom_species(all_species_lists: List[List[str]], selected_species: List[str]) -> List[str]:
    eligible: set[str] = set()
    for species_labels in all_species_lists:
        eligible.update(expand_selected_species(species_labels, selected_species))

    grouped: Dict[str, List[str]] = {category: [] for category in CATEGORY_ORDER}
    uncategorized: List[str] = []

    for species in eligible:
        category = ELEMENT_CATEGORY.get(element_root(species))
        if category is None:
            uncategorized.append(species)
        else:
            grouped[category].append(species)

    ordered: List[str] = []
    for category in CATEGORY_ORDER:
        ordered.extend(sorted(grouped[category], key=atom_species_sort_key))
    ordered.extend(sorted(uncategorized, key=atom_species_sort_key))
    return ordered


def selected_molecule_species(species_labels: List[str]) -> List[str]:
    if not SELECTED_SPECIES:
        return list(species_labels)
    selected = set(SELECTED_SPECIES)
    return [species for species in species_labels if species in selected]


def atom_contrast_limits() -> Tuple[float, float]:
    return float(ATOM_BETA_LOW), float(ATOM_BETA_HIGH)


def build_atom_colormap() -> Tuple[LinearSegmentedColormap, Normalize]:
    vmin, vmax = atom_contrast_limits()
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    green_low_position = float(norm(ATOM_BETA_GREEN_LOW))
    green_high_position = float(norm(ATOM_BETA_GREEN_HIGH))
    blue_plateau_position = float(norm(ATOM_BETA_HIGH))
    cmap = LinearSegmentedColormap.from_list(
        "beta_threshold",
        [
            (0.0, "#7f0000"),
            (green_low_position, "#d73027"),
            (0.5, "#00a651"),
            (green_high_position, "#74add1"),
            (blue_plateau_position, "#08306b"),
            (1.0, "#08306b"),
        ],
        N=256,
    )
    cmap.set_bad(color="#d9d9d9")
    return cmap, norm


def plot_atom_triptych(
    datasets: List[Tuple[str, pathlib.Path, Dict[str, Any], np.ndarray, np.ndarray, List[str], List[str]]]
) -> None:
    species_order = ordered_atom_species([species_labels for _, _, _, _, _, species_labels, _ in datasets], SELECTED_SPECIES)
    if not species_order:
        raise ValueError("No selected atomic species were available to plot.")

    triptych_row_height = ROW_HEIGHT * 0.88
    fig_height = max(MIN_FIGHEIGHT * 1.65, 2.0 + triptych_row_height * len(species_order))
    fig_width = max(FIGWIDTH * 1.45, 4.3 * len(datasets) + 1.0)
    fig, axes = plt.subplots(1, len(datasets), figsize=(fig_width, fig_height), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    cmap, norm = build_atom_colormap()
    image = None

    for ax, (_target_label, _path, metadata, x_values, y_matrix, species_labels, _) in zip(axes, datasets):
        lookup = build_species_lookup(species_labels)
        panel_matrix = np.full((len(species_order), len(x_values)), np.nan, dtype=float)

        for row_idx, species in enumerate(species_order):
            col_idx = lookup.get(species)
            if col_idx is not None:
                panel_matrix[row_idx, :] = y_matrix[:, col_idx]

        masked_panel = np.ma.masked_invalid(panel_matrix)
        image = ax.imshow(
            masked_panel,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            norm=norm,
        )

        stellar_teff = dataset_stellar_teff(metadata)
        teff_text = "unknown" if stellar_teff is None else f"{int(round(stellar_teff))} K"
        ax.set_title(teff_text, fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel(r"$T_{\rm exc}$ [$10^3$ K]", fontsize=AXIS_LABEL_SIZE)
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([f"{(value / 1e3):g}" for value in x_values], rotation=45, ha="right")
        ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", which="major", labelsize=YTICK_LABEL_SIZE)
        ax.set_xticks(np.arange(-0.5, len(x_values), 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, len(species_order), 1.0), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.28, alpha=0.18)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_edgecolor((0, 0, 0, 0.35))

    axes[0].set_yticks(np.arange(len(species_order)))
    axes[0].set_yticklabels([format_atom_species_label(species) for species in species_order])
    for ax in axes[1:]:
        ax.tick_params(axis="y", which="major", labelleft=False)

    fig.suptitle(r"$\beta$ heatmap vs $T_{\rm exc}$ | $N_{\rm col}=0$", fontsize=TITLE_SIZE)
    fig.subplots_adjust(right=0.885, top=0.915, bottom=0.075, wspace=0.08)

    if image is None:
        raise ValueError("No atom heatmap image was created.")

    cbar_ax = fig.add_axes([0.902, 0.075, 0.018, 0.84])
    cbar = fig.colorbar(image, cax=cbar_ax, extend="both")
    cbar.set_ticks(
        [
            ATOM_BETA_LOW,
            ATOM_BETA_GREEN_LOW,
            1.0,
            ATOM_BETA_GREEN_HIGH,
            ATOM_BETA_HIGH,
        ]
    )
    cbar.set_ticklabels(
        [
            f"{ATOM_BETA_LOW:g}",
            "0.95",
            "1",
            "1.05",
            f"{ATOM_BETA_HIGH:g}",
        ]
    )
    cbar.set_label(r"$\beta$", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / OUTPUT_NAME_ATOMS
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


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

    nonempty_groups = [(title, group) for title, group in category_groups if group]
    max_rows = max(len(group) for _, group in nonempty_groups)
    fig_height = max(MIN_FIGHEIGHT * 1.6, 2.6 + ROW_HEIGHT * max_rows)
    fig, axes = plt.subplots(2, 3, figsize=(FIGWIDTH * 1.5, fig_height), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    ncols = 3
    cmap, norm = build_atom_colormap()
    image = None

    for ax, (panel_title, panel_species) in zip(axes, category_groups):
        if not panel_species:
            ax.set_axis_off()
            continue

        column_indices = [species_labels.index(species) for species in panel_species]
        panel_matrix = y_matrix[:, column_indices].T
        masked_panel = np.ma.masked_invalid(panel_matrix)

        image = ax.imshow(
            masked_panel,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            norm=norm,
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
        ax.set_yticklabels([format_atom_species_label(species) for species in panel_species])
        ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", which="major", labelsize=YTICK_LABEL_SIZE)

        if ANNOTATE_CELLS and panel_matrix.shape[0] <= 8:
            for row_idx, species in enumerate(panel_species):
                for col_idx, beta_value in enumerate(panel_matrix[row_idx]):
                    if np.isfinite(beta_value) and beta_value > 0:
                        ax.text(
                            col_idx,
                            row_idx,
                            f"{beta_value:.2f}",
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

    title_parts = [r"$\beta$ heatmap vs $T_{\rm exc}$", mode_label]

    fig.suptitle(" | ".join(title_parts), fontsize=TITLE_SIZE)
    fig.subplots_adjust(right=0.92, top=0.92, bottom=0.14, wspace=0.22, hspace=0.10)
    if image is None:
        raise ValueError(f"No heatmap image was created for {mode_label}.")
    cbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, extend="both")
    cbar.set_ticks(
        [
            ATOM_BETA_LOW,
            ATOM_BETA_GREEN_LOW,
            1.0,
            ATOM_BETA_GREEN_HIGH,
            ATOM_BETA_HIGH,
        ]
    )
    cbar.set_ticklabels(
        [
            f"{ATOM_BETA_LOW:g}",
            "0.95",
            "1",
            "1.05",
            f"{ATOM_BETA_HIGH:g}",
        ]
    )
    cbar.set_label(r"$\beta$", fontsize=AXIS_LABEL_SIZE)
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
    plotted_any = False
    if PLOT_ATOMS:
        atom_files = discover_atom_table_files()
        atom_datasets = [(path, *read_plotdata_txt(path)) for path in atom_files]
        selected_atom_datasets = select_atom_target_datasets(atom_datasets)

        if not selected_atom_datasets:
            print(f"Skipping missing atom txt files matching: {ATOM_FILE_GLOB}")
        else:
            plot_atom_triptych(selected_atom_datasets)
            for target_label, path, metadata_tau1, *_rest in selected_atom_datasets:
                stellar_teff = dataset_stellar_teff(metadata_tau1)
                teff_text = "unknown" if stellar_teff is None else f"{int(round(stellar_teff))} K"
                print(f"Used atom dataset for {target_label} ({teff_text}): {path}")
            plotted_any = True

    if PLOT_MOLECULES:
        table_path_tau1 = TABLES_ROOT / TXT_NAME_MOLECULES
        if not table_path_tau1.exists():
            print(f"Skipping missing txt file: {table_path_tau1}")
        else:
            metadata_tau1, x_tau1, y_tau1, species_tau1, _ = read_plotdata_txt(table_path_tau1)
            plot_dataset(
                metadata_tau1,
                x_tau1,
                y_tau1,
                species_tau1,
                output_name=OUTPUT_NAME_MOLECULES,
                mode_label=rf"$N_{{\rm col}}=0$",
            )
            plotted_any = True

    if not plotted_any:
        raise FileNotFoundError("No enabled beta(T_exc) txt files were found to plot.")



if __name__ == "__main__":
    main()
