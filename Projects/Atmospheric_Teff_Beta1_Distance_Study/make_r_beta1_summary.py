import csv
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
from astropy import constants as const

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import plot_by_txt_file as plot_txt
from project_utils.exobase_table_path import resolve_exobase_table_path
from project_utils.r_beta1_table_sources import discover_rbeta1_table_files
from Templates.Atoms.atom_species import ATOM_SPECIES
from Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from Templates.Planets.planet_templates import PLANET_TEMPLATES


TABLES_BASE_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Teff_study"
OUTPUT_DIR = TABLES_BASE_DIR / "r_at_beta1" / "summary"
EXOBASE_TABLE = (
    resolve_exobase_table_path(pathlib.Path(__file__).resolve().parents[2])
)

CATEGORY_ORDER = [
    "rocky",
    "mini_neptune",
    "sub_neptune",
    "neptune",
    "gas_giant",
]

PLOT_TITLE_SIZE = 23
PLOT_LABEL_SIZE = 20
PLOT_TICK_SIZE = 17

# Use clearly separated palettes so temperature and distance are never confused.
# Temperature: warm yellow -> orange -> red.
# Distance: cool purple -> teal -> blue-green.
TEMP_DISCRETE_CMAP = "YlOrRd"
DISTANCE_DISCRETE_CMAP = "PuBuGn"


def load_exobase_heights(table_path: pathlib.Path) -> dict[tuple[str, str], float]:
    heights: dict[tuple[str, str], float] = {}
    if not table_path.exists():
        return heights

    with table_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            planet = str(row.get("planet", "")).strip()
            species = str(row.get("species", "")).strip()
            z_value = row.get("z_exobase_km", "")
            if not planet or not species:
                continue
            try:
                heights[(planet, species)] = float(z_value)
            except (TypeError, ValueError):
                continue
    return heights


def neutral_exobase_species(species: str) -> str:
    parts = str(species).split()
    if len(parts) != 2:
        return str(species)

    element, stage = parts
    if stage in {"II", "III", "IV"}:
        return f"{element} I"
    return str(species)


def pretty_planet_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def pretty_category_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def pretty_species_label(species: str) -> str:
    parts = str(species).split()
    if len(parts) != 2:
        return str(species)

    element, stage = parts
    charge_map = {
        "I": element,
        "II": rf"{element}$^+$",
        "III": rf"{element}$^{{++}}$",
        "IV": rf"{element}$^{{3+}}$",
    }
    return charge_map.get(stage, str(species))


def species_type(species: str) -> str:
    if species in ATOM_SPECIES:
        return "atom"
    if species in MOLECULE_TEMPLATES:
        return "molecule"
    return "unknown"


def planet_sort_key(planet_key: str) -> tuple[int, str]:
    category = PLANET_TEMPLATES.get(planet_key, {}).get("category", "")
    try:
        category_index = CATEGORY_ORDER.index(category)
    except ValueError:
        category_index = len(CATEGORY_ORDER)
    return category_index, pretty_planet_name(planet_key)


def parse_series_values(metadata: dict[str, str], columns: list[str]) -> np.ndarray:
    raw = metadata.get("series_values", "")
    if raw:
        return np.asarray([float(value.strip()) for value in raw.split(",")], dtype=float)

    return np.asarray(
        [float(series_value) for series_value, _ in plot_txt.extract_series(columns)],
        dtype=float,
    )


def exobase_height_km(metadata: dict[str, str], exobase_heights: dict[tuple[str, str], float]) -> float | None:
    planet = metadata.get("planet", "")
    species = metadata.get("species", "")
    if not planet or not species:
        return None

    z_exobase_km = exobase_heights.get((planet, species))
    if z_exobase_km is None:
        z_exobase_km = exobase_heights.get((planet, neutral_exobase_species(species)))
    return z_exobase_km


def planet_radius_km(metadata: dict[str, str]) -> float | None:
    try:
        radius_rjup = float(metadata.get("planet_radius_Rjup", ""))
    except (TypeError, ValueError):
        return None
    return radius_rjup * const.R_jup.to_value("km")


def safe_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def nanmedian_or_nan(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return np.nan
    return float(np.nanmedian(np.asarray(finite_values, dtype=float)))


def nanmin_or_nan(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return np.nan
    return float(np.nanmin(np.asarray(finite_values, dtype=float)))


def nanmax_or_nan(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return np.nan
    return float(np.nanmax(np.asarray(finite_values, dtype=float)))


def hardest_guaranteed_threshold(
    below_exobase_mask: np.ndarray,
    teff_values: np.ndarray,
    distance_values_au: np.ndarray,
) -> tuple[float, float]:
    if below_exobase_mask.size == 0:
        return np.nan, np.nan

    best_distance = np.nan
    best_teff = np.nan

    for j, distance_value in enumerate(distance_values_au):
        candidate_teffs = []
        for i, teff_value in enumerate(teff_values):
            if not below_exobase_mask[i, j]:
                continue
            favorable_quadrant = below_exobase_mask[i:, : j + 1]
            if favorable_quadrant.size == 0:
                continue
            if np.all(favorable_quadrant):
                candidate_teffs.append(float(teff_value))

        if not candidate_teffs:
            continue

        current_coolest_teff = float(np.min(candidate_teffs))
        if not np.isfinite(best_distance) or float(distance_value) > best_distance:
            best_distance = float(distance_value)
            best_teff = current_coolest_teff
        elif np.isclose(float(distance_value), best_distance) and current_coolest_teff < best_teff:
            best_teff = current_coolest_teff

    return best_distance, best_teff


def build_row_summary(
    table_path: pathlib.Path,
    exobase_heights: dict[tuple[str, str], float],
) -> dict[str, object]:
    metadata, columns, data = plot_txt.parse_header_and_table(table_path)
    teff_values = np.asarray(data[:, 0], dtype=float)
    rbeta_matrix = np.asarray(data[:, 1:], dtype=float)
    distance_values_au = parse_series_values(metadata, columns)

    planet = metadata.get("planet", table_path.parent.name.replace("_r_beta1", ""))
    species = metadata.get("species", table_path.stem.replace("_r_beta1", ""))
    category = PLANET_TEMPLATES.get(planet, {}).get("category", "unknown")
    current_species_type = species_type(species)

    z_exobase_km = exobase_height_km(metadata, exobase_heights)
    radius_km = planet_radius_km(metadata)

    if z_exobase_km is None or radius_km is None or not np.isfinite(radius_km) or radius_km <= 0:
        r_exo_over_rp = np.nan
        eta_matrix = np.full_like(rbeta_matrix, np.nan, dtype=float)
    else:
        r_exo_over_rp = 1.0 + float(z_exobase_km) / float(radius_km)
        eta_matrix = rbeta_matrix / r_exo_over_rp

    finite_mask = np.isfinite(rbeta_matrix)
    finite_eta_mask = np.isfinite(eta_matrix)
    below_exobase_mask = finite_eta_mask & (eta_matrix <= 1.0)

    n_total = int(rbeta_matrix.size)
    n_finite = int(np.count_nonzero(finite_mask))
    n_below = int(np.count_nonzero(below_exobase_mask))

    summary: dict[str, object] = {
        "species_type": current_species_type,
        "category": category,
        "planet": planet,
        "planet_label": pretty_planet_name(planet),
        "species": species,
        "table_path": str(table_path),
        "z_exobase_km": safe_float(z_exobase_km),
        "r_exo_over_Rp": safe_float(r_exo_over_rp),
        "n_total": n_total,
        "n_finite": n_finite,
        "finite_fraction": float(n_finite / n_total) if n_total else np.nan,
        "n_below_exobase": n_below,
        "below_exobase_fraction": float(n_below / n_total) if n_total else np.nan,
    }

    if np.any(finite_mask):
        finite_rbeta = rbeta_matrix[finite_mask]
        summary["min_r_beta1_over_Rp"] = float(np.nanmin(finite_rbeta))
        summary["median_r_beta1_over_Rp"] = float(np.nanmedian(finite_rbeta))
        summary["max_r_beta1_over_Rp"] = float(np.nanmax(finite_rbeta))
    else:
        summary["min_r_beta1_over_Rp"] = np.nan
        summary["median_r_beta1_over_Rp"] = np.nan
        summary["max_r_beta1_over_Rp"] = np.nan

    if np.any(finite_eta_mask):
        finite_eta = eta_matrix[finite_eta_mask]
        summary["min_eta"] = float(np.nanmin(finite_eta))
        summary["median_eta"] = float(np.nanmedian(finite_eta))
        summary["max_eta"] = float(np.nanmax(finite_eta))

        masked_eta = np.where(finite_eta_mask, eta_matrix, np.nan)
        min_flat_index = int(np.nanargmin(masked_eta))
        min_i, min_j = np.unravel_index(min_flat_index, eta_matrix.shape)
        summary["best_case_teff_K"] = float(teff_values[min_i])
        summary["best_case_distance_AU"] = float(distance_values_au[min_j])
        summary["best_case_r_beta1_over_Rp"] = float(rbeta_matrix[min_i, min_j])
    else:
        summary["min_eta"] = np.nan
        summary["median_eta"] = np.nan
        summary["max_eta"] = np.nan
        summary["best_case_teff_K"] = np.nan
        summary["best_case_distance_AU"] = np.nan
        summary["best_case_r_beta1_over_Rp"] = np.nan

    if np.any(below_exobase_mask):
        distance_hit_mask = np.any(below_exobase_mask, axis=0)
        teff_hit_mask = np.any(below_exobase_mask, axis=1)
        summary["min_distance_below_exobase_AU"] = float(np.min(distance_values_au[distance_hit_mask]))
        summary["max_distance_below_exobase_AU"] = float(np.max(distance_values_au[distance_hit_mask]))
        summary["min_teff_below_exobase_K"] = float(np.min(teff_values[teff_hit_mask]))
        summary["max_teff_below_exobase_K"] = float(np.max(teff_values[teff_hit_mask]))

        max_distance_value = float(np.max(distance_values_au[distance_hit_mask]))
        max_distance_mask = np.isclose(distance_values_au, max_distance_value)
        teffs_at_max_distance = teff_values[np.any(below_exobase_mask[:, max_distance_mask], axis=1)]
        summary["coolest_teff_at_max_distance_K"] = float(np.min(teffs_at_max_distance))

        coolest_teff_value = float(np.min(teff_values[teff_hit_mask]))
        coolest_teff_mask = np.isclose(teff_values, coolest_teff_value)
        distances_at_coolest_teff = distance_values_au[np.any(below_exobase_mask[coolest_teff_mask, :], axis=0)]
        summary["coolest_teff_below_exobase_K"] = coolest_teff_value
        summary["largest_distance_at_coolest_teff_AU"] = float(np.max(distances_at_coolest_teff))
        summary["smallest_distance_at_coolest_teff_AU"] = float(np.min(distances_at_coolest_teff))

        threshold_distance_au, threshold_teff_k = hardest_guaranteed_threshold(
            below_exobase_mask,
            teff_values,
            distance_values_au,
        )
        summary["threshold_distance_AU"] = threshold_distance_au
        summary["threshold_teff_K"] = threshold_teff_k
    else:
        summary["min_distance_below_exobase_AU"] = np.nan
        summary["max_distance_below_exobase_AU"] = np.nan
        summary["min_teff_below_exobase_K"] = np.nan
        summary["max_teff_below_exobase_K"] = np.nan
        summary["coolest_teff_at_max_distance_K"] = np.nan
        summary["coolest_teff_below_exobase_K"] = np.nan
        summary["largest_distance_at_coolest_teff_AU"] = np.nan
        summary["smallest_distance_at_coolest_teff_AU"] = np.nan
        summary["threshold_distance_AU"] = np.nan
        summary["threshold_teff_K"] = np.nan

    return summary


def collect_row_summaries(exobase_heights: dict[tuple[str, str], float]) -> list[dict[str, object]]:
    row_summaries: list[dict[str, object]] = []
    for table_path in discover_rbeta1_table_files(TABLES_BASE_DIR):
        row_summaries.append(build_row_summary(table_path, exobase_heights))
    return row_summaries


def aggregate_category_species_rows(row_summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in row_summaries:
        key = (str(row["species_type"]), str(row["category"]), str(row["species"]))
        grouped[key].append(row)

    aggregated_rows: list[dict[str, object]] = []
    for (current_species_type, category, species), rows in sorted(grouped.items()):
        min_eta_values = [safe_float(row["min_eta"]) for row in rows]
        max_distance_values = [safe_float(row["max_distance_below_exobase_AU"]) for row in rows]
        threshold_distance_values = [safe_float(row["threshold_distance_AU"]) for row in rows]
        threshold_teff_values = [safe_float(row["threshold_teff_K"]) for row in rows]
        aggregated_rows.append(
            {
                "species_type": current_species_type,
                "category": category,
                "species": species,
                "n_planets": len(rows),
                "median_min_eta_across_planets": nanmedian_or_nan(
                    [safe_float(row["min_eta"]) for row in rows]
                ),
                "median_median_eta_across_planets": nanmedian_or_nan(
                    [safe_float(row["median_eta"]) for row in rows]
                ),
                "median_below_exobase_fraction_across_planets": nanmedian_or_nan(
                    [safe_float(row["below_exobase_fraction"]) for row in rows]
                ),
                "median_max_distance_below_exobase_AU_across_planets": nanmedian_or_nan(
                    max_distance_values
                ),
                "min_min_eta_any_planet": nanmin_or_nan(min_eta_values),
                "max_distance_below_exobase_AU_any_planet": nanmax_or_nan(max_distance_values),
                "median_threshold_distance_AU_across_planets": nanmedian_or_nan(
                    threshold_distance_values
                ),
                "median_threshold_teff_K_across_planets": nanmedian_or_nan(
                    threshold_teff_values
                ),
            }
        )

    return aggregated_rows


def write_csv(rows: list[dict[str, object]], path: pathlib.Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary_notes(path: pathlib.Path) -> None:
    lines = [
        "Summary outputs for Teff_study/r_at_beta1",
        "",
        "Metric definitions",
        "min_eta:",
        "eta = r_beta1 / r_exo, with r_exo / R_p = 1 + z_exobase / R_p.",
        "Values below 1 mean beta = 1 is reached below the exobase.",
        "",
        "below_exobase_fraction:",
        "Fraction of all sampled (Teff, distance) grid points with eta <= 1.",
        "",
        "max_distance_below_exobase_AU:",
        "Largest orbital distance in the sampled grid where eta <= 1 for at least one stellar Teff.",
        "",
        "coolest_teff_at_max_distance_K:",
        "At that furthest sampled orbital distance, the coolest sampled stellar Teff where eta <= 1 still occurs.",
        "",
        "coolest_teff_below_exobase_K:",
        "Coolest sampled stellar Teff where eta <= 1 occurs.",
        "",
        "largest_distance_at_coolest_teff_AU:",
        "At that coolest sampled Teff, the largest sampled orbital distance where eta <= 1 still occurs.",
        "",
        "threshold_distance_AU and threshold_teff_K:",
        "Hardest sampled threshold pair that guarantees below-exobase loss in the favorable direction.",
        "That means every sampled point with T_eff >= threshold_teff_K and distance <= threshold_distance_AU",
        "also has eta <= 1.",
        "",
        "Category-level heatmaps use medians across planets in each planet category.",
        "Planet-level heatmaps show one cell per (planet, species) table file.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_planet_metric_matrix(
    row_summaries: list[dict[str, object]],
    current_species_type: str,
    metric_field: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    filtered_rows = [
        row for row in row_summaries if str(row["species_type"]) == current_species_type
    ]
    species_labels = sorted({str(row["species"]) for row in filtered_rows})
    planet_keys = sorted({str(row["planet"]) for row in filtered_rows}, key=planet_sort_key)

    row_index = {species: i for i, species in enumerate(species_labels)}
    col_index = {planet: j for j, planet in enumerate(planet_keys)}
    matrix = np.full((len(species_labels), len(planet_keys)), np.nan, dtype=float)

    for row in filtered_rows:
        i = row_index[str(row["species"])]
        j = col_index[str(row["planet"])]
        matrix[i, j] = safe_float(row[metric_field])

    pretty_planet_labels = [pretty_planet_name(planet) for planet in planet_keys]
    return matrix, species_labels, pretty_planet_labels


def build_category_metric_matrix(
    aggregated_rows: list[dict[str, object]],
    current_species_type: str,
    metric_field: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    filtered_rows = [
        row for row in aggregated_rows if str(row["species_type"]) == current_species_type
    ]
    species_labels = sorted({str(row["species"]) for row in filtered_rows})
    categories = [category for category in CATEGORY_ORDER if any(
        str(row["category"]) == category for row in filtered_rows
    )]

    row_index = {species: i for i, species in enumerate(species_labels)}
    col_index = {category: j for j, category in enumerate(categories)}
    matrix = np.full((len(species_labels), len(categories)), np.nan, dtype=float)

    for row in filtered_rows:
        i = row_index[str(row["species"])]
        j = col_index[str(row["category"])]
        matrix[i, j] = safe_float(row[metric_field])

    pretty_category_labels = [pretty_category_name(category) for category in categories]
    return matrix, species_labels, pretty_category_labels


def figsize_for_heatmap(n_rows: int, n_cols: int) -> tuple[float, float]:
    width = max(7.5, 0.6 * n_cols + 3.0)
    height = max(4.5, 0.24 * n_rows + 2.0)
    return width, height


def plot_log_eta_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: pathlib.Path,
    x_label: str,
) -> None:
    if matrix.size == 0:
        return

    log_matrix = np.full_like(matrix, np.nan, dtype=float)
    positive_mask = np.isfinite(matrix) & (matrix > 0.0)
    log_matrix[positive_mask] = np.log10(matrix[positive_mask])
    finite_values = log_matrix[np.isfinite(log_matrix)]
    if finite_values.size == 0:
        return

    vmin = min(-0.5, float(np.nanpercentile(finite_values, 5)))
    vmax = max(0.5, float(np.nanpercentile(finite_values, 95)))
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad("#f0f0f0")

    fig, ax = plt.subplots(figsize=figsize_for_heatmap(len(row_labels), len(col_labels)))
    image = ax.imshow(np.ma.masked_invalid(log_matrix), aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Species")
    ax.tick_params(axis="both", labelsize=8)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"$\log_{10}(r_{\beta=1} / r_{\rm exo})$")
    colorbar.formatter = FuncFormatter(lambda value, _: f"{10**value:.2g}")
    colorbar.update_ticks()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fraction_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: pathlib.Path,
    x_label: str,
) -> None:
    if matrix.size == 0:
        return

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f0f0f0")

    fig, ax = plt.subplots(figsize=figsize_for_heatmap(len(row_labels), len(col_labels)))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Species")
    ax.tick_params(axis="both", labelsize=8)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Fraction of sampled grid with $r_{\\beta=1} \\leq r_{\\rm exo}$")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distance_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: pathlib.Path,
    x_label: str,
) -> None:
    if matrix.size == 0:
        return

    allowed_distances = np.asarray(
        sorted({float(value) for value in matrix[np.isfinite(matrix)]}),
        dtype=float,
    )
    if allowed_distances.size == 0:
        return
    value_to_index = {value: index for index, value in enumerate(allowed_distances)}

    discrete_matrix = np.full_like(matrix, np.nan, dtype=float)
    for distance_value, distance_index in value_to_index.items():
        discrete_matrix[np.isclose(matrix, distance_value, equal_nan=False)] = float(distance_index)

    cmap = plt.get_cmap(DISTANCE_DISCRETE_CMAP)(np.linspace(0.16, 0.92, len(allowed_distances)))
    cmap = colors.ListedColormap(cmap)
    cmap.set_bad("#f0f0f0")
    norm = colors.BoundaryNorm(np.arange(-0.5, len(allowed_distances) + 0.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=figsize_for_heatmap(len(row_labels), len(col_labels)))
    image = ax.imshow(
        np.ma.masked_invalid(discrete_matrix),
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Species")
    ax.tick_params(axis="both", labelsize=8)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02, ticks=np.arange(len(allowed_distances)))
    colorbar.ax.set_yticklabels([str(value) for value in allowed_distances])
    colorbar.set_label("Largest sampled distance with $r_{\\beta=1} \\leq r_{\\rm exo}$ [AU]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def discrete_palette(base_cmap_name: str, n_colors: int, start: float = 0.08, stop: float = 0.92) -> np.ndarray:
    cmap = plt.get_cmap(base_cmap_name)
    return cmap(np.linspace(start, stop, n_colors))


def select_palette_colors(base_colors: np.ndarray, n_values: int) -> np.ndarray:
    if n_values <= 0:
        return np.empty((0, 4), dtype=float)
    if n_values == 1:
        return np.asarray([base_colors[len(base_colors) // 2]], dtype=float)

    positions = np.linspace(0, len(base_colors) - 1, n_values).round().astype(int)
    return np.asarray([base_colors[position] for position in positions], dtype=float)


def plot_split_metric_heatmap(
    temp_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: pathlib.Path,
    x_label: str,
) -> None:
    if temp_matrix.size == 0 or distance_matrix.size == 0:
        return

    finite_temp_values = sorted({float(value) for value in temp_matrix[np.isfinite(temp_matrix)]})
    finite_distance_values = sorted({float(value) for value in distance_matrix[np.isfinite(distance_matrix)]})
    if not finite_temp_values or not finite_distance_values:
        return

    temp_base_colors = discrete_palette(TEMP_DISCRETE_CMAP, len(finite_temp_values), start=0.16, stop=0.95)
    dist_base_colors = discrete_palette(DISTANCE_DISCRETE_CMAP, len(finite_distance_values), start=0.16, stop=0.92)
    temp_colors = select_palette_colors(temp_base_colors, len(finite_temp_values))
    dist_colors = select_palette_colors(dist_base_colors, len(finite_distance_values))

    temp_color_map = {value: color for value, color in zip(finite_temp_values, temp_colors)}
    dist_color_map = {value: color for value, color in zip(finite_distance_values, dist_colors)}
    display_row_labels = [pretty_species_label(label) for label in row_labels]

    fig, ax = plt.subplots(figsize=figsize_for_heatmap(len(row_labels), len(col_labels)))
    fig.subplots_adjust(right=0.82)
    no_hit_color = "#e6e6e6"

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            temp_value = safe_float(temp_matrix[i, j])
            distance_value = safe_float(distance_matrix[i, j])
            x0 = j - 0.5
            y0 = i - 0.5

            if not np.isfinite(temp_value) or not np.isfinite(distance_value):
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        1.0,
                        1.0,
                        facecolor=no_hit_color,
                        edgecolor="white",
                        linewidth=0.4,
                    )
                )
                continue

            ax.add_patch(
                Polygon(
                    [(x0, y0), (x0 + 1.0, y0), (x0, y0 + 1.0)],
                    closed=True,
                    facecolor=temp_color_map[float(temp_value)],
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Polygon(
                    [(x0 + 1.0, y0), (x0 + 1.0, y0 + 1.0), (x0, y0 + 1.0)],
                    closed=True,
                    facecolor=dist_color_map[float(distance_value)],
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    1.0,
                    1.0,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=0.4,
                )
            )
            ax.plot([x0, x0 + 1.0], [y0 + 1.0, y0], color="white", linewidth=0.4)

    ax.set_title(title, fontsize=PLOT_TITLE_SIZE)
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=PLOT_TICK_SIZE)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(display_row_labels, fontsize=PLOT_TICK_SIZE)
    ax.set_xlabel(x_label, fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel("Species", fontsize=PLOT_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)
    ax.set_facecolor("white")

    temp_cmap = colors.ListedColormap([temp_color_map[value] for value in finite_temp_values])
    temp_norm = colors.BoundaryNorm(np.arange(-0.5, len(finite_temp_values) + 0.5, 1.0), temp_cmap.N)
    temp_mappable = plt.cm.ScalarMappable(cmap=temp_cmap, norm=temp_norm)
    temp_mappable.set_array([])

    dist_cmap = colors.ListedColormap([dist_color_map[value] for value in finite_distance_values])
    dist_norm = colors.BoundaryNorm(np.arange(-0.5, len(finite_distance_values) + 0.5, 1.0), dist_cmap.N)
    dist_mappable = plt.cm.ScalarMappable(cmap=dist_cmap, norm=dist_norm)
    dist_mappable.set_array([])

    temp_cax = fig.add_axes([0.845, 0.58, 0.028, 0.28])
    temp_colorbar = fig.colorbar(
        temp_mappable,
        cax=temp_cax,
        ticks=np.arange(len(finite_temp_values)),
    )
    temp_colorbar.ax.set_yticklabels(
        [f"{value / 1e4:g}" for value in finite_temp_values],
        fontsize=PLOT_TICK_SIZE,
    )
    temp_colorbar.set_label(
        r"$T_{\rm eff}\,[10^4\,\mathrm{K}]$",
        fontsize=PLOT_LABEL_SIZE,
    )
    temp_colorbar.ax.tick_params(labelsize=PLOT_TICK_SIZE)

    dist_cax = fig.add_axes([0.845, 0.18, 0.028, 0.28])
    dist_colorbar = fig.colorbar(
        dist_mappable,
        cax=dist_cax,
        ticks=np.arange(len(finite_distance_values)),
    )
    dist_colorbar.ax.set_yticklabels([f"{value:g}" for value in finite_distance_values], fontsize=PLOT_TICK_SIZE)
    dist_colorbar.set_label(
        "Distance [AU]",
        fontsize=PLOT_LABEL_SIZE,
    )
    dist_colorbar.ax.tick_params(labelsize=PLOT_TICK_SIZE)

    no_hit_handle = Patch(facecolor=no_hit_color, edgecolor="none", label="No $r_{\\beta=1} \\leq r_{\\rm exo}$")
    no_hit_legend = ax.legend(
        handles=[no_hit_handle],
        loc="upper left",
        bbox_to_anchor=(1.01, 0.04),
        frameon=True,
        fontsize=PLOT_TICK_SIZE,
    )
    ax.add_artist(no_hit_legend)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_heatmaps(
    row_summaries: list[dict[str, object]],
    aggregated_rows: list[dict[str, object]],
) -> None:
    for current_species_type in ["atom", "molecule"]:
        planet_min_eta, species_labels, planet_labels = build_planet_metric_matrix(
            row_summaries,
            current_species_type,
            "min_eta",
        )
        planet_fraction, _, _ = build_planet_metric_matrix(
            row_summaries,
            current_species_type,
            "below_exobase_fraction",
        )
        planet_distance, _, _ = build_planet_metric_matrix(
            row_summaries,
            current_species_type,
            "max_distance_below_exobase_AU",
        )
        planet_coolest_teff, _, _ = build_planet_metric_matrix(
            row_summaries,
            current_species_type,
            "threshold_teff_K",
        )
        planet_threshold_distance, _, _ = build_planet_metric_matrix(
            row_summaries,
            current_species_type,
            "threshold_distance_AU",
        )

        category_min_eta, category_species_labels, category_labels = build_category_metric_matrix(
            aggregated_rows,
            current_species_type,
            "median_min_eta_across_planets",
        )
        category_fraction, _, _ = build_category_metric_matrix(
            aggregated_rows,
            current_species_type,
            "median_below_exobase_fraction_across_planets",
        )
        category_distance, _, _ = build_category_metric_matrix(
            aggregated_rows,
            current_species_type,
            "max_distance_below_exobase_AU_any_planet",
        )

        plot_log_eta_heatmap(
            planet_min_eta,
            species_labels,
            planet_labels,
            f"{current_species_type.title()}s: best-case $r_{{\\beta=1}}/r_{{\\rm exo}}$ by planet",
            OUTPUT_DIR / f"{current_species_type}s_min_eta_by_planet.pdf",
            "Planet",
        )
        plot_fraction_heatmap(
            planet_fraction,
            species_labels,
            planet_labels,
            f"{current_species_type.title()}s: fraction with $r_{{\\beta=1}} \\leq r_{{\\rm exo}}$ by planet",
            OUTPUT_DIR / f"{current_species_type}s_below_exobase_fraction_by_planet.pdf",
            "Planet",
        )
        plot_distance_heatmap(
            planet_distance,
            species_labels,
            planet_labels,
            f"{current_species_type.title()}s: largest sampled distance with $r_{{\\beta=1}} \\leq r_{{\\rm exo}}$",
            OUTPUT_DIR / f"{current_species_type}s_max_distance_below_exobase_by_planet.pdf",
            "Planet",
        )
        plot_split_metric_heatmap(
            planet_coolest_teff,
            planet_threshold_distance,
            species_labels,
            planet_labels,
            f"{current_species_type.title()}s: Threshold for below-exobase loss",
            OUTPUT_DIR / f"{current_species_type}s_coolest_teff_and_distance_by_planet.pdf",
            "Planet",
        )

        plot_log_eta_heatmap(
            category_min_eta,
            category_species_labels,
            category_labels,
            f"{current_species_type.title()}s: median best-case $r_{{\\beta=1}}/r_{{\\rm exo}}$ by planet category",
            OUTPUT_DIR / f"{current_species_type}s_min_eta_by_category.pdf",
            "Planet category",
        )
        plot_fraction_heatmap(
            category_fraction,
            category_species_labels,
            category_labels,
            f"{current_species_type.title()}s: median below-exobase fraction by planet category",
            OUTPUT_DIR / f"{current_species_type}s_below_exobase_fraction_by_category.pdf",
            "Planet category",
        )
        plot_distance_heatmap(
            category_distance,
            category_species_labels,
            category_labels,
            f"{current_species_type.title()}s: largest sampled distance reached by any planet in category",
            OUTPUT_DIR / f"{current_species_type}s_max_distance_below_exobase_by_category.pdf",
            "Planet category",
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exobase_heights = load_exobase_heights(EXOBASE_TABLE)
    row_summaries = collect_row_summaries(exobase_heights)
    if not row_summaries:
        raise ValueError(f"No r_beta1 txt files found under discovered roots in {TABLES_BASE_DIR}")

    row_summaries = sorted(
        row_summaries,
        key=lambda row: (
            0 if str(row["species_type"]) == "atom" else 1,
            planet_sort_key(str(row["planet"])),
            str(row["species"]),
        ),
    )

    aggregated_rows = aggregate_category_species_rows(row_summaries)
    aggregated_rows = sorted(
        aggregated_rows,
        key=lambda row: (
            0 if str(row["species_type"]) == "atom" else 1,
            CATEGORY_ORDER.index(str(row["category"])) if str(row["category"]) in CATEGORY_ORDER else len(CATEGORY_ORDER),
            str(row["species"]),
        ),
    )

    write_csv(
        row_summaries,
        OUTPUT_DIR / "r_beta1_summary_by_planet_species.csv",
        [
            "species_type",
            "category",
            "planet",
            "planet_label",
            "species",
            "z_exobase_km",
            "r_exo_over_Rp",
            "n_total",
            "n_finite",
            "finite_fraction",
            "n_below_exobase",
            "below_exobase_fraction",
            "min_r_beta1_over_Rp",
            "median_r_beta1_over_Rp",
            "max_r_beta1_over_Rp",
            "min_eta",
            "median_eta",
            "max_eta",
            "best_case_teff_K",
            "best_case_distance_AU",
            "best_case_r_beta1_over_Rp",
            "min_distance_below_exobase_AU",
            "max_distance_below_exobase_AU",
            "min_teff_below_exobase_K",
            "max_teff_below_exobase_K",
            "coolest_teff_at_max_distance_K",
            "coolest_teff_below_exobase_K",
            "largest_distance_at_coolest_teff_AU",
            "smallest_distance_at_coolest_teff_AU",
            "threshold_distance_AU",
            "threshold_teff_K",
            "table_path",
        ],
    )
    write_csv(
        aggregated_rows,
        OUTPUT_DIR / "r_beta1_summary_by_category_species.csv",
        [
            "species_type",
            "category",
            "species",
            "n_planets",
            "median_min_eta_across_planets",
            "median_median_eta_across_planets",
            "median_below_exobase_fraction_across_planets",
            "median_max_distance_below_exobase_AU_across_planets",
            "min_min_eta_any_planet",
            "max_distance_below_exobase_AU_any_planet",
            "median_threshold_distance_AU_across_planets",
            "median_threshold_teff_K_across_planets",
        ],
    )
    write_summary_notes(OUTPUT_DIR / "summary_metric_definitions.txt")
    make_heatmaps(row_summaries, aggregated_rows)

    print(f"Wrote summaries to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
