import csv
import math
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from Templates.Atoms.atom_species import ATOM_SPECIES
from Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from Templates.Planets.planet_templates import PLANET_TEMPLATES


OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Teff_study" / "R_p_exo_beta1"
RAW_FILES = [
    OUTPUT_DIR / "R_p_exo_beta1_atoms.txt",
    OUTPUT_DIR / "R_p_exo_beta1_molecules.txt",
    OUTPUT_DIR / "R_p_exo_beta1_all.txt",
]

CATEGORY_ORDER = [
    "rocky",
    "mini_neptune",
    "sub_neptune",
    "neptune",
    "gas_giant",
]

TITLE_SIZE = 35
LABEL_SIZE = 39
Y_TICK_SIZE = 28
X_TICK_SIZE = 32
CMAP_TICK_SIZE = 32
TEMP_CMAP_NAME = "YlOrRd"
DISTANCE_CMAP_NAME = "PuBuGn"
DISTANCE_DISCRETE_COLORS = [
    "#08306b",
    "#2171b5",
    "#41ab5d",
    "#feb24c",
    "#f03b20",
    "#7a0177",
    "#252525",
    "#08519c",
]
PERIODIC_ELEMENTS_THROUGH_FE = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe",
]
ELEMENT_ORDER = {element: index for index, element in enumerate(PERIODIC_ELEMENTS_THROUGH_FE)}
IONIZATION_STAGE_ORDER = {"I": 0, "II": 1, "III": 2, "IV": 3}


def pretty_planet_name(name: str) -> str:
    short_names = {
        "alkali_exosphere_rocky": "Alkali Rocky",
        "metal_rich_secondary": "Metal rich",
        "super_earth_rocky": "Super earth",
    }
    if name in short_names:
        return short_names[name]
    return str(name).replace("_", " ").title()


def pretty_species_label(species: str) -> str:
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


def species_sort_key(species: str) -> tuple:
    parts = str(species).split()
    if len(parts) == 2:
        element, stage = parts
        if stage in IONIZATION_STAGE_ORDER:
            return (
                0,
                IONIZATION_STAGE_ORDER[stage],
                ELEMENT_ORDER.get(element, len(ELEMENT_ORDER)),
                str(species),
            )
    return (1, pretty_species_label(species))


def planet_sort_key(planet_key: str) -> tuple[int, str]:
    category = PLANET_TEMPLATES.get(planet_key, {}).get("category", "")
    try:
        category_index = CATEGORY_ORDER.index(category)
    except ValueError:
        category_index = len(CATEGORY_ORDER)
    return category_index, pretty_planet_name(planet_key)


def classify_species(species: str) -> str:
    if species in ATOM_SPECIES:
        return "atom"
    if species in MOLECULE_TEMPLATES:
        return "molecule"
    return "unknown"


def load_raw_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def hardest_guaranteed_threshold(
    hit_mask: np.ndarray,
    teff_values: np.ndarray,
    distance_values_au: np.ndarray,
) -> tuple[float, float]:
    if hit_mask.size == 0:
        return np.nan, np.nan

    best_distance = np.nan
    best_teff = np.nan

    for j, distance_value in enumerate(distance_values_au):
        candidate_teffs = []
        for i, teff_value in enumerate(teff_values):
            if not hit_mask[i, j]:
                continue
            favorable_quadrant = hit_mask[i:, : j + 1]
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


def build_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["planet"], row["species"])
        grouped_rows.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (planet_key, species), group_rows in sorted(grouped_rows.items()):
        teff_values = sorted({float(row["stellar_teff_K"]) for row in group_rows})
        distance_values = sorted({float(row["distance_AU"]) for row in group_rows})
        teff_index = {value: i for i, value in enumerate(teff_values)}
        distance_index = {value: j for j, value in enumerate(distance_values)}
        hit_matrix = np.full((len(teff_values), len(distance_values)), np.nan, dtype=float)
        z_exobase_values: list[float] = []

        for row in group_rows:
            teff = float(row["stellar_teff_K"])
            distance = float(row["distance_AU"])
            hit_matrix[teff_index[teff], distance_index[distance]] = float(row["beta1_hit_below_exobase"])
            try:
                z_exobase = float(row["z_exobase_km"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(z_exobase):
                z_exobase_values.append(z_exobase)

        finite_mask = np.isfinite(hit_matrix)
        hit_mask = finite_mask & (hit_matrix >= 0.5)
        n_total = int(hit_matrix.size)
        n_finite = int(np.count_nonzero(finite_mask))
        n_hits = int(np.count_nonzero(hit_mask))

        teff_array = np.asarray(teff_values, dtype=float)
        distance_array = np.asarray(distance_values, dtype=float)
        hit_teff_candidates = teff_array[np.any(hit_mask, axis=1)]
        hit_distance_candidates = distance_array[np.any(hit_mask, axis=0)]
        threshold_distance_au, threshold_teff_k = hardest_guaranteed_threshold(
            hit_mask,
            teff_array,
            distance_array,
        )

        summary_rows.append(
            {
                "species_type": classify_species(species),
                "planet": planet_key,
                "category": PLANET_TEMPLATES.get(planet_key, {}).get("category", "unknown"),
                "species": species,
                "z_exobase_km": float(np.nanmax(np.asarray(z_exobase_values, dtype=float))) if z_exobase_values else np.nan,
                "n_total": n_total,
                "n_finite": n_finite,
                "finite_fraction": float(n_finite / n_total) if n_total else np.nan,
                "n_hits": n_hits,
                "hit_fraction": float(n_hits / n_total) if n_total else np.nan,
                "coolest_teff_with_hit_K": float(np.min(hit_teff_candidates)) if hit_teff_candidates.size else np.nan,
                "max_distance_with_hit_AU": float(np.max(hit_distance_candidates)) if hit_distance_candidates.size else np.nan,
                "threshold_teff_K": threshold_teff_k,
                "threshold_distance_AU": threshold_distance_au,
            }
        )

    return summary_rows


def save_summary_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    fieldnames = [
        "species_type",
        "planet",
        "category",
        "species",
        "z_exobase_km",
        "n_total",
        "n_finite",
        "finite_fraction",
        "n_hits",
        "hit_fraction",
        "coolest_teff_with_hit_K",
        "max_distance_with_hit_AU",
        "threshold_teff_K",
        "threshold_distance_AU",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discrete_value_map(values: list[float], cmap_name: str) -> tuple[dict[float, tuple[float, float, float, float]], colors.ListedColormap]:
    clean_values = sorted({float(value) for value in values if np.isfinite(value)})
    if not clean_values:
        cmap = colors.ListedColormap(["#ffffff"])
        return {}, cmap

    base_cmap = plt.get_cmap(cmap_name)
    color_positions = np.linspace(0.15, 0.9, len(clean_values))
    rgba_colors = [base_cmap(pos) for pos in color_positions]
    value_to_color = {value: color for value, color in zip(clean_values, rgba_colors)}
    cmap = colors.ListedColormap(rgba_colors)
    return value_to_color, cmap


def discrete_distance_value_map(values: list[float]) -> tuple[dict[float, tuple[float, float, float, float]], colors.ListedColormap]:
    clean_values = sorted({float(value) for value in values if np.isfinite(value)})
    if not clean_values:
        cmap = colors.ListedColormap(["#ffffff"])
        return {}, cmap

    if len(clean_values) > len(DISTANCE_DISCRETE_COLORS):
        raise ValueError(
            f"Not enough distinct distance colors for {len(clean_values)} values. "
            f"Extend DISTANCE_DISCRETE_COLORS in {__file__}."
        )

    rgba_colors = [colors.to_rgba(color) for color in DISTANCE_DISCRETE_COLORS[: len(clean_values)]]
    value_to_color = {value: color for value, color in zip(clean_values, rgba_colors)}
    cmap = colors.ListedColormap(rgba_colors)
    return value_to_color, cmap


def format_tex_integer(value: float) -> str:
    return f"{int(round(float(value))):,}".replace(",", r"\,")


def title_teff_label(summary_rows: list[dict[str, object]]) -> str:
    teff_values = sorted(
        {
            float(row["threshold_teff_K"])
            for row in summary_rows
            if np.isfinite(float(row["threshold_teff_K"]))
        }
    )
    if len(teff_values) == 1:
        return rf"$T_{{\rm eff}}={format_tex_integer(teff_values[0])}\ \mathrm{{K}}$"
    return r"$T_{\rm eff}$ grid"


def plot_threshold_heatmap(summary_rows: list[dict[str, object]], title: str, output_path: pathlib.Path) -> bool:
    if not summary_rows:
        if output_path.exists():
            output_path.unlink()
        print(f"No threshold rows available; removed stale plot at {output_path}")
        return False

    planet_order = sorted({row["planet"] for row in summary_rows}, key=planet_sort_key)
    lookup = {(row["species"], row["planet"]): row for row in summary_rows}
    species_with_data = {
        row["species"]
        for row in summary_rows
        if np.isfinite(float(row["threshold_teff_K"])) and np.isfinite(float(row["threshold_distance_AU"]))
    }
    species_order = sorted(species_with_data, key=species_sort_key)
    if not species_order:
        if output_path.exists():
            output_path.unlink()
        print(f"No finite threshold values available; removed stale plot at {output_path}")
        return False

    distance_values = [float(row["threshold_distance_AU"]) for row in summary_rows if np.isfinite(float(row["threshold_distance_AU"]))]
    distance_to_color, distance_cmap = discrete_distance_value_map(distance_values)

    fig_width = max(12, 1.1 * len(planet_order) + 5)
    fig_height = max(8, 0.42 * len(species_order) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    missing_color = "#d9d9d9"
    for y_idx, species in enumerate(species_order):
        for x_idx, planet in enumerate(planet_order):
            row = lookup.get((species, planet))
            x0 = x_idx
            y0 = y_idx
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    1.0,
                    1.0,
                    facecolor=missing_color,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            if row is None:
                continue

            teff_value = float(row["threshold_teff_K"])
            distance_value = float(row["threshold_distance_AU"])
            if not np.isfinite(teff_value) or not np.isfinite(distance_value):
                continue

            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    1.0,
                    1.0,
                    facecolor=distance_to_color[distance_value],
                    edgecolor="white",
                    linewidth=0.8,
                )
            )

    ax.set_xlim(0, len(planet_order))
    ax.set_ylim(0, len(species_order))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(planet_order)) + 0.5)
    ax.set_yticks(np.arange(len(species_order)) + 0.5)
    ax.set_xticklabels([pretty_planet_name(planet) for planet in planet_order], rotation=45, ha="right")
    ax.set_yticklabels([pretty_species_label(species) for species in species_order])
    ax.set_xlabel("Planet", fontsize=LABEL_SIZE)
    ax.set_ylabel("Species", fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=X_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(right=0.84)

    if distance_values:
        distance_norm = colors.BoundaryNorm(np.arange(len(distance_to_color) + 1), distance_cmap.N)
        sm_distance = plt.cm.ScalarMappable(norm=distance_norm, cmap=distance_cmap)
        sm_distance.set_array([])
        cax_distance = fig.add_axes([0.855, 0.38, 0.022, 0.28])
        cbar_distance = fig.colorbar(sm_distance, cax=cax_distance)
        ordered_distance = sorted(distance_to_color)
        cbar_distance.set_ticks(np.arange(len(ordered_distance)) + 0.5)
        cbar_distance.set_ticklabels([f"{value:g}" for value in ordered_distance])
        cbar_distance.ax.tick_params(labelsize=CMAP_TICK_SIZE)
        cbar_distance.set_label("Distance [AU]", fontsize=LABEL_SIZE)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def species_title_from_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "Absorbers"
    species_type = rows[0]["species_type"]
    if species_type == "atom":
        return "Atoms"
    if species_type == "molecule":
        return "Molecules"
    return "Absorbers"


def ionization_stage(species: str) -> str | None:
    parts = str(species).split()
    if len(parts) != 2:
        return None
    stage = parts[1]
    if stage in {"I", "II", "III"}:
        return stage
    return None


def finite_threshold_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    kept = []
    for row in rows:
        try:
            td = float(row["threshold_distance_AU"])
            tt = float(row["threshold_teff_K"])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(td) and math.isfinite(tt)):
            continue
        copied = dict(row)
        copied["_threshold_distance_AU"] = td
        copied["_threshold_teff_K"] = tt
        copied["_hit_fraction"] = float(row["hit_fraction"])
        kept.append(copied)
    return kept


def distance_count_columns(distance_values: list[float]) -> list[str]:
    return [f"count_distance_ge_{value:g}_AU" for value in sorted(distance_values, reverse=True)]


def temperature_count_columns(teff_values: list[float]) -> list[str]:
    return [f"count_teff_le_{int(value):d}_K" for value in sorted(teff_values)]


def ranking_key_distance(row: dict[str, object], distance_values: list[float]) -> tuple:
    return (
        *[-int(row[f"count_distance_ge_{value:g}_AU"]) for value in sorted(distance_values, reverse=True)],
        float(row["best_threshold_teff_K"]),
        -float(row["best_threshold_distance_AU"]),
        str(row["label"]),
    )


def ranking_key_temperature(row: dict[str, object], teff_values: list[float]) -> tuple:
    return (
        *[-int(row[f"count_teff_le_{int(value):d}_K"]) for value in sorted(teff_values)],
        -float(row["best_threshold_distance_AU"]),
        float(row["best_threshold_teff_K"]),
        str(row["label"]),
    )


def aggregate_species_rows(
    rows: list[dict[str, object]],
    stages: list[str],
    distance_values: list[float],
    teff_values: list[float],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for stage in stages:
        species_rows = [row for row in rows if ionization_stage(str(row["species"])) == stage]
        by_species: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in species_rows:
            by_species[str(row["species"])].append(row)

        aggregated: list[dict[str, object]] = []
        for species, group_rows in sorted(by_species.items()):
            best_row = min(
                group_rows,
                key=lambda r: (-float(r["_threshold_distance_AU"]), float(r["_threshold_teff_K"]), -float(r["_hit_fraction"])),
            )
            record: dict[str, object] = {
                "label": species,
                "species": species,
                "ionization_stage": stage,
                "n_planets_with_threshold": len(group_rows),
                "best_planet": best_row["planet"],
                "best_threshold_distance_AU": float(best_row["_threshold_distance_AU"]),
                "best_threshold_teff_K": float(best_row["_threshold_teff_K"]),
                "mean_hit_fraction": float(np.mean([float(row["_hit_fraction"]) for row in group_rows])),
            }
            for value in sorted(distance_values, reverse=True):
                record[f"count_distance_ge_{value:g}_AU"] = int(
                    sum(float(row["_threshold_distance_AU"]) >= value for row in group_rows)
                )
            for value in sorted(teff_values):
                record[f"count_teff_le_{int(value):d}_K"] = int(
                    sum(float(row["_threshold_teff_K"]) <= value for row in group_rows)
                )
            aggregated.append(record)
        grouped[stage] = aggregated
    return grouped


def aggregate_planet_rows(
    rows: list[dict[str, object]],
    distance_values: list[float],
    teff_values: list[float],
) -> list[dict[str, object]]:
    by_planet: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_planet[str(row["planet"])].append(row)

    aggregated: list[dict[str, object]] = []
    for planet, group_rows in sorted(by_planet.items(), key=lambda item: planet_sort_key(item[0])):
        best_row = min(
            group_rows,
            key=lambda r: (-float(r["_threshold_distance_AU"]), float(r["_threshold_teff_K"]), -float(r["_hit_fraction"])),
        )
        record: dict[str, object] = {
            "label": planet,
            "planet": planet,
            "category": PLANET_TEMPLATES.get(planet, {}).get("category", "unknown"),
            "n_species_with_threshold": len(group_rows),
            "best_species": best_row["species"],
            "best_threshold_distance_AU": float(best_row["_threshold_distance_AU"]),
            "best_threshold_teff_K": float(best_row["_threshold_teff_K"]),
            "mean_hit_fraction": float(np.mean([float(row["_hit_fraction"]) for row in group_rows])),
        }
        for value in sorted(distance_values, reverse=True):
            record[f"count_distance_ge_{value:g}_AU"] = int(
                sum(float(row["_threshold_distance_AU"]) >= value for row in group_rows)
            )
        for value in sorted(teff_values):
            record[f"count_teff_le_{int(value):d}_K"] = int(
                sum(float(row["_threshold_teff_K"]) <= value for row in group_rows)
            )
        aggregated.append(record)
    return aggregated


def write_csv_rows(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_ranking_tables(summary_rows: list[dict[str, object]], raw_file: pathlib.Path) -> None:
    filtered_rows = finite_threshold_rows(summary_rows)
    if not filtered_rows:
        return

    atom_rows = [row for row in filtered_rows if str(row.get("species_type", "")) == "atom"]
    distance_values = sorted({float(row["_threshold_distance_AU"]) for row in filtered_rows})
    teff_values = sorted({float(row["_threshold_teff_K"]) for row in filtered_rows})

    species_groups = aggregate_species_rows(atom_rows, ["I", "II", "III"], distance_values, teff_values)
    species_base_fields = [
        "rank",
        "species",
        "ionization_stage",
        "n_planets_with_threshold",
        "best_planet",
        "best_threshold_distance_AU",
        "best_threshold_teff_K",
        "mean_hit_fraction",
    ]
    species_distance_fields = species_base_fields + distance_count_columns(distance_values)
    species_temperature_fields = species_base_fields + temperature_count_columns(teff_values)

    stage_name_map = {"I": "neutral", "II": "singly_ionized", "III": "doubly_ionized"}
    for stage, rows in species_groups.items():
        rows_distance = [dict(row) for row in sorted(rows, key=lambda row: ranking_key_distance(row, distance_values))]
        rows_temperature = [dict(row) for row in sorted(rows, key=lambda row: ranking_key_temperature(row, teff_values))]
        rows_distance = rows_distance[:10]
        rows_temperature = rows_temperature[:10]

        for rank, row in enumerate(rows_distance, start=1):
            row["rank"] = rank
        for rank, row in enumerate(rows_temperature, start=1):
            row["rank"] = rank

        distance_path = raw_file.with_name(f"{raw_file.stem}_species_rank_by_distance_{stage_name_map[stage]}.csv")
        temperature_path = raw_file.with_name(f"{raw_file.stem}_species_rank_by_temperature_{stage_name_map[stage]}.csv")
        write_csv_rows(distance_path, species_distance_fields, rows_distance)
        write_csv_rows(temperature_path, species_temperature_fields, rows_temperature)
        print(f"Saved species distance ranking to {distance_path}")
        print(f"Saved species temperature ranking to {temperature_path}")

    planet_rows = aggregate_planet_rows(filtered_rows, distance_values, teff_values)
    planet_base_fields = [
        "rank",
        "planet",
        "category",
        "n_species_with_threshold",
        "best_species",
        "best_threshold_distance_AU",
        "best_threshold_teff_K",
        "mean_hit_fraction",
    ]
    planet_distance_fields = planet_base_fields + distance_count_columns(distance_values)
    planet_temperature_fields = planet_base_fields + temperature_count_columns(teff_values)

    planets_distance = [dict(row) for row in sorted(planet_rows, key=lambda row: ranking_key_distance(row, distance_values))]
    planets_temperature = [dict(row) for row in sorted(planet_rows, key=lambda row: ranking_key_temperature(row, teff_values))]
    planets_distance = planets_distance[:8]
    planets_temperature = planets_temperature[:8]
    for rank, row in enumerate(planets_distance, start=1):
        row["rank"] = rank
    for rank, row in enumerate(planets_temperature, start=1):
        row["rank"] = rank

    overall_distance_path = raw_file.with_name(f"{raw_file.stem}_planet_rank_by_distance_overall.csv")
    overall_temperature_path = raw_file.with_name(f"{raw_file.stem}_planet_rank_by_temperature_overall.csv")
    write_csv_rows(overall_distance_path, planet_distance_fields, planets_distance)
    write_csv_rows(overall_temperature_path, planet_temperature_fields, planets_temperature)
    print(f"Saved planet distance ranking to {overall_distance_path}")
    print(f"Saved planet temperature ranking to {overall_temperature_path}")


def main():
    for raw_file in RAW_FILES:
        if not raw_file.exists():
            continue

        rows = load_raw_rows(raw_file)
        if not rows:
            continue

        summary_rows = build_summary_rows(rows)
        summary_csv_path = raw_file.with_name(f"{raw_file.stem}_summary.csv")
        save_summary_csv(summary_rows, summary_csv_path)
        save_ranking_tables(summary_rows, raw_file)

        pdf_path = raw_file.with_name(f"{raw_file.stem}_threshold_by_planet.pdf")
        plot_written = plot_threshold_heatmap(
            summary_rows,
            f"Threshold for below-exobase $\\beta = 1$ at {title_teff_label(summary_rows)}",
            pdf_path,
        )
        print(f"Saved summary CSV to {summary_csv_path}")
        if plot_written:
            print(f"Saved threshold plot to {pdf_path}")
        else:
            print(f"Skipped threshold plot for {raw_file}; no finite thresholds were found.")


if __name__ == "__main__":
    main()
