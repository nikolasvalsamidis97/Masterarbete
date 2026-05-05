import csv
import gc
import json
import pathlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import astropy.units as u
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.Molecule import Molecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem
from project_classes.Star import Star
from project_func.exobase_table_path import resolve_exobase_table_path
from project_func.Templates.Atoms.atom_species import ATOM_SPECIES
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Planets.planet_templates import PLANET_TEMPLATES, get_planet_template
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template


SELECTED_ATOMIC_SPECIES = None
SELECTED_MOLECULAR_SPECIES = None
SKIP_ATOMS = True
SKIP_MOLECULES = False
RUN_ALL_ABSORBERS_IF_UNSPECIFIED = True
START_FRESH_RUN = True
FRESH_RUN_LABEL = "fresh_run_molecules_only_1"
USE_COMPOSITION_MIXING_RATIOS = False


stellar_models = STAR_TEMPLATES

DEFAULT_PLANET_KEYS = list(PLANET_TEMPLATES.keys())
SELECTED_PLANET_SPECIES = {
    planet_key: None
    for planet_key in DEFAULT_PLANET_KEYS
    if planet_key in PLANET_TEMPLATES
}

DISTANCE_LIST = [0.1, 0.5, 1.0, 5.0, 10.0, 100.0] * u.AU
SELECTED_STARS = None
TARGET_TEFFS_K = [3000, 5000, 6000, 8000, 10000, 15000, 20000, 30000, 50000]

# Fast Rp-exobase mode: evaluate only at the exobase radius.
EVALUATION_MODE = "rexo_only"
ATOMIC_COLUMN_CHUNK_SIZE = 8
# Set to "serial" to disable parallelism completely.
# Set to "star" to compute different stellar templates in parallel.
PARALLEL_TASK_MODE = "serial"
STAR_MAX_WORKERS = 2
# If True, evaluate distances from smallest to largest and stop after the
# first fail for a given (planet, species, star), marking all larger
# distances as failed by monotonicity.
DISTANCE_PRUNING_ASSUME_MONOTONIC = True
PRINT_TRACEBACKS = False
SAVE_OUTPUT_TXT = True
USE_CHECKPOINT = True

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
b_atom = 1 * u.km / u.s
npts_atom = 150
b_molecule = 1 * u.km / u.s

star_cache = {}
profile_cache = {}

TEFF_STUDY_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "Plots"
    / "Atmospheric test"
    / "Teff_study"
)
OUTPUT_DIR = TEFF_STUDY_OUTPUT_DIR / "R_p_exo_beta1"
EXOBASE_TABLE = (
    resolve_exobase_table_path(pathlib.Path(__file__).resolve().parents[2])
)


def output_stem() -> str:
    if SKIP_ATOMS and not SKIP_MOLECULES:
        return "R_p_exo_beta1_molecules"
    if SKIP_MOLECULES and not SKIP_ATOMS:
        return "R_p_exo_beta1_atoms"
    return "R_p_exo_beta1_all"


RAW_OUTPUT_PATH = OUTPUT_DIR / f"{output_stem()}.txt"
FRESH_RUN_MARKER_PATH = OUTPUT_DIR / f"{output_stem()}_fresh_run.json"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / f"{output_stem()}_summary.csv"


def safe_name(value):
    return str(value).replace(" ", "").replace("/", "_")


def maybe_print_traceback():
    if PRINT_TRACEBACKS:
        traceback.print_exc()


def raw_fieldnames():
    return [
        "planet",
        "category",
        "species_type",
        "species",
        "star",
        "stellar_teff_K",
        "distance_AU",
        "z_exobase_km",
        "beta_at_Rexo",
        "beta1_hit_below_exobase",
        "first_hit_over_Rp_topdown",
        "points_evaluated",
        "result_source",
        "run_signature",
    ]


def summary_fieldnames():
    return [
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


def lookup_exobase_height_km(
    planet_key: str,
    species: str,
    exobase_heights: dict[tuple[str, str], float],
) -> float | None:
    z_exobase_km = exobase_heights.get((planet_key, species))
    if z_exobase_km is None:
        z_exobase_km = exobase_heights.get((planet_key, neutral_exobase_species(species)))
    return z_exobase_km


def current_run_signature(star_keys_sorted, distance_values_au):
    return json.dumps(
        {
            "distance_grid_au": [f"{float(value):.12g}" for value in distance_values_au],
            "star_keys_sorted": list(star_keys_sorted),
            "target_teffs_k": [float(value) for value in TARGET_TEFFS_K],
            "selected_atomic_species": SELECTED_ATOMIC_SPECIES,
            "selected_molecular_species": SELECTED_MOLECULAR_SPECIES,
            "skip_atoms": bool(SKIP_ATOMS),
            "skip_molecules": bool(SKIP_MOLECULES),
            "run_all_absorbers_if_unspecified": bool(RUN_ALL_ABSORBERS_IF_UNSPECIFIED),
            "use_composition_mixing_ratios": bool(USE_COMPOSITION_MIXING_RATIOS),
            "evaluation_mode": EVALUATION_MODE,
            "atomic_column_chunk_size": int(ATOMIC_COLUMN_CHUNK_SIZE),
            "parallel_task_mode": PARALLEL_TASK_MODE,
            "star_max_workers": int(STAR_MAX_WORKERS),
            "distance_pruning_assume_monotonic": bool(DISTANCE_PRUNING_ASSUME_MONOTONIC),
        },
        sort_keys=True,
    )


def validate_saved_rows(rows, expected_run_signature):
    if not rows:
        return

    for row in rows:
        if row.get("run_signature") != expected_run_signature:
            raise ValueError(
                "Existing Rp-exobase raw file was created with different run settings. "
                f"Delete {RAW_OUTPUT_PATH} or restore the old settings before resuming."
            )


def load_saved_rows(path, expected_run_signature):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    validate_saved_rows(rows, expected_run_signature)
    return rows


def save_rows_tsv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp_path.replace(path)


def save_rows_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp_path.replace(path)


def checkpoint_key(planet_key, species, star_key, distance_au):
    return planet_key, species, star_key, f"{float(distance_au):.12g}"


def completed_checkpoint_keys(rows):
    return {
        checkpoint_key(row["planet"], row["species"], row["star"], float(row["distance_AU"]))
        for row in rows
    }


def classify_species(species: str) -> str:
    if species in ATOM_SPECIES:
        return "atom"
    if species in MOLECULE_TEMPLATES:
        return "molecule"
    return "unknown"


def get_molecule_fetch(species):
    template = MOLECULE_TEMPLATES[species]
    return {
        "source": template.get("source", "exomol").lower(),
        "fetch_kwargs": template["fetch_kwargs"],
    }


def get_star(star_key):
    if star_key not in star_cache:
        s = stellar_models[star_key]
        star_cache[star_key] = Star(
            s["path"],
            s["radius"],
            s["mass"],
            vsini=s["vsini"],
            epsilon=s["epsilon"],
        )
    return star_cache[star_key]


def select_star_keys_by_target_teff(target_teffs_k):
    all_keys = list(stellar_models.keys())
    teff_map = {key: float(infer_teff_from_star_template(key)) for key in all_keys}

    selected_keys = []
    used_keys = set()
    for target_teff in target_teffs_k:
        remaining = [key for key in all_keys if key not in used_keys]
        if not remaining:
            break
        best_key = min(remaining, key=lambda key: abs(teff_map[key] - float(target_teff)))
        selected_keys.append(best_key)
        used_keys.add(best_key)

    return selected_keys


def get_profile(species):
    if species in profile_cache:
        return profile_cache[species]

    if species in MOLECULE_TEMPLATES:
        molecule_fetch = get_molecule_fetch(species)
        mol = Molecule(species, wavemin, wavemax)
        fetch_kwargs = molecule_fetch["fetch_kwargs"]
        if molecule_fetch["source"] == "hitran":
            mol.fetch_hitran(**fetch_kwargs)
        else:
            mol.fetch_exomol(
                path=fetch_kwargs["path"],
                database=fetch_kwargs["database"],
                localdatabase=fetch_kwargs.get("localdatabase", "exomol_data"),
            )
        print(f"[{species}] molecule loaded")
        print(f"Starting to build broadening profile for {species}...")
        profile = BroadeningProfileMolecule(mol, b_molecule, profileType="Voigt")
        print(f"[{species}] broadening profile built")
        if hasattr(profile, "temp_strength_rel_cutoff"):
            profile.temp_strength_rel_cutoff = 1e-8
    else:
        atom = Atom(species, wavemin, wavemax)
        profile = BroadeningProfile(atom, b_atom, npts_atom, "Voigt")

    profile_cache[species] = profile
    return profile


def is_atomic_species(species):
    return species in ATOM_SPECIES


def is_molecular_species(species):
    return species in MOLECULE_TEMPLATES


def species_matches_run_filters(species):
    if is_atomic_species(species):
        if SKIP_ATOMS:
            return False
        return SELECTED_ATOMIC_SPECIES is None or species in SELECTED_ATOMIC_SPECIES

    if is_molecular_species(species):
        if SKIP_MOLECULES:
            return False
        return SELECTED_MOLECULAR_SPECIES is None or species in SELECTED_MOLECULAR_SPECIES

    return False


def ordered_all_absorber_species():
    atomic_species = sorted(
        species for species in ATOM_SPECIES
        if species_matches_run_filters(species)
    )
    molecular_species = sorted(
        species for species in MOLECULE_TEMPLATES
        if species_matches_run_filters(species)
    )
    return atomic_species + molecular_species


def species_mixing_ratio(planet_case, species):
    if not USE_COMPOSITION_MIXING_RATIOS:
        return 1.0

    ratio = planet_case.get("composition", {}).get(species, 1.0)
    if isinstance(ratio, u.Quantity):
        ratio = ratio.to_value(u.dimensionless_unscaled)
    return float(ratio)


def get_total_slant_column(system_obj, z, ncol_cache=None, ncol_key=None):
    if ncol_key is None:
        ncol_key = float(z.to_value(u.cm))

    if ncol_cache is not None and ncol_key in ncol_cache:
        return ncol_cache[ncol_key]

    ncol_total = np.array([
        system_obj.planet.slant_column_density(z).to_value(1 / u.cm**2)
    ]) / u.cm**2
    if ncol_cache is not None:
        ncol_cache[ncol_key] = ncol_total
    return ncol_total


def build_species_column_grid(system_obj, z_grid, abundance, ncol_cache=None):
    z_cm_values = np.asarray([float(z.to_value(u.cm)) for z in z_grid], dtype=float)
    ncol_values = np.empty(len(z_grid), dtype=float)
    missing_indices = []

    for i, z_cm_value in enumerate(z_cm_values):
        if ncol_cache is not None and z_cm_value in ncol_cache:
            ncol_values[i] = float(
                np.squeeze((abundance * ncol_cache[z_cm_value]).to_value(1 / u.cm**2))
            )
        else:
            missing_indices.append(i)

    for i in missing_indices:
        z = z_grid[i]
        ncol_total = get_total_slant_column(system_obj, z, ncol_cache, z_cm_values[i])
        ncol_values[i] = float(
            np.squeeze((abundance * ncol_total).to_value(1 / u.cm**2))
        )

    return ncol_values / u.cm**2


def evaluate_beta_grid(pp, ncol_grid, temp_atm, distance, planet_mass, r_grid, chunk_size=1):
    F_ph_tot_grid, F_ph_tot_err_grid, _, _ = pp.calc_PhotonPressure(
        ncol_grid,
        temp_atm,
        distance,
        chunk_size=max(1, int(chunk_size)),
    )
    beta_species_grid, _ = pp.beta_Values(
        F_ph_tot_grid,
        F_ph_tot_err_grid,
        planet_mass,
        r_grid,
    )
    return np.asarray(beta_species_grid.value, dtype=float).reshape(-1)


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


def clear_species_runtime_caches(species):
    profile = profile_cache.pop(species, None)
    if profile is not None and hasattr(profile, "clear_temperature_cache"):
        profile.clear_temperature_cache()
    PhotonPressure.clear_molecule_flux_cache()
    gc.collect()


def should_parallelize_species(species: str) -> bool:
    if PARALLEL_TASK_MODE == "serial":
        return False
    if PARALLEL_TASK_MODE != "star":
        raise ValueError(
            f"Unknown PARALLEL_TASK_MODE={PARALLEL_TASK_MODE!r}. Use 'serial' or 'star'."
        )
    if is_molecular_species(species):
        return False
    return True


def fresh_run_marker_matches(run_signature):
    if not FRESH_RUN_MARKER_PATH.exists():
        return False

    try:
        marker_payload = json.loads(FRESH_RUN_MARKER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        marker_payload.get("run_signature") == run_signature
        and marker_payload.get("fresh_run_label") == FRESH_RUN_LABEL
    )


def write_fresh_run_marker(run_signature):
    FRESH_RUN_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRESH_RUN_MARKER_PATH.write_text(
        json.dumps(
            {
                "run_signature": run_signature,
                "fresh_run_label": FRESH_RUN_LABEL,
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def reset_run_outputs():
    if RAW_OUTPUT_PATH.exists():
        print(f"Removing old raw output file: {RAW_OUTPUT_PATH}")
        RAW_OUTPUT_PATH.unlink()

    if SUMMARY_OUTPUT_PATH.exists():
        SUMMARY_OUTPUT_PATH.unlink()

    if FRESH_RUN_MARKER_PATH.exists():
        FRESH_RUN_MARKER_PATH.unlink()


def find_beta1_from_exobase(
    system_obj,
    planet_case,
    planet_key,
    star_key,
    species,
    exobase_heights,
    ncol_cache=None,
):
    z_exobase_km = lookup_exobase_height_km(planet_key, species, exobase_heights)
    metrics = {
        "z_exobase_km": np.nan if z_exobase_km is None else float(z_exobase_km),
        "beta_at_Rexo": np.nan,
        "beta1_hit_below_exobase": np.nan,
        "first_hit_over_Rp_topdown": np.nan,
        "points_evaluated": 0,
    }
    if z_exobase_km is None or not np.isfinite(z_exobase_km) or z_exobase_km <= 0.0:
        return metrics

    try:
        profile = get_profile(species)
        pp = PhotonPressure(profile, system_obj.star)
    except Exception as exc:
        print(f"Skipping {species} for {star_key} before Rp-exobase search: {exc}")
        maybe_print_traceback()
        return metrics

    abundance = species_mixing_ratio(planet_case, species)
    planet_radius = system_obj.planet.radius.to(u.cm)
    z_exo = float(z_exobase_km) * 1.0e5 * u.cm
    r_exo = planet_radius + z_exo
    z_grid = u.Quantity([z_exo])
    r_grid = u.Quantity([r_exo])
    ncol_exo = build_species_column_grid(system_obj, z_grid, abundance, ncol_cache=ncol_cache)
    chunk_size = ATOMIC_COLUMN_CHUNK_SIZE if is_atomic_species(species) else 1

    try:
        beta_values = evaluate_beta_grid(
            pp,
            ncol_exo,
            planet_case["T"],
            system_obj.distance,
            system_obj.planet.mass,
            r_grid,
            chunk_size=chunk_size,
        )
    except Exception as exc:
        stage = "molecular" if is_molecular_species(species) else "atomic"
        print(f"Skipping {species} for {star_key} in Rp-exobase {stage} evaluation: {exc}")
        maybe_print_traceback()
        return metrics

    metrics["points_evaluated"] = 1
    if beta_values.size == 0 or not np.isfinite(beta_values[0]):
        return metrics

    beta_at_rexo = float(beta_values[0])
    metrics["beta_at_Rexo"] = beta_at_rexo
    if beta_at_rexo >= 1.0:
        metrics["beta1_hit_below_exobase"] = 1.0
        metrics["first_hit_over_Rp_topdown"] = float((r_exo / planet_radius).decompose().value)
    else:
        metrics["beta1_hit_below_exobase"] = 0.0
    return metrics


def make_inferred_fail_metrics(z_exobase_km):
    return {
        "z_exobase_km": np.nan if z_exobase_km is None else float(z_exobase_km),
        "beta_at_Rexo": np.nan,
        "beta1_hit_below_exobase": 0.0,
        "first_hit_over_Rp_topdown": np.nan,
        "points_evaluated": 0,
    }


def compute_star_rows_worker(args):
    (
        selected_planet,
        selected_species,
        star_key,
        distance_values_au,
        exobase_heights,
        run_signature,
    ) = args

    planet_case = get_planet_template(selected_planet)
    planet_obj = Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )
    ncol_cache = {}
    rows = []
    star = get_star(star_key)
    z_exobase_km = lookup_exobase_height_km(selected_planet, selected_species, exobase_heights)
    fail_encountered = False

    for dist_value_au in sorted(distance_values_au):
        if DISTANCE_PRUNING_ASSUME_MONOTONIC and fail_encountered:
            rows.append(
                make_raw_row(
                    selected_planet,
                    selected_species,
                    star_key,
                    dist_value_au,
                    make_inferred_fail_metrics(z_exobase_km),
                    run_signature,
                    result_source="inferred_fail_monotonic_distance",
                )
            )
            continue

        dist = dist_value_au * u.AU
        star = get_star(star_key)
        system_obj = PlanetarySystem(planet_obj, star, dist)
        metrics = find_beta1_from_exobase(
            system_obj,
            planet_case,
            selected_planet,
            star_key,
            selected_species,
            exobase_heights,
            ncol_cache=ncol_cache,
        )
        hit_value = metrics["beta1_hit_below_exobase"]
        rows.append(
            make_raw_row(
                selected_planet,
                selected_species,
                star_key,
                dist_value_au,
                metrics,
                run_signature,
            )
        )
        if DISTANCE_PRUNING_ASSUME_MONOTONIC and np.isfinite(hit_value) and hit_value < 0.5:
            fail_encountered = True

    return star_key, rows


def make_raw_row(
    planet_key,
    species,
    star_key,
    distance_au,
    metrics,
    run_signature,
    result_source="computed",
):
    return {
        "planet": planet_key,
        "category": PLANET_TEMPLATES.get(planet_key, {}).get("category", "unknown"),
        "species_type": classify_species(species),
        "species": species,
        "star": star_key,
        "stellar_teff_K": float(infer_teff_from_star_template(star_key)),
        "distance_AU": float(distance_au),
        "z_exobase_km": float(metrics["z_exobase_km"]) if np.isfinite(metrics["z_exobase_km"]) else np.nan,
        "beta_at_Rexo": float(metrics["beta_at_Rexo"]) if np.isfinite(metrics["beta_at_Rexo"]) else np.nan,
        "beta1_hit_below_exobase": (
            float(metrics["beta1_hit_below_exobase"])
            if np.isfinite(metrics["beta1_hit_below_exobase"])
            else np.nan
        ),
        "first_hit_over_Rp_topdown": (
            float(metrics["first_hit_over_Rp_topdown"])
            if np.isfinite(metrics["first_hit_over_Rp_topdown"])
            else np.nan
        ),
        "points_evaluated": int(metrics["points_evaluated"]),
        "result_source": result_source,
        "run_signature": run_signature,
    }


def build_summary_rows(all_rows, star_keys_sorted, distance_values_au):
    teff_reference = np.asarray(
        [infer_teff_from_star_template(star_key) for star_key in star_keys_sorted],
        dtype=float,
    )
    distance_reference = np.asarray(distance_values_au, dtype=float)
    star_index = {star_key: i for i, star_key in enumerate(star_keys_sorted)}
    distance_index = {f"{float(value):.12g}": j for j, value in enumerate(distance_values_au)}

    grouped_rows = {}
    for row in all_rows:
        key = (row["planet"], row["species"])
        grouped_rows.setdefault(key, []).append(row)

    summary_rows = []
    for (planet_key, species), rows in sorted(grouped_rows.items()):
        hit_matrix = np.full((len(teff_reference), len(distance_reference)), np.nan, dtype=float)
        z_exobase_values = []
        for row in rows:
            star_key = row["star"]
            distance_key = f"{float(row['distance_AU']):.12g}"
            if star_key not in star_index or distance_key not in distance_index:
                continue
            i = star_index[star_key]
            j = distance_index[distance_key]
            hit_matrix[i, j] = float(row["beta1_hit_below_exobase"])
            try:
                z_exobase_values.append(float(row["z_exobase_km"]))
            except (TypeError, ValueError):
                continue

        finite_mask = np.isfinite(hit_matrix)
        hit_mask = finite_mask & (hit_matrix >= 0.5)
        n_total = int(hit_matrix.size)
        n_finite = int(np.count_nonzero(finite_mask))
        n_hits = int(np.count_nonzero(hit_mask))

        hit_teff_candidates = teff_reference[np.any(hit_mask, axis=1)]
        hit_distance_candidates = distance_reference[np.any(hit_mask, axis=0)]
        threshold_distance_au, threshold_teff_k = hardest_guaranteed_threshold(
            hit_mask,
            teff_reference,
            distance_reference,
        )

        summary_rows.append(
            {
                "species_type": classify_species(species),
                "planet": planet_key,
                "category": PLANET_TEMPLATES.get(planet_key, {}).get("category", "unknown"),
                "species": species,
                "z_exobase_km": (
                    float(np.nanmax(np.asarray(z_exobase_values, dtype=float)))
                    if z_exobase_values
                    else np.nan
                ),
                "n_total": n_total,
                "n_finite": n_finite,
                "finite_fraction": float(n_finite / n_total) if n_total else np.nan,
                "n_hits": n_hits,
                "hit_fraction": float(n_hits / n_total) if n_total else np.nan,
                "coolest_teff_with_hit_K": (
                    float(np.min(hit_teff_candidates)) if hit_teff_candidates.size else np.nan
                ),
                "max_distance_with_hit_AU": (
                    float(np.max(hit_distance_candidates)) if hit_distance_candidates.size else np.nan
                ),
                "threshold_teff_K": threshold_teff_k,
                "threshold_distance_AU": threshold_distance_au,
            }
        )

    return summary_rows


def persist_progress(all_rows, star_keys_sorted, distance_values_au):
    if USE_CHECKPOINT or SAVE_OUTPUT_TXT:
        save_rows_tsv(all_rows, RAW_OUTPUT_PATH, raw_fieldnames())
    summary_rows = build_summary_rows(all_rows, star_keys_sorted, distance_values_au)
    save_rows_csv(summary_rows, SUMMARY_OUTPUT_PATH, summary_fieldnames())


def main():
    exobase_heights = load_exobase_heights(EXOBASE_TABLE)

    if SELECTED_STARS is None:
        star_keys_sorted = select_star_keys_by_target_teff(TARGET_TEFFS_K)
    else:
        invalid_star_keys = [star_key for star_key in SELECTED_STARS if star_key not in stellar_models]
        if invalid_star_keys:
            raise ValueError(
                f"Unknown stars in SELECTED_STARS: {invalid_star_keys}. "
                f"Available stars: {list(stellar_models.keys())}"
            )
        star_keys_sorted = sorted(SELECTED_STARS, key=infer_teff_from_star_template)

    distance_values_au = [dist.to_value(u.AU) for dist in DISTANCE_LIST]
    run_signature = current_run_signature(star_keys_sorted, distance_values_au)

    if START_FRESH_RUN and not fresh_run_marker_matches(run_signature):
        reset_run_outputs()
        write_fresh_run_marker(run_signature)
        all_rows = []
        completed_keys = set()
    elif USE_CHECKPOINT:
        all_rows = load_saved_rows(RAW_OUTPUT_PATH, run_signature)
        completed_keys = completed_checkpoint_keys(all_rows)
        if all_rows:
            print(f"Loaded {len(all_rows)} saved Rp-exobase rows from {RAW_OUTPUT_PATH}")
    else:
        all_rows = []
        completed_keys = set()

    row_lookup = {
        checkpoint_key(row["planet"], row["species"], row["star"], float(row["distance_AU"])): row
        for row in all_rows
    }

    for selected_planet, requested_species in SELECTED_PLANET_SPECIES.items():
        planet_case = get_planet_template(selected_planet)
        planet_obj = Planet(
            radius=planet_case["radius"],
            mass=planet_case["mass"],
            T=planet_case["T"],
            mu=planet_case["mu"],
            P0=planet_case["P0"],
        )
        ncol_cache = {}

        if requested_species is None:
            if RUN_ALL_ABSORBERS_IF_UNSPECIFIED:
                requested_species = ordered_all_absorber_species()
                print(
                    f"No species specified for {selected_planet}; using all filtered absorbers: {requested_species}"
                )
            else:
                composition_species = list(planet_case["composition"].keys())
                requested_species = [
                    species
                    for species in composition_species
                    if species_matches_run_filters(species) and species not in {"O2", "OH"}
                ]
                print(
                    f"No species specified for {selected_planet}; using filtered planet composition species: {requested_species}"
                )
        else:
            requested_species = [
                species for species in requested_species
                if species_matches_run_filters(species)
            ]
            print(
                f"Using explicitly requested filtered species for {selected_planet}: {requested_species}"
            )

        if not requested_species:
            raise ValueError(
                f"No species selected for {selected_planet}. "
                f"Check SELECTED_ATOMIC_SPECIES={SELECTED_ATOMIC_SPECIES}, "
                f"SELECTED_MOLECULAR_SPECIES={SELECTED_MOLECULAR_SPECIES}, "
                f"SKIP_ATOMS={SKIP_ATOMS}, and SKIP_MOLECULES={SKIP_MOLECULES}."
            )

        teff_labels = [f"{infer_teff_from_star_template(star_key):.0f}" for star_key in star_keys_sorted]

        for selected_species in requested_species:
            species_start_time = time.perf_counter()
            total_systems = len(distance_values_au) * len(star_keys_sorted)
            existing_rows = [
                row
                for row in all_rows
                if row.get("planet") == selected_planet and row.get("species") == selected_species
            ]
            if existing_rows:
                print(
                    f"Resuming {selected_planet} / {selected_species}: "
                    f"{len(existing_rows)}/{total_systems} systems already saved"
                )

            use_parallel = should_parallelize_species(selected_species)
            if PARALLEL_TASK_MODE == "star" and is_molecular_species(selected_species):
                print(f"{selected_species} is molecular; forcing serial mode for safety.")

            if use_parallel:
                tasks = []
                for star_key in star_keys_sorted:
                    has_missing_distance = any(
                        checkpoint_key(selected_planet, selected_species, star_key, dist_value_au)
                        not in completed_keys
                        for dist_value_au in distance_values_au
                    )
                    if not has_missing_distance:
                        continue
                    print(
                        f"species={selected_species}, "
                        f"temp_atm={planet_case['T'].to_value(u.K):.0f} K, "
                        f"star={star_key}, "
                        f"Teff={infer_teff_from_star_template(star_key):.0f} K, "
                        f"distances={[float(value) for value in distance_values_au]} AU"
                    )
                    tasks.append(
                        (
                            selected_planet,
                            selected_species,
                            star_key,
                            distance_values_au,
                            exobase_heights,
                            run_signature,
                        )
                    )

                if tasks:
                    with ProcessPoolExecutor(max_workers=min(STAR_MAX_WORKERS, len(tasks))) as executor:
                        futures = [executor.submit(compute_star_rows_worker, task) for task in tasks]
                        for future in as_completed(futures):
                            _star_key, new_rows = future.result()
                            filtered_rows = []
                            for row in new_rows:
                                row_key = checkpoint_key(
                                    row["planet"],
                                    row["species"],
                                    row["star"],
                                    float(row["distance_AU"]),
                                )
                                if row_key in completed_keys:
                                    continue
                                filtered_rows.append(row)
                                completed_keys.add(row_key)
                                row_lookup[row_key] = row

                            if filtered_rows:
                                all_rows.extend(filtered_rows)
                                persist_progress(all_rows, star_keys_sorted, distance_values_au)
            else:
                z_exobase_km = lookup_exobase_height_km(selected_planet, selected_species, exobase_heights)
                ordered_distances = sorted(distance_values_au)
                for star_key in star_keys_sorted:
                    print(
                        f"species={selected_species}, "
                        f"temp_atm={planet_case['T'].to_value(u.K):.0f} K, "
                        f"star={star_key}, "
                        f"Teff={infer_teff_from_star_template(star_key):.0f} K, "
                        f"distances={ordered_distances} AU"
                    )
                    star = get_star(star_key)
                    fail_encountered = False
                    new_rows = []
                    for dist_value_au in ordered_distances:
                        row_key = checkpoint_key(selected_planet, selected_species, star_key, dist_value_au)
                        if row_key in completed_keys:
                            existing_row = row_lookup.get(row_key)
                            if (
                                DISTANCE_PRUNING_ASSUME_MONOTONIC
                                and existing_row is not None
                                and str(existing_row.get("beta1_hit_below_exobase", "")).strip() == "0.0"
                            ):
                                fail_encountered = True
                            continue

                        if DISTANCE_PRUNING_ASSUME_MONOTONIC and fail_encountered:
                            metrics = make_inferred_fail_metrics(z_exobase_km)
                            row = make_raw_row(
                                selected_planet,
                                selected_species,
                                star_key,
                                dist_value_au,
                                metrics,
                                run_signature,
                                result_source="inferred_fail_monotonic_distance",
                            )
                        else:
                            dist = dist_value_au * u.AU
                            system_obj = PlanetarySystem(planet_obj, star, dist)
                            metrics = find_beta1_from_exobase(
                                system_obj,
                                planet_case,
                                selected_planet,
                                star_key,
                                selected_species,
                                exobase_heights,
                                ncol_cache=ncol_cache,
                            )
                            row = make_raw_row(
                                selected_planet,
                                selected_species,
                                star_key,
                                dist_value_au,
                                metrics,
                                run_signature,
                            )
                            hit_value = metrics["beta1_hit_below_exobase"]
                            if DISTANCE_PRUNING_ASSUME_MONOTONIC and np.isfinite(hit_value) and hit_value < 0.5:
                                fail_encountered = True

                        new_rows.append(
                            row
                        )
                        completed_keys.add(row_key)
                        row_lookup[row_key] = row

                    if new_rows:
                        all_rows.extend(new_rows)
                        persist_progress(all_rows, star_keys_sorted, distance_values_au)

            species_elapsed_s = time.perf_counter() - species_start_time
            print(f"Used species: {selected_species}")
            print(f"Total time for {selected_species}: {species_elapsed_s:.2f} s")
            clear_species_runtime_caches(selected_species)

    persist_progress(all_rows, star_keys_sorted, distance_values_au)
    print(f"Saved raw data to {RAW_OUTPUT_PATH}")
    print(f"Saved summary data to {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    freeze_support()
    main()
