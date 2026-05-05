import csv
import gc
import json
import sys
import pathlib
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import numpy as np
import astropy.units as u

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem
from project_classes.Star import Star
from project_func.Templates.Planets.planet_templates import PLANET_TEMPLATES, get_planet_template
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Atoms.atom_species import ATOM_SPECIES
from project_func.plotdata_to_txt import save_plotdata_txt
import traceback

# Optional explicit species filters when a planet entry is set to None.
# Use exact species names, for example:
# SELECTED_ATOMIC_SPECIES = ["Na I", "Na II", "Fe I", "Fe III"]
# SELECTED_MOLECULAR_SPECIES = ["H2O", "CO", "CH4"]
SELECTED_ATOMIC_SPECIES = None
SELECTED_MOLECULAR_SPECIES = None
SKIP_ATOMS = False
SKIP_MOLECULES = True
RUN_ALL_ABSORBERS_IF_UNSPECIFIED = True
START_FRESH_RUN = True
FRESH_RUN_LABEL = "fresh_run_atoms_only"
USE_COMPOSITION_MIXING_RATIOS = False


# Use global templates for stellar models
stellar_models = STAR_TEMPLATES

# -----------------------------------------------------------------------------
# Fixed setup for the first Teff study
# -----------------------------------------------------------------------------
# All planets
DEFAULT_PLANET_KEYS = list(PLANET_TEMPLATES.keys())
# For quick testing, you can set this to a subset of planets, for example:
# DEFAULT_PLANET_KEYS = [
#     "hot_jupiter",
# ]
SELECTED_PLANET_SPECIES = {
    planet_key: None
    for planet_key in DEFAULT_PLANET_KEYS
    if planet_key in PLANET_TEMPLATES
}

DISTANCE_LIST = [0.1, 0.5, 1.0, 5.0, 10.0, 100.0] * u.AU
# DISTANCE_LIST = [0.05, 0.1, 1, 10, 100] * u.AU
SELECTED_STARS = None
# SELECTED_STARS = ["O0", "B0", "A0"]
TARGET_TEFFS_K = [3000, 5000, 6000, 8000, 10000, 15000, 20000, 30000, 50000]
STAR_STRIDE = 5
DISTANCE_MAX_WORKERS = 1
STAR_MAX_WORKERS = 1
OUTPUT_ROOT_NAME = None
# Choose outer parallelization strategy:
# "distance"      -> one worker per distance (reuses stars serially inside each worker)
# "distance_star" -> one worker per (distance, star) pair (more workers, less reuse)
# "serial"        -> no outer parallelism
PARALLEL_TASK_MODE = "serial"
N_HEIGHT_POINTS = 150
COARSE_HEIGHT_POINTS = 30
REFINE_HEIGHT_POINTS = 30
### For faster testing ###
# COARSE_HEIGHT_POINTS = 20
# REFINE_HEIGHT_POINTS = 15
##########################
COARSE_GRID_POWER = 3.0
PRINT_TRACEBACKS = False
SAVE_OUTPUT_TXT = True
USE_CHECKPOINT = True
ATOMIC_COLUMN_CHUNK_SIZE = 8

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


def default_output_root_name() -> str:
    if SKIP_ATOMS and not SKIP_MOLECULES:
        return "r_at_beta1_molecules"
    if SKIP_MOLECULES and not SKIP_ATOMS:
        return "r_at_beta1_atoms"
    return "r_at_beta1"


ROOT_OUTPUT_DIR = TEFF_STUDY_OUTPUT_DIR / (OUTPUT_ROOT_NAME or default_output_root_name())
CHECKPOINT_PATH = ROOT_OUTPUT_DIR / "teff_planet_beta1_dist_checkpoint.csv"
FRESH_RUN_MARKER_PATH = ROOT_OUTPUT_DIR / "teff_planet_beta1_dist_fresh_run.json"





# Helper to sanitize names for saving
def safe_name(value):
    return str(value).replace(" ", "").replace("/", "_")


def maybe_print_traceback():
    if PRINT_TRACEBACKS:
        traceback.print_exc()


def checkpoint_fieldnames():
    return [
        "planet",
        "species",
        "star",
        "stellar_teff_K",
        "distance_AU",
        "r_beta1_over_Rp",
        "coarse_height_points",
        "refine_height_points",
        "coarse_grid_power",
        "run_signature",
    ]


def current_run_signature(star_keys_sorted, distance_values_au):
    return json.dumps(
        {
            "coarse_height_points": int(COARSE_HEIGHT_POINTS),
            "refine_height_points": int(REFINE_HEIGHT_POINTS),
            "coarse_grid_power": float(COARSE_GRID_POWER),
            "distance_grid_au": [f"{float(value):.12g}" for value in distance_values_au],
            "star_keys_sorted": list(star_keys_sorted),
            "selected_atomic_species": SELECTED_ATOMIC_SPECIES,
            "selected_molecular_species": SELECTED_MOLECULAR_SPECIES,
            "skip_atoms": bool(SKIP_ATOMS),
            "skip_molecules": bool(SKIP_MOLECULES),
            "run_all_absorbers_if_unspecified": bool(RUN_ALL_ABSORBERS_IF_UNSPECIFIED),
            "use_composition_mixing_ratios": bool(USE_COMPOSITION_MIXING_RATIOS),
        },
        sort_keys=True,
    )


def validate_checkpoint_rows(rows, expected_run_signature):
    if not rows:
        return

    for row in rows:
        if row.get("run_signature") != expected_run_signature:
            raise ValueError(
                "Existing Teff checkpoint was created with different run settings. "
                f"Delete {CHECKPOINT_PATH} or restore the old settings before resuming."
            )


def load_checkpoint_rows(path, expected_run_signature):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    validate_checkpoint_rows(rows, expected_run_signature)
    return rows


def save_checkpoint_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=checkpoint_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in checkpoint_fieldnames()})
    tmp_path.replace(path)


def checkpoint_key(planet_key, species, star_key, distance_au):
    return planet_key, species, star_key, f"{float(distance_au):.12g}"


def completed_checkpoint_keys(rows):
    return {
        checkpoint_key(row["planet"], row["species"], row["star"], float(row["distance_AU"]))
        for row in rows
    }


def make_checkpoint_row(planet_key, species, star_key, distance_au, rbeta_value, run_signature):
    return {
        "planet": planet_key,
        "species": species,
        "star": star_key,
        "stellar_teff_K": infer_teff_from_star_template(star_key),
        "distance_AU": float(distance_au),
        "r_beta1_over_Rp": float(rbeta_value) if np.isfinite(rbeta_value) else np.nan,
        "coarse_height_points": COARSE_HEIGHT_POINTS,
        "refine_height_points": REFINE_HEIGHT_POINTS,
        "coarse_grid_power": COARSE_GRID_POWER,
        "run_signature": run_signature,
    }


def save_species_table_from_rows(
    rows,
    selected_planet,
    selected_species,
    planet_case,
    star_keys_sorted,
    distance_values_au,
):
    matching_rows = [
        row
        for row in rows
        if row.get("planet") == selected_planet and row.get("species") == selected_species
    ]
    if not matching_rows:
        return None

    planet_save_name = safe_name(selected_planet)
    species_save_name = safe_name(selected_species)
    output_dir = ROOT_OUTPUT_DIR / f"{planet_save_name}_r_beta1"
    output_dir.mkdir(parents=True, exist_ok=True)

    teff_reference = np.asarray(
        [infer_teff_from_star_template(star_key) for star_key in star_keys_sorted],
        dtype=float,
    )
    star_index = {star_key: i for i, star_key in enumerate(star_keys_sorted)}
    distance_index = {
        f"{float(distance_au):.12g}": j
        for j, distance_au in enumerate(distance_values_au)
    }
    rbeta_matrix = np.full((len(teff_reference), len(distance_values_au)), np.nan, dtype=float)

    for row in matching_rows:
        star_key = row["star"]
        distance_key = f"{float(row['distance_AU']):.12g}"
        if star_key not in star_index or distance_key not in distance_index:
            continue
        i = star_index[star_key]
        j = distance_index[distance_key]
        rbeta_matrix[i, j] = float(row["r_beta1_over_Rp"])

    table_path = output_dir / f"{species_save_name}_r_beta1.txt"
    if SAVE_OUTPUT_TXT:
        planet_mass_metadata = {}
        if planet_case["mass"] < 0.1 * u.M_jup:
            planet_mass_metadata["planet_mass_Mearth"] = planet_case["mass"].to_value(u.M_earth)
        else:
            planet_mass_metadata["planet_mass_Mjup"] = planet_case["mass"].to_value(u.M_jup)

        save_plotdata_txt(
            table_path,
            dataset_name=f"{species_save_name}_r_beta1",
            x_label="Stellar Teff",
            x_unit="K",
            y_label="r_beta1 / R_p",
            y_unit="dimensionless",
            x_values=teff_reference,
            y_matrix=rbeta_matrix,
            series_values=list(distance_values_au),
            series_label="distance",
            series_unit="AU",
            extra_metadata={
                "planet": selected_planet,
                "species": selected_species,
                "planet_radius_Rjup": planet_case["radius"].to_value(u.R_jup),
                "planet_temperature_K": planet_case["T"].to_value(u.K),
                "planet_mu": float(planet_case["mu"]),
                "species_mixing_ratio": species_mixing_ratio(planet_case, selected_species),
                "stellar_teff_grid_K": teff_reference.tolist(),
                **planet_mass_metadata,
            },
        )
    return table_path


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

    return z_cm_values, ncol_values / u.cm**2


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


def fallback_lowest_height_success_ratio(beta_values: np.ndarray, r_grid, planet_radius) -> float:
    finite_indices = np.flatnonzero(np.isfinite(beta_values))
    if finite_indices.size == 0:
        return np.nan

    first_index = int(finite_indices[0])
    if beta_values[first_index] >= 1.0:
        return float((r_grid[first_index] / planet_radius).decompose().value)
    return np.nan


def persist_species_progress(
    all_results,
    selected_planet,
    selected_species,
    planet_case,
    star_keys_sorted,
    distance_values_au,
):
    if USE_CHECKPOINT:
        save_checkpoint_rows(all_results, CHECKPOINT_PATH)
    save_species_table_from_rows(
        all_results,
        selected_planet,
        selected_species,
        planet_case,
        star_keys_sorted,
        distance_values_au,
    )


def clear_species_runtime_caches(species):
    profile = profile_cache.pop(species, None)
    if profile is not None and hasattr(profile, "clear_temperature_cache"):
        profile.clear_temperature_cache()
    PhotonPressure.clear_molecule_flux_cache()
    gc.collect()


def reset_selected_planet_outputs():
    if CHECKPOINT_PATH.exists():
        print(f"Removing old checkpoint: {CHECKPOINT_PATH}")
        CHECKPOINT_PATH.unlink()

    if FRESH_RUN_MARKER_PATH.exists():
        FRESH_RUN_MARKER_PATH.unlink()

    for planet_key in SELECTED_PLANET_SPECIES:
        output_dir = ROOT_OUTPUT_DIR / f"{safe_name(planet_key)}_r_beta1"
        if output_dir.exists():
            print(f"Removing old output directory: {output_dir}")
            shutil.rmtree(output_dir)


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


def r_beta1_over_R(
    system_obj,
    planet_case,
    star_key,
    species,
    geometry_cache=None,
    ncol_cache=None,
    coarse_points=None,
    refine_points=None,
):
    """
    Build the beta profile for one selected species as a function of
    atmospheric height z using the local slant column density and local
    planetary gravity, then find the first radius where beta(z) = 1.
    """
    if coarse_points is None:
        coarse_points = COARSE_HEIGHT_POINTS
    if refine_points is None:
        refine_points = REFINE_HEIGHT_POINTS
    geometry_cache_key = (
        star_key,
        float(system_obj.distance.to_value(u.AU)),
    )

    if geometry_cache is not None and geometry_cache_key in geometry_cache:
        hill_radius, planet_radius, z_grid = geometry_cache[geometry_cache_key]
    else:
        hill_radius = system_obj.hill_radius().to(u.cm)
        planet_radius = system_obj.planet.radius.to(u.cm)

        if hill_radius <= planet_radius:
            return np.nan

        z_max_cm = (hill_radius - planet_radius).to_value(u.cm)
        coarse_fraction = np.linspace(0.0, 1.0, coarse_points) ** COARSE_GRID_POWER
        z_grid = (coarse_fraction * z_max_cm) * u.cm

        if geometry_cache is not None:
            geometry_cache[geometry_cache_key] = (hill_radius, planet_radius, z_grid)

    try:
        profile = get_profile(species)
        pp = PhotonPressure(profile, system_obj.star)
    except Exception as exc:
        print(f"Skipping {species} for {star_key} before height loop: {exc}")
        maybe_print_traceback()
        return np.nan

    abundance = species_mixing_ratio(planet_case, species)

    if is_molecular_species(species):
        _, ncol_grid = build_species_column_grid(system_obj, z_grid, abundance, ncol_cache)
        r_grid = planet_radius + z_grid

        try:
            beta_values_grid = evaluate_beta_grid(
                pp,
                ncol_grid,
                planet_case["T"],
                system_obj.distance,
                system_obj.planet.mass,
                r_grid,
                chunk_size=1,
            )
        except Exception as exc:
            print(f"Skipping {species} for {star_key} in coarse molecular grid: {exc}")
            maybe_print_traceback()
            return np.nan

        exact_matches = np.where(np.isclose(beta_values_grid, 1.0, atol=1e-6))[0]
        if len(exact_matches) > 0:
            idx = int(exact_matches[0])
            return float((r_grid[idx] / planet_radius).decompose().value)

        for i in range(1, len(beta_values_grid)):
            b1 = beta_values_grid[i - 1]
            b2 = beta_values_grid[i]
            if not np.isfinite(b1) or not np.isfinite(b2):
                continue

            crossed = ((b1 < 1.0 and b2 > 1.0) or (b1 > 1.0 and b2 < 1.0))
            if not crossed:
                continue

            z1 = z_grid[i - 1]
            z2 = z_grid[i]
            r1 = (planet_radius + z1).to_value(u.cm)
            r2 = (planet_radius + z2).to_value(u.cm)

            if np.isclose(b2, b1):
                r_beta1_cm = r1
            else:
                r_beta1_cm = r1 + (1.0 - b1) * (r2 - r1) / (b2 - b1)

            z_refine_grid = np.linspace(z1.to_value(u.cm), z2.to_value(u.cm), refine_points) * u.cm
            _, ncol_refine_grid = build_species_column_grid(
                system_obj,
                z_refine_grid,
                abundance,
                ncol_cache,
            )
            r_refine_grid = planet_radius + z_refine_grid

            try:
                beta_values_refine = evaluate_beta_grid(
                    pp,
                    ncol_refine_grid,
                    planet_case["T"],
                    system_obj.distance,
                    system_obj.planet.mass,
                    r_refine_grid,
                    chunk_size=1,
                )
            except Exception as exc:
                print(f"Skipping {species} for {star_key} in refined molecular grid: {exc}")
                maybe_print_traceback()
                return np.nan

            exact_matches_refine = np.where(np.isclose(beta_values_refine, 1.0, atol=1e-6))[0]
            if len(exact_matches_refine) > 0:
                j = int(exact_matches_refine[0])
                return float((r_refine_grid[j] / planet_radius).decompose().value)

            for j in range(1, len(beta_values_refine)):
                br1 = beta_values_refine[j - 1]
                br2 = beta_values_refine[j]
                if not np.isfinite(br1) or not np.isfinite(br2):
                    continue

                crossed_refine = ((br1 < 1.0 and br2 > 1.0) or (br1 > 1.0 and br2 < 1.0))
                if crossed_refine:
                    rr1 = r_refine_grid[j - 1].to_value(u.cm)
                    rr2 = r_refine_grid[j].to_value(u.cm)
                    if np.isclose(br2, br1):
                        r_beta1_refine_cm = rr1
                    else:
                        r_beta1_refine_cm = rr1 + (1.0 - br1) * (rr2 - rr1) / (br2 - br1)
                    return float(((r_beta1_refine_cm * u.cm) / planet_radius).decompose().value)

            return float(((r_beta1_cm * u.cm) / planet_radius).decompose().value)

        fallback_ratio = fallback_lowest_height_success_ratio(beta_values_grid, r_grid, planet_radius)
        if np.isfinite(fallback_ratio):
            return fallback_ratio
        return np.nan

    _, ncol_grid = build_species_column_grid(system_obj, z_grid, abundance, ncol_cache)
    r_grid = planet_radius + z_grid
    atomic_chunk_size = min(ATOMIC_COLUMN_CHUNK_SIZE, max(1, coarse_points))

    try:
        beta_values_grid = evaluate_beta_grid(
            pp,
            ncol_grid,
            planet_case["T"],
            system_obj.distance,
            system_obj.planet.mass,
            r_grid,
            chunk_size=atomic_chunk_size,
        )
    except Exception as exc:
        print(f"Skipping {species} for {star_key} in coarse atomic grid: {exc}")
        maybe_print_traceback()
        return np.nan

    exact_matches = np.where(np.isclose(beta_values_grid, 1.0, atol=1e-6))[0]
    if len(exact_matches) > 0:
        idx = int(exact_matches[0])
        return float((r_grid[idx] / planet_radius).decompose().value)

    for i in range(1, len(beta_values_grid)):
        b1 = beta_values_grid[i - 1]
        b2 = beta_values_grid[i]
        if not np.isfinite(b1) or not np.isfinite(b2):
            continue

        crossed = ((b1 < 1.0 and b2 > 1.0) or (b1 > 1.0 and b2 < 1.0))
        if not crossed:
            continue

        z1 = z_grid[i - 1]
        z2 = z_grid[i]
        r1 = (planet_radius + z1).to_value(u.cm)
        r2 = (planet_radius + z2).to_value(u.cm)

        if np.isclose(b2, b1):
            r_beta1_cm = r1
        else:
            r_beta1_cm = r1 + (1.0 - b1) * (r2 - r1) / (b2 - b1)

        z_refine_grid = np.linspace(z1.to_value(u.cm), z2.to_value(u.cm), refine_points) * u.cm
        _, ncol_refine_grid = build_species_column_grid(
            system_obj,
            z_refine_grid,
            abundance,
            ncol_cache,
        )
        r_refine_grid = planet_radius + z_refine_grid
        refine_chunk_size = min(ATOMIC_COLUMN_CHUNK_SIZE, max(1, refine_points))

        try:
            beta_values_refine = evaluate_beta_grid(
                pp,
                ncol_refine_grid,
                planet_case["T"],
                system_obj.distance,
                system_obj.planet.mass,
                r_refine_grid,
                chunk_size=refine_chunk_size,
            )
        except Exception as exc:
            print(f"Skipping {species} for {star_key} in refined atomic grid: {exc}")
            maybe_print_traceback()
            return np.nan

        exact_matches_refine = np.where(np.isclose(beta_values_refine, 1.0, atol=1e-6))[0]
        if len(exact_matches_refine) > 0:
            j = int(exact_matches_refine[0])
            return float((r_refine_grid[j] / planet_radius).decompose().value)

        for j in range(1, len(beta_values_refine)):
            br1 = beta_values_refine[j - 1]
            br2 = beta_values_refine[j]
            if not np.isfinite(br1) or not np.isfinite(br2):
                continue

            crossed_refine = ((br1 < 1.0 and br2 > 1.0) or (br1 > 1.0 and br2 < 1.0))
            if crossed_refine:
                rr1 = r_refine_grid[j - 1].to_value(u.cm)
                rr2 = r_refine_grid[j].to_value(u.cm)
                if np.isclose(br2, br1):
                    r_beta1_refine_cm = rr1
                else:
                    r_beta1_refine_cm = rr1 + (1.0 - br1) * (rr2 - rr1) / (br2 - br1)
                return float(((r_beta1_refine_cm * u.cm) / planet_radius).decompose().value)

        return float(((r_beta1_cm * u.cm) / planet_radius).decompose().value)

    fallback_ratio = fallback_lowest_height_success_ratio(beta_values_grid, r_grid, planet_radius)
    if np.isfinite(fallback_ratio):
        return fallback_ratio
    return np.nan







# New worker function for parallel distance column computation
def compute_distance_column_worker(args):
    selected_planet, selected_species, planet_case, planet_obj, dist_value_au, star_keys_sorted = args

    dist = dist_value_au * u.AU
    geometry_cache = {}
    ncol_cache = {}
    results = []

    for star_key in star_keys_sorted:
        teff = infer_teff_from_star_template(star_key)
        star = get_star(star_key)
        system_obj = PlanetarySystem(planet_obj, star, dist)
        value = r_beta1_over_R(
            system_obj,
            planet_case,
            star_key,
            selected_species,
            geometry_cache=geometry_cache,
            ncol_cache=ncol_cache,
        )
        results.append((star_key, teff, value))

    results.sort(key=lambda item: item[1])
    return dist_value_au, results


# New worker function for parallel distance-star computation
def compute_distance_star_worker(args):
    selected_planet, selected_species, planet_case, planet_obj, dist_value_au, star_key = args

    dist = dist_value_au * u.AU
    star = get_star(star_key)
    system_obj = PlanetarySystem(planet_obj, star, dist)
    teff = infer_teff_from_star_template(star_key)
    value = r_beta1_over_R(
        system_obj,
        planet_case,
        star_key,
        selected_species,
    )
    return dist_value_au, star_key, teff, value










def main():
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
        reset_selected_planet_outputs()
        write_fresh_run_marker(run_signature)
        all_results = []
        completed_keys = set()
    elif USE_CHECKPOINT:
        all_results = load_checkpoint_rows(CHECKPOINT_PATH, run_signature)
        completed_keys = completed_checkpoint_keys(all_results)
        if all_results:
            print(f"Loaded {len(all_results)} saved Teff beta1 checkpoint rows from {CHECKPOINT_PATH}")
    else:
        all_results = []
        completed_keys = set()

    for selected_planet, requested_species in SELECTED_PLANET_SPECIES.items():
        planet_case = get_planet_template(selected_planet)
        planet_obj = Planet(
            radius=planet_case["radius"],
            mass=planet_case["mass"],
            T=planet_case["T"],
            mu=planet_case["mu"],
            P0=planet_case["P0"],
        )
        planet_save_name = safe_name(selected_planet)
        geometry_cache = {}
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

        if PARALLEL_TASK_MODE not in {"distance", "distance_star", "serial"}:
            raise ValueError(
                f"Unknown PARALLEL_TASK_MODE={PARALLEL_TASK_MODE!r}. "
                f"Use 'distance', 'distance_star', or 'serial'."
            )
        use_parallel = PARALLEL_TASK_MODE != "serial"

        for selected_species in requested_species:
            species_start_time = time.perf_counter()
            species_save_name = safe_name(selected_species)
            output_dir = ROOT_OUTPUT_DIR / f"{planet_save_name}_r_beta1"
            output_dir.mkdir(parents=True, exist_ok=True)

            teff_labels = [f"{infer_teff_from_star_template(star_key):.0f}" for star_key in star_keys_sorted]
            total_systems = len(distance_values_au) * len(star_keys_sorted)
            existing_species_rows = [
                row
                for row in all_results
                if row.get("planet") == selected_planet and row.get("species") == selected_species
            ]

            if existing_species_rows:
                print(
                    f"Resuming {selected_planet} / {selected_species}: "
                    f"{len(existing_species_rows)}/{total_systems} systems already saved"
                )
                save_species_table_from_rows(
                    all_results,
                    selected_planet,
                    selected_species,
                    planet_case,
                    star_keys_sorted,
                    distance_values_au,
                )

            if use_parallel and PARALLEL_TASK_MODE == "distance":
                tasks = []
                for dist in DISTANCE_LIST:
                    dist_value_au = float(dist.to_value(u.AU))
                    missing_star_keys = [
                        star_key
                        for star_key in star_keys_sorted
                        if checkpoint_key(selected_planet, selected_species, star_key, dist_value_au)
                        not in completed_keys
                    ]
                    if not missing_star_keys:
                        continue
                    print(
                        f"species={selected_species}, "
                        f"temp_atm={planet_case['T'].to_value(u.K):.0f} K, "
                        f"planet_distance={dist.to_value(u.AU):g} AU, "
                        f"Teff={teff_labels} K"
                    )
                    tasks.append(
                        (
                            selected_planet,
                            selected_species,
                            planet_case,
                            planet_obj,
                            dist_value_au,
                            missing_star_keys,
                        )
                    )

                if tasks:
                    with ProcessPoolExecutor(max_workers=min(DISTANCE_MAX_WORKERS, len(tasks))) as executor:
                        futures = [executor.submit(compute_distance_column_worker, task) for task in tasks]
                        for future in as_completed(futures):
                            dist_value_au, results = future.result()
                            new_rows = []
                            for star_key, _teff, rbeta_value in results:
                                row_key = checkpoint_key(
                                    selected_planet,
                                    selected_species,
                                    star_key,
                                    dist_value_au,
                                )
                                if row_key in completed_keys:
                                    continue
                                new_rows.append(
                                    make_checkpoint_row(
                                        selected_planet,
                                        selected_species,
                                        star_key,
                                        dist_value_au,
                                        rbeta_value,
                                        run_signature,
                                    )
                                )
                                completed_keys.add(row_key)

                            if new_rows:
                                all_results.extend(new_rows)
                                persist_species_progress(
                                    all_results,
                                    selected_planet,
                                    selected_species,
                                    planet_case,
                                    star_keys_sorted,
                                    distance_values_au,
                                )
            elif use_parallel and PARALLEL_TASK_MODE == "distance_star":
                tasks = []
                for dist in DISTANCE_LIST:
                    dist_value_au = float(dist.to_value(u.AU))
                    print(
                        f"species={selected_species}, "
                        f"temp_atm={planet_case['T'].to_value(u.K):.0f} K, "
                        f"planet_distance={dist.to_value(u.AU):g} AU, "
                        f"Teff={teff_labels} K"
                    )
                    for star_key in star_keys_sorted:
                        if checkpoint_key(selected_planet, selected_species, star_key, dist_value_au) in completed_keys:
                            continue
                        tasks.append(
                            (
                                selected_planet,
                                selected_species,
                                planet_case,
                                planet_obj,
                                dist_value_au,
                                star_key,
                            )
                        )

                if tasks:
                    with ProcessPoolExecutor(max_workers=min(STAR_MAX_WORKERS, len(tasks))) as executor:
                        futures = [executor.submit(compute_distance_star_worker, task) for task in tasks]
                        for future in as_completed(futures):
                            dist_value_au, star_key, _teff, rbeta_value = future.result()
                            row_key = checkpoint_key(
                                selected_planet,
                                selected_species,
                                star_key,
                                dist_value_au,
                            )
                            if row_key in completed_keys:
                                continue

                            all_results.append(
                                make_checkpoint_row(
                                    selected_planet,
                                    selected_species,
                                    star_key,
                                    dist_value_au,
                                    rbeta_value,
                                    run_signature,
                                )
                            )
                            completed_keys.add(row_key)
                            persist_species_progress(
                                all_results,
                                selected_planet,
                                selected_species,
                                planet_case,
                                star_keys_sorted,
                                distance_values_au,
                            )
            else:
                for dist in DISTANCE_LIST:
                    dist_value_au = float(dist.to_value(u.AU))
                    new_rows = []
                    print(
                        f"species={selected_species}, "
                        f"temp_atm={planet_case['T'].to_value(u.K):.0f} K, "
                        f"planet_distance={dist.to_value(u.AU):g} AU, "
                        f"Teff={teff_labels} K"
                    )
                    for star_key in star_keys_sorted:
                        row_key = checkpoint_key(selected_planet, selected_species, star_key, dist_value_au)
                        if row_key in completed_keys:
                            continue

                        star = get_star(star_key)
                        system_obj = PlanetarySystem(planet_obj, star, dist)
                        value = r_beta1_over_R(
                            system_obj,
                            planet_case,
                            star_key,
                            selected_species,
                            geometry_cache=geometry_cache,
                            ncol_cache=ncol_cache,
                        )
                        new_rows.append(
                            make_checkpoint_row(
                                selected_planet,
                                selected_species,
                                star_key,
                                dist_value_au,
                                value,
                                run_signature,
                            )
                        )
                        completed_keys.add(row_key)

                    if new_rows:
                        all_results.extend(new_rows)
                        persist_species_progress(
                            all_results,
                            selected_planet,
                            selected_species,
                            planet_case,
                            star_keys_sorted,
                            distance_values_au,
                        )

            species_rows = [
                row
                for row in all_results
                if row.get("planet") == selected_planet and row.get("species") == selected_species
            ]
            if not species_rows:
                raise ValueError(f"No data were computed for planet={selected_planet}, species={selected_species}")

            persist_species_progress(
                all_results,
                selected_planet,
                selected_species,
                planet_case,
                star_keys_sorted,
                distance_values_au,
            )
            table_path = output_dir / f"{species_save_name}_r_beta1.txt"
            species_elapsed_s = time.perf_counter() - species_start_time
            print(f"Used species: {selected_species}")
            if SAVE_OUTPUT_TXT:
                print(f"Saved table to {table_path}")
            else:
                print("SAVE_OUTPUT_TXT=False, skipping table save")
            print(f"Total time for {selected_species}: {species_elapsed_s:.2f} s")
            clear_species_runtime_caches(selected_species)


if __name__ == "__main__":
    freeze_support()
    main()
