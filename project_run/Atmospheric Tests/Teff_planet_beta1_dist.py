import csv
import sys
import pathlib
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
SKIP_MOLECULES = False


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

# DISTANCE_LIST = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0] * u.AU
DISTANCE_LIST = [0.05, 0.1, 1, 10, 100] * u.AU
SELECTED_STARS = None
# SELECTED_STARS = ["O0", "B0", "A0"]
TARGET_TEFFS_K = [3000, 5000, 6000, 8000, 10000, 15000, 20000, 30000, 50000]
STAR_STRIDE = 5
DISTANCE_MAX_WORKERS = 4
STAR_MAX_WORKERS = 15
# Choose outer parallelization strategy:
# "distance"      -> one worker per distance (reuses stars serially inside each worker)
# "distance_star" -> one worker per (distance, star) pair (more workers, less reuse)
# "serial"        -> no outer parallelism
PARALLEL_TASK_MODE = "distance"
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

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
b_atom = 1 * u.km / u.s
npts_atom = 150
b_molecule = 1 * u.km / u.s

star_cache = {}
profile_cache = {}

ROOT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "Plots"
    / "Atmospheric test"
    / "Teff_study"
    / "r_at_beta1"
)
CHECKPOINT_PATH = ROOT_OUTPUT_DIR / "teff_planet_beta1_dist_checkpoint.csv"





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
    ]


def current_checkpoint_config():
    return {
        "coarse_height_points": int(COARSE_HEIGHT_POINTS),
        "refine_height_points": int(REFINE_HEIGHT_POINTS),
        "coarse_grid_power": float(COARSE_GRID_POWER),
    }


def validate_checkpoint_rows(rows):
    if not rows:
        return

    expected = current_checkpoint_config()
    for row in rows:
        actual = {
            "coarse_height_points": int(float(row.get("coarse_height_points", np.nan))),
            "refine_height_points": int(float(row.get("refine_height_points", np.nan))),
            "coarse_grid_power": float(row.get("coarse_grid_power", np.nan)),
        }
        if actual != expected:
            raise ValueError(
                "Existing Teff checkpoint was created with different grid settings. "
                f"Delete {CHECKPOINT_PATH} or restore the old settings before resuming."
            )


def load_checkpoint_rows(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    validate_checkpoint_rows(rows)
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


def make_checkpoint_row(planet_key, species, star_key, distance_au, rbeta_value):
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


def species_mixing_ratio(planet_case, species):
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
        z_cm_values = np.asarray([float(z.to_value(u.cm)) for z in z_grid], dtype=float)
        ncol_values = np.empty(len(z_grid), dtype=float)
        missing_indices = []

        for i, z_cm_value in enumerate(z_cm_values):
            ncol_key = z_cm_value
            if ncol_cache is not None and ncol_key in ncol_cache:
                ncol_values[i] = float(np.squeeze((abundance * ncol_cache[ncol_key]).to_value(1 / u.cm**2)))
            else:
                missing_indices.append(i)

        for i in missing_indices:
            z = z_grid[i]
            ncol_total = get_total_slant_column(system_obj, z, ncol_cache, z_cm_values[i])
            ncol_values[i] = float(np.squeeze((abundance * ncol_total).to_value(1 / u.cm**2)))

        ncol_grid = ncol_values / u.cm**2
        r_grid = planet_radius + z_grid

        try:
            F_ph_tot_grid, F_ph_tot_err_grid, _, _ = pp.calc_PhotonPressure(
                ncol_grid,
                planet_case["T"],
                system_obj.distance,
            )
            beta_species_grid, _ = pp.beta_Values(
                F_ph_tot_grid,
                F_ph_tot_err_grid,
                system_obj.planet.mass,
                r_grid,
            )
            beta_values_grid = np.asarray(beta_species_grid.value, dtype=float).reshape(-1)
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
            z_refine_cm_values = np.asarray([float(z.to_value(u.cm)) for z in z_refine_grid], dtype=float)
            ncol_refine_values = np.empty(len(z_refine_grid), dtype=float)
            missing_refine_indices = []

            for j, z_refine_cm_value in enumerate(z_refine_cm_values):
                ncol_refine_key = z_refine_cm_value
                if ncol_cache is not None and ncol_refine_key in ncol_cache:
                    ncol_refine_values[j] = float(np.squeeze((abundance * ncol_cache[ncol_refine_key]).to_value(1 / u.cm**2)))
                else:
                    missing_refine_indices.append(j)

            for j in missing_refine_indices:
                z_refine = z_refine_grid[j]
                ncol_total_refine = get_total_slant_column(system_obj, z_refine, ncol_cache, z_refine_cm_values[j])
                ncol_refine_values[j] = float(np.squeeze((abundance * ncol_total_refine).to_value(1 / u.cm**2)))

            ncol_refine_grid = ncol_refine_values / u.cm**2
            r_refine_grid = planet_radius + z_refine_grid

            try:
                F_ph_tot_refine, F_ph_tot_err_refine, _, _ = pp.calc_PhotonPressure(
                    ncol_refine_grid,
                    planet_case["T"],
                    system_obj.distance,
                )
                beta_species_refine, _ = pp.beta_Values(
                    F_ph_tot_refine,
                    F_ph_tot_err_refine,
                    system_obj.planet.mass,
                    r_refine_grid,
                )
                beta_values_refine = np.asarray(beta_species_refine.value, dtype=float).reshape(-1)
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

        return np.nan

    prev_r_local = None
    prev_beta_value = None
    prev_z = None

    for z in z_grid:
        r_local = planet_radius + z
        z_cm_value = float(z.to_value(u.cm))
        ncol_key = z_cm_value

        ncol_local = abundance * get_total_slant_column(system_obj, z, ncol_cache, ncol_key)

        try:
            F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(
                ncol_local,
                planet_case["T"],
                system_obj.distance,
            )
            beta_species, _ = pp.beta_Values(
                F_ph_tot,
                F_ph_tot_err,
                system_obj.planet.mass,
                r_local,
            )
            beta_value = float(np.squeeze(beta_species.value))
        except Exception as exc:
            print(f"Skipping {species} for {star_key} at z={z.to_value(u.km):.3f} km: {exc}")
            maybe_print_traceback()
            return np.nan

        if np.isclose(beta_value, 1.0, atol=1e-6):
            return float((r_local / planet_radius).decompose().value)

        if prev_beta_value is not None:
            crossed = ((prev_beta_value < 1.0 and beta_value > 1.0) or
                       (prev_beta_value > 1.0 and beta_value < 1.0))
            if crossed:
                r1 = prev_r_local.to_value(u.cm)
                r2 = r_local.to_value(u.cm)
                b1 = prev_beta_value
                b2 = beta_value

                if np.isclose(b2, b1):
                    r_beta1_cm = r1
                else:
                    r_beta1_cm = r1 + (1.0 - b1) * (r2 - r1) / (b2 - b1)

                r_beta1 = r_beta1_cm * u.cm

                z_refine_grid = np.linspace(prev_z.to_value(u.cm), z.to_value(u.cm), refine_points) * u.cm
                for z_refine in z_refine_grid:
                    r_local_refine = planet_radius + z_refine
                    z_refine_cm_value = float(z_refine.to_value(u.cm))
                    ncol_refine_key = z_refine_cm_value

                    if ncol_cache is not None and ncol_refine_key in ncol_cache:
                        ncol_local_refine = abundance * ncol_cache[ncol_refine_key]
                    else:
                        ncol_local_refine = abundance * get_total_slant_column(system_obj, z_refine, ncol_cache, ncol_refine_key)
                    try:
                        F_ph_tot_refine, F_ph_tot_err_refine, _, _ = pp.calc_PhotonPressure(
                            ncol_local_refine,
                            planet_case["T"],
                            system_obj.distance,
                        )
                        beta_species_refine, _ = pp.beta_Values(
                            F_ph_tot_refine,
                            F_ph_tot_err_refine,
                            system_obj.planet.mass,
                            r_local_refine,
                        )
                        beta_value_refine = float(np.squeeze(beta_species_refine.value))
                    except Exception as exc:
                        print(f"Skipping {species} for {star_key} at refined z={z_refine.to_value(u.km):.3f} km: {exc}")
                        maybe_print_traceback()
                        return np.nan

                    if np.isclose(beta_value_refine, 1.0, atol=1e-6):
                        return float((r_local_refine / planet_radius).decompose().value)

                return float((r_beta1 / planet_radius).decompose().value)

        prev_r_local = r_local
        prev_beta_value = beta_value
        prev_z = z

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

    if USE_CHECKPOINT:
        all_results = load_checkpoint_rows(CHECKPOINT_PATH)
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

            distance_values_au = [dist.to_value(u.AU) for dist in DISTANCE_LIST]
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
                                    )
                                )
                                completed_keys.add(row_key)

                            if new_rows:
                                all_results.extend(new_rows)
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
                                )
                            )
                            completed_keys.add(row_key)
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
            else:
                for dist in DISTANCE_LIST:
                    dist_value_au = float(dist.to_value(u.AU))
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
                        all_results.append(
                            make_checkpoint_row(
                                selected_planet,
                                selected_species,
                                star_key,
                                dist_value_au,
                                value,
                            )
                        )
                        completed_keys.add(row_key)
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

            species_rows = [
                row
                for row in all_results
                if row.get("planet") == selected_planet and row.get("species") == selected_species
            ]
            if not species_rows:
                raise ValueError(f"No data were computed for planet={selected_planet}, species={selected_species}")

            table_path = save_species_table_from_rows(
                all_results,
                selected_planet,
                selected_species,
                planet_case,
                star_keys_sorted,
                distance_values_au,
            )
            species_elapsed_s = time.perf_counter() - species_start_time
            print(f"Used species: {selected_species}")
            if SAVE_OUTPUT_TXT:
                print(f"Saved table to {table_path}")
            else:
                print("SAVE_OUTPUT_TXT=False, skipping table save")
            print(f"Total time for {selected_species}: {species_elapsed_s:.2f} s")


if __name__ == "__main__":
    freeze_support()
    main()
