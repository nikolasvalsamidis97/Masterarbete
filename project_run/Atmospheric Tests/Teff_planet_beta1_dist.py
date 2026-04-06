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
SELECTED_MOLECULAR_SPECIES = ["CO", "NO", "H2O"]
SKIP_ATOMS = True
SKIP_MOLECULES = False


# Use global templates for stellar models
stellar_models = STAR_TEMPLATES

# -----------------------------------------------------------------------------
# Fixed setup for the first Teff study
# -----------------------------------------------------------------------------
DEFAULT_PLANET_KEYS = [
    "hot_jupiter",
]
SELECTED_PLANET_SPECIES = {
    planet_key: None
    for planet_key in DEFAULT_PLANET_KEYS
    if planet_key in PLANET_TEMPLATES
}

DISTANCE_LIST = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0] * u.AU
# DISTANCE_LIST = [0.1, 0.5, 1.0, 5.0, 10.0] * u.AU
SELECTED_STARS = None
STAR_STRIDE = 15
STAR_MAX_WORKERS = 10
N_HEIGHT_POINTS = 150
# COARSE_HEIGHT_POINTS = 30
# REFINE_HEIGHT_POINTS = 30
### For faster testing ###
COARSE_HEIGHT_POINTS = 20
REFINE_HEIGHT_POINTS = 15
##########################
COARSE_GRID_POWER = 3.0
PRINT_TRACEBACKS = False

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
b_atom = 1 * u.km / u.s
npts_atom = 150
b_molecule = 1 * u.km / u.s

star_cache = {}
profile_cache = {}





# Helper to sanitize names for saving
def safe_name(value):
    return str(value).replace(" ", "").replace("/", "_")


def maybe_print_traceback():
    if PRINT_TRACEBACKS:
        traceback.print_exc()


def get_molecule_fetch(species):
    template = MOLECULE_TEMPLATES[species]
    fetch_kwargs = template["fetch_kwargs"]
    return {
        "path": fetch_kwargs["path"],
        "database": fetch_kwargs["database"],
        "localdatabase": fetch_kwargs.get("localdatabase", "exomol_data"),
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


def get_profile(species):
    if species in profile_cache:
        return profile_cache[species]

    if species in MOLECULE_TEMPLATES:
        molecule_fetch = get_molecule_fetch(species)
        mol = Molecule(species, wavemin, wavemax)
        mol.fetch_exomol(
            path=molecule_fetch["path"],
            database=molecule_fetch["database"],
            localdatabase=molecule_fetch["localdatabase"],
        )
        print(f"[{species}] molecule loaded")
        profile = BroadeningProfileMolecule(mol, b_molecule, profileType="Voigt")
        if hasattr(profile, "temp_strength_rel_cutoff"):
            profile.temp_strength_rel_cutoff = 1e-8
    else:
        atom = Atom(species, wavemin, wavemax)
        profile = BroadeningProfile(atom, b_atom, npts_atom, "Voigt")

    profile_cache[species] = profile
    return profile


def is_atomic_species(species):
    return species not in MOLECULE_TEMPLATES


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
    cache_key = (
        star_key,
        float(system_obj.distance.to_value(u.AU)),
    )

    if geometry_cache is not None and cache_key in geometry_cache:
        hill_radius, planet_radius, z_grid = geometry_cache[cache_key]
    else:
        hill_radius = system_obj.hill_radius().to(u.cm)
        planet_radius = system_obj.planet.radius.to(u.cm)

        if hill_radius <= planet_radius:
            return np.nan

        z_max_cm = (hill_radius - planet_radius).to_value(u.cm)
        coarse_fraction = np.linspace(0.0, 1.0, coarse_points) ** COARSE_GRID_POWER
        z_grid = (coarse_fraction * z_max_cm) * u.cm

        if geometry_cache is not None:
            geometry_cache[cache_key] = (hill_radius, planet_radius, z_grid)

    try:
        profile = get_profile(species)
        pp = PhotonPressure(profile, system_obj.star)
    except Exception as exc:
        print(f"Skipping {species} for {star_key} before height loop: {exc}")
        maybe_print_traceback()
        return np.nan

    prev_r_local = None
    prev_beta_value = None
    prev_z = None

    for z in z_grid:
        r_local = planet_radius + z
        z_cm_value = float(z.to_value(u.cm))
        ncol_key = (cache_key, z_cm_value)

        if ncol_cache is not None and ncol_key in ncol_cache:
            ncol_local = ncol_cache[ncol_key]
        else:
            ncol_local = np.array([
                system_obj.planet.slant_column_density(z).to_value(1 / u.cm**2)
            ]) / u.cm**2
            if ncol_cache is not None:
                ncol_cache[ncol_key] = ncol_local

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
                    ncol_refine_key = (cache_key, z_refine_cm_value)

                    if ncol_cache is not None and ncol_refine_key in ncol_cache:
                        ncol_local_refine = ncol_cache[ncol_refine_key]
                    else:
                        ncol_local_refine = np.array([
                            system_obj.planet.slant_column_density(z_refine).to_value(1 / u.cm**2)
                        ]) / u.cm**2
                        if ncol_cache is not None:
                            ncol_cache[ncol_refine_key] = ncol_local_refine
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


def compute_rbeta_for_star(args):
    selected_planet, selected_species, planet_case, planet_obj, dist_value_au, star_key = args

    dist = dist_value_au * u.AU
    star = get_star(star_key)
    system_obj = PlanetarySystem(planet_obj, star, dist)
    teff = infer_teff_from_star_template(star_key)
    value = r_beta1_over_R(system_obj, planet_case, star_key, selected_species)
    return teff, value



def compute_distance_column_parallel(selected_planet, selected_species, planet_case, planet_obj, dist, star_keys_sorted):
    dist_value_au = float(dist.to_value(u.AU))
    tasks = [
        (selected_planet, selected_species, planet_case, planet_obj, dist_value_au, star_key)
        for star_key in star_keys_sorted
    ]

    results = []
    with ProcessPoolExecutor(max_workers=min(STAR_MAX_WORKERS, len(tasks))) as executor:
        futures = [executor.submit(compute_rbeta_for_star, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item[0])
    teff_values = np.asarray([item[0] for item in results], dtype=float)
    rbeta_values = np.asarray([item[1] for item in results], dtype=float)
    return teff_values, rbeta_values










def main():
    all_star_keys_sorted = sorted(stellar_models.keys(), key=infer_teff_from_star_template)
    if SELECTED_STARS is None:
        star_keys_sorted = all_star_keys_sorted[::STAR_STRIDE]
    else:
        invalid_star_keys = [star_key for star_key in SELECTED_STARS if star_key not in stellar_models]
        if invalid_star_keys:
            raise ValueError(
                f"Unknown stars in SELECTED_STARS: {invalid_star_keys}. "
                f"Available stars: {list(stellar_models.keys())}"
            )
        star_keys_sorted = sorted(SELECTED_STARS, key=infer_teff_from_star_template)

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

        if requested_species is None:
            requested_species = [
                species for species in (
                    (SELECTED_ATOMIC_SPECIES or []) + (SELECTED_MOLECULAR_SPECIES or [])
                )
                if species_matches_run_filters(species)
            ]
            print(
                f"No species specified for {selected_planet}; using filtered explicitly selected species: {requested_species}"
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



        use_parallel = len(requested_species) > 1

        for selected_species in requested_species:
            species_start_time = time.perf_counter()
            species_save_name = safe_name(selected_species)
            output_dir = (
                pathlib.Path(__file__).resolve().parents[2]
                / "Plots"
                / "Atmospheric test"
                / "Teff_study"
                / "r_at_beta1"
                / f"{planet_save_name}_r_beta1"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            star_objects = {
                star_key: get_star(star_key)
                for star_key in star_keys_sorted
            }
            geometry_cache = {}
            ncol_cache = {}


            distance_values_au = [dist.to_value(u.AU) for dist in DISTANCE_LIST]
            teff_labels = [f"{infer_teff_from_star_template(star_key):.0f}" for star_key in star_keys_sorted]
            teff_reference = None
            rbeta_columns = []

            for dist in DISTANCE_LIST:
                print(
                    f"species={selected_species}, "
                    f"temp_atm={planet_case['T'].to_value(u.K):.0f} K, "
                    f"planet_distance={dist.to_value(u.AU):g} AU, "
                    f"Teff={teff_labels} K"
                )
                if use_parallel:
                    teff_values, rbeta_values = compute_distance_column_parallel(
                        selected_planet,
                        selected_species,
                        planet_case,
                        planet_obj,
                        dist,
                        star_keys_sorted,
                    )
                else:
                    teff_pairs = []
                    for star_key in star_keys_sorted:
                        teff = infer_teff_from_star_template(star_key)
                        star = star_objects[star_key]
                        system_obj = PlanetarySystem(planet_obj, star, dist)
                        value = r_beta1_over_R(
                            system_obj,
                            planet_case,
                            star_key,
                            selected_species,
                            geometry_cache=geometry_cache,
                            ncol_cache=ncol_cache,
                        )
                        teff_pairs.append((teff, value))

                    teff_pairs.sort(key=lambda item: item[0])
                    teff_values = np.asarray([item[0] for item in teff_pairs], dtype=float)
                    rbeta_values = np.asarray([item[1] for item in teff_pairs], dtype=float)

                if teff_reference is None:
                    teff_reference = teff_values.copy()
                elif not np.array_equal(teff_reference, teff_values):
                    raise ValueError("Teff grid changed between distances; cannot save a consistent table.")

                rbeta_columns.append(rbeta_values)

            if teff_reference is None or not rbeta_columns:
                raise ValueError(f"No data were computed for planet={selected_planet}, species={selected_species}")

            rbeta_matrix = np.column_stack(rbeta_columns)
            table_path = output_dir / f"{species_save_name}_r_beta1.txt"
            save_plotdata_txt(
                table_path,
                dataset_name=f"{species_save_name}_r_beta1",
                x_label="Stellar Teff",
                x_unit="K",
                y_label="r_beta1 / R_p",
                y_unit="dimensionless",
                x_values=teff_reference,
                y_matrix=rbeta_matrix,
                series_values=distance_values_au,
                series_label="distance",
                series_unit="AU",
                extra_metadata={
                    "planet": selected_planet,
                    "species": selected_species,
                    "planet_radius_Rjup": planet_case["radius"].to_value(u.R_jup),
                    "planet_mass_Mjup": planet_case["mass"].to_value(u.M_jup),
                    "planet_temperature_K": planet_case["T"].to_value(u.K),
                    "planet_mu": float(planet_case["mu"]),
                    "stellar_teff_grid_K": teff_reference.tolist(),
                },
            )
            species_elapsed_s = time.perf_counter() - species_start_time
            print(f"Used species: {selected_species}")
            print(f"Saved table to {table_path}")
            print(f"Total time for {selected_species}: {species_elapsed_s:.2f} s")


if __name__ == "__main__":
    freeze_support()
    main()
