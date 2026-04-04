import sys
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import astropy.constants as const
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
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.plotdata_to_txt import save_plotdata_txt


def get_star_teff(star_key):
    star = get_star(star_key)
    teff_value = star.header.get("teff", {}).get("value", None)
    if teff_value is None:
        raise ValueError(f"Star header for {star_key} does not contain teff.")
    return float(teff_value)


# Use global templates for stellar models
stellar_models = STAR_TEMPLATES

# -----------------------------------------------------------------------------
# Fixed setup for the Teff study
# -----------------------------------------------------------------------------
SELECTED_PLANET_SPECIES = {
    "hot_jupiter": ["H2O"],
}
DISTANCE_LIST = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0] * u.AU
STAR_STRIDE = 4
TEMPERATURE_LIST = [300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0] * u.K
TEMP_MAX_WORKERS = 6
SKIP_MOLECULES = False

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
b_atom = 1 * u.km / u.s
npts_atom = 150
b_molecule = 1 * u.km / u.s
ncol = np.array([0.0]) * u.cm ** -2

star_cache = {}
profile_cache = {}
beta_planet_cache = {}





def scalar_value(x):
    if isinstance(x, u.Quantity):
        return float(np.squeeze(x.value))
    return float(np.squeeze(x))

# Helper to sanitize names for saving
def safe_name(value):
    return str(value).replace(" ", "").replace("/", "_")


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
        profile = BroadeningProfileMolecule(mol, b_molecule, profileType="Voigt")
        if hasattr(profile, "temp_strength_rel_cutoff"):
            profile.temp_strength_rel_cutoff = 1e-8
    else:
        atom = Atom(species, wavemin, wavemax)
        profile = BroadeningProfile(atom, b_atom, npts_atom, "Voigt")

    profile_cache[species] = profile
    return profile




# Calculate beta relative to the planet's gravity
def beta_against_planet_gravity(species, system_obj, T_atm):
    cache_key = (
        species,
        system_obj.star.filePath if hasattr(system_obj.star, "filePath") else str(system_obj.star),
        float(system_obj.distance.to_value(u.AU)),
        float(system_obj.planet.radius.to_value(u.cm)),
        float(system_obj.planet.mass.to_value(u.g)),
        float(T_atm.to_value(u.K)),
    )
    if cache_key in beta_planet_cache:
        return beta_planet_cache[cache_key]

    profile = get_profile(species)
    pp = PhotonPressure(profile, system_obj.star)
    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(ncol, T_atm, system_obj.distance)
    beta, _ = pp.beta_Values(F_ph_tot, F_ph_tot_err, system_obj.planet.mass, system_obj.planet.radius)
    beta_val = scalar_value(beta)
    beta_planet_cache[cache_key] = beta_val
    return beta_val



def r_beta1_over_R(system_obj, planet_case, star_key, species):
    """
    Build the beta profile for one selected species as a function of
    atmospheric height z using the local slant column density and local
    planetary gravity, then find the first radius where beta(z) = 1.
    """
    if SKIP_MOLECULES and species in MOLECULE_TEMPLATES:
        return np.nan

    hill_radius = system_obj.hill_radius().to(u.cm)
    planet_radius = system_obj.planet.radius.to(u.cm)

    if hill_radius <= planet_radius:
        return np.nan

    z_grid = np.linspace(0.0, (hill_radius - planet_radius).value, 300) * u.cm
    r_grid = planet_radius + z_grid
    beta_profile = np.full(len(z_grid), np.nan, dtype=float)

    for iz, z in enumerate(z_grid):
        r_local = r_grid[iz]
        ncol_local = np.array([
            system_obj.planet.slant_column_density(z).to_value(1 / u.cm**2)
        ]) / u.cm**2

        try:
            profile = get_profile(species)
            pp = PhotonPressure(profile, system_obj.star)
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
            beta_profile[iz] = float(np.squeeze(beta_species.value))
        except Exception as exc:
            print(f"Skipping {species} for {star_key} at z={z.to_value(u.km):.3f} km: {exc}")
            return np.nan

    finite_mask = np.isfinite(beta_profile)
    if np.count_nonzero(finite_mask) < 2:
        return np.nan

    r_grid = r_grid[finite_mask]
    beta_profile = beta_profile[finite_mask]
    diff = beta_profile - 1.0

    exact_idx = np.where(np.isclose(diff, 0.0, atol=1e-6))[0]
    if len(exact_idx) > 0:
        r_beta1 = r_grid[exact_idx[0]]
        return float((r_beta1 / planet_radius).decompose().value)

    sign_change_idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(sign_change_idx) == 0:
        return np.nan

    i = sign_change_idx[0]
    r1 = r_grid[i].to_value(u.cm)
    r2 = r_grid[i + 1].to_value(u.cm)
    b1 = beta_profile[i]
    b2 = beta_profile[i + 1]

    if np.isclose(b2, b1):
        r_beta1_cm = r1
    else:
        r_beta1_cm = r1 + (1.0 - b1) * (r2 - r1) / (b2 - b1)

    r_beta1 = r_beta1_cm * u.cm
    return float((r_beta1 / planet_radius).decompose().value)


def compute_species_for_temperature(selected_planet, selected_species, temperature_quantity, star_keys_sorted):
    planet_save_name = safe_name(selected_planet)
    species_save_name = safe_name(selected_species)
    temp_value_k = float(temperature_quantity.to_value(u.K))
    temp_save_name = safe_name(f"{temp_value_k:.0f}K")

    planet_case = get_planet_template(selected_planet).copy()
    planet_case["T"] = temperature_quantity

    planet_obj = Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )

    output_dir = (
        pathlib.Path(__file__).resolve().parents[2]
        / "Plots"
        / "Atmospheric test"
        / "Teff_study"
        / "r_at_beta1"
        / f"{planet_save_name}_r_beta1"
        / temp_save_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    distance_values_au = [dist.to_value(u.AU) for dist in DISTANCE_LIST]
    teff_reference = None
    rbeta_columns = []

    for dist in DISTANCE_LIST:
        print(
            f"Processing planet={selected_planet}, species={selected_species}, "
            f"T={temp_value_k:.0f} K, distance={dist}"
        )
        teff_values = []
        rbeta_values = []

        for star_key in star_keys_sorted:
            star = get_star(star_key)
            system_obj = PlanetarySystem(planet_obj, star, dist)
            teff = get_star_teff(star_key)
            value = r_beta1_over_R(system_obj, planet_case, star_key, selected_species)
            teff_values.append(teff)
            rbeta_values.append(value)

        teff_values = np.asarray(teff_values, dtype=float)
        rbeta_values = np.asarray(rbeta_values, dtype=float)

        if teff_reference is None:
            teff_reference = teff_values.copy()
        elif not np.array_equal(teff_reference, teff_values):
            raise ValueError("Teff grid changed between distances; cannot save a consistent table.")

        rbeta_columns.append(rbeta_values)

    if teff_reference is None or not rbeta_columns:
        raise ValueError(
            f"No data were computed for planet={selected_planet}, species={selected_species}, T={temp_value_k:.0f} K"
        )

    rbeta_matrix = np.column_stack(rbeta_columns)
    table_path = output_dir / f"{species_save_name}_r_beta1.txt"
    save_plotdata_txt(
        table_path,
        dataset_name=f"{species_save_name}_r_beta1_{temp_save_name}",
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
            "temperature_K": temp_value_k,
        },
    )

    return {
        "selected_planet": selected_planet,
        "selected_species": selected_species,
        "temperature_k": temp_value_k,
        "table_path": str(table_path),
    }


def run_temperature_batch(selected_planet, selected_species, star_keys_sorted):
    max_workers = min(TEMP_MAX_WORKERS, len(TEMPERATURE_LIST))
    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for temperature_quantity in TEMPERATURE_LIST:
            futures.append(
                executor.submit(
                    compute_species_for_temperature,
                    selected_planet,
                    selected_species,
                    temperature_quantity,
                    star_keys_sorted,
                )
            )

        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["temperature_k"])
    return results


def main():
    star_keys_sorted = sorted(stellar_models.keys(), key=get_star_teff)
    star_keys_sorted = star_keys_sorted[::STAR_STRIDE]

    for selected_planet, requested_species in SELECTED_PLANET_SPECIES.items():
        planet_case = get_planet_template(selected_planet)

        if not requested_species:
            raise ValueError(f"No species specified for {selected_planet}.")

        for selected_species in requested_species:
            print(
                f"Starting batch for planet={selected_planet}, species={selected_species}, "
                f"temperatures={[float(t.to_value(u.K)) for t in TEMPERATURE_LIST]} K"
            )
            results = run_temperature_batch(selected_planet, selected_species, star_keys_sorted)

            for result in results:
                print(
                    f"Saved planet={result['selected_planet']}, species={result['selected_species']}, "
                    f"T={result['temperature_k']:.0f} K -> {result['table_path']}"
                )


main()
