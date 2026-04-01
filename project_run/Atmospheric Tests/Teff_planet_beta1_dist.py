import sys
import pathlib

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

def get_star_teff(star_key):
    star = get_star(star_key)
    teff_value = star.header.get("teff", {}).get("value", None)
    if teff_value is None:
        raise ValueError(f"Star header for {star_key} does not contain teff.")
    return float(teff_value)


# Use global templates for stellar models
stellar_models = STAR_TEMPLATES

# -----------------------------------------------------------------------------
# Fixed setup for the first Teff study
# -----------------------------------------------------------------------------
SELECTED_PLANET_SPECIES = {
    "hot_jupiter": ["H I", "Na I"],
    "inflated_hot_jupiter": ["H I", "Na I"],
    "ultra_hot_jupiter": ["H I", "Na I", "Na II", "K I", "K II"],
    "earth_like": ["O I", "N I"],
    "mercury_like": ["Na I", "K I"],
}
DISTANCE_LIST = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0] * u.AU
STAR_STRIDE = 4
MOLECULE_FETCH = {
    "CO": {"path": "CO/12C-16O/Li2015", "database": "Li2015"},
    "NO": {"path": "NO/14N-16O/XABC", "database": "XABC"},
}
SKIP_MOLECULES = True

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


# Helper to save r_beta1_over_R data as a structured text table
def save_rbeta_table_txt(output_path, selected_planet, selected_species, teff_values, distance_values_au, rbeta_matrix):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "# r_beta1_over_R data table",
        f"# planet: {selected_planet}",
        f"# species: {selected_species}",
        "# x_label: Stellar Teff [K]",
        "# y_label: r_beta1 / R_p",
        "# description: Each distance column contains r_beta1_over_R evaluated as a function of stellar Teff.",
        "# distance_unit: AU",
        f"# distances_au: {', '.join(f'{d:g}' for d in distance_values_au)}",
        f"# star_stride: {STAR_STRIDE}",
        "#",
    ]

    column_names = ["Teff_K"] + [f"r_beta1_over_R__{d:g}_AU" for d in distance_values_au]

    with open(output_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("\t".join(column_names) + "\n")

        for i, teff in enumerate(teff_values):
            row = [f"{float(teff):.6g}"]
            for j in range(len(distance_values_au)):
                value = rbeta_matrix[i, j]
                row.append("nan" if not np.isfinite(value) else f"{float(value):.10g}")
            f.write("\t".join(row) + "\n")


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

    if species in MOLECULE_FETCH:
        mol = Molecule(species, wavemin, wavemax)
        mol.fetch_exomol(
            path=MOLECULE_FETCH[species]["path"],
            database=MOLECULE_FETCH[species]["database"],
            localdatabase="exomol_data",
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
    if SKIP_MOLECULES and species in MOLECULE_FETCH:
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


def main():
    star_keys_sorted = sorted(stellar_models.keys(), key=get_star_teff)
    star_keys_sorted = star_keys_sorted[::STAR_STRIDE]

    for selected_planet, requested_species in SELECTED_PLANET_SPECIES.items():
        planet_save_name = safe_name(selected_planet)
        planet_case = get_planet_template(selected_planet)

        if not requested_species:
            requested_species = list(planet_case["composition"].keys())
            print(f"No species specified for {selected_planet}; using all composition species: {requested_species}")

        invalid_species = [sp for sp in requested_species if sp not in planet_case["composition"]]
        if invalid_species:
            raise ValueError(
                f"Selected species for {selected_planet} must be in the planet composition. "
                f"Invalid entries: {invalid_species}. "
                f"Available species: {list(planet_case['composition'].keys())}"
            )

        planet_obj = Planet(
            radius=planet_case["radius"],
            mass=planet_case["mass"],
            T=planet_case["T"],
            mu=planet_case["mu"],
            P0=planet_case["P0"],
        )

        for selected_species in requested_species:
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

            used_species_summary = [selected_species]
            distance_values_au = [dist.to_value(u.AU) for dist in DISTANCE_LIST]
            teff_reference = None
            rbeta_columns = []

            for dist in DISTANCE_LIST:
                print(f"Processing planet={selected_planet}, species={selected_species}, distance={dist}")
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
                raise ValueError(f"No data were computed for planet={selected_planet}, species={selected_species}")

            rbeta_matrix = np.column_stack(rbeta_columns)
            table_path = output_dir / f"{species_save_name}_r_beta1.txt"
            save_rbeta_table_txt(
                table_path,
                selected_planet,
                selected_species,
                teff_reference,
                distance_values_au,
                rbeta_matrix,
            )
            print(f"Used species: {used_species_summary}")
            print(f"Saved table to {table_path}")


main()
