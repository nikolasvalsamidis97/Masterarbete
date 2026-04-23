

import sys
import pathlib
import time
import traceback

import numpy as np
import astropy.units as u
import astropy.constants as const

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem
from project_classes.Star import Star
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.plotdata_to_txt import save_plotdata_txt


# -----------------------------------------------------------------------------
# Gravity study configuration
# -----------------------------------------------------------------------------
# Fixed mock-planet properties
MOCK_PLANET_LABEL = "mock_neutral_planet"
PLANET_RADIUS = 1.0 * const.R_jup
PLANET_T_ATM = 1400.0 * u.K
PLANET_MU = 2.3 * u.dimensionless_unscaled
PLANET_P0 = 1.0e-3 * u.bar

# Vary only mass, then compute surface gravity from each mass.
PLANET_MASS_GRID = np.array([0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00, 5.00]) * const.M_jup

# Fixed irradiation setup
STAR_KEY = "A0"
DISTANCE_AU = 0.1

# Strong absorbers plus one molecule
SELECTED_SPECIES = [
    "H I",
    "Na I",
    "Li I",
    "Fe I",
    "NO",
]

# Height-search settings
COARSE_HEIGHT_POINTS = 60
REFINE_HEIGHT_POINTS = 30
COARSE_GRID_POWER = 3.0
PRINT_TRACEBACKS = False
SAVE_OUTPUT_TXT = True

# Line-profile setup (kept similar to the other studies)
wavemin = 150 * u.AA
wavemax = 50000 * u.AA
b_atom = 1 * u.km / u.s
npts_atom = 150
b_molecule = 1 * u.km / u.s

star_cache = {}
profile_cache = {}
stellar_models = STAR_TEMPLATES


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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



def is_molecular_species(species):
    return species in MOLECULE_TEMPLATES



def make_mock_planet(mass):
    return Planet(
        radius=PLANET_RADIUS,
        mass=mass,
        T=PLANET_T_ATM,
        mu=PLANET_MU,
        P0=PLANET_P0,
    )



def surface_gravity_m_s2(mass):
    return (const.G * mass / PLANET_RADIUS**2).to_value(u.m / u.s**2)



def max_beta_over_height(
    system_obj,
    species,
    geometry_cache=None,
    ncol_cache=None,
    coarse_points=None,
    refine_points=None,
):
    if coarse_points is None:
        coarse_points = COARSE_HEIGHT_POINTS
    if refine_points is None:
        refine_points = REFINE_HEIGHT_POINTS

    geometry_cache_key = (
        float(system_obj.planet.mass.to_value(const.M_jup.unit)),
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
        print(f"Skipping {species} before height loop: {exc}")
        maybe_print_traceback()
        return np.nan

    if is_molecular_species(species):
        z_cm_values = np.asarray([float(z.to_value(u.cm)) for z in z_grid], dtype=float)
        ncol_values = np.empty(len(z_grid), dtype=float)
        missing_indices = []

        for i, z_cm_value in enumerate(z_cm_values):
            ncol_key = z_cm_value
            if ncol_cache is not None and ncol_key in ncol_cache:
                ncol_values[i] = float(np.squeeze(ncol_cache[ncol_key].to_value(1 / u.cm**2)))
            else:
                missing_indices.append(i)

        for i in missing_indices:
            z = z_grid[i]
            ncol_local = np.array([
                system_obj.planet.slant_column_density(z).to_value(1 / u.cm**2)
            ]) / u.cm**2
            if ncol_cache is not None:
                ncol_cache[z_cm_values[i]] = ncol_local
            ncol_values[i] = float(np.squeeze(ncol_local.to_value(1 / u.cm**2)))

        ncol_grid = ncol_values / u.cm**2
        r_grid = planet_radius + z_grid

        try:
            F_ph_tot_grid, F_ph_tot_err_grid, _, _ = pp.calc_PhotonPressure(
                ncol_grid,
                PLANET_T_ATM,
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
            print(f"Skipping {species} in coarse molecular grid: {exc}")
            maybe_print_traceback()
            return np.nan
        finite_beta = beta_values_grid[np.isfinite(beta_values_grid)]
        if finite_beta.size == 0:
            return np.nan
        return float(np.nanmax(finite_beta))

    beta_values = []

    for z in z_grid:
        r_local = planet_radius + z
        z_cm_value = float(z.to_value(u.cm))
        ncol_key = z_cm_value

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
                PLANET_T_ATM,
                system_obj.distance,
            )
            beta_species, _ = pp.beta_Values(
                F_ph_tot,
                F_ph_tot_err,
                system_obj.planet.mass,
                r_local,
            )
            beta_value = float(np.squeeze(beta_species.value))
            if np.isfinite(beta_value):
                beta_values.append(beta_value)
        except Exception as exc:
            print(f"Skipping {species} at z={z.to_value(u.km):.3f} km: {exc}")
            maybe_print_traceback()
            return np.nan

    if not beta_values:
        return np.nan
    return float(np.nanmax(np.asarray(beta_values, dtype=float)))


# -----------------------------------------------------------------------------
# Main study
# -----------------------------------------------------------------------------
def main():
    star = get_star(STAR_KEY)
    distance = DISTANCE_AU * u.AU
    geometry_cache = {}

    print(f"\n=== Running gravity study for {MOCK_PLANET_LABEL}, star={STAR_KEY}, distance={DISTANCE_AU:g} AU ===")
    print(f"Species list: {SELECTED_SPECIES}")
    print(f"Mass grid [M_jup]: {[float(m.to_value(const.M_jup)) for m in PLANET_MASS_GRID]}")

    output_dir = (
        pathlib.Path(__file__).resolve().parents[2]
        / "Plots"
        / "Atmospheric test"
        / "gravity_vs_maxbeta"
        / f"{safe_name(MOCK_PLANET_LABEL)}_max_beta"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for selected_species in SELECTED_SPECIES:
        species_start_time = time.perf_counter()
        species_save_name = safe_name(selected_species)

        gravity_values = []
        maxbeta_values = []

        for mass in PLANET_MASS_GRID:
            g_value = surface_gravity_m_s2(mass)
            print(
                f"planet={MOCK_PLANET_LABEL}, species={selected_species}, star={STAR_KEY}, "
                f"distance={DISTANCE_AU:g} AU, mass={mass.to_value(const.M_jup):.3f} M_jup, "
                f"g={g_value:.3f} m/s^2"
            )

            planet_obj = make_mock_planet(mass)
            system_obj = PlanetarySystem(planet_obj, star, distance)
            ncol_cache = {}
            value = max_beta_over_height(
                system_obj,
                selected_species,
                geometry_cache=geometry_cache,
                ncol_cache=ncol_cache,
            )
            gravity_values.append(g_value)
            maxbeta_values.append(value)

        gravity_values = np.asarray(gravity_values, dtype=float)
        maxbeta_values = np.asarray(maxbeta_values, dtype=float).reshape(-1, 1)
        table_path = output_dir / f"{species_save_name}_max_beta.txt"

        if SAVE_OUTPUT_TXT:
            save_plotdata_txt(
                table_path,
                dataset_name=f"{species_save_name}_max_beta",
                x_label="Surface gravity",
                x_unit="m/s^2",
                y_label="max beta",
                y_unit="dimensionless",
                x_values=gravity_values,
                y_matrix=maxbeta_values,
                series_values=[DISTANCE_AU],
                series_label="distance",
                series_unit="AU",
                extra_metadata={
                    "planet": MOCK_PLANET_LABEL,
                    "species": selected_species,
                    "star": STAR_KEY,
                    "planet_radius_Rjup": PLANET_RADIUS.to_value(u.R_jup),
                    "planet_temperature_K": PLANET_T_ATM.to_value(u.K),
                    "planet_mu": float(PLANET_MU),
                    "distance_AU": DISTANCE_AU,
                    "mass_grid_Mjup": [float(m.to_value(const.M_jup)) for m in PLANET_MASS_GRID],
                    "gravity_grid_m_s2": gravity_values.tolist(),
                    "species_color_family": (
                        "H" if selected_species.startswith("H ") else
                        "Na" if selected_species.startswith("Na ") else
                        "Fe" if selected_species.startswith("Fe ") else
                        selected_species
                    ),
                    "species_marker_type": (
                        "double_ion" if "III" in selected_species else
                        "ion" if "II" in selected_species else
                        "molecule" if selected_species == "NO" else
                        "neutral"
                    ),
                },
            )

        species_elapsed_s = time.perf_counter() - species_start_time
        print(f"Used species: {selected_species}")
        if SAVE_OUTPUT_TXT:
            print(f"Saved table to {table_path}")
        else:
            print("SAVE_OUTPUT_TXT=False, skipping table save")
        print(f"Total time for {selected_species}: {species_elapsed_s:.2f} s")


if __name__ == "__main__":
    main()