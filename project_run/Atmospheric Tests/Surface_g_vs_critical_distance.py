import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from astropy import units as u

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem
from project_classes.Star import Star
from project_func.Templates.Planets.planet_templates import PLANET_TEMPLATES, get_planet_template
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template
from project_func.plotdata_to_txt import save_plotdata_txt


# -----------------------------------------------------------------------------
# Surface gravity vs critical distance test
# One planet-star case, all atomic species from H I to Fe III
# -----------------------------------------------------------------------------

ELEMENTS_H_TO_FE = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe",
]
ION_STAGES = ["I", "II", "III"]
SELECTED_SPECIES_LIST = [f"{element} {stage}" for element in ELEMENTS_H_TO_FE for stage in ION_STAGES]
SELECTED_STARS = ["A0"]
SELECTED_PLANET_KEY = "hot_jupiter"
DISTANCE_GRID = np.logspace(-2, 0, 20) * u.AU
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_ATOM = 1 * u.km / u.s
NPTS_ATOM = 150
N_Z = 300


star_cache = {}
profile_cache = {}


def safe_name(value):
    return str(value).replace(" ", "").replace("/", "_")


def get_star(star_key):
    if star_key not in star_cache:
        params = STAR_TEMPLATES[star_key]
        star_cache[star_key] = Star(
            params["path"],
            params["radius"],
            params["mass"],
            vsini=params["vsini"],
            epsilon=params["epsilon"],
        )
    return star_cache[star_key]


def get_profile(species):
    if species not in profile_cache:
        atom = Atom(species, WAVEMIN, WAVEMAX)
        profile_cache[species] = BroadeningProfile(atom, B_ATOM, NPTS_ATOM, "Voigt")
    return profile_cache[species]


def species_available_in_planet(planet_case, species):
    composition = planet_case.get("composition", {})
    return species in composition and composition[species] > 0


def r_beta1_over_R(system_obj, planet_case, species):
    hill_radius = system_obj.hill_radius().to(u.cm)
    planet_radius = system_obj.planet.radius.to(u.cm)

    if hill_radius <= planet_radius:
        return np.nan

    z_grid = np.linspace(0.0, (hill_radius - planet_radius).value, N_Z) * u.cm
    r_grid = planet_radius + z_grid
    beta_profile = np.full(len(z_grid), np.nan, dtype=float)

    try:
        profile = get_profile(species)
    except Exception as exc:
        print(f"Skipping profile build for {species}: {exc}")
        return np.nan

    pp = PhotonPressure(profile, system_obj.star)

    for iz, z in enumerate(z_grid):
        r_local = r_grid[iz]
        ncol_local = np.array([
            system_obj.planet.slant_column_density(z).to_value(1 / u.cm**2)
        ]) / u.cm**2

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
            beta_profile[iz] = float(np.squeeze(beta_species.value))
        except Exception as exc:
            print(
                f"Skipping {species} for distance={system_obj.distance.to_value(u.AU):.5g} AU "
                f"at z={z.to_value(u.km):.3f} km: {exc}"
            )
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


def critical_distance_for_case(planet_case, star_key, species):
    planet_obj = Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )
    star = get_star(star_key)

    largest_distance = np.nan

    for dist in DISTANCE_GRID:
        system_obj = PlanetarySystem(planet_obj, star, dist)
        r_beta1 = r_beta1_over_R(system_obj, planet_case, species)
        if np.isfinite(r_beta1):
            largest_distance = dist.to_value(u.AU)

    return largest_distance


def main():
    if SELECTED_PLANET_KEY not in PLANET_TEMPLATES:
        raise ValueError(f"Planet template '{SELECTED_PLANET_KEY}' was not found.")

    planet_case = get_planet_template(SELECTED_PLANET_KEY)
    planet_obj = Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )

    x_surface_g = np.asarray([
        planet_obj.gravity(0 * u.cm).to_value(u.m / u.s**2)
    ], dtype=float)

    for star_key in SELECTED_STARS:
        teff = infer_teff_from_star_template(star_key)
        print(f"Running all species for {SELECTED_PLANET_KEY} around {star_key} (Teff={teff} K)")

        species_labels = []
        critical_distances = []

        for species in SELECTED_SPECIES_LIST:
            print(f"  Testing {species}")
            try:
                value = critical_distance_for_case(planet_case, star_key, species)
            except Exception as exc:
                print(f"    Skipping {species}: {exc}")
                value = np.nan

            species_labels.append(species)
            critical_distances.append(value)
            print(f"    critical distance = {value} AU")

        y_matrix = np.asarray([critical_distances], dtype=float)

        output_dir = (
            pathlib.Path(__file__).resolve().parents[2]
            / "Plots"
            / "Atmospheric test"
            / "Surface_g_vs_critical_dist"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{safe_name(SELECTED_PLANET_KEY)}_{safe_name(star_key)}_surface_g_vs_critical_dist.txt"
        output_path = output_dir / output_filename

        save_plotdata_txt(
            output_path,
            dataset_name=f"{safe_name(SELECTED_PLANET_KEY)}_{safe_name(star_key)}_surface_g_vs_critical_dist",
            x_label="Surface gravity",
            x_unit="m s^-2",
            y_label="Critical distance",
            y_unit="AU",
            x_values=x_surface_g,
            y_matrix=y_matrix,
            series_values=species_labels,
            series_label="species",
            series_unit="label",
            extra_metadata={
                "planet_key": SELECTED_PLANET_KEY,
                "star_key": star_key,
                "teff_K": teff,
                "distance_grid_AU": ", ".join(f"{dist.to_value(u.AU):.6g}" for dist in DISTANCE_GRID),
            },
        )

        print(f"Saved plot data to: {output_path}")


if __name__ == "__main__":
    main()