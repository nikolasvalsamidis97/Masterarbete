

import sys
import pathlib
import time
import traceback

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
from project_func.Templates.Planets.planet_templates import get_planet_template
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.plotdata_to_txt import save_plotdata_txt


# -----------------------------------------------------------------------------
# Study configuration
# -----------------------------------------------------------------------------
SELECTED_PLANET_KEYS = [
    "earth_like",
    "sub_neptune",
    "hot_jupiter",
    "super_puff",
]

SELECTED_SPECIES = [
    "H I",
    "Na I",
    "Na II",
    "Fe I",
    "Fe II",
    "NO",
]

STAR_KEY = "O0"
DISTANCE_AU = 1
T_ATM_GRID = np.array([150, 350, 500.0, 800.0, 1200.0, 1800.0, 2500.0, 3500.0]) * u.K

COARSE_HEIGHT_POINTS = 40
REFINE_HEIGHT_POINTS = 30
COARSE_GRID_POWER = 3.0
PRINT_TRACEBACKS = False
SAVE_OUTPUT_TXT = True

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
b_atom = 1 * u.km / u.s
npts_atom = 200
b_molecule = 1 * u.km / u.s

star_cache = {}
profile_cache = {}
stellar_models = STAR_TEMPLATES

# Optional plotting style hints for later plotting scripts.
SPECIES_COLORS = {
    "H": "tab:blue",
    "Na": "tab:orange",
    "Fe": "tab:red",
    "CO": "tab:green",
    "NO": "tab:purple",
}
SPECIES_MARKERS = {
    "neutral": "o",
    "ion": "s",
    "molecule": "^",
}


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
                    ncol_refine_values[j] = float(np.squeeze(ncol_cache[ncol_refine_key].to_value(1 / u.cm**2)))
                else:
                    missing_refine_indices.append(j)

            for j in missing_refine_indices:
                z_refine = z_refine_grid[j]
                ncol_local_refine = np.array([
                    system_obj.planet.slant_column_density(z_refine).to_value(1 / u.cm**2)
                ]) / u.cm**2
                if ncol_cache is not None:
                    ncol_cache[z_refine_cm_values[j]] = ncol_local_refine
                ncol_refine_values[j] = float(np.squeeze(ncol_local_refine.to_value(1 / u.cm**2)))

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
                    ncol_refine_key = z_refine_cm_value

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


# -----------------------------------------------------------------------------
# Main study
# -----------------------------------------------------------------------------

def main():
    star = get_star(STAR_KEY)
    distance = DISTANCE_AU * u.AU

    for selected_planet in SELECTED_PLANET_KEYS:
        planet_case_template = get_planet_template(selected_planet)
        planet_save_name = safe_name(selected_planet)
        geometry_cache = {}

        print(f"\n=== Running T_atm study for planet={selected_planet}, star={STAR_KEY}, distance={DISTANCE_AU:g} AU ===")
        print(f"Species list: {SELECTED_SPECIES}")
        print(f"T_atm grid [K]: {[float(temp.to_value(u.K)) for temp in T_ATM_GRID]}")

        for selected_species in SELECTED_SPECIES:
            species_start_time = time.perf_counter()
            species_save_name = safe_name(selected_species)
            output_dir = (
                pathlib.Path(__file__).resolve().parents[2]
                / "Plots"
                / "Atmospheric test"
                / "T_atm_vs_beta1"
                / f"{planet_save_name}_r_beta1"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            tatm_values_k = []
            rbeta_values = []

            for temp_atm in T_ATM_GRID:
                temp_k = float(temp_atm.to_value(u.K))
                print(
                    f"planet={selected_planet}, species={selected_species}, star={STAR_KEY}, "
                    f"distance={DISTANCE_AU:g} AU, T_atm={temp_k:.0f} K"
                )

                planet_case = dict(planet_case_template)
                planet_case["T"] = temp_atm
                planet_obj = Planet(
                    radius=planet_case["radius"],
                    mass=planet_case["mass"],
                    T=planet_case["T"],
                    mu=planet_case["mu"],
                    P0=planet_case["P0"],
                )
                system_obj = PlanetarySystem(planet_obj, star, distance)

                ncol_cache = {}
                value = r_beta1_over_R(
                    system_obj,
                    planet_case,
                    STAR_KEY,
                    selected_species,
                    geometry_cache=geometry_cache,
                    ncol_cache=ncol_cache,
                )
                tatm_values_k.append(temp_k)
                rbeta_values.append(value)

            tatm_values_k = np.asarray(tatm_values_k, dtype=float)
            rbeta_values = np.asarray(rbeta_values, dtype=float).reshape(-1, 1)
            table_path = output_dir / f"{species_save_name}_r_beta1.txt"

            if SAVE_OUTPUT_TXT:
                save_plotdata_txt(
                    table_path,
                    dataset_name=f"{species_save_name}_r_beta1",
                    x_label="Atmospheric temperature",
                    x_unit="K",
                    y_label="r_beta1 / R_p",
                    y_unit="dimensionless",
                    x_values=tatm_values_k,
                    y_matrix=rbeta_values,
                    series_values=[DISTANCE_AU],
                    series_label="distance",
                    series_unit="AU",
                    extra_metadata={
                        "planet": selected_planet,
                        "species": selected_species,
                        "star": STAR_KEY,
                        "planet_radius_Rjup": planet_case_template["radius"].to_value(u.R_jup),
                        "planet_mass_Mjup": planet_case_template["mass"].to_value(u.M_jup),
                        "planet_mu": float(planet_case_template["mu"]),
                        "distance_AU": DISTANCE_AU,
                        "t_atm_grid_K": tatm_values_k.tolist(),
                        "species_color_family": (
                            "H" if selected_species.startswith("H ") else
                            "Na" if selected_species.startswith("Na ") else
                            "Fe" if selected_species.startswith("Fe ") else
                            selected_species
                        ),
                        "species_marker_type": (
                            "ion" if "II" in selected_species or "III" in selected_species else
                            "molecule" if selected_species in {"CO", "NO"} else
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