

import importlib.util
import pathlib
import numpy as np
from astropy import units as u


SPECIES = "CO"
PLANET_KEY = "hot_jupiter"
TARGET_TEFF_K = 10000.0
DISTANCE_AU = 0.1
N_PRINT_POINTS = 10


def load_teff_script_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    script_path = (
        repo_root
        / "Projects"
        / "Atmospheric_Teff_Beta1_Distance_Study"
        / "Teff_planet_beta1_dist.py"
    )

    spec = importlib.util.spec_from_file_location("teff_planet_beta1_dist_testmod", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_nearest_star_key(module, target_teff_k: float):
    all_star_keys = list(module.STAR_TEMPLATES.keys())
    return min(all_star_keys, key=lambda key: abs(module.infer_teff_from_star_template(key) - target_teff_k))


def main():
    module = load_teff_script_module()

    planet_case = module.get_planet_template(PLANET_KEY)
    planet_obj = module.Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )

    star_key = find_nearest_star_key(module, TARGET_TEFF_K)
    star_teff = module.infer_teff_from_star_template(star_key)
    star = module.get_star(star_key)
    distance = DISTANCE_AU * u.AU
    system_obj = module.PlanetarySystem(planet_obj, star, distance)

    print(f"Testing species={SPECIES}, planet={PLANET_KEY}, star={star_key} (Teff={star_teff:.0f} K), distance={DISTANCE_AU} AU")

    profile = module.get_profile(SPECIES)
    pp = module.PhotonPressure(profile, system_obj.star)

    hill_radius = system_obj.hill_radius().to(u.cm)
    planet_radius = system_obj.planet.radius.to(u.cm)
    if hill_radius <= planet_radius:
        raise ValueError("Hill radius is not larger than planet radius for this test case.")

    z_max = hill_radius - planet_radius
    z_grid = np.linspace(0.0, z_max.to_value(u.cm), N_PRINT_POINTS) * u.cm

    print("\nFirst test heights and beta values:")
    print("index | z [km] | r/Rp | beta")

    for i, z in enumerate(z_grid):
        r_local = planet_radius + z
        ncol_local = np.array([
            system_obj.planet.slant_column_density(z).to_value(1 / u.cm**2)
        ]) / u.cm**2

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
        r_over_rp = float((r_local / planet_radius).decompose().value)

        print(f"{i:5d} | {z.to_value(u.km):8.3f} | {r_over_rp:5.3f} | {beta_value:.6e}")


if __name__ == "__main__":
    main()
