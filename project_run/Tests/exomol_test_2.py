

import sys
import pathlib
import importlib.util
import numpy as np
from astropy import units as u
from astropy import constants as const

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


def load_teff_module():
    script_path = pathlib.Path(__file__).resolve().parents[1] / "Atmospheric Tests" / "Teff_planet_beta1_dist.py"
    spec = importlib.util.spec_from_file_location("teff_planet_beta1_dist_reuse", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_teff_module()

    species = "O2"
    planet_key = "hot_jupiter"
    star_key = "G4"
    distance = 0.1 * u.AU
    z_test = 0.1 * const.R_jup

    planet_case = module.get_planet_template(planet_key)
    planet_obj = module.Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )
    star = module.get_star(star_key)
    system_obj = module.PlanetarySystem(planet_obj, star, distance)

    print(f"Testing HITRAN pipeline for species={species}")
    print(
        f"planet={planet_key}, star={star_key}, distance={distance.to_value(u.AU):g} AU, "
        f"T_atm={planet_case['T'].to_value(u.K):.0f} K"
    )

    profile = module.get_profile(species)
    pp = module.PhotonPressure(profile, system_obj.star)

    ncol_test = np.array([
        system_obj.planet.slant_column_density(z_test).to_value(1 / u.cm**2)
    ]) / u.cm**2
    r_test = system_obj.planet.radius.to(u.cm) + z_test.to(u.cm)

    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(
        ncol_test,
        planet_case["T"],
        system_obj.distance,
    )
    beta_species, beta_err = pp.beta_Values(
        F_ph_tot,
        F_ph_tot_err,
        system_obj.planet.mass,
        r_test,
    )

    print("HITRAN test completed successfully.")
    print(f"z_test = {z_test.to_value(u.km):.3f} km")
    print(f"Ncol = {float(np.squeeze(ncol_test.to_value(1 / u.cm**2))):.6e} 1/cm^2")
    print(f"beta = {float(np.squeeze(beta_species.value)):.6e}")
    print(f"beta_err = {float(np.squeeze(beta_err.value)):.6e}")


if __name__ == "__main__":
    main()