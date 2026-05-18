import sys
import pathlib
import importlib.util
import numpy as np
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))


def load_teff_module():
    script_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "Atmospheric_Teff_Beta1_Distance_Study"
        / "Teff_planet_beta1_dist.py"
    )
    spec = importlib.util.spec_from_file_location("teff_planet_beta1_dist_reuse", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_system(module, planet_key: str, star_key: str, distance_au: float):
    if planet_key == "55_Cnc_e":
        planet_case = {
            "label": "55 Cnc e (temporary example)",
            "category": "rocky",
            "radius": 1.875 * const.R_earth,
            "mass": 7.99 * const.M_earth,
            "T": 2000 * u.K,
            "mu": 44.0 * u.dimensionless_unscaled,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "O I": 0.50,
                "O II": 0.15,
                "N I": 0.18,
                "N II": 0.02,
                "Na I": 0.08,
                "Na II": 0.02,
                "K I": 0.04,
                "K II": 0.01,
                "CO": 0.02,
                "NO": 0.01,
            },
        }
    else:
        planet_case = module.get_planet_template(planet_key)

    planet_obj = module.Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )
    star = module.get_star(star_key)
    system_obj = module.PlanetarySystem(planet_obj, star, distance_au * u.AU)
    return planet_case, system_obj


def compute_beta_profile(module, species: str, planet_case, system_obj, n_points: int = 120):
    hill_radius = system_obj.hill_radius().to(u.cm)
    planet_radius = system_obj.planet.radius.to(u.cm)
    if hill_radius <= planet_radius:
        raise ValueError("Hill radius is not larger than planet radius.")

    z_max_cm = (hill_radius - planet_radius).to_value(u.cm)
    fraction = np.linspace(0.0, 1.0, n_points) ** 2.0
    z_grid = (fraction * z_max_cm) * u.cm

    ncol_grid = np.array([
        system_obj.planet.slant_column_density(z_i).to_value(1 / u.cm**2)
        for z_i in z_grid
    ]) / u.cm**2
    r_grid = planet_radius + z_grid

    profile = module.get_profile(species)
    pp = module.PhotonPressure(profile, system_obj.star)
    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(
        ncol_grid,
        planet_case["T"],
        system_obj.distance,
    )
    beta_species, beta_err = pp.beta_Values(
        F_ph_tot,
        F_ph_tot_err,
        system_obj.planet.mass,
        r_grid,
    )

    return {
        "z_km": z_grid.to_value(u.km),
        "r_over_rp": (r_grid / planet_radius).decompose().value,
        "beta": np.asarray(beta_species.value, dtype=float).reshape(-1),
        "beta_err": np.asarray(beta_err.value, dtype=float).reshape(-1),
        "hill_height_km": (hill_radius - planet_radius).to_value(u.km),
    }


def save_validation_table(output_path: pathlib.Path, planet_key: str, star_key: str, distance_au: float, fe_data, no_data):
    lines = [
        "# dataset_name: Fe_NO_beta_validation",
        f"# planet: {planet_key}",
        f"# star: {star_key}",
        f"# distance_AU: {distance_au}",
        "# x_label: Height",
        "# x_unit: km",
        "# y_label: beta",
        "# y_unit: dimensionless",
        "# series_label: species",
        "# series_values: Fe I, NO",
        "#",
        "x__Height_km\tbeta__Fe_I\tbeta__NO\tr_over_rp__Fe_I\tr_over_rp__NO",
    ]

    n_rows = min(len(fe_data["z_km"]), len(no_data["z_km"]))
    for i in range(n_rows):
        lines.append(
            f"{fe_data['z_km'][i]:.9g}\t"
            f"{fe_data['beta'][i]:.9g}\t"
            f"{no_data['beta'][i]:.9g}\t"
            f"{fe_data['r_over_rp'][i]:.9g}\t"
            f"{no_data['r_over_rp'][i]:.9g}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved validation table to {output_path}")


def make_plot(output_pdf: pathlib.Path, planet_key: str, star_key: str, distance_au: float, fe_data, no_data):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(fe_data["z_km"], fe_data["beta"], linewidth=1.6, label="Fe I")
    ax.plot(no_data["z_km"], no_data["beta"], linewidth=1.6, label="NO")
    ax.axhline(1.0, linestyle="-", color="0.45", linewidth=1.2, label=r"$\beta = 1$")
    ax.axvline(fe_data["hill_height_km"], linestyle="-.", linewidth=1.2, label="Hill limit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Height [km]")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(f"Fe I vs NO validation | {planet_key.replace('_', ' ')} | {star_key} | {distance_au:g} AU")
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation plot to {output_pdf}")


def main():
    module = load_teff_module()

    planet_key = "55_Cnc_e"
    star_key = "G8"
    distance_au = 0.01544

    print("Running Fe I vs NO validation test")
    print(f"planet={planet_key}, star={star_key}, distance={distance_au} AU")

    planet_case, system_obj = build_system(module, planet_key, star_key, distance_au)

    fe_data = compute_beta_profile(module, "Fe I", planet_case, system_obj)
    no_data = compute_beta_profile(module, "NO", planet_case, system_obj)

    output_dir = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Beta_validation"
    txt_path = output_dir / "55_Cnc_e_G8_0.01544AU_FeI_vs_NO_validation.txt"
    pdf_path = output_dir / "55_Cnc_e_G8_0.01544AU_FeI_vs_NO_validation.pdf"

    save_validation_table(txt_path, planet_key, star_key, distance_au, fe_data, no_data)
    make_plot(pdf_path, planet_key, star_key, distance_au, fe_data, no_data)

    print(f"Fe I max beta = {np.nanmax(fe_data['beta']):.6e}")
    print(f"NO   max beta = {np.nanmax(no_data['beta']):.6e}")


if __name__ == "__main__":
    main()
