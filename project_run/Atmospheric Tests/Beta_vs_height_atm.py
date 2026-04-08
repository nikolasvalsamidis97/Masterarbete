
import sys
import pathlib
import importlib.util

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from astropy import units as u


# -----------------------------------------------------------------------------
# Beta vs height for one planet-star-distance configuration
# Reuses the same optimized profile/template logic as Teff_planet_beta1_dist.py
# and saves all selected species for the configuration in one txt table.
# -----------------------------------------------------------------------------

PLANET_KEY = "55_Cnc_e"
STAR_KEY = "G8"
DISTANCE_AU = 0.01544

# None -> use all species present in the selected planet composition, after skip
# filters below. Otherwise give a list, for example ["Na I", "K I", "CO"].
SELECTED_SPECIES = None

SKIP_ATOMS = False
SKIP_MOLECULES = True

# Height grid from the planet surface up to the Hill limit.
HEIGHT_POINTS = 400
HEIGHT_GRID_POWER = 2.0

# Save only txt table by default. Set True if you also want a quick PDF preview.
SAVE_PREVIEW_PDF = False

PRINT_TRACEBACKS = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_teff_module():
    script_path = pathlib.Path(__file__).resolve().parent / "Teff_planet_beta1_dist.py"
    spec = importlib.util.spec_from_file_location("teff_planet_beta1_dist_reuse", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def species_is_atom(module, species: str) -> bool:
    return species in set(module.ATOM_SPECIES)


def species_is_molecule(module, species: str) -> bool:
    return species in set(module.MOLECULE_TEMPLATES.keys())


def species_matches_run_filters(module, species: str) -> bool:
    if SKIP_ATOMS and species_is_atom(module, species):
        return False
    if SKIP_MOLECULES and species_is_molecule(module, species):
        return False
    return True


def resolve_species_list(module, composition_species):
    if SELECTED_SPECIES is None:
        requested_species = [
            species
            for species in composition_species
            if species_matches_run_filters(module, species)
        ]
        print(
            f"No species specified for {PLANET_KEY}; using filtered composition species: {requested_species}"
        )
    else:
        requested_species = [
            species
            for species in SELECTED_SPECIES
            if species in composition_species and species_matches_run_filters(module, species)
        ]
        print(f"Using explicitly selected species: {requested_species}")

    if not requested_species:
        raise ValueError("No species selected after applying composition and skip filters.")

    return requested_species


def build_height_grid(system_obj, n_points: int, grid_power: float):
    hill_radius = system_obj.hill_radius().to(u.cm)
    planet_radius = system_obj.planet.radius.to(u.cm)
    if hill_radius <= planet_radius:
        raise ValueError("Hill radius is not larger than planet radius.")

    z_max_cm = (hill_radius - planet_radius).to_value(u.cm)
    fraction = np.linspace(0.0, 1.0, n_points) ** grid_power
    z_grid = (fraction * z_max_cm) * u.cm
    return z_grid.to(u.km), hill_radius.to(u.km), planet_radius.to(u.km)


def save_beta_table(table_path, metadata, x_values_km, species_columns):
    lines = []
    for key, value in metadata.items():
        lines.append(f"# {key}: {value}")
    lines.append("#")

    header = ["x__Height"] + [f"beta__{species.replace(' ', '_')}" for species in species_columns.keys()]
    lines.append("\t".join(header))

    n_rows = len(x_values_km)
    for i in range(n_rows):
        row = [f"{x_values_km[i]:.9g}"]
        for values in species_columns.values():
            value = values[i]
            if np.isfinite(value):
                row.append(f"{value:.9g}")
            else:
                row.append("nan")
        lines.append("\t".join(row))

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved table to {table_path}")


def save_preview_pdf(module, pdf_path, z_values_km, species_columns, system_label):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for species, values in species_columns.items():
        ax.plot(z_values_km, values, linewidth=1.0, label=species)

    ax.axhline(1.0, linestyle="-", color="gray", linewidth=1, label=r"$\beta = 1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Height [km]")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(f"Beta vs height | {system_label}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview plot to {pdf_path}")


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------

def main():
    module = load_teff_module()

    planet_case = module.get_planet_template(PLANET_KEY)
    star = module.get_star(STAR_KEY)
    planet_obj = module.Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )
    distance = DISTANCE_AU * u.AU
    system_obj = module.PlanetarySystem(planet_obj, star, distance)

    composition_species = list(planet_case["composition"].keys())
    requested_species = resolve_species_list(module, composition_species)

    z_grid_km, hill_radius_km, planet_radius_km = build_height_grid(
        system_obj,
        n_points=HEIGHT_POINTS,
        grid_power=HEIGHT_GRID_POWER,
    )
    z_grid_cm = z_grid_km.to(u.cm)

    ncol_z = np.array([
        system_obj.planet.slant_column_density(z_i).to_value(1 / u.cm**2)
        for z_i in z_grid_cm
    ]) / u.cm**2
    r_values = system_obj.planet.radius.to(u.cm) + z_grid_cm

    species_columns = {}
    used_species = []

    for species in requested_species:
        try:
            print(
                f"species={species}, planet={PLANET_KEY}, star={STAR_KEY}, "
                f"temp_atm={planet_case['T']}, distance={distance}"
            )
            profile = module.get_profile(species)
            pp = module.PhotonPressure(profile, system_obj.star)

            F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(
                ncol_z,
                planet_case["T"],
                system_obj.distance,
            )
            beta_species, _ = pp.beta_Values(
                F_ph_tot,
                F_ph_tot_err,
                system_obj.planet.mass,
                r_values,
            )
            beta_values = np.asarray(beta_species.value).reshape(-1).astype(float)

            species_columns[species] = beta_values
            used_species.append(species)
        except Exception as exc:
            print(f"Failed for species {species}: {exc}")
            if PRINT_TRACEBACKS:
                import traceback
                traceback.print_exc()
            species_columns[species] = np.full(len(z_grid_km), np.nan, dtype=float)

    if not used_species:
        raise RuntimeError("No species completed successfully.")

    planet_safe = module.safe_name(PLANET_KEY)
    star_safe = module.safe_name(STAR_KEY)
    distance_safe = module.safe_name(f"{DISTANCE_AU:g}AU")

    output_dir = (
        pathlib.Path(__file__).resolve().parents[2]
        / "Plots"
        / "Atmospheric test"
        / "Beta_vs_height_system"
    )
    table_path = output_dir / f"{planet_safe}_{star_safe}_{distance_safe}_beta_vs_height.txt"

    metadata = {
        "dataset_name": f"{planet_safe}_{star_safe}_{distance_safe}_beta_vs_height",
        "x_label": "Height",
        "x_unit": "km",
        "y_label": "beta",
        "y_unit": "dimensionless",
        "series_label": "species",
        "series_unit": "",
        "series_values": ", ".join(species_columns.keys()),
        "planet": PLANET_KEY,
        "star": STAR_KEY,
        "distance_AU": DISTANCE_AU,
        "planet_radius_Rjup": (system_obj.planet.radius / module.const.R_jup).decompose().value,
        "planet_mass_Mjup": (system_obj.planet.mass / module.const.M_jup).decompose().value,
        "planet_temperature_K": planet_case["T"].to_value(u.K),
        "planet_mu": planet_case["mu"].value,
        "hill_radius_km": hill_radius_km.to_value(u.km),
        "planet_radius_km": planet_radius_km.to_value(u.km),
        "height_grid_points": HEIGHT_POINTS,
        "height_grid_power": HEIGHT_GRID_POWER,
        "used_species": ", ".join(used_species),
    }

    save_beta_table(
        table_path,
        metadata,
        z_grid_km.to_value(u.km),
        species_columns,
    )

    if SAVE_PREVIEW_PDF:
        pdf_path = table_path.with_suffix(".pdf")
        save_preview_pdf(
            module,
            pdf_path,
            z_grid_km.to_value(u.km),
            species_columns,
            system_label=f"{PLANET_KEY} | {STAR_KEY} | {DISTANCE_AU:g} AU",
        )


if __name__ == "__main__":
    main()