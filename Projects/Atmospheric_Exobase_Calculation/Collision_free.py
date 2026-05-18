import os
import pathlib
import sys

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Planet import Planet
from project_utils.exobase_table_path import (
    canonical_exobase_table_path,
    legacy_exobase_table_path,
)
from Templates.Atoms.atom_species import ATOM_SPECIES
from Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from Templates.Planets.planet_templates import PLANET_TEMPLATES, get_planet_template


# -----------------------------------------------------------------------------
# Exobase calculation for all template planets
# -----------------------------------------------------------------------------
# This script computes the exobase height for every supported neutral atom and
# molecule across the full template catalog, for every planet template. The
# exobase is defined here in the same way as in the older script: the height
# where the mean free path is comparable to the local scale height,
#
#     lambda_mfp(z) = H(z)
#
# with
#
#     lambda_mfp = 1 / sum_j [ n_j(z) * sigma_ij ]
#
# using hard-sphere collision cross sections
#
#     sigma_ij = pi (r_i + r_j)^2 .
#
# The search domain is Hill-limited, as in the older script.

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_PATH = canonical_exobase_table_path(REPO_ROOT)
LEGACY_OUTPUT_PATH = legacy_exobase_table_path(REPO_ROOT)
N_Z = 10000
Z_MAX_SCALE_HEIGHTS = 100.0
Z_MAX_MIN = 1000.0 * u.km


# -----------------------------------------------------------------------------
# Neutral-species properties (representative hard-sphere radii and masses)
# -----------------------------------------------------------------------------
# Radii are approximate effective neutral hard-sphere radii in Angstrom.
# Masses are in atomic mass units.
ATOMIC_PROPERTIES = {
    "H I": {"radius": 1.20 * u.AA, "mass_u": 1.008},
    "He I": {"radius": 1.43 * u.AA, "mass_u": 4.0026},
    "Li I": {"radius": 1.82 * u.AA, "mass_u": 6.94},
    "Be I": {"radius": 1.53 * u.AA, "mass_u": 9.0122},
    "B I": {"radius": 1.92 * u.AA, "mass_u": 10.81},
    "C I": {"radius": 1.77 * u.AA, "mass_u": 12.011},
    "N I": {"radius": 1.66 * u.AA, "mass_u": 14.007},
    "O I": {"radius": 1.50 * u.AA, "mass_u": 15.999},
    "F I": {"radius": 1.47 * u.AA, "mass_u": 18.998403},
    "Ne I": {"radius": 1.54 * u.AA, "mass_u": 20.1797},
    "Na I": {"radius": 2.50 * u.AA, "mass_u": 22.989769},
    "Mg I": {"radius": 2.51 * u.AA, "mass_u": 24.305},
    "Al I": {"radius": 2.25 * u.AA, "mass_u": 26.981538},
    "Si I": {"radius": 2.19 * u.AA, "mass_u": 28.085},
    "P I": {"radius": 1.90 * u.AA, "mass_u": 30.973762},
    "S I": {"radius": 1.89 * u.AA, "mass_u": 32.06},
    "Cl I": {"radius": 1.75 * u.AA, "mass_u": 35.45},
    "Ar I": {"radius": 1.88 * u.AA, "mass_u": 39.948},
    "K I": {"radius": 2.73 * u.AA, "mass_u": 39.0983},
    "Ca I": {"radius": 2.62 * u.AA, "mass_u": 40.078},
    "Sc I": {"radius": 2.58 * u.AA, "mass_u": 44.955908},
    "Ti I": {"radius": 2.46 * u.AA, "mass_u": 47.867},
    "V I": {"radius": 2.42 * u.AA, "mass_u": 50.9415},
    "Cr I": {"radius": 2.45 * u.AA, "mass_u": 51.9961},
    "Mn I": {"radius": 2.45 * u.AA, "mass_u": 54.938044},
    "Fe I": {"radius": 2.44 * u.AA, "mass_u": 55.845},
}

MOLECULAR_PROPERTIES = {
    "H2": {"radius": 1.45 * u.AA, "mass_u": 2.01588},
    "N2": {"radius": 1.82 * u.AA, "mass_u": 28.0134},
    "O2": {"radius": 2.05 * u.AA, "mass_u": 31.9988},
    "OH": {"radius": 1.75 * u.AA, "mass_u": 17.00734},
    "H2O": {"radius": 2.75 * u.AA, "mass_u": 18.01528},
    "CO": {"radius": 2.10 * u.AA, "mass_u": 28.0101},
    "CO2": {"radius": 2.30 * u.AA, "mass_u": 44.0095},
    "NO": {"radius": 2.15 * u.AA, "mass_u": 30.0061},
    "SO": {"radius": 2.20 * u.AA, "mass_u": 48.059},
    "CH4": {"radius": 2.00 * u.AA, "mass_u": 16.04246},
    "NH3": {"radius": 2.20 * u.AA, "mass_u": 17.031},
    "TiO": {"radius": 2.30 * u.AA, "mass_u": 63.866},
    "SiO": {"radius": 2.20 * u.AA, "mass_u": 44.084},
    "NaCl": {"radius": 2.80 * u.AA, "mass_u": 58.443},
    "PH3": {"radius": 2.35 * u.AA, "mass_u": 33.998},
    "H2S": {"radius": 2.90 * u.AA, "mass_u": 34.081},
    "SO2": {"radius": 2.65 * u.AA, "mass_u": 64.066},
    "HCN": {"radius": 2.25 * u.AA, "mass_u": 27.026},
    "C3": {"radius": 2.40 * u.AA, "mass_u": 36.033},
    "OCS": {"radius": 2.55 * u.AA, "mass_u": 60.075},
}


SPECIES_PROPERTIES = {}
SPECIES_PROPERTIES.update(ATOMIC_PROPERTIES)
SPECIES_PROPERTIES.update(MOLECULAR_PROPERTIES)


def get_species_radius(species: str) -> u.Quantity:
    """Return the hard-sphere radius for a neutral species."""
    if species not in SPECIES_PROPERTIES:
        raise KeyError(
            f"No hard-sphere radius is defined for species '{species}'. "
            "Add it to ATOMIC_PROPERTIES or MOLECULAR_PROPERTIES."
        )
    return SPECIES_PROPERTIES[species]["radius"].to(u.cm)


def get_species_mass(species: str) -> u.Quantity:
    """Return the particle mass for a neutral species."""
    if species not in SPECIES_PROPERTIES:
        raise KeyError(
            f"No mass is defined for species '{species}'. "
            "Add it to ATOMIC_PROPERTIES or MOLECULAR_PROPERTIES."
        )
    return (SPECIES_PROPERTIES[species]["mass_u"] * const.u).to(u.g)


def collision_cross_section(species_1: str, species_2: str) -> u.Quantity:
    """Hard-sphere collision cross section sigma_ij = pi (r_i + r_j)^2."""
    r1 = get_species_radius(species_1)
    r2 = get_species_radius(species_2)
    return (np.pi * (r1 + r2) ** 2).to(u.cm**2)


def neutral_species_only(composition: dict) -> dict:
    """
    Keep only neutral species present in the composition that have hardcoded
    neutral radii/masses in this script.

    This includes neutral atoms like 'Na I' and neutral molecules like 'H2O'.
    Ionized species like 'Na II' are excluded because this script is only meant
    to evaluate collision-free heights for neutral species.
    """
    selected = {}
    for species, frac in composition.items():
        if species in SPECIES_PROPERTIES:
            selected[species] = frac
    return selected


def all_supported_target_species() -> list[str]:
    """
    Build the full target list for which exobase heights should be evaluated.

    Neutral atoms are taken from the global atom catalog by selecting only stage I
    species. Molecules are taken from the molecule template catalog. Every target
    species must have hard-sphere properties defined in this script.
    """
    neutral_atoms = [species for species in ATOM_SPECIES if species.endswith(" I")]
    molecule_species = list(MOLECULE_TEMPLATES.keys())
    target_species = list(dict.fromkeys(neutral_atoms + molecule_species))

    missing = [species for species in target_species if species not in SPECIES_PROPERTIES]
    if missing:
        raise KeyError(
            "The following target species are missing hard-sphere radii/masses in "
            f"Collision_free.py: {missing}"
        )
    return target_species


# -----------------------------------------------------------------------------
# Exobase calculation
# -----------------------------------------------------------------------------
def calc_exobase_for_planet(template_name: str, n_z: int = N_Z):
    """
    For one planet template, calculate the exobase height for each neutral species.

    Exobase condition:
        lambda_mfp(z) ~ H(z)
    implemented by selecting the height where |lambda_mfp / H - 1| is minimal.

    No stellar or orbital information is used here; this is purely an
    atmospheric calculation in the planet's own atmosphere.
    """
    params = get_planet_template(template_name)
    planet = Planet(
        params["radius"],
        params["mass"],
        params["T"],
        params["mu"],
        params["P0"],
    )

    composition = neutral_species_only(params["composition"])

    if not composition:
        return []

    target_species = all_supported_target_species()

    # Use a purely planetary search grid based on the atmospheric scale height.
    H0 = planet.scale_height(0 * u.km).to(u.km)
    z_max = max(Z_MAX_MIN, Z_MAX_SCALE_HEIGHTS * H0)
    z = np.linspace(0.0, z_max.to_value(u.km), n_z) * u.km

    n_total = np.array([
        planet.number_density(zi).to_value(1 / u.cm**3) for zi in z
    ]) / u.cm**3

    H_local = np.array([
        planet.scale_height(zi).to_value(u.cm) for zi in z
    ]) * u.cm

    results = []

    for species_i in target_species:
        collision_rate = np.zeros_like(n_total.value) / u.cm

        for species_j, frac_j in composition.items():
            sigma_ij = collision_cross_section(species_i, species_j)
            n_j = frac_j * n_total
            collision_rate += n_j * sigma_ij

        lambda_mfp = (1 / collision_rate).to(u.cm)
        knudsen = (lambda_mfp / H_local).decompose()
        idx = int(np.nanargmin(np.abs(knudsen.value - 1.0)))

        results.append({
            "planet_name": template_name,
            "species": species_i,
            "z_collision_free": z[idx].to(u.km),
            "knudsen_at_height": knudsen[idx],
            "lambda_mfp_at_height": lambda_mfp[idx].to(u.km),
            "mass_amu": (get_species_mass(species_i) / const.u.to(u.g)).value,
            "radius_AA": get_species_radius(species_i).to_value(u.AA),
            "z_max_km": z_max.to_value(u.km),
            "H0_km": H0.to_value(u.km),
        })

    return results


# -----------------------------------------------------------------------------
# Run for all template planets and save table
# -----------------------------------------------------------------------------
def main() -> None:
    all_results = []

    for template_name in PLANET_TEMPLATES.keys():
        planet_results = calc_exobase_for_planet(template_name, n_z=N_Z)
        all_results.extend(planet_results)

    rows = []
    for result in all_results:
        rows.append({
            "planet": result["planet_name"],
            "species": result["species"],
            "z_exobase_km": result["z_collision_free"].to_value(u.km),
            "knudsen": result["knudsen_at_height"].value,
            "lambda_mfp_km": result["lambda_mfp_at_height"].to_value(u.km),
            "mass_amu": result["mass_amu"],
            "radius_AA": result["radius_AA"],
            "z_max_km": result["z_max_km"],
            "H0_km": result["H0_km"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No exobase results were produced.")

    df = df.sort_values(["planet", "z_exobase_km", "species"]).reset_index(drop=True)
    df["z_exobase_km"] = df["z_exobase_km"].round(0).astype(int)
    df["knudsen"] = df["knudsen"].round(2)
    df["lambda_mfp_km"] = df["lambda_mfp_km"].round(0).astype(int)
    df["mass_amu"] = df["mass_amu"].round(3)
    df["radius_AA"] = df["radius_AA"].round(2)
    df["z_max_km"] = df["z_max_km"].round(0).astype(int)
    df["H0_km"] = df["H0_km"].round(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved exobase table to: {OUTPUT_PATH}")

    LEGACY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEGACY_OUTPUT_PATH, index=False)
    print(f"Saved legacy exobase table to: {LEGACY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
