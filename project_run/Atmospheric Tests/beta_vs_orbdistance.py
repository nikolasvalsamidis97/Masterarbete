import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
from scipy.integrate import trapezoid

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem


planet_cases = {
    "Mercury": {
        "radius": 2440 * u.km,
        "mass": 3.301e23 * u.kg,
        "T": 440 * u.K,
        "mu": 23.0 * u.dimensionless_unscaled,
        "P0": 1.0e-9 * u.bar,
        "composition": {"H I": 0.22, "He I": 0.06, "O I": 0.42, "Na I": 0.29, "K I": 0.01},
    },
    "Earth": {
        "radius": 1.0 * const.R_earth,
        "mass": 1.0 * const.M_earth,
        "T": 288 * u.K,
        "mu": 28.97 * u.dimensionless_unscaled,
        "P0": 1.0 * u.bar,
        "composition": {"N I": 0.7808, "O I": 0.2095, "He I": 5.24e-6, "H I": 5.5e-7, "Na I": 1e-9, "K I": 1e-10},
    },
    "HAT_P_11_b": {
        "radius": 4.84 * const.R_earth,
        "mass": 25.0 * const.M_earth,
        "T": 880 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {"H I": 0.85, "He I": 0.14, "O I": 1e-3, "Na I": 1e-5, "K I": 1e-6},
    },
    "HD_209458_b": {
        "radius": 1.39 * const.R_jup,
        "mass": 0.73 * const.M_jup,
        "T": 1400 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "H I": 0.8997989,
            "He I": 0.1,
            "CO": 1e-4,
            "O I": 1e-4,
            "Na I": 1e-6,
            "K I": 1e-7,
        },
    },
    "WASP_121_b": {
        "radius": 1.742 * const.R_jup,
        "mass": 1.17 * const.M_jup,
        "T": 2350 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-7 * u.bar,
        "composition": {
            "H I": 0.880899,
            "He I": 0.09,
            "He II": 0.01,
            "O I": 0.015,
            "O II": 0.003,
            "Na I": 1.5e-3,
            "Na II": 3.5e-4,
            "K I": 1.2e-4,
            "K II": 3.0e-5,
            "CO": 1e-4,
            "NO": 1e-6,
        },
    },
}

stellar_models = {
    "G1": {
        "path": "TS/Spectral_type/G/G1/lte058-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
        "radius": 1.05 * const.R_sun,
        "mass": 1.05 * const.M_sun,
        "vsini": 5 * u.km / u.s,
        "epsilon": 0.6 * u.dimensionless_unscaled,
    },
    "K1": {
        "path": "TS/Spectral_type/K/K1/lte050-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
        "radius": 0.82 * const.R_sun,
        "mass": 0.82 * const.M_sun,
        "vsini": 3 * u.km / u.s,
        "epsilon": 0.6 * u.dimensionless_unscaled,
    },
    "F0": {
        "path": "TS/Spectral_type/F/F0/lte072-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
        "radius": 1.6 * const.R_sun,
        "mass": 1.6 * const.M_sun,
        "vsini": 20 * u.km / u.s,
        "epsilon": 0.6 * u.dimensionless_unscaled,
    },
}

systems = {
    "Mercury-Sun": {"planet": planet_cases["Mercury"], "star_key": "G1"},
    "Earth-Sun": {"planet": planet_cases["Earth"], "star_key": "G1"},
    "HAT-P-11 b": {"planet": planet_cases["HAT_P_11_b"], "star_key": "K1"},
    "HD 209458 b": {"planet": planet_cases["HD_209458_b"], "star_key": "G1"},
    "WASP-121 b": {"planet": planet_cases["WASP_121_b"], "star_key": "F0"},
}

MOLECULE_FETCH = {
    "CO": {"path": "CO/12C-16O/Li2015", "database": "Li2015"},
    "NO": {"path": "NO/14N-16O/XABC", "database": "XABC"},
}

selected_species = ["H I", "O I", "Na I", "K I"]

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
b_atom = 1 * u.km / u.s
npts_atom = 150
b_molecule = 1 * u.km / u.s
ncol = np.array([0.0]) * u.cm ** -2

star_cache = {}
profile_cache = {}
beta_star_cache = {}
planet_cache = {}


def scalar_value(x):
    if isinstance(x, u.Quantity):
        return float(np.squeeze(x.value))
    return float(np.squeeze(x))


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

def get_planet(system_name):
    if system_name not in planet_cache:
        p = systems[system_name]["planet"]
        planet_cache[system_name] = Planet(
            radius=p["radius"],
            mass=p["mass"],
            T=p["T"],
            mu=p["mu"],
            P0=p["P0"],
        )
    return planet_cache[system_name]


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


def beta_against_stellar_gravity(species, star_key, T_atm):
    cache_key = (species, star_key, float(T_atm.to_value(u.K)))
    if cache_key in beta_star_cache:
        return beta_star_cache[cache_key]

    star = get_star(star_key)
    profile = get_profile(species)
    pp = PhotonPressure(profile, star)
    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(ncol, T_atm, star.radius)
    beta, _ = pp.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, star.radius)
    beta_val = scalar_value(beta)
    beta_star_cache[cache_key] = beta_val
    return beta_val


def mean_exospheric_beta(beta_star_species, system_obj):
    hill_radius = system_obj.hill_radius().to(u.cm)
    z_top = (hill_radius - system_obj.planet.radius.to(u.cm)).to(u.cm)
    if z_top <= 0 * u.cm:
        return np.nan

    z = np.linspace(0.0, z_top.value, 1500) * u.cm
    r = (system_obj.planet.radius.to(u.cm) + z).to(u.cm)

    weights = system_obj.planet.number_density(z).to_value(1 / u.cm**3)

    beta_z = beta_star_species * (system_obj.star.mass / system_obj.planet.mass).decompose().value * (r / system_obj.distance.to(u.cm)) ** 2
    return float(trapezoid(beta_z.value * weights, z.value) / trapezoid(weights, z.value))

def main():
    output_dir = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Atmospheric test"
    output_dir.mkdir(parents=True, exist_ok=True)

    distance_grid = np.linspace(0.02, 2.0, 160) * u.AU

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, species in zip(axes, selected_species):
        for system_name, system in systems.items():
            planet_case = system["planet"]
            star = get_star(system["star_key"])
            planet_obj = get_planet(system_name)

            if species not in planet_case["composition"]:
                continue

            try:
                beta_species_star = beta_against_stellar_gravity(species, system["star_key"], planet_obj.T)
            except Exception as exc:
                print(f"Skipping {species} in {system_name}: {exc}")
                continue

            mean_betas = []
            for a in distance_grid:
                system_obj = PlanetarySystem(planet_obj, star, a)
                mean_beta = mean_exospheric_beta(beta_species_star, system_obj)
                mean_betas.append(mean_beta)

            mean_betas = np.asarray(mean_betas)
            ax.plot(distance_grid.to_value(u.AU), mean_betas, lw=2, label=system_name)

        ax.set_yscale("log")
        ax.set_title(species)
        ax.axhline(1.0, ls="--", lw=1)
        ax.grid(True, alpha=0.25)

    for ax in axes[2:]:
        ax.set_xlabel("Orbital distance [AU]")
    axes[0].set_ylabel(r"Mean exospheric $\beta$")
    axes[2].set_ylabel(r"Mean exospheric $\beta$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Species-specific mean exospheric beta vs orbital distance", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    pdf_path = output_dir / "Exospheric_beta_vs_d.pdf"
    plt.savefig(pdf_path)
    print(f"Saved plot to {pdf_path}")

if __name__ == "__main__":
    main()