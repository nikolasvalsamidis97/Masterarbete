import copy
import re

import astropy.constants as const
import astropy.units as u


ATOMIC_MASSES_AMU = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
}

FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def species_molecular_weight_amu(species: str) -> float:
    formula = str(species).split()[0]
    position = 0
    molecular_weight = 0.0
    for match in FORMULA_TOKEN_RE.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Could not parse species formula '{species}'.")
        element, count_text = match.groups()
        if element not in ATOMIC_MASSES_AMU:
            raise KeyError(f"No atomic mass available for element '{element}' in species '{species}'.")
        molecular_weight += ATOMIC_MASSES_AMU[element] * int(count_text or 1)
        position = match.end()
    if position != len(formula):
        raise ValueError(f"Could not parse species formula '{species}'.")
    return molecular_weight


def mean_molecular_weight_from_composition(composition: dict[str, float]) -> float:
    total_fraction = sum(float(fraction) for fraction in composition.values())
    if total_fraction <= 0.0:
        raise ValueError("Composition fractions must have a positive sum.")
    weighted_mass = sum(
        float(fraction) * species_molecular_weight_amu(species)
        for species, fraction in composition.items()
    )
    return weighted_mass / total_fraction


def rounded_mean_molecular_weight_from_composition(composition: dict[str, float]) -> float:
    return round(mean_molecular_weight_from_composition(composition), 2)


def assign_composition_mean_molecular_weights(systems: dict) -> None:
    for system_def in systems.values():
        planet = system_def["planet"]
        planet["mu"] = (
            rounded_mean_molecular_weight_from_composition(planet["composition"])
            * u.dimensionless_unscaled
        )


REAL_MASS_LOSS_REFERENCE_SYSTEMS = {
    "gj1132_b": {
        "system_name": "GJ 1132 b",
        "category": "rocky",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201132",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201132",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201132",
        "spectrum_template_key": "M4",
        "exobase_template_key": "super_earth_rocky",
        "star": {
            "label": "GJ 1132",
            "path": "Templates/TS/Spectral_type/M/M4/lte032-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
            "teff_K": 3229.0,
            "radius": 0.2211 * const.R_sun,
            "mass": 0.1945 * const.M_sun,
            "vsini": 2.0 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "GJ 1132 b",
            "radius": 1.192 * const.R_earth,
            "mass": 1.84 * const.M_earth,
            "T": 583.8 * u.K,
            "P0": 1.0 * u.bar,
            "composition": {
                "H2": 0.55,
                "H I": 0.20,
                "HCN": 0.20,
                "N I": 0.03,
                "C I": 0.02,
            },
            "notes": (
                "Real rocky comparison system. Composition is a normalized proxy based on the "
                "claimed H2/HCN secondary-atmosphere interpretation, with the later JWST "
                "non-detection noted as an uncertainty."
            ),
        },
        "distance_au": 0.01570,
    },
    "gj1214_b": {
        "system_name": "GJ 1214 b",
        "category": "sub_neptune",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201214",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201214",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201214",
        "spectrum_template_key": "M6",
        "exobase_template_key": "sub_neptune",
        "star": {
            "label": "GJ 1214",
            "path": "Templates/TS/Spectral_type/M/M6/lte030-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 3101.0,
            "radius": 0.2162 * const.R_sun,
            "mass": 0.1820 * const.M_sun,
            "vsini": 2.0 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "GJ 1214 b",
            "radius": 2.733 * const.R_earth,
            "mass": 8.41 * const.M_earth,
            "T": 567.0 * u.K,
            "P0": 1.0e-4 * u.bar,
            "composition": {
                "H2": 0.40,
                "He I": 0.10,
                "H2O": 0.30,
                "CO2": 0.10,
                "H I": 0.05,
                "CO": 0.03,
                "O I": 0.01,
                "Na I": 0.01,
            },
            "notes": (
                "Real sub-Neptune comparison system. Composition is a normalized proxy for a "
                "high-metallicity, cloudy/hazy atmosphere with H2O as the main inferred absorber."
            ),
        },
        "distance_au": 0.01505,
    },
    "gj436_b": {
        "system_name": "GJ 436 b",
        "category": "neptune",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20436%20b",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20436",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20436%20b",
        "spectrum_template_key": "M1",
        "exobase_template_key": "hot_neptune",
        "star": {
            "label": "GJ 436",
            "path": "Templates/TS/Spectral_type/M/M1/lte036-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
            "teff_K": 3500.0,
            "radius": 0.422 * const.R_sun,
            "mass": 0.445 * const.M_sun,
            "vsini": 0.33 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "GJ 436 b",
            "radius": 4.17 * const.R_earth,
            "mass": 22.1 * const.M_earth,
            "T": 686.0 * u.K,
            "P0": 1.0e-4 * u.bar,
            "composition": {
                "H2": 0.30,
                "H I": 0.20,
                "He I": 0.10,
                "CO": 0.25,
                "H2O": 0.08,
                "CO2": 0.04,
                "O I": 0.02,
                "Na I": 0.01,
            },
            "notes": (
                "Real Neptune comparison system. Composition is a normalized proxy for a "
                "high-metallicity, CO-rich, methane-poor atmosphere with an escaping hydrogen component."
            ),
        },
        "distance_au": 0.0291,
    },
    "hd209458_b": {
        "system_name": "HD 209458 b",
        "category": "gas_giant",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/HD%20209458%20b",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/HD%20209458%20b",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/HD%20209458%20b",
        "spectrum_template_key": "F8",
        "exobase_template_key": "inflated_hot_jupiter",
        "star": {
            "label": "HD 209458",
            "path": "Templates/TS/Spectral_type/F/F8/lte060-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
            "teff_K": 6026.35,
            "radius": 1.199976 * const.R_sun,
            "mass": 1.069175 * const.M_sun,
            "vsini": 4.49 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "HD 209458 b",
            "radius": 1.39 * const.R_jup,
            "mass": 0.73 * const.M_jup,
            "T": 1448.0 * u.K,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H2": 0.70,
                "He I": 0.15,
                "H I": 0.10,
                "H2O": 0.03,
                "Na I": 0.01,
                "NH3": 0.005,
                "HCN": 0.005,
            },
            "notes": (
                "Real gas-giant comparison system. Composition is a normalized hot-Jupiter proxy "
                "based on H2/He dominance, sodium absorption, escaping hydrogen, H2O, and possible "
                "nitrogen chemistry."
            ),
        },
        "distance_au": 0.04634,
    },
    "55cnc_e": {
        "system_name": "55 Cnc e",
        "category": "rocky",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/55%20Cnc%20e",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/55%20Cnc",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/55%20Cnc%20e",
        "spectrum_template_key": "G8",
        "exobase_template_key": "lava_world",
        "star": {
            "label": "55 Cnc",
            "path": "Templates/TS/Spectral_type/G/G8/lte052-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 5172.0,
            "radius": 0.943 * const.R_sun,
            "mass": 0.905 * const.M_sun,
            "vsini": 1.06 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "55 Cnc e",
            "radius": 1.875 * const.R_earth,
            "mass": 7.99 * const.M_earth,
            "T": 1958.0 * u.K,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "CO2": 0.45,
                "CO": 0.35,
                "H2O": 0.08,
                "H I": 0.04,
                "O I": 0.04,
                "HCN": 0.02,
                "SiO": 0.02,
            },
            "notes": (
                "Real ultra-hot rocky comparison system. Composition is a normalized proxy for a "
                "secondary volatile atmosphere rich in CO2 or CO, not a primordial H2/He atmosphere."
            ),
        },
        "distance_au": 0.01544,
    },
    "hd56414_b": {
        "system_name": "HD 56414 b",
        "category": "sub_neptune",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/HD%2056414%20b",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/HD%2056414",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/HD%2056414%20b",
        "spectrum_template_key": "A4",
        "exobase_template_key": "sub_neptune",
        "star": {
            "label": "HD 56414",
            "path": "Templates/TS/Spectral_type/A/A4/lte086-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 8500.0,
            "radius": 1.751 * const.R_sun,
            "mass": 1.89 * const.M_sun,
            "vsini": 59.4 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "HD 56414 b",
            "radius": 3.71 * const.R_earth,
            "mass": 13.8 * const.M_earth,
            "T": 1133.0 * u.K,
            "P0": 1.0e-4 * u.bar,
            "composition": {
                "H2": 0.65,
                "He I": 0.15,
                "H I": 0.10,
                "H2O": 0.05,
                "CO": 0.03,
                "O I": 0.01,
                "Na I": 0.01,
            },
            "notes": (
                "Real irradiated sub-Neptune comparison system. Composition is a normalized light-atmosphere "
                "proxy for a Neptune-size planet expected to retain most of its atmosphere."
            ),
        },
        "distance_au": 0.229,
    },
    "wasp193_b": {
        "system_name": "WASP-193 b",
        "category": "gas_giant",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/WASP-193%20b",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/WASP-193",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/WASP-193%20b",
        "spectrum_template_key": "F8",
        "exobase_template_key": "inflated_hot_jupiter",
        "star": {
            "label": "WASP-193",
            "path": "Templates/TS/Spectral_type/F/F8/lte060-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
            "teff_K": 6080.0,
            "radius": 1.213 * const.R_sun,
            "mass": 1.018 * const.M_sun,
            "vsini": 5.11 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "WASP-193 b",
            "radius": 1.319 * const.R_jup,
            "mass": 0.112 * const.M_jup,
            "T": 1250.0 * u.K,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H2": 0.75,
                "He I": 0.15,
                "H I": 0.08,
                "Na I": 0.01,
                "H2O": 0.01,
            },
            "notes": (
                "Real inflated hot-Jupiter comparison system. Composition is a normalized H2/He proxy "
                "for an extremely low-density planet with an extended light atmosphere."
            ),
        },
        "distance_au": 0.0668,
    },
    "wasp174_b": {
        "system_name": "WASP-174 b",
        "category": "gas_giant",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/WASP-174%20b",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/WASP-174",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/WASP-174%20b",
        "spectrum_template_key": "F5",
        "exobase_template_key": "inflated_hot_jupiter",
        "star": {
            "label": "WASP-174",
            "path": "Templates/TS/Spectral_type/F/F5/lte064-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 6399.0,
            "radius": 1.347 * const.R_sun,
            "mass": 1.24 * const.M_sun,
            "vsini": 16.24 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "WASP-174 b",
            "radius": 1.437 * const.R_jup,
            "mass": 0.33 * const.M_jup,
            "T": 1528.0 * u.K,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H2": 0.72,
                "He I": 0.15,
                "H I": 0.08,
                "H2O": 0.03,
                "Na I": 0.01,
                "O I": 0.01,
            },
            "notes": (
                "Real inflated hot-Jupiter comparison system. Composition is a normalized H2/He hot-Jupiter "
                "proxy for a highly inflated giant planet."
            ),
        },
        "distance_au": 0.05503,
    },
    "51peg_b": {
        "system_name": "51 Peg b",
        "category": "gas_giant",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/51%20Peg%20b",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/51%20Peg",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/51%20Peg%20b",
        "spectrum_template_key": "G1",
        "exobase_template_key": "inflated_hot_jupiter",
        "star": {
            "label": "51 Peg",
            "path": "Templates/TS/Spectral_type/G/G1/lte058-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 5760.76,
            "radius": 1.186744 * const.R_sun,
            "mass": 1.069296 * const.M_sun,
            "vsini": 2.6 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "51 Peg b",
            "radius": 1.2 * const.R_jup,
            "mass": 0.46 * const.M_jup,
            "T": 1250.0 * u.K,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H2": 0.72,
                "He I": 0.15,
                "H I": 0.08,
                "H2O": 0.04,
                "O I": 0.005,
                "Na I": 0.005,
            },
            "notes": (
                "Real hot-Jupiter comparison system. Source-based mass/radius/orbit from the Exoplanet Archive; "
                "temperature adopts the dayside value from the 51 Pegasi b literature. Composition is a normalized "
                "H2/He hot-Jupiter proxy with detected H2O absorption."
            ),
        },
        "distance_au": 0.052,
    },
    "kelt9_b": {
        "system_name": "KELT-9 b",
        "category": "gas_giant",
        "planet_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/KELT-9%20b",
        "star_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/KELT-9",
        "orbit_source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/KELT-9%20b",
        "spectrum_template_key": "A0",
        "exobase_template_key": "ultra_hot_jupiter",
        "star": {
            "label": "KELT-9",
            "path": "Templates/TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 10170.0,
            "radius": 2.362 * const.R_sun,
            "mass": 2.52 * const.M_sun,
            "vsini": 111.4 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "KELT-9 b",
            "radius": 1.891 * const.R_jup,
            "mass": 2.88 * const.M_jup,
            "T": 4050.0 * u.K,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H I": 0.765,
                "He I": 0.13,
                "He II": 0.02,
                "H2": 0.03,
                "O I": 0.02,
                "Fe I": 0.008,
                "Fe II": 0.008,
                "Ti II": 0.004,
                "Mg I": 0.005,
                "Ca II": 0.005,
                "Na I": 0.005,
            },
            "notes": (
                "Real ultra-hot Jupiter comparison system. Source-based mass/radius/orbit from the Exoplanet Archive; "
                "composition is a normalized ultra-hot-Jupiter proxy with an extended hydrogen envelope and "
                "atomic/ionized metal absorbers."
            ),
        },
        "distance_au": 0.03462,
    },
}


assign_composition_mean_molecular_weights(REAL_MASS_LOSS_REFERENCE_SYSTEMS)


def get_real_mass_loss_reference_system(name):
    if name not in REAL_MASS_LOSS_REFERENCE_SYSTEMS:
        available = ", ".join(sorted(REAL_MASS_LOSS_REFERENCE_SYSTEMS))
        raise KeyError(f"Unknown real mass-loss reference system '{name}'. Available systems: {available}")
    return copy.deepcopy(REAL_MASS_LOSS_REFERENCE_SYSTEMS[name])


def list_real_mass_loss_reference_systems():
    return sorted(REAL_MASS_LOSS_REFERENCE_SYSTEMS)
