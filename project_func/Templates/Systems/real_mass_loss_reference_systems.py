import copy

import astropy.constants as const
import astropy.units as u


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
            "path": "TS/Spectral_type/M/M4/lte032-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
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
            "mu": 25.0 * u.dimensionless_unscaled,
            "P0": 1.0 * u.bar,
            "composition": {
                "O I": 0.30,
                "N I": 0.15,
                "Na I": 0.15,
                "K I": 0.05,
                "CO2": 0.35,
            },
            "notes": (
                "Real rocky comparison system. Composition trimmed to the four most abundant atoms "
                "plus the most abundant molecule from the rocky super-Earth analogue and renormalized."
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
            "path": "TS/Spectral_type/M/M6/lte030-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 3026.0,
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
            "mu": 2.5 * u.dimensionless_unscaled,
            "P0": 1.0e-4 * u.bar,
            "composition": {
                "H I": 0.56,
                "He I": 0.14,
                "O I": 0.01,
                "Na I": 0.01,
                "H2": 0.28,
            },
            "notes": (
                "Real sub-Neptune comparison system. Composition trimmed to the four most abundant atoms "
                "plus the dominant molecule from the sub-Neptune analogue and renormalized."
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
            "path": "TS/Spectral_type/M/M1/lte036-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
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
            "mu": 2.5 * u.dimensionless_unscaled,
            "P0": 1.0e-4 * u.bar,
            "composition": {
                "H I": 0.58,
                "He I": 0.14,
                "O I": 0.01,
                "Na I": 0.01,
                "H2": 0.26,
            },
            "notes": (
                "Real Neptune comparison system. Composition trimmed to the four most abundant atoms "
                "plus the dominant molecule from the hot-Neptune analogue and renormalized."
            ),
        },
        "distance_au": 0.0282,
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
            "path": "TS/Spectral_type/F/F8/lte060-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
            "teff_K": 6065.0,
            "radius": 1.20 * const.R_sun,
            "mass": 1.07 * const.M_sun,
            "vsini": 4.5 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "HD 209458 b",
            "radius": 1.359 * const.R_jup,
            "mass": 0.685 * const.M_jup,
            "T": 1459.0 * u.K,
            "mu": 2.5 * u.dimensionless_unscaled,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H I": 0.60,
                "He I": 0.10,
                "O I": 0.01,
                "Na I": 0.01,
                "H2": 0.28,
            },
            "notes": (
                "Real gas-giant comparison system. Composition trimmed to the four most abundant atoms "
                "plus the dominant molecule from the inflated hot-Jupiter analogue and renormalized."
            ),
        },
        "distance_au": 0.04707,
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
            "path": "TS/Spectral_type/G/G8/lte052-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 5172.0,
            "radius": 0.943 * const.R_sun,
            "mass": 0.905 * const.M_sun,
            "vsini": 1.23 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "55 Cnc e",
            "radius": 1.875 * const.R_earth,
            "mass": 7.99 * const.M_earth,
            "T": 1958.0 * u.K,
            "mu": 35.0 * u.dimensionless_unscaled,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "O I": 0.1,
                "Na I": 0.2,
                "K I": 0.1,
                "Si I": 0.1,
                "Ca I": 0.1,
                "SiO": 0.2,
                "NaCl": 0.1,
                "O2": 0.05,
                "SO2": 0.05,
            },
            "notes": (
                "Real ultra-hot rocky comparison system. Source-based mass/radius/orbit from the Exoplanet Archive; "
                "composition adopts the generic lava-world analogue."
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
            "path": "TS/Spectral_type/A/A4/lte086-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 8500.0,
            "radius": 1.751 * const.R_sun,
            "mass": 1.89 * const.M_sun,
            "vsini": 59.4 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "HD 56414 b",
            "radius": 3.71 * const.R_earth,
            "mass": 13.3 * const.M_earth,
            "T": 1133.0 * u.K,
            "mu": 2.5 * u.dimensionless_unscaled,
            "P0": 1.0e-4 * u.bar,
            "composition": {
                "H I": 0.4,
                "He I": 0.11,
                "H2": 0.24,
                "H2O": 0.1,
                "CO": 0.1,
                "CO2": 0.03,
                "NH3": 0.02,
            },
            "notes": (
                "Real irradiated sub-Neptune comparison system. Source-based mass/radius/orbit from the Exoplanet Archive; "
                "composition adopts the generic sub-Neptune analogue."
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
            "path": "TS/Spectral_type/F/F8/lte060-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
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
            "mu": 2.5 * u.dimensionless_unscaled,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H I": 0.60,
                "He I": 0.1,
                "O I": 0.01,
                "Na I": 0.01,
                "H2": 0.28,
            },
            "notes": (
                "Real inflated hot-Jupiter comparison system. Source-based mass/radius/orbit from the Exoplanet Archive; "
                "composition adopts the generic inflated hot-Jupiter analogue."
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
            "path": "TS/Spectral_type/F/F5/lte064-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
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
            "mu": 2.5 * u.dimensionless_unscaled,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H I": 0.60,
                "He I": 0.1,
                "O I": 0.01,
                "Na I": 0.01,
                "H2": 0.28,
            },
            "notes": (
                "Real inflated hot-Jupiter comparison system. Source-based mass/radius/orbit from the Exoplanet Archive; "
                "composition adopts the generic inflated hot-Jupiter analogue."
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
            "path": "TS/Spectral_type/G/G1/lte058-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
            "teff_K": 5758.0,
            "radius": 1.175610 * const.R_sun,
            "mass": 1.03 * const.M_sun,
            "vsini": 2.2 * u.km / u.s,
            "epsilon": 0.5 * u.dimensionless_unscaled,
        },
        "planet": {
            "label": "51 Peg b",
            "radius": 1.27 * const.R_jup,
            "mass": 0.46 * const.M_jup,
            "T": 1311.0 * u.K,
            "mu": 2.5 * u.dimensionless_unscaled,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H I": 0.60,
                "He I": 0.1,
                "O I": 0.01,
                "Na I": 0.01,
                "H2": 0.28,
            },
            "notes": (
                "Real hot-Jupiter comparison system. Source-based mass/radius/orbit from the Exoplanet Archive; "
                "equilibrium temperature estimated from archive stellar properties and orbital distance because "
                "no archive pl_eqt entry is available. Composition adopts the generic inflated hot-Jupiter analogue."
            ),
        },
        "distance_au": 0.0527,
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
            "path": "TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
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
            "mu": 2.5 * u.dimensionless_unscaled,
            "P0": 1.0e-3 * u.bar,
            "composition": {
                "H I": 0.6,
                "He I": 0.1,
                "He II": 0.01,
                "O I": 0.01,
                "H2": 0.1,
                "CO": 0.11,
                "H2O": 0.04,
                "NO": 0.02,
                "OH": 0.01,
            },
            "notes": (
                "Real ultra-hot Jupiter comparison system. Source-based mass/radius/orbit from the Exoplanet Archive; "
                "composition adopts the generic ultra-hot-Jupiter analogue."
            ),
        },
        "distance_au": 0.03462,
    },
}


def get_real_mass_loss_reference_system(name):
    if name not in REAL_MASS_LOSS_REFERENCE_SYSTEMS:
        available = ", ".join(sorted(REAL_MASS_LOSS_REFERENCE_SYSTEMS))
        raise KeyError(f"Unknown real mass-loss reference system '{name}'. Available systems: {available}")
    return copy.deepcopy(REAL_MASS_LOSS_REFERENCE_SYSTEMS[name])


def list_real_mass_loss_reference_systems():
    return sorted(REAL_MASS_LOSS_REFERENCE_SYSTEMS)
