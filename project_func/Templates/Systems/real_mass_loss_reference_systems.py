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
}


def get_real_mass_loss_reference_system(name):
    if name not in REAL_MASS_LOSS_REFERENCE_SYSTEMS:
        available = ", ".join(sorted(REAL_MASS_LOSS_REFERENCE_SYSTEMS))
        raise KeyError(f"Unknown real mass-loss reference system '{name}'. Available systems: {available}")
    return copy.deepcopy(REAL_MASS_LOSS_REFERENCE_SYSTEMS[name])


def list_real_mass_loss_reference_systems():
    return sorted(REAL_MASS_LOSS_REFERENCE_SYSTEMS)
