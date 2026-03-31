

import copy

import astropy.constants as const
import astropy.units as u


PLANET_TEMPLATES = {
    # -------------------------------------------------------------------------
    # Rocky / terrestrial templates
    # -------------------------------------------------------------------------
    "mercury_like": {
        "label": "Mercury-like rocky planet",
        "category": "rocky",
        "radius": 0.383 * const.R_earth,
        "mass": 0.055 * const.M_earth,
        "T": 440 * u.K,
        "mu": 23.0 * u.dimensionless_unscaled,
        "P0": 1.0e-9 * u.bar,
        "composition": {
            "H I": 0.22,
            "He I": 0.06,
            "O I": 0.42,
            "Na I": 0.29,
            "K I": 0.01,
        },
        "notes": "Na-rich, tenuous rocky exosphere inspired by Mercury.",
    },
    "earth_like": {
        "label": "Earth-like rocky planet",
        "category": "rocky",
        "radius": 1.0 * const.R_earth,
        "mass": 1.0 * const.M_earth,
        "T": 288 * u.K,
        "mu": 28.97 * u.dimensionless_unscaled,
        "P0": 1.0 * u.bar,
        "composition": {
            "N I": 0.7808,
            "O I": 0.2095,
            "He I": 5.24e-6,
            "H I": 5.5e-7,
            "Na I": 1e-9,
            "K I": 1e-10,
        },
        "notes": "Secondary atmosphere, Earth-like reference case.",
    },
    "mars_like": {
        "label": "Mars-like rocky planet",
        "category": "rocky",
        "radius": 0.532 * const.R_earth,
        "mass": 0.107 * const.M_earth,
        "T": 210 * u.K,
        "mu": 43.0 * u.dimensionless_unscaled,
        "P0": 6.0e-3 * u.bar,
        "composition": {
            "N I": 0.027,
            "O I": 0.0013,
            "He I": 1e-5,
            "H I": 1e-6,
        },
        "notes": "Thin CO2-dominated secondary atmosphere approximated by large mean molecular weight.",
    },
    "super_earth_rocky": {
        "label": "Super-Earth rocky planet",
        "category": "rocky",
        "radius": 1.7 * const.R_earth,
        "mass": 5.0 * const.M_earth,
        "T": 700 * u.K,
        "mu": 30.0 * u.dimensionless_unscaled,
        "P0": 1.0e-2 * u.bar,
        "composition": {
            "O I": 0.45,
            "N I": 0.35,
            "Na I": 0.10,
            "K I": 0.05,
            "H I": 0.05,
        },
        "notes": "Generic hot rocky super-Earth with a heavy secondary atmosphere.",
    },
    "lava_world": {
        "label": "Lava world",
        "category": "rocky",
        "radius": 1.9 * const.R_earth,
        "mass": 8.0 * const.M_earth,
        "T": 2200 * u.K,
        "mu": 35.0 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "O I": 0.40,
            "Na I": 0.25,
            "K I": 0.10,
            "Si I": 0.15,
            "Ca I": 0.10,
        },
        "notes": "Generic silicate-vapor / alkali-rich ultra-hot rocky planet.",
    },

    # -------------------------------------------------------------------------
    # Sub-Neptunes / mini-Neptunes / Neptunes
    # -------------------------------------------------------------------------
    "mini_neptune_cool": {
        "label": "Cool mini-Neptune",
        "category": "mini_neptune",
        "radius": 2.3 * const.R_earth,
        "mass": 7.0 * const.M_earth,
        "T": 400 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-5 * u.bar,
        "composition": {
            "H I": 0.84,
            "He I": 0.15,
            "O I": 0.009,
            "Na I": 5e-4,
            "K I": 5e-5,
        },
        "notes": "H/He-dominated mini-Neptune with trace heavy species.",
    },
    "mini_neptune_warm": {
        "label": "Warm mini-Neptune",
        "category": "mini_neptune",
        "radius": 2.7 * const.R_earth,
        "mass": 9.0 * const.M_earth,
        "T": 700 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-5 * u.bar,
        "composition": {
            "H I": 0.82,
            "He I": 0.15,
            "O I": 0.02,
            "Na I": 8e-4,
            "K I": 8e-5,
        },
        "notes": "Warm mini-Neptune appropriate for irradiation studies.",
    },
    "sub_neptune": {
        "label": "Sub-Neptune",
        "category": "sub_neptune",
        "radius": 3.4 * const.R_earth,
        "mass": 12.0 * const.M_earth,
        "T": 850 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-4 * u.bar,
        "composition": {
            "H I": 0.85,
            "He I": 0.14,
            "O I": 0.009,
            "Na I": 7e-4,
            "K I": 7e-5,
            "CO": 3e-4,
        },
        "notes": "Generic irradiated sub-Neptune with H/He envelope.",
    },
    "warm_neptune": {
        "label": "Warm Neptune",
        "category": "neptune",
        "radius": 4.0 * const.R_earth,
        "mass": 17.0 * const.M_earth,
        "T": 900 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "H I": 0.84,
            "He I": 0.15,
            "O I": 0.008,
            "Na I": 8e-4,
            "K I": 8e-5,
            "CO": 4e-4,
        },
        "notes": "Representative warm Neptune atmosphere.",
    },
    "hot_neptune": {
        "label": "Hot Neptune",
        "category": "neptune",
        "radius": 4.3 * const.R_earth,
        "mass": 20.0 * const.M_earth,
        "T": 1200 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-4 * u.bar,
        "composition": {
            "H I": 0.83,
            "He I": 0.14,
            "O I": 0.015,
            "Na I": 1.0e-3,
            "K I": 1.0e-4,
            "CO": 7e-4,
            "NO": 1e-5,
        },
        "notes": "Hot Neptune with stronger heavy-species signature.",
    },

    # -------------------------------------------------------------------------
    # Gas giants
    # -------------------------------------------------------------------------
    "warm_jupiter": {
        "label": "Warm Jupiter",
        "category": "gas_giant",
        "radius": 1.0 * const.R_jup,
        "mass": 1.0 * const.M_jup,
        "T": 600 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-2 * u.bar,
        "composition": {
            "H I": 0.899,
            "He I": 0.10,
            "O I": 7e-4,
            "Na I": 5e-5,
            "K I": 5e-6,
            "CO": 2e-4,
        },
        "notes": "Generic warm Jupiter.",
    },
    "hot_jupiter": {
        "label": "Hot Jupiter",
        "category": "gas_giant",
        "radius": 1.2 * const.R_jup,
        "mass": 0.9 * const.M_jup,
        "T": 1400 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "H I": 0.899,
            "He I": 0.10,
            "O I": 1e-3,
            "Na I": 1e-4,
            "K I": 1e-5,
            "CO": 5e-4,
            "NO": 1e-5,
        },
        "notes": "Representative hot Jupiter for irradiation studies.",
    },
    "inflated_hot_jupiter": {
        "label": "Inflated hot Jupiter",
        "category": "gas_giant",
        "radius": 1.6 * const.R_jup,
        "mass": 0.6 * const.M_jup,
        "T": 1700 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-4 * u.bar,
        "composition": {
            "H I": 0.898,
            "He I": 0.10,
            "O I": 1.2e-3,
            "Na I": 2e-4,
            "K I": 2e-5,
            "CO": 8e-4,
            "NO": 2e-5,
        },
        "notes": "Low-gravity inflated hot Jupiter.",
    },
    "ultra_hot_jupiter": {
        "label": "Ultra-hot Jupiter",
        "category": "gas_giant",
        "radius": 1.75 * const.R_jup,
        "mass": 1.2 * const.M_jup,
        "T": 2500 * u.K,
        "mu": 2.3 * u.dimensionless_unscaled,
        "P0": 1.0e-7 * u.bar,
        "composition": {
            "H I": 0.88,
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
        "notes": "Ultra-hot giant with partial ionization and strong heavy-species signatures.",
    },

    # -------------------------------------------------------------------------
    # Optional composition-specialized templates
    # -------------------------------------------------------------------------
    "alkali_exosphere_rocky": {
        "label": "Alkali-rich rocky exosphere",
        "category": "rocky",
        "radius": 0.8 * const.R_earth,
        "mass": 0.4 * const.M_earth,
        "T": 1200 * u.K,
        "mu": 25.0 * u.dimensionless_unscaled,
        "P0": 1.0e-8 * u.bar,
        "composition": {
            "Na I": 0.60,
            "K I": 0.10,
            "O I": 0.20,
            "H I": 0.05,
            "He I": 0.05,
        },
        "notes": "Useful for emphasizing Na/K radiation-pressure signatures.",
    },
    "metal_rich_secondary": {
        "label": "Metal-rich secondary atmosphere",
        "category": "rocky",
        "radius": 2.0 * const.R_earth,
        "mass": 6.0 * const.M_earth,
        "T": 1000 * u.K,
        "mu": 35.0 * u.dimensionless_unscaled,
        "P0": 1.0e-2 * u.bar,
        "composition": {
            "O I": 0.40,
            "N I": 0.25,
            "Na I": 0.10,
            "K I": 0.05,
            "CO": 0.15,
            "NO": 0.05,
        },
        "notes": "Heavy secondary atmosphere for parameter studies beyond H/He envelopes.",
    },
}


def get_planet_template(name):
    if name not in PLANET_TEMPLATES:
        available = ", ".join(sorted(PLANET_TEMPLATES.keys()))
        raise KeyError(f"Unknown planet template '{name}'. Available templates: {available}")
    return copy.deepcopy(PLANET_TEMPLATES[name])


def list_planet_templates():
    return sorted(PLANET_TEMPLATES.keys())