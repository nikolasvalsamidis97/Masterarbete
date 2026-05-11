import copy
import re

import astropy.constants as const
import astropy.units as u


ATOMIC_MASSES_AMU = {
    "H": 1.008,
    "He": 4.002602,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Na": 22.98976928,
    "Si": 28.085,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.0983,
}

FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
COMPOSITION_MU_TEMPLATE_KEYS = {
    "super_earth_rocky",
    "alkali_exosphere_rocky",
    "metal_rich_secondary",
}


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


def assign_composition_mean_molecular_weights(templates: dict, template_keys: set[str]) -> None:
    for template_key in template_keys:
        template = templates[template_key]
        template["mu"] = (
            rounded_mean_molecular_weight_from_composition(template["composition"])
            * u.dimensionless_unscaled
        )


PLANET_TEMPLATES_UPDATED = {
    "mercury_like": {
        "label": "Mercury-like rocky planet",
        "category": "rocky",
        "radius": 0.383 * const.R_earth,
        "mass": 0.055 * const.M_earth,
        "T": 440 * u.K,
        "mu": 23.0 * u.dimensionless_unscaled,
        "P0": 1.0e-9 * u.bar,
        "composition": {
            "H I": 0.18,
            "He I": 0.05,
            "O I": 0.34,
            "Na I": 0.22,
            "K I": 0.01,
            "NaCl": 0.06,
            "SiO": 0.08,
            "O2": 0.06,
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
            "N I": 0.35,
            "O I": 0.10,
            "N2": 0.42,
            "O2": 0.11,
            "H2O": 0.008,
            "CO2": 0.012,
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
            "N I": 0.01,
            "O I": 0.004,
            "CO2": 0.94,
            "CO": 0.035,
            "O2": 0.008,
            "H2O": 0.003,
        },
        "notes": "Thin CO2-dominated secondary atmosphere approximated by large mean molecular weight.",
    },
    "super_earth_rocky": {
        "label": "Super earth",
        "category": "rocky",
        "radius": 2.0 * const.R_earth,
        "mass": 4.0 * const.M_earth,
        "T": 300 * u.K,
        "P0": 1.0 * u.bar,
        "composition": {
            "O I": 0.2,
            "N I": 0.1,
            "Na I": 0.1,
            "K I": 0.03,
            "H I": 0.01,
            "CO2": 0.25,
            "SO2": 0.1,
            "CO": 0.1,
            "H2O": 0.08,
            "SiO": 0.03,
        },
        "notes": "Generic rocky super-Earth",
    },
    "lava_world": {
        "label": "Lava world",
        "category": "rocky",
        "radius": 2.0 * const.R_earth,
        "mass": 8.0 * const.M_earth,
        "T": 2000 * u.K,
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
        "notes": "Generic silicate-vapor / alkali-rich ultra-hot rocky planet.",
    },
    "volatile_super_earth": {
        "label": "Volatile-rich hot super-Earth",
        "category": "rocky",
        "radius": 2.0 * const.R_earth,
        "mass": 6.5 * const.M_earth,
        "T": 900 * u.K,
        "mu": 6.0 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "H I": 0.1,
            "He I": 0.03,
            "O I": 0.04,
            "Na I": 0.01,
            "K I": 0.01,
            "H2": 0.36,
            "H2O": 0.2,
            "CO": 0.1,
            "CO2": 0.1,
            "NO": 0.03,
            "NH3": 0.02,
        },
        "notes": "Transitional hot super-Earth with a volatile-rich, partially retained light envelope.",
    },
    "mini_neptune_cool": {
        "label": "Cool mini-Neptune",
        "category": "mini_neptune",
        "radius": 2.5 * const.R_earth,
        "mass": 7.0 * const.M_earth,
        "T": 400 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-4 * u.bar,
        "composition": {
            "H I": 0.4,
            "He I": 0.1,
            "H2": 0.35,
            "H2O": 0.1,
            "NH3": 0.02,
            "CO": 0.03,
        },
        "notes": "H/He-dominated mini-Neptune with trace heavy species.",
    },
    "mini_neptune_warm": {
        "label": "Warm mini-Neptune",
        "category": "mini_neptune",
        "radius": 2.5 * const.R_earth,
        "mass": 9.0 * const.M_earth,
        "T": 700 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-4 * u.bar,
        "composition": {
            "H I": 0.35,
            "He I": 0.1,
            "H2": 0.32,
            "H2O": 0.1,
            "CO": 0.1,
            "CO2": 0.03,
        },
        "notes": "Warm mini-Neptune appropriate for irradiation studies.",
    },
    "sub_neptune": {
        "label": "Sub-Neptune",
        "category": "sub_neptune",
        "radius": 3.5 * const.R_earth,
        "mass": 12.0 * const.M_earth,
        "T": 900 * u.K,
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
        "notes": "Generic irradiated sub-Neptune with H/He envelope.",
    },
    "warm_neptune": {
        "label": "Warm Neptune",
        "category": "neptune",
        "radius": 4.0 * const.R_earth,
        "mass": 17.0 * const.M_earth,
        "T": 900 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-4 * u.bar,
        "composition": {
            "H I": 0.45,
            "He I": 0.1,
            "H2": 0.23,
            "H2O": 0.1,
            "CO": 0.08,
            "CO2": 0.03,
            "NO": 0.01,
        },
        "notes": "Representative warm Neptune atmosphere.",
    },
    "hot_neptune": {
        "label": "Hot Neptune",
        "category": "neptune",
        "radius": 4.5 * const.R_earth,
        "mass": 20.0 * const.M_earth,
        "T": 1000 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-4 * u.bar,
        "composition": {
            "H I": 0.47,
            "He I": 0.1,
            "O I": 0.01,
            "H2": 0.2,
            "H2O": 0.1,
            "CO": 0.08,
            "CO2": 0.03,
            "NO": 0.01,
        },
        "notes": "Hot Neptune with stronger heavy-species signature.",
    },
    "super_puff": {
        "label": "Super-puff sub-Neptune",
        "category": "sub_neptune",
        "radius": 6.5 * const.R_earth,
        "mass": 4.5 * const.M_earth,
        "T": 700 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-4 * u.bar,
        "composition": {
            "H I": 0.35,
            "He I": 0.1,
            "H2": 0.37,
            "H2O": 0.1,
            "CO": 0.05,
            "CO2": 0.02,
            "NH3": 0.01,
        },
        "notes": "Very low-gravity inflated sub-Neptune / super-puff for escape sensitivity tests.",
    },
    "cold_jupiter": {
        "label": "Cold Jupiter",
        "category": "gas_giant",
        "radius": 1.0 * const.R_jup,
        "mass": 1.0 * const.M_jup,
        "T": 200 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "H I": 0.49,
            "He I": 0.1,
            "H2": 0.35,
            "NH3": 0.03,
            "H2O": 0.02,
            "CO": 0.01,
        },
        "notes": "Cold giant baseline with weak irradiation and molecule-rich atmosphere.",
    },
    "warm_jupiter": {
        "label": "Warm Jupiter",
        "category": "gas_giant",
        "radius": 1.0 * const.R_jup,
        "mass": 1.0 * const.M_jup,
        "T": 600 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "H I": 0.5,
            "He I": 0.1,
            "H2": 0.33,
            "H2O": 0.03,
            "CO": 0.02,
            "NH3": 0.02,
        },
        "notes": "Generic warm Jupiter.",
    },
    "hot_jupiter": {
        "label": "Hot Jupiter",
        "category": "gas_giant",
        "radius": 1.0 * const.R_jup,
        "mass": 1.0 * const.M_jup,
        "T": 1500 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "H I": 0.49,
            "He I": 0.1,
            "H2": 0.2,
            "H2O": 0.1,
            "CO": 0.08,
            "CO2": 0.01,
            "NO": 0.01,
            "OH": 0.01,
        },
        "notes": "Representative hot Jupiter for irradiation studies.",
    },
    "inflated_hot_jupiter": {
        "label": "Inflated hot Jupiter",
        "category": "gas_giant",
        "radius": 1.5 * const.R_jup,
        "mass": 0.5 * const.M_jup,
        "T": 1500 * u.K,
        "mu": 2.5 * u.dimensionless_unscaled,
        "P0": 1.0e-3 * u.bar,
        "composition": {
            "H I": 0.60,
            "He I": 0.1,
            "O I": 0.01,
            "Na I": 0.01,
            "H2": 0.28,
        },
        "notes": "Low-gravity inflated hot Jupiter with an HD 209458 b-like H/He-dominated composition.",
    },
    "ultra_hot_jupiter": {
        "label": "Ultra-hot Jupiter",
        "category": "gas_giant",
        "radius": 2.0 * const.R_jup,
        "mass": 1.0 * const.M_jup,
        "T": 2500 * u.K,
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
        "notes": "Ultra-hot giant with partial ionization and strong heavy-species signatures.",
    },
    "alkali_exosphere_rocky": {
        "label": "Alkali Rocky",
        "category": "rocky",
        "radius": 0.8 * const.R_earth,
        "mass": 0.4 * const.M_earth,
        "T": 1000 * u.K,
        "P0": 1.0e-8 * u.bar,
        "composition": {
            "Na I": 0.4,
            "K I": 0.1,
            "O I": 0.1,
            "H I": 0.03,
            "He I": 0.03,
            "NaCl": 0.1,
            "SiO": 0.1,
            "SO2": 0.1,
            "O2": 0.04,
        },
        "notes": "Useful for emphasizing Na/K radiation-pressure signatures.",
    },
    "metal_rich_secondary": {
        "label": "Metal rich",
        "category": "rocky",
        "radius": 2.0 * const.R_earth,
        "mass": 6.0 * const.M_earth,
        "T": 1000 * u.K,
        "P0": 1.0e-2 * u.bar,
        "composition": {
            "O I": 0.15,
            "N I": 0.1,
            "Na I": 0.1,
            "K I": 0.03,
            "CO": 0.15,
            "NO": 0.08,
            "CO2": 0.15,
            "SO2": 0.1,
            "H2O": 0.1,
            "SiO": 0.04,
        },
        "notes": "Heavy secondary atmosphere for parameter studies beyond H/He envelopes.",
    },
}

assign_composition_mean_molecular_weights(PLANET_TEMPLATES_UPDATED, COMPOSITION_MU_TEMPLATE_KEYS)

PLANET_TEMPLATES = PLANET_TEMPLATES_UPDATED


def get_planet_template(name):
    if name not in PLANET_TEMPLATES:
        available = ", ".join(sorted(PLANET_TEMPLATES.keys()))
        raise KeyError(f"Unknown planet template '{name}'. Available templates: {available}")
    return copy.deepcopy(PLANET_TEMPLATES[name])


def list_planet_templates():
    return sorted(PLANET_TEMPLATES.keys())
