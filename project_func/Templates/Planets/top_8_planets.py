import copy

from project_func.Templates.Planets.planet_templates_updated import PLANET_TEMPLATES_UPDATED


TOP_8_PLANET_KEYS = [
    "mercury_like",
    "mars_like",
    "alkali_exosphere_rocky",
    "inflated_hot_jupiter",
    "hot_neptune",
    "sub_neptune",
    "warm_neptune",
    "ultra_hot_jupiter",
]

TOP_8_PLANETS = {key: copy.deepcopy(PLANET_TEMPLATES_UPDATED[key]) for key in TOP_8_PLANET_KEYS}


def get_top_8_planet_template(name):
    if name not in TOP_8_PLANETS:
        available = ", ".join(TOP_8_PLANET_KEYS)
        raise KeyError(f"Unknown top-8 planet template '{name}'. Available templates: {available}")
    return copy.deepcopy(TOP_8_PLANETS[name])


def list_top_8_planets():
    return list(TOP_8_PLANET_KEYS)
