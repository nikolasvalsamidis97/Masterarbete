import copy
import re

import astropy.constants as const
import astropy.units as u


STAR_TEMPLATES_UPDATED = {
    "O0": {"label": "O0 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O0/lte500-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 12.0 * const.R_sun, "mass": 60.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O1": {"label": "O1 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O1/lte480-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 11.0 * const.R_sun, "mass": 50.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O2": {"label": "O2 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O2/lte460-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 10.0 * const.R_sun, "mass": 45.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O3": {"label": "O3 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O3/lte440-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 9.5 * const.R_sun, "mass": 40.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O4": {"label": "O4 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O4/lte420-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 9.0 * const.R_sun, "mass": 35.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O5": {"label": "O5 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O5/lte400-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 8.5 * const.R_sun, "mass": 30.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O6": {"label": "O6 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O6/lte380-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 8.0 * const.R_sun, "mass": 28.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O7": {"label": "O7 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O7/lte360-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 7.5 * const.R_sun, "mass": 25.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O8": {"label": "O8 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O8/lte340-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 7.0 * const.R_sun, "mass": 22.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "O9": {"label": "O9 star", "category": "O", "path": "Templates/TS/Spectral_type/O/O9/lte320-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 6.5 * const.R_sun, "mass": 20.0 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},

    "B0": {"label": "B0 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B0/lte300-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 6.0 * const.R_sun, "mass": 18.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B1": {"label": "B1 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B1/lte250-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 5.5 * const.R_sun, "mass": 14.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B2": {"label": "B2 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B2/lte220-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 5.0 * const.R_sun, "mass": 10.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B3": {"label": "B3 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B3/lte190-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 4.5 * const.R_sun, "mass": 8.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B4": {"label": "B4 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B4/lte170-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 4.3 * const.R_sun, "mass": 7.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B5": {"label": "B5 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B5/lte150-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 4.1 * const.R_sun, "mass": 6.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B6": {"label": "B6 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B6/lte140-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.9 * const.R_sun, "mass": 5.5 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B7": {"label": "B7 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B7/lte130-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.7 * const.R_sun, "mass": 5.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B8": {"label": "B8 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B8/lte120-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.4 * const.R_sun, "mass": 4.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "B9": {"label": "B9 star", "category": "B", "path": "Templates/TS/Spectral_type/B/B9/lte110-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.1 * const.R_sun, "mass": 3.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},

    "A0": {"label": "A0 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.5 * const.R_sun, "mass": 2.5 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A1": {"label": "A1 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A1/lte094-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.3 * const.R_sun, "mass": 2.4 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A2": {"label": "A2 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A2/lte090-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.2 * const.R_sun, "mass": 2.3 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A3": {"label": "A3 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A3/lte088-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.1 * const.R_sun, "mass": 2.2 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A4": {"label": "A4 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A4/lte086-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.0 * const.R_sun, "mass": 2.1 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A5": {"label": "A5 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A5/lte082-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.9 * const.R_sun, "mass": 2.0 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A6": {"label": "A6 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A6/lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.8 * const.R_sun, "mass": 1.9 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A7": {"label": "A7 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A7/lte078-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.8 * const.R_sun, "mass": 1.9 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A8": {"label": "A8 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A8/lte076-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.7 * const.R_sun, "mass": 1.8 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "A9": {"label": "A9 star", "category": "A", "path": "Templates/TS/Spectral_type/A/A9/lte074-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.7 * const.R_sun, "mass": 1.8 * const.M_sun, "vsini": 60 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},

    "F0": {"label": "F0 star", "category": "F", "path": "Templates/TS/Spectral_type/F/F0/lte072-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.6 * const.R_sun, "mass": 1.6 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "F1": {"label": "F1 star", "category": "F", "path": "Templates/TS/Spectral_type/F/F1/lte070-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.5 * const.R_sun, "mass": 1.5 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "F2": {"label": "F2 star", "category": "F", "path": "Templates/TS/Spectral_type/F/F2/lte068-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.5 * const.R_sun, "mass": 1.5 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "F4": {"label": "F4 star", "category": "F", "path": "Templates/TS/Spectral_type/F/F4/lte066-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 1.4 * const.R_sun, "mass": 1.4 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "F5": {"label": "F5 star", "category": "F", "path": "Templates/TS/Spectral_type/F/F5/lte064-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.4 * const.R_sun, "mass": 1.4 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "F6": {"label": "F6 star", "category": "F", "path": "Templates/TS/Spectral_type/F/F6/lte062-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.4 * const.R_sun, "mass": 1.3 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "F8": {"label": "F8 star", "category": "F", "path": "Templates/TS/Spectral_type/F/F8/lte060-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 1.2 * const.R_sun, "mass": 1.2 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},

    "G1": {"label": "G1 star", "category": "G", "path": "Templates/TS/Spectral_type/G/G1/lte058-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.0 * const.R_sun, "mass": 1.0 * const.M_sun, "vsini": 10 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "G4": {"label": "G4 star", "category": "G", "path": "Templates/TS/Spectral_type/G/G4/lte056-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.0 * const.R_sun, "mass": 1.0 * const.M_sun, "vsini": 10 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "G6": {"label": "G6 star", "category": "G", "path": "Templates/TS/Spectral_type/G/G6/lte054-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.9 * const.R_sun, "mass": 0.9 * const.M_sun, "vsini": 10 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "G8": {"label": "G8 star", "category": "G", "path": "Templates/TS/Spectral_type/G/G8/lte052-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.9 * const.R_sun, "mass": 0.9 * const.M_sun, "vsini": 10 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},

    "K1": {"label": "K1 star", "category": "K", "path": "Templates/TS/Spectral_type/K/K1/lte050-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.8 * const.R_sun, "mass": 0.8 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "K3": {"label": "K3 star", "category": "K", "path": "Templates/TS/Spectral_type/K/K3/lte048-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.8 * const.R_sun, "mass": 0.8 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "K4": {"label": "K4 star", "category": "K", "path": "Templates/TS/Spectral_type/K/K4/lte046-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.7 * const.R_sun, "mass": 0.7 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "K5": {"label": "K5 star", "category": "K", "path": "Templates/TS/Spectral_type/K/K5/lte044-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.7 * const.R_sun, "mass": 0.7 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "K6": {"label": "K6 star", "category": "K", "path": "Templates/TS/Spectral_type/K/K6/lte042-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.7 * const.R_sun, "mass": 0.7 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "K7": {"label": "K7 star", "category": "K", "path": "Templates/TS/Spectral_type/K/K7/lte040-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.6 * const.R_sun, "mass": 0.6 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "K9": {"label": "K9 star", "category": "K", "path": "Templates/TS/Spectral_type/K/K9/lte038-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.6 * const.R_sun, "mass": 0.6 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},

    "M1": {"label": "M1 star", "category": "M", "path": "Templates/TS/Spectral_type/M/M1/lte036-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 0.5 * const.R_sun, "mass": 0.5 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "M2": {"label": "M2 star", "category": "M", "path": "Templates/TS/Spectral_type/M/M2/lte034-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 0.4 * const.R_sun, "mass": 0.4 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "M4": {"label": "M4 star", "category": "M", "path": "Templates/TS/Spectral_type/M/M4/lte032-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 0.3 * const.R_sun, "mass": 0.3 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "M6": {"label": "M6 star", "category": "M", "path": "Templates/TS/Spectral_type/M/M6/lte030-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.2 * const.R_sun, "mass": 0.2 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "M8": {"label": "M8 star", "category": "M", "path": "Templates/TS/Spectral_type/M/M8/lte028-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.2 * const.R_sun, "mass": 0.2 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
    "M9": {"label": "M9 star", "category": "M", "path": "Templates/TS/Spectral_type/M/M9/lte026-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.1 * const.R_sun, "mass": 0.1 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
}

STAR_TEMPLATES = STAR_TEMPLATES_UPDATED


def get_star_template(name):
    if name not in STAR_TEMPLATES:
        available = ", ".join(sorted(STAR_TEMPLATES.keys()))
        raise KeyError(f"Unknown star template '{name}'. Available templates: {available}")
    return copy.deepcopy(STAR_TEMPLATES[name])


def list_star_templates():
    return sorted(STAR_TEMPLATES.keys())


def infer_teff_from_star_template(template_or_name):
    if isinstance(template_or_name, str):
        template = get_star_template(template_or_name)
    else:
        template = template_or_name

    path = template["path"]
    match = re.search(r"lte(\d{2,3})", path)
    if match is None:
        raise ValueError(f"Could not infer Teff from path: {path}")
    return int(match.group(1)) * 100
