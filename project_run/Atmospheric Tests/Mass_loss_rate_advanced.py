import copy
import concurrent.futures as cf
import csv
from dataclasses import dataclass
import os
import pathlib
import tempfile
import sys
import time
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.integrate import trapezoid

# Avoid RADIS/numba cache issues when PhotonPressure imports the molecule stack.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(tempfile.gettempdir()) / "matplotlib_mass_loss_advanced"))
os.environ.setdefault("XDG_CACHE_HOME", str(pathlib.Path(tempfile.gettempdir()) / "xdg_cache_mass_loss_advanced"))

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem
from project_classes.Molecule import Molecule
from project_classes.Star import Star
from project_func.exobase_table_path import resolve_exobase_table_path
from project_func.plotdata_to_txt import save_plotdata_txt
from project_func.Templates.Atoms.atom_species import ATOM_SPECIES
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Planets.planet_templates_updated import PLANET_TEMPLATES, get_planet_template
from project_func.Templates.Stars.stars_templates_updated import STAR_TEMPLATES, infer_teff_from_star_template

base_mass_loss = SimpleNamespace()


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str_list(name: str):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


# -----------------------------------------------------------------------------
# Advanced run configuration
# -----------------------------------------------------------------------------
SKIP_ATOMS = _env_flag("MLA_SKIP_ATOMS", False)
SKIP_MOLECULES = _env_flag("MLA_SKIP_MOLECULES", False)
SELECTED_ATOMIC_SPECIES = _env_str_list("MLA_SELECTED_ATOMIC_SPECIES")
SELECTED_MOLECULAR_SPECIES = _env_str_list("MLA_SELECTED_MOLECULAR_SPECIES")
INCLUDE_STELLAR_GRAVITY = True
USE_CHECKPOINT = _env_flag("MLA_USE_CHECKPOINT", True)
MLA_SPECIES_MAX_WORKERS = max(1, _env_int("MLA_SPECIES_MAX_WORKERS", 4))
MLA_PLANET_MAX_WORKERS = max(1, _env_int("MLA_PLANET_MAX_WORKERS", 1))

RUN_FAMILY = os.environ.get("MLA_RUN_FAMILY", "solar_system_fixed")
# Allowed values:
#   solar_system_fixed
#   real_reference_systems
#   distance_sweep
#   p0_sweep
#   mu_sweep
#   surface_gravity_sweep
#   all
SELECTED_SYSTEM_KEYS = None

# Moderate-resolution advanced run for one representative case.
N_RHO = 32
N_X = 120
RHO_GRID_POWER = 4.0
X_GRID_POWER = 3.0
COLUMN_STEPS = 160
COLUMN_GRID_POWER = 3.0

# Photon-pressure lookup cache on logarithmic column bins.
LOG_COLUMN_BIN_DEX = 0.02
MIN_COLUMN_CM2 = 1.0e-60

# Trajectory integration controls.
# Use the finest tested timestep for thesis-grade production runs.
# It is slower, but it is the most conservative timestep setting that has
# already been benchmarked against coarser alternatives.
DT_FRACTION = 0.02
DT_MIN_S = 1.0e-3
DT_MAX_S = 1.0e2
DT_LENGTH_FLOOR_CM = 1.0e5
MAX_STEP_LENGTH_FRACTION = 0.05
MAX_STEPS = 8000
MAX_TIME_S = 5.0e7

WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_ATOM = 1.0 * u.km / u.s
NPTS_ATOM = 150

_DEFAULT_EXTERNAL_OUTPUT_DIR = pathlib.Path.home() / "DATA" / "results" / "Mass_loss_rate_advanced"
_DEFAULT_OUTPUT_DIR = (
    _DEFAULT_EXTERNAL_OUTPUT_DIR
    if (pathlib.Path.home() / "DATA").exists()
    else pathlib.Path(__file__).resolve().parent / "results"
)
OUTPUT_DIR = pathlib.Path(os.environ.get("MLA_OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR))).expanduser()
CHECKPOINT_DIR = OUTPUT_DIR / "_checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "standard_systems_checkpoint.csv"
EXOBASE_TABLE = (
    resolve_exobase_table_path(pathlib.Path(__file__).resolve().parents[2])
)

SOLAR_SYSTEM_STAR_KEY = "G1"
DISTANCE_SWEEP_PLANET_KEY = "inflated_hot_jupiter"
DISTANCE_SWEEP_STAR_KEY = "F8"
DISTANCE_SWEEP_DISTANCES_AU = [0.03, 0.05, 0.1, 0.3]
HOT_SWEEP_DISTANCES_AU = DISTANCE_SWEEP_DISTANCES_AU
PLAUSIBLE_PLANET_DISTANCE_AU = 0.1
MU_SWEEP_STAR_KEY = "F8"
MU_SWEEP_DISTANCE_AU = 0.05
SURFACE_GRAVITY_STAR_KEY = "F8"
SURFACE_GRAVITY_DISTANCE_AU = 0.1

P0_SWEEP_PLANET_KEY = "inflated_hot_jupiter"
P0_SWEEP_STAR_KEY = "F8"
P0_SWEEP_DISTANCE_AU = 0.05
P0_SWEEP_VALUES_BAR = np.array([1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
P0_SWEEP_TEST_FAMILY = "p0_sweep"
P0_SWEEP_OUTPUT_DIR = OUTPUT_DIR

MU_SWEEP_VALUES_AMU = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
MU_SWEEP_HOT_JUPITER_FAMILY = "mu_sweep"
MU_SWEEP_OUTPUT_DIR = OUTPUT_DIR

SURFACE_GRAVITY_MASS_SCALE_VALUES = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
SURFACE_GRAVITY_SWEEP_FAMILY = "surface_gravity_sweep"
SURFACE_GRAVITY_SWEEP_OUTPUT_DIR = OUTPUT_DIR

M_NEPTUNE = 17.147 * u.M_earth
ROCKY_CATEGORIES = {"rocky"}
NEPTUNE_LIKE_CATEGORIES = {"mini_neptune", "sub_neptune", "neptune"}
JUPITER_LIKE_CATEGORIES = {"gas_giant"}

star_cache: Dict[str, Star] = {}
profile_cache: Dict[str, BroadeningProfile] = {}

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
            "composition": {"O I": 0.30, "N I": 0.15, "Na I": 0.15, "K I": 0.05, "CO2": 0.35},
            "notes": "Real rocky comparison system with trimmed 4-atom + 1-molecule composition.",
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
            "composition": {"H I": 0.56, "He I": 0.14, "O I": 0.01, "Na I": 0.01, "H2": 0.28},
            "notes": "Real sub-Neptune comparison system with trimmed 4-atom + 1-molecule composition.",
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
            "composition": {"H I": 0.58, "He I": 0.14, "O I": 0.01, "Na I": 0.01, "H2": 0.26},
            "notes": "Real Neptune comparison system with trimmed 4-atom + 1-molecule composition.",
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
            "composition": {"H I": 0.60, "He I": 0.10, "O I": 0.01, "Na I": 0.01, "H2": 0.28},
            "notes": "Real gas-giant comparison system with trimmed 4-atom + 1-molecule composition.",
        },
        "distance_au": 0.04707,
    },
}


def get_real_mass_loss_reference_system(name: str) -> dict:
    if name not in REAL_MASS_LOSS_REFERENCE_SYSTEMS:
        available = ", ".join(sorted(REAL_MASS_LOSS_REFERENCE_SYSTEMS))
        raise KeyError(f"Unknown real mass-loss reference system '{name}'. Available systems: {available}")
    return copy.deepcopy(REAL_MASS_LOSS_REFERENCE_SYSTEMS[name])


@dataclass(frozen=True)
class AdvancedSystem:
    test_family: str
    planet_key: str
    star_key: str
    distance_au: float
    exobase_planet_key: str | None = None


def build_p0_sweep_system() -> AdvancedSystem:
    return AdvancedSystem(
        P0_SWEEP_TEST_FAMILY,
        P0_SWEEP_PLANET_KEY,
        P0_SWEEP_STAR_KEY,
        P0_SWEEP_DISTANCE_AU,
    )


def build_mu_sweep_systems() -> List[AdvancedSystem]:
    return [
        AdvancedSystem(MU_SWEEP_HOT_JUPITER_FAMILY, "inflated_hot_jupiter", MU_SWEEP_STAR_KEY, MU_SWEEP_DISTANCE_AU),
    ]


def build_surface_gravity_sweep_system() -> AdvancedSystem:
    return AdvancedSystem(
        SURFACE_GRAVITY_SWEEP_FAMILY,
        "super_earth_rocky",
        SURFACE_GRAVITY_STAR_KEY,
        SURFACE_GRAVITY_DISTANCE_AU,
    )


def build_advanced_systems() -> List[AdvancedSystem]:
    systems: List[AdvancedSystem] = []

    if family_enabled("solar_system_fixed"):
        systems.extend(
            [
                AdvancedSystem("solar_system_fixed", "mercury_like", SOLAR_SYSTEM_STAR_KEY, 0.387),
                AdvancedSystem("solar_system_fixed", "earth_like", SOLAR_SYSTEM_STAR_KEY, 1.0),
                AdvancedSystem("solar_system_fixed", "mars_like", SOLAR_SYSTEM_STAR_KEY, 1.524),
                AdvancedSystem("solar_system_fixed", "cold_jupiter", SOLAR_SYSTEM_STAR_KEY, 5.204),
            ]
        )

    if family_enabled("real_reference_systems"):
        for system_key, system_def in REAL_MASS_LOSS_REFERENCE_SYSTEMS.items():
            systems.append(
                AdvancedSystem(
                    "real_reference_systems",
                    system_key,
                    system_key,
                    float(system_def["distance_au"]),
                    exobase_planet_key=str(system_def["exobase_template_key"]),
                )
            )

    if family_enabled("distance_sweep"):
        for distance_au in DISTANCE_SWEEP_DISTANCES_AU:
            systems.append(
                AdvancedSystem("distance_sweep", DISTANCE_SWEEP_PLANET_KEY, DISTANCE_SWEEP_STAR_KEY, float(distance_au))
            )

    return systems


def safe_name(value) -> str:
    return str(value).replace(" ", "").replace("/", "_")


def load_exobase_table(path: pathlib.Path) -> Dict[Tuple[str, str], dict]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find exobase table: {path}")

    rows: Dict[Tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            planet = row.get("planet", "").strip()
            species = row.get("species", "").strip()
            if not planet or not species:
                continue
            rows[(planet, species)] = row
    return rows


def get_star(star_key: str) -> Star:
    if star_key not in star_cache:
        params = STAR_TEMPLATES[star_key]
        star_cache[star_key] = Star(
            params["path"],
            params["radius"],
            params["mass"],
            vsini=params["vsini"],
            epsilon=params["epsilon"],
        )
    return star_cache[star_key]


def get_atom_profile(species: str) -> BroadeningProfile:
    if species not in profile_cache:
        atom = Atom(species, WAVEMIN, WAVEMAX)
        profile_cache[species] = BroadeningProfile(atom, B_ATOM, NPTS_ATOM, "Voigt")
    return profile_cache[species]


def species_mixing_ratio(planet_case: dict, species: str) -> float:
    value = planet_case["composition"].get(species, 0.0)
    if isinstance(value, u.Quantity):
        value = value.to_value(u.dimensionless_unscaled)
    return float(value)


def exobase_height(planet_key: str, species: str, exobase_rows: Dict[Tuple[str, str], dict]) -> u.Quantity | None:
    row = exobase_rows.get((planet_key, species))
    if row is None:
        return None
    try:
        return float(row["z_exobase_km"]) * u.km
    except (KeyError, TypeError, ValueError):
        return None


def planet_category(planet_case: dict) -> str:
    return str(planet_case.get("category", "")).strip().lower()


def is_rocky_planet_case(planet_case: dict) -> bool:
    return planet_category(planet_case) in ROCKY_CATEGORIES


def rate_unit_info_for_planet(planet_case: dict) -> Tuple[str, u.Quantity]:
    category = planet_category(planet_case)
    if category in JUPITER_LIKE_CATEGORIES:
        return "Mjup", const.M_jup.to(u.g)
    if category in NEPTUNE_LIKE_CATEGORIES:
        return "Mnep", M_NEPTUNE.to(u.g)
    return "Mearth", const.M_earth.to(u.g)


def representative_atmosphere_top(
    planet_key: str,
    planet_case: dict,
    exobase_rows: Dict[Tuple[str, str], dict],
) -> Tuple[u.Quantity, str]:
    candidate_species = list(planet_case.get("composition", {}).keys())
    heights = []
    for species in candidate_species:
        z_exo = exobase_height(planet_key, species, exobase_rows)
        if z_exo is not None:
            heights.append((species, z_exo.to(u.km)))

    if heights:
        species, z_top = max(heights, key=lambda item: item[1].to_value(u.km))
        return z_top, f"max_composition_exobase:{species}"

    planet = build_planet(planet_case)
    fallback = (10.0 * planet.scale_height()).to(u.km)
    return fallback, "fallback:10_scale_heights"


def integrated_atmosphere_mass(
    planet_key: str,
    planet_case: dict,
    exobase_rows: Dict[Tuple[str, str], dict],
    n_z: int = 2048,
) -> Tuple[u.Quantity, u.Quantity, str]:
    planet = build_planet(planet_case)
    z_top, source = representative_atmosphere_top(planet_key, planet_case, exobase_rows)

    z_grid = np.linspace(0.0, z_top.to_value(u.cm), int(n_z)) * u.cm
    r_grid = planet.radius.to(u.cm) + z_grid
    n_grid = planet.number_density(z_grid)
    rho_grid = (planet.mu * const.u).to(u.g) * n_grid
    integrand = (4.0 * np.pi * r_grid**2 * rho_grid).to(u.g / u.cm)
    mass_g = trapezoid(integrand.to_value(u.g / u.cm), z_grid.to_value(u.cm)) * u.g
    return mass_g.to(u.g), z_top, source


def rocky_atmosphere_mass(
    planet_key: str,
    planet_case: dict,
    exobase_rows: Dict[Tuple[str, str], dict],
    n_z: int = 2048,
) -> Tuple[u.Quantity, u.Quantity, str]:
    return integrated_atmosphere_mass(planet_key, planet_case, exobase_rows, n_z=n_z)


def planet_reference_mass_info(
    planet_key: str,
    planet_case: dict,
    exobase_rows: Dict[Tuple[str, str], dict],
) -> dict:
    unit_name, unit_mass = rate_unit_info_for_planet(planet_case)
    if is_rocky_planet_case(planet_case):
        reference_mass, z_top, source = rocky_atmosphere_mass(planet_key, planet_case, exobase_rows)
        reference_kind = "whole_atmosphere"
    else:
        reference_mass = planet_case["mass"].to(u.g)
        z_top = None
        source = "planet_mass"
        reference_kind = "whole_planet"

    return {
        "planet_category": planet_category(planet_case),
        "reference_mass_kind": reference_kind,
        "reference_mass_g": reference_mass.to_value(u.g),
        "reference_mass_unit_name": unit_name,
        "reference_mass_in_unit": (reference_mass / unit_mass).decompose().value,
        "rate_unit_name": f"{unit_name}/yr",
        "reference_top_km": "" if z_top is None else z_top.to_value(u.km),
        "reference_top_source": source,
        "unit_mass_g": unit_mass.to_value(u.g),
    }


def mass_loss_in_planet_unit_per_year(mdot_g_s: np.ndarray, planet_case: dict) -> np.ndarray:
    _, unit_mass = rate_unit_info_for_planet(planet_case)
    return (np.asarray(mdot_g_s, dtype=float) * u.g / u.s * u.yr / unit_mass).decompose().value


def mass_loss_over_reference_mass_per_second(
    mdot_g_s: np.ndarray,
    planet_key: str,
    planet_case: dict,
    exobase_rows: Dict[Tuple[str, str], dict],
) -> Tuple[np.ndarray, dict]:
    info = planet_reference_mass_info(planet_key, planet_case, exobase_rows)
    reference_mass_g = float(info["reference_mass_g"])
    ratio = np.asarray(mdot_g_s, dtype=float) / reference_mass_g
    return ratio, info


def save_mass_loss_matrix_txt(
    output_path: pathlib.Path,
    dataset_name: str,
    x_label: str,
    x_unit: str,
    y_label: str,
    y_unit: str,
    x_values: Iterable[float],
    y_matrix: np.ndarray,
    series_values: Iterable[str | float],
    series_label: str,
    series_unit: str,
    column_names: List[str],
    extra_metadata: dict,
) -> None:
    save_plotdata_txt(
        output_path,
        dataset_name=dataset_name,
        x_label=x_label,
        x_unit=x_unit,
        y_label=y_label,
        y_unit=y_unit,
        x_values=np.asarray(list(x_values), dtype=float),
        y_matrix=np.asarray(y_matrix, dtype=float),
        series_values=list(series_values),
        series_label=series_label,
        series_unit=series_unit,
        column_names=column_names,
        extra_metadata=extra_metadata,
    )


def build_planet(planet_case: dict) -> Planet:
    return Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )


def integrate_density_segment(
    planet: Planet,
    abundance: float,
    rho_cm: float,
    x0_cm: float,
    x1_cm: float,
    r_exobase_cm: float,
    cluster_at: str,
) -> float:
    if x1_cm <= x0_cm:
        return 0.0

    t = np.linspace(0.0, 1.0, COLUMN_STEPS)
    if cluster_at == "left":
        t = t**COLUMN_GRID_POWER
    elif cluster_at == "right":
        t = 1.0 - (1.0 - t) ** COLUMN_GRID_POWER

    x_cm = x0_cm + (x1_cm - x0_cm) * t
    r_cm = np.sqrt(x_cm**2 + rho_cm**2)
    shell_mask = r_cm >= r_exobase_cm
    if not np.any(shell_mask):
        return 0.0

    n_species_cm3 = np.zeros_like(x_cm)
    z_cm = (r_cm[shell_mask] - planet.radius.to_value(u.cm)) * u.cm
    n_species_cm3[shell_mask] = abundance * planet.number_density(z_cm).to_value(1 / u.cm**3)
    return float(trapezoid(n_species_cm3, x_cm))


def integrate_density_segment_closest(
    planet: Planet,
    abundance: float,
    rho_cm: float,
    x0_cm: float,
    x1_cm: float,
    r_exobase_cm: float,
) -> float:
    if x1_cm <= x0_cm:
        return 0.0

    if x0_cm <= 0.0 <= x1_cm:
        return (
            integrate_density_segment(planet, abundance, rho_cm, x0_cm, 0.0, r_exobase_cm, "right")
            + integrate_density_segment(planet, abundance, rho_cm, 0.0, x1_cm, r_exobase_cm, "left")
        )

    if abs(x0_cm) <= abs(x1_cm):
        return integrate_density_segment(planet, abundance, rho_cm, x0_cm, x1_cm, r_exobase_cm, "left")
    return integrate_density_segment(planet, abundance, rho_cm, x0_cm, x1_cm, r_exobase_cm, "right")


def upstream_shell_column_to_cell(
    planet: Planet,
    abundance: float,
    rho_cm: float,
    x_cell_cm: float,
    r_exobase_cm: float,
    hill_cm: float,
) -> float:
    x_hill = np.sqrt(max(hill_cm**2 - rho_cm**2, 0.0))
    x_left = -x_hill

    if x_cell_cm <= x_left:
        return 0.0

    if rho_cm < r_exobase_cm:
        x_inner = np.sqrt(max(r_exobase_cm**2 - rho_cm**2, 0.0))
        if x_cell_cm <= -x_inner:
            return integrate_density_segment_closest(
                planet, abundance, rho_cm, x_left, x_cell_cm, r_exobase_cm
            )

        column = integrate_density_segment_closest(
            planet, abundance, rho_cm, x_left, -x_inner, r_exobase_cm
        )
        if x_cell_cm > x_inner:
            column += integrate_density_segment_closest(
                planet, abundance, rho_cm, x_inner, x_cell_cm, r_exobase_cm
            )
        return column

    return integrate_density_segment_closest(
        planet, abundance, rho_cm, x_left, x_cell_cm, r_exobase_cm
    )


def spherical_shell_cells(
    planet: Planet,
    abundance: float,
    species_mass_g: float,
    r_exobase_cm: float,
    hill_cm: float,
) -> dict:
    radial_fraction = np.linspace(0.0, 1.0, N_RHO + 1) ** RHO_GRID_POWER
    r_edges = r_exobase_cm + (hill_cm - r_exobase_cm) * radial_fraction
    theta_edges = np.linspace(0.0, np.pi, N_X + 1)

    x_values = []
    rho_values = []
    r_values = []
    dm_values = []
    ncol_values = []

    for i in range(N_RHO):
        r1 = r_edges[i]
        r2 = r_edges[i + 1]
        r_mid = 0.5 * (r1 + r2)
        z_mid = (r_mid - planet.radius.to_value(u.cm)) * u.cm
        n_species_cm3 = abundance * float(np.squeeze(planet.number_density(z_mid).to_value(1 / u.cm**3)))

        for j in range(N_X):
            theta1 = theta_edges[j]
            theta2 = theta_edges[j + 1]
            theta_mid = 0.5 * (theta1 + theta2)
            rho_cm = r_mid * np.sin(theta_mid)
            if rho_cm < r_exobase_cm:
                continue

            dV_cm3 = (2.0 * np.pi / 3.0) * (r2**3 - r1**3) * (np.cos(theta1) - np.cos(theta2))
            dm_g = n_species_cm3 * species_mass_g * dV_cm3
            if not np.isfinite(dm_g) or dm_g <= 0.0:
                continue

            x_cm = r_mid * np.cos(theta_mid)
            ncol_cm2 = upstream_shell_column_to_cell(
                planet,
                abundance,
                rho_cm,
                x_cm,
                r_exobase_cm,
                hill_cm,
            )

            x_values.append(x_cm)
            rho_values.append(rho_cm)
            r_values.append(r_mid)
            dm_values.append(dm_g)
            ncol_values.append(ncol_cm2)

    return {
        "x_cm": np.asarray(x_values, dtype=float),
        "rho_cm": np.asarray(rho_values, dtype=float),
        "r_cm": np.asarray(r_values, dtype=float),
        "dm_g": np.asarray(dm_values, dtype=float),
        "ncol_cm2": np.asarray(ncol_values, dtype=float),
    }


def initialize_base_namespace() -> None:
    base_mass_loss.N_RHO = N_RHO
    base_mass_loss.N_X = N_X
    base_mass_loss.RHO_GRID_POWER = RHO_GRID_POWER
    base_mass_loss.X_GRID_POWER = X_GRID_POWER
    base_mass_loss.COLUMN_STEPS = COLUMN_STEPS
    base_mass_loss.COLUMN_GRID_POWER = COLUMN_GRID_POWER
    base_mass_loss.WAVEMIN = WAVEMIN
    base_mass_loss.WAVEMAX = WAVEMAX
    base_mass_loss.B_ATOM = B_ATOM
    base_mass_loss.NPTS_ATOM = NPTS_ATOM
    base_mass_loss.ATOM_SPECIES = ATOM_SPECIES
    base_mass_loss.MOLECULE_TEMPLATES = MOLECULE_TEMPLATES
    base_mass_loss.PhotonPressure = PhotonPressure
    base_mass_loss.PlanetarySystem = PlanetarySystem
    base_mass_loss.STAR_TEMPLATES = STAR_TEMPLATES
    base_mass_loss.PLANET_TEMPLATES = PLANET_TEMPLATES
    base_mass_loss.infer_teff_from_star_template = infer_teff_from_star_template
    base_mass_loss.get_planet_template = get_planet_template
    base_mass_loss.get_star = get_star
    base_mass_loss.get_atom_profile = get_atom_profile
    base_mass_loss.species_mixing_ratio = species_mixing_ratio
    base_mass_loss.exobase_height = exobase_height
    base_mass_loss.load_exobase_table = load_exobase_table
    base_mass_loss.safe_name = safe_name
    base_mass_loss.integrated_atmosphere_mass = integrated_atmosphere_mass
    base_mass_loss.planet_reference_mass_info = planet_reference_mass_info
    base_mass_loss.mass_loss_in_planet_unit_per_year = mass_loss_in_planet_unit_per_year
    base_mass_loss.mass_loss_over_reference_mass_per_second = mass_loss_over_reference_mass_per_second
    base_mass_loss.save_mass_loss_matrix_txt = save_mass_loss_matrix_txt
    base_mass_loss.build_planet = build_planet
    base_mass_loss.upstream_shell_column_to_cell = upstream_shell_column_to_cell
    base_mass_loss.spherical_shell_cells = spherical_shell_cells


def family_enabled(name: str) -> bool:
    selected = str(RUN_FAMILY).strip().lower()
    if selected == "all":
        return True
    return selected == str(name).strip().lower()


def is_real_system_key(system_key: str) -> bool:
    return system_key in REAL_MASS_LOSS_REFERENCE_SYSTEMS


def get_system_star_case(system: AdvancedSystem) -> dict:
    if is_real_system_key(system.planet_key):
        return dict(get_real_mass_loss_reference_system(system.planet_key)["star"])
    return dict(base_mass_loss.STAR_TEMPLATES[system.star_key])


def get_system_planet_case(system: AdvancedSystem) -> dict:
    if is_real_system_key(system.planet_key):
        return dict(get_real_mass_loss_reference_system(system.planet_key)["planet"])
    return dict(base_mass_loss.get_planet_template(system.planet_key))


def get_system_display_names(system: AdvancedSystem) -> Tuple[str, str]:
    if is_real_system_key(system.planet_key):
        system_def = get_real_mass_loss_reference_system(system.planet_key)
        return system_def["planet"]["label"], system_def["star"]["label"]
    planet_case = base_mass_loss.get_planet_template(system.planet_key)
    star_case = base_mass_loss.STAR_TEMPLATES[system.star_key]
    return planet_case.get("label", system.planet_key), star_case.get("label", system.star_key)


def get_system_source_urls(system: AdvancedSystem) -> Tuple[str, str, str]:
    if is_real_system_key(system.planet_key):
        system_def = get_real_mass_loss_reference_system(system.planet_key)
        return (
            str(system_def.get("planet_source_url", "")),
            str(system_def.get("star_source_url", "")),
            str(system_def.get("orbit_source_url", "")),
        )
    return "", "", ""


def get_system_actual_teff_k(system: AdvancedSystem) -> float:
    star_case = get_system_star_case(system)
    if "teff_K" in star_case:
        return float(star_case["teff_K"])
    return float(base_mass_loss.infer_teff_from_star_template(system.star_key))


def get_system_spectrum_template_key(system: AdvancedSystem) -> str:
    if is_real_system_key(system.planet_key):
        return str(get_real_mass_loss_reference_system(system.planet_key)["spectrum_template_key"])
    return system.star_key


def get_system_exobase_planet_key(system: AdvancedSystem) -> str:
    return system.exobase_planet_key or system.planet_key


def build_system_star(system: AdvancedSystem) -> Star:
    star_case = get_system_star_case(system)
    return Star(
        star_case["path"],
        star_case["radius"],
        star_case["mass"],
        vsini=star_case["vsini"],
        epsilon=star_case["epsilon"],
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def configure_base_module() -> None:
    base_mass_loss.N_RHO = N_RHO
    base_mass_loss.N_X = N_X
    base_mass_loss.RHO_GRID_POWER = RHO_GRID_POWER
    base_mass_loss.X_GRID_POWER = X_GRID_POWER
    base_mass_loss.COLUMN_STEPS = COLUMN_STEPS
    base_mass_loss.COLUMN_GRID_POWER = COLUMN_GRID_POWER


def is_atomic_species(species: str) -> bool:
    return species in base_mass_loss.ATOM_SPECIES


def is_molecular_species(species: str) -> bool:
    return species in base_mass_loss.MOLECULE_TEMPLATES


def selected_species_for_planet(planet_case: dict) -> List[str]:
    selected = []
    for species in planet_case["composition"].keys():
        if is_atomic_species(species):
            if SKIP_ATOMS:
                continue
            if SELECTED_ATOMIC_SPECIES is not None and species not in SELECTED_ATOMIC_SPECIES:
                continue
            selected.append(species)
            continue

        if is_molecular_species(species):
            if SKIP_MOLECULES:
                continue
            if SELECTED_MOLECULAR_SPECIES is not None and species not in SELECTED_MOLECULAR_SPECIES:
                continue
            selected.append(species)

    return selected


molecule_profile_cache: Dict[str, BroadeningProfileMolecule] = {}


def get_molecule_profile(species: str) -> BroadeningProfileMolecule:
    cached = molecule_profile_cache.get(species)
    if cached is not None:
        return cached

    template = base_mass_loss.MOLECULE_TEMPLATES[species]
    fetch_kwargs = template["fetch_kwargs"]
    molecule = Molecule(species, base_mass_loss.WAVEMIN, base_mass_loss.WAVEMAX)
    source = template.get("source", "exomol").lower()

    if source == "hitran":
        molecule.fetch_hitran(**fetch_kwargs)
    else:
        molecule.fetch_exomol(
            path=fetch_kwargs["path"],
            database=fetch_kwargs["database"],
            localdatabase=os.environ.get(
                "EXOMOL_LOCALDATABASE",
                fetch_kwargs.get("localdatabase", "exomol_data"),
            ),
        )

    profile = BroadeningProfileMolecule(molecule, base_mass_loss.B_ATOM, profileType="Voigt")
    if hasattr(profile, "temp_strength_rel_cutoff"):
        profile.temp_strength_rel_cutoff = 1e-8
    molecule_profile_cache[species] = profile
    return profile


def get_species_profile(species: str):
    if is_molecular_species(species):
        return get_molecule_profile(species)
    return base_mass_loss.get_atom_profile(species)


class PhotonPressureBinnedCache:
    def __init__(
        self,
        photon_pressure,
        species_mass: u.Quantity,
        planet_temperature: u.Quantity,
        distance: u.Quantity,
        log_column_bin_dex: float,
    ) -> None:
        self.photon_pressure = photon_pressure
        self.species_mass = species_mass.to(u.g)
        self.planet_temperature = planet_temperature
        self.distance = distance
        self.log_column_bin_dex = float(log_column_bin_dex)
        self.cache: Dict[int, float] = {}

    def acceleration(self, ncol_cm2: float) -> float:
        ncol_cm2 = max(float(ncol_cm2), MIN_COLUMN_CM2)
        key = int(np.round(np.log10(ncol_cm2) / self.log_column_bin_dex))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        eval_column_cm2 = 10.0 ** (key * self.log_column_bin_dex)
        force, _, _, _ = self.photon_pressure.calc_PhotonPressure(
            np.array([eval_column_cm2]) / u.cm**2,
            self.planet_temperature,
            self.distance,
        )
        accel = (np.ravel(force.to(u.dyn))[0] / self.species_mass).to_value(u.cm / u.s**2)
        self.cache[key] = accel
        return accel

    @property
    def n_bins(self) -> int:
        return len(self.cache)


_WORKER_EXOBASE_ROWS = None


def build_species_task(
    system_def: AdvancedSystem,
    species: str,
    planet_case: dict,
    extra_fields: dict | None = None,
) -> dict:
    planet_label, star_label = get_system_display_names(system_def)
    actual_teff_k = get_system_actual_teff_k(system_def)
    planet_source_url, star_source_url, orbit_source_url = get_system_source_urls(system_def)
    exobase_planet_key = get_system_exobase_planet_key(system_def)
    spectrum_template_key = get_system_spectrum_template_key(system_def)
    return {
        "system_def": system_def,
        "species": species,
        "planet_case": planet_case,
        "planet_label": planet_label,
        "star_label": star_label,
        "actual_teff_k": actual_teff_k,
        "planet_source_url": planet_source_url,
        "star_source_url": star_source_url,
        "orbit_source_url": orbit_source_url,
        "exobase_planet_key": exobase_planet_key,
        "spectrum_template_key": spectrum_template_key,
        "extra_fields": dict(extra_fields or {}),
    }


def _species_worker_init(exobase_rows) -> None:
    global _WORKER_EXOBASE_ROWS
    configure_base_module()
    initialize_base_namespace()
    _WORKER_EXOBASE_ROWS = exobase_rows


def _compute_species_task(task: dict, exobase_rows) -> dict:
    system_def: AdvancedSystem = task["system_def"]
    species = task["species"]
    planet_case = dict(task["planet_case"])
    start_time = time.perf_counter()

    try:
        planet = base_mass_loss.build_planet(planet_case)
        star = build_system_star(system_def)
        system = base_mass_loss.PlanetarySystem(planet, star, system_def.distance_au * u.AU)
        row = mass_loss_for_species_advanced(
            system_def.test_family,
            system_def.planet_key,
            system_def.star_key,
            system_def.distance_au,
            species,
            planet_case,
            system,
            exobase_rows,
            exobase_planet_key=task["exobase_planet_key"],
            target_stellar_teff_k=task["actual_teff_k"],
            planet_label=task["planet_label"],
            star_label=task["star_label"],
            spectrum_template_key=task["spectrum_template_key"],
            planet_source_url=task["planet_source_url"],
            star_source_url=task["star_source_url"],
            orbit_source_url=task["orbit_source_url"],
        )
        row.update(task["extra_fields"])
        return {
            "ok": True,
            "species": species,
            "system_def": system_def,
            "planet_label": task["planet_label"],
            "star_label": task["star_label"],
            "row": row,
            "elapsed_s": time.perf_counter() - start_time,
        }
    except Exception as exc:
        return {
            "ok": False,
            "species": species,
            "system_def": system_def,
            "planet_label": task["planet_label"],
            "star_label": task["star_label"],
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "elapsed_s": time.perf_counter() - start_time,
        }


def _species_worker_entry(task: dict) -> dict:
    return _compute_species_task(task, _WORKER_EXOBASE_ROWS)


def effective_species_max_workers(n_tasks: int, n_systems: int = 1) -> int:
    if n_tasks <= 1:
        return 1
    total_budget = max(1, int(MLA_SPECIES_MAX_WORKERS)) * max(1, min(int(MLA_PLANET_MAX_WORKERS), int(n_systems)))
    return max(1, min(int(total_budget), int(n_tasks)))


def iter_species_task_results(tasks: List[dict], exobase_rows, n_systems: int = 1):
    max_workers = effective_species_max_workers(len(tasks), n_systems=n_systems)
    if max_workers == 1:
        for task in tasks:
            yield _compute_species_task(task, exobase_rows)
        return

    with cf.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_species_worker_init,
        initargs=(exobase_rows,),
    ) as executor:
        futures = [executor.submit(_species_worker_entry, task) for task in tasks]
        for future in cf.as_completed(futures):
            yield future.result()


def batched(items: List, size: int):
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def is_total_species_label(species: str) -> bool:
    return str(species).startswith("TOTAL")


def local_acceleration(
    x_cm: float,
    y_cm: float,
    planet,
    abundance: float,
    r_exobase_cm: float,
    hill_cm: float,
    photon_cache: PhotonPressureBinnedCache,
    mu_planet: float,
    mu_star: float,
    distance_cm: float,
    initial_ncol_cm2: float | None = None,
) -> Tuple[float, float, float, float]:
    r_cm = float(np.hypot(x_cm, y_cm))
    if r_cm <= r_exobase_cm:
        raise ValueError("Particle fell below the exobase during trajectory integration.")

    impact_cm = abs(float(y_cm))
    ncol_cm2 = (
        float(initial_ncol_cm2)
        if initial_ncol_cm2 is not None
        else base_mass_loss.upstream_shell_column_to_cell(
            planet,
            abundance,
            impact_cm,
            float(x_cm),
            r_exobase_cm,
            hill_cm,
        )
    )

    a_rad = photon_cache.acceleration(ncol_cm2)
    ax = a_rad - mu_planet * x_cm / r_cm**3
    ay = -mu_planet * y_cm / r_cm**3

    if INCLUDE_STELLAR_GRAVITY:
        norm = ((distance_cm + x_cm) ** 2 + y_cm**2) ** 1.5
        ax += -mu_star * (distance_cm + x_cm) / norm + mu_star / distance_cm**2
        ay += -mu_star * y_cm / norm

    return ax, ay, ncol_cm2, a_rad


def choose_timestep(
    x_cm: float,
    y_cm: float,
    vx_cm_s: float,
    vy_cm_s: float,
    ax_cm_s2: float,
    ay_cm_s2: float,
    hill_cm: float,
) -> float:
    speed_cm_s = max(float(np.hypot(vx_cm_s, vy_cm_s)), 1.0e-30)
    accel_cm_s2 = max(float(np.hypot(ax_cm_s2, ay_cm_s2)), 1.0e-30)
    r_cm = float(np.hypot(x_cm, y_cm))
    impact_cm = min(abs(float(y_cm)), hill_cm * (1.0 - 1.0e-12))
    x_hill_cm = float(np.sqrt(max(hill_cm**2 - impact_cm**2, DT_LENGTH_FLOOR_CM**2)))
    remaining_x_cm = max(x_hill_cm - x_cm, DT_LENGTH_FLOOR_CM)
    remaining_r_cm = max(hill_cm - r_cm, DT_LENGTH_FLOOR_CM)
    length_scale_cm = max(
        min(remaining_x_cm, remaining_r_cm, MAX_STEP_LENGTH_FRACTION * hill_cm),
        DT_LENGTH_FLOOR_CM,
    )

    dt_acc_s = DT_FRACTION * np.sqrt(length_scale_cm / accel_cm_s2)
    dt_vel_s = DT_FRACTION * length_scale_cm / speed_cm_s
    dt_s = min(dt_acc_s, dt_vel_s, DT_MAX_S)
    return max(dt_s, DT_MIN_S)


def right_hill_crossing_fraction(
    x0_cm: float,
    y0_cm: float,
    x1_cm: float,
    y1_cm: float,
    hill_cm: float,
) -> float:
    f0 = x0_cm**2 + y0_cm**2 - hill_cm**2
    f1 = x1_cm**2 + y1_cm**2 - hill_cm**2
    if f1 == f0:
        return 1.0
    fraction = -f0 / (f1 - f0)
    return float(np.clip(fraction, 0.0, 1.0))


def integrate_escape_trajectory(
    x0_cm: float,
    y0_cm: float,
    initial_ncol_cm2: float,
    planet,
    abundance: float,
    r_exobase_cm: float,
    hill_cm: float,
    photon_cache: PhotonPressureBinnedCache,
    mu_planet: float,
    mu_star: float,
    distance_cm: float,
) -> dict:
    x_cm = float(x0_cm)
    y_cm = float(y0_cm)
    vx_cm_s = 0.0
    vy_cm_s = 0.0
    time_s = 0.0
    step_count = 0

    ax_cm_s2, ay_cm_s2, _, a_rad0 = local_acceleration(
        x_cm,
        y_cm,
        planet,
        abundance,
        r_exobase_cm,
        hill_cm,
        photon_cache,
        mu_planet,
        mu_star,
        distance_cm,
        initial_ncol_cm2=initial_ncol_cm2,
    )

    while step_count < MAX_STEPS and time_s < MAX_TIME_S:
        if x_cm < 0.0 and vx_cm_s <= 0.0 and ax_cm_s2 <= 0.0:
            break

        dt_s = choose_timestep(x_cm, y_cm, vx_cm_s, vy_cm_s, ax_cm_s2, ay_cm_s2, hill_cm)
        x_new_cm = x_cm + vx_cm_s * dt_s + 0.5 * ax_cm_s2 * dt_s**2
        y_new_cm = y_cm + vy_cm_s * dt_s + 0.5 * ay_cm_s2 * dt_s**2
        r_new_cm = float(np.hypot(x_new_cm, y_new_cm))

        if r_new_cm <= r_exobase_cm:
            break

        if r_new_cm >= hill_cm:
            if x_new_cm > 0.0:
                crossing_fraction = right_hill_crossing_fraction(x_cm, y_cm, x_new_cm, y_new_cm, hill_cm)
                return {
                    "escaped": True,
                    "escape_time_s": time_s + crossing_fraction * dt_s,
                    "step_count": step_count + 1,
                    "initial_a_rad_cm_s2": a_rad0,
                }
            break

        ax_new_cm_s2, ay_new_cm_s2, _, _ = local_acceleration(
            x_new_cm,
            y_new_cm,
            planet,
            abundance,
            r_exobase_cm,
            hill_cm,
            photon_cache,
            mu_planet,
            mu_star,
            distance_cm,
        )

        vx_cm_s += 0.5 * (ax_cm_s2 + ax_new_cm_s2) * dt_s
        vy_cm_s += 0.5 * (ay_cm_s2 + ay_new_cm_s2) * dt_s
        x_cm = x_new_cm
        y_cm = y_new_cm
        ax_cm_s2 = ax_new_cm_s2
        ay_cm_s2 = ay_new_cm_s2
        time_s += dt_s
        step_count += 1

    return {
        "escaped": False,
        "escape_time_s": np.nan,
        "step_count": step_count,
        "initial_a_rad_cm_s2": a_rad0,
    }


def output_fieldnames() -> List[str]:
    return [
        "test_family",
        "planet",
        "planet_label",
        "star",
        "star_label",
        "spectrum_template_key",
        "planet_source_url",
        "star_source_url",
        "orbit_source_url",
        "exobase_template_key",
        "target_stellar_teff_K",
        "actual_stellar_teff_K",
        "distance_AU",
        "P0_bar",
        "mu_amu",
        "surface_gravity_m_s2",
        "mass_scale",
        "species",
        "mixing_ratio",
        "z_exobase_km",
        "r_exobase_over_Rp",
        "hill_radius_over_Rp",
        "total_shell_mass_g",
        "escaping_shell_mass_g",
        "mass_loss_rate_g_s",
        "mass_loss_rate_kg_s",
        "mass_loss_rate_Mearth_yr",
        "mass_lost_g_1Myr",
        "mass_lost_g_1Gyr",
        "mass_lost_Mearth_1Myr",
        "mass_lost_Mearth_1Gyr",
        "mean_escape_time_s",
        "median_escape_time_s",
        "min_escape_time_s",
        "mass_weighted_initial_beta",
        "mean_steps_escaped",
        "max_steps_any_cell",
        "photon_pressure_cache_bins",
        "n_cells",
        "n_escape_cells",
        "include_stellar_gravity",
        "method",
        "n_rho",
        "n_x",
        "column_steps",
        "rho_grid_power",
        "log_column_bin_dex",
        "dt_min_s",
        "dt_max_s",
        "max_steps",
        "max_time_s",
    ]


def checkpoint_fieldnames() -> List[str]:
    return output_fieldnames()


def current_checkpoint_config() -> dict:
    return {
        "include_stellar_gravity": bool(INCLUDE_STELLAR_GRAVITY),
        "method": "advanced_trajectory_recomputed_acceleration",
        "n_rho": int(N_RHO),
        "n_x": int(N_X),
        "column_steps": int(COLUMN_STEPS),
        "rho_grid_power": float(RHO_GRID_POWER),
        "log_column_bin_dex": float(LOG_COLUMN_BIN_DEX),
        "dt_min_s": float(DT_MIN_S),
        "dt_max_s": float(DT_MAX_S),
        "max_steps": int(MAX_STEPS),
        "max_time_s": float(MAX_TIME_S),
    }


def validate_checkpoint_rows(rows: List[dict]) -> None:
    if not rows:
        return

    expected = current_checkpoint_config()
    for row in rows:
        required_keys = {"test_family", "planet", "species", "star", "distance_AU"}
        if not required_keys.issubset(row.keys()):
            raise ValueError(
                "Existing advanced mass-loss checkpoint is from an older layout and cannot be resumed. "
                f"Delete {CHECKPOINT_PATH} before running the new fixed-system version."
            )
        actual = {
            "include_stellar_gravity": parse_bool(row.get("include_stellar_gravity", "")),
            "method": row.get("method", ""),
            "n_rho": int(float(row.get("n_rho", np.nan))),
            "n_x": int(float(row.get("n_x", np.nan))),
            "column_steps": int(float(row.get("column_steps", np.nan))),
            "rho_grid_power": float(row.get("rho_grid_power", np.nan)),
            "log_column_bin_dex": float(row.get("log_column_bin_dex", np.nan)),
            "dt_min_s": float(row.get("dt_min_s", np.nan)),
            "dt_max_s": float(row.get("dt_max_s", np.nan)),
            "max_steps": int(float(row.get("max_steps", np.nan))),
            "max_time_s": float(row.get("max_time_s", np.nan)),
        }
        if actual != expected:
            raise ValueError(
                "Existing advanced mass-loss checkpoint was created with different settings. "
                f"Delete {CHECKPOINT_PATH} or restore the old settings before resuming."
            )


def load_checkpoint_rows(path: pathlib.Path) -> List[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            row for row in reader
            if row.get("species") and not is_total_species_label(row.get("species", ""))
        ]

    validate_checkpoint_rows(rows)
    return rows


def save_checkpoint_rows(rows: List[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=checkpoint_fieldnames())
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in checkpoint_fieldnames()})
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def family_results_path(test_family: str) -> pathlib.Path:
    return OUTPUT_DIR / f"{base_mass_loss.safe_name(test_family)}.txt"


def load_family_rows(test_family: str) -> List[dict]:
    path = family_results_path(test_family)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [
            row
            for row in reader
            if row.get("species") and not is_total_species_label(row.get("species", ""))
        ]


def sort_rows_for_output(rows: Iterable[dict]) -> List[dict]:
    def sort_key(row: dict):
        value_keys = (
            float(row.get("distance_AU") or np.nan),
            float(row.get("P0_bar") or np.nan) if row.get("P0_bar", "") != "" else np.nan,
            float(row.get("mu_amu") or np.nan) if row.get("mu_amu", "") != "" else np.nan,
            float(row.get("surface_gravity_m_s2") or np.nan) if row.get("surface_gravity_m_s2", "") != "" else np.nan,
            row.get("planet", ""),
            row.get("star", ""),
            row.get("species", ""),
        )
        return value_keys

    return sorted(rows, key=sort_key)


def family_case_key(row: dict) -> Tuple[str, str, str, str, str, str]:
    return (
        row.get("planet", ""),
        row.get("star", ""),
        f"{float(row.get('distance_AU', np.nan)):.12g}",
        row.get("P0_bar", ""),
        row.get("mu_amu", ""),
        row.get("surface_gravity_m_s2", ""),
    )


def write_family_results_txt(test_family: str, rows: List[dict]) -> pathlib.Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = family_results_path(test_family)
    species_rows = sort_rows_for_output(rows)
    rows_to_write: List[dict] = []
    current_case = None
    current_species_rows: List[dict] = []

    for row in species_rows:
        case_key = family_case_key(row)
        if current_case is None:
            current_case = case_key
        if case_key != current_case:
            rows_to_write.extend(current_species_rows)
            total_row = total_row_from_species_rows(current_species_rows)
            if total_row is not None:
                rows_to_write.append(total_row)
            current_case = case_key
            current_species_rows = []
        current_species_rows.append(row)

    if current_species_rows:
        rows_to_write.extend(current_species_rows)
        total_row = total_row_from_species_rows(current_species_rows)
        if total_row is not None:
            rows_to_write.append(total_row)

    fd, tmp_name = tempfile.mkstemp(prefix=output_path.stem + "_", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames(), delimiter="\t")
            writer.writeheader()
            for row in rows_to_write:
                writer.writerow({key: row.get(key, "") for key in output_fieldnames()})
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return output_path


def row_key_from_values(
    test_family: str,
    planet_key: str,
    species: str,
    actual_star_key: str,
    distance_au: float,
) -> Tuple[str, str, str, str, str]:
    return test_family, planet_key, species, actual_star_key, f"{float(distance_au):.12g}"


def row_key_from_row(row: dict) -> Tuple[str, str, str, str, str]:
    return row_key_from_values(
        row["test_family"],
        row["planet"],
        row["species"],
        row["star"],
        float(row["distance_AU"]),
    )


def completed_row_keys(rows: List[dict]) -> set[Tuple[str, str, str, str, str]]:
    return {row_key_from_row(row) for row in rows}


def write_results_csv(rows: List[dict], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in output_fieldnames()})


def write_text_atomic(text: str, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=output_path.stem + "_", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def format_numeric_list(values) -> str:
    return ", ".join(f"{float(value):.12g}" for value in values)


def format_selected_species(planet_case: dict) -> str:
    species = selected_species_for_planet(planet_case)
    if not species:
        return "none"
    return ", ".join(species)


def format_composition(planet_case: dict) -> str:
    composition = []
    for species, value in planet_case["composition"].items():
        if isinstance(value, u.Quantity):
            numeric_value = value.to_value(u.dimensionless_unscaled)
        else:
            numeric_value = float(value)
        composition.append(f"{species}={numeric_value:.12g}")
    return ", ".join(composition)


def common_run_parameter_lines() -> List[str]:
    return [
        f"skip_atoms: {SKIP_ATOMS}",
        f"skip_molecules: {SKIP_MOLECULES}",
        f"selected_atomic_species: {SELECTED_ATOMIC_SPECIES if SELECTED_ATOMIC_SPECIES is not None else 'all available'}",
        f"selected_molecular_species: {SELECTED_MOLECULAR_SPECIES if SELECTED_MOLECULAR_SPECIES is not None else 'all available'}",
        f"include_stellar_gravity: {INCLUDE_STELLAR_GRAVITY}",
        f"use_checkpoint: {USE_CHECKPOINT}",
        f"species_max_workers: {MLA_SPECIES_MAX_WORKERS}",
        f"planet_max_workers: {MLA_PLANET_MAX_WORKERS}",
        f"selected_system_keys: {SELECTED_SYSTEM_KEYS if SELECTED_SYSTEM_KEYS is not None else 'all enabled systems'}",
        f"n_rho: {N_RHO}",
        f"n_x: {N_X}",
        f"rho_grid_power: {RHO_GRID_POWER}",
        f"x_grid_power: {X_GRID_POWER}",
        f"column_steps: {COLUMN_STEPS}",
        f"column_grid_power: {COLUMN_GRID_POWER}",
        f"log_column_bin_dex: {LOG_COLUMN_BIN_DEX}",
        f"dt_fraction: {DT_FRACTION}",
        f"dt_min_s: {DT_MIN_S}",
        f"dt_max_s: {DT_MAX_S}",
        f"dt_length_floor_cm: {DT_LENGTH_FLOOR_CM}",
        f"max_step_length_fraction: {MAX_STEP_LENGTH_FRACTION}",
        f"max_steps: {MAX_STEPS}",
        f"max_time_s: {MAX_TIME_S}",
        f"wavemin_AA: {base_mass_loss.WAVEMIN.to_value(u.AA):.12g}",
        f"wavemax_AA: {base_mass_loss.WAVEMAX.to_value(u.AA):.12g}",
        f"atom_b_kms: {base_mass_loss.B_ATOM.to_value(u.km / u.s):.12g}",
        f"atom_npts: {base_mass_loss.NPTS_ATOM}",
        f"output_dir: {OUTPUT_DIR}",
    ]


def star_parameter_lines(star_key: str) -> List[str]:
    star_case = base_mass_loss.STAR_TEMPLATES[star_key]
    return [
        f"star_template: {star_key}",
        f"star_label: {star_case.get('label', '')}",
        f"star_category: {star_case.get('category', '')}",
        f"stellar_teff_K: {base_mass_loss.infer_teff_from_star_template(star_key)}",
        f"stellar_radius_Rsun: {star_case['radius'].to_value(const.R_sun):.12g}",
        f"stellar_mass_Msun: {star_case['mass'].to_value(const.M_sun):.12g}",
        f"stellar_vsini_kms: {star_case['vsini'].to_value(u.km / u.s):.12g}",
        f"stellar_epsilon: {star_case['epsilon'].to_value(u.dimensionless_unscaled):.12g}",
        f"stellar_spectrum_path: {star_case['path']}",
    ]


def planet_parameter_lines(planet_key: str, planet_case: dict | None = None) -> List[str]:
    active_planet_case = dict(base_mass_loss.get_planet_template(planet_key) if planet_case is None else planet_case)
    return [
        f"planet_template: {planet_key}",
        f"planet_label: {active_planet_case.get('label', '')}",
        f"planet_category: {active_planet_case.get('category', '')}",
        f"planet_radius_Rearth: {active_planet_case['radius'].to_value(const.R_earth):.12g}",
        f"planet_mass_Mearth: {active_planet_case['mass'].to_value(const.M_earth):.12g}",
        f"planet_temperature_K: {active_planet_case['T'].to_value(u.K):.12g}",
        f"planet_mu_amu: {active_planet_case['mu'].to_value(u.dimensionless_unscaled):.12g}",
        f"planet_P0_bar: {active_planet_case['P0'].to_value(u.bar):.12g}",
        f"planet_surface_gravity_m_s2: {surface_gravity_m_s2_for_planet_case(active_planet_case):.12g}",
        f"selected_species: {format_selected_species(active_planet_case)}",
        f"composition: {format_composition(active_planet_case)}",
        f"planet_notes: {active_planet_case.get('notes', '')}",
    ]


def standard_family_parameter_path(test_family: str) -> pathlib.Path:
    family_dir = OUTPUT_DIR / base_mass_loss.safe_name(test_family)
    return family_dir / f"{base_mass_loss.safe_name(test_family)}_run_parameters.txt"


def write_standard_family_run_parameters(test_family: str, systems: List[AdvancedSystem]) -> pathlib.Path:
    lines = [
        "Advanced Mass-Loss Standard Family Parameters",
        "============================================",
        f"test_family: {test_family}",
        f"n_systems: {len(systems)}",
    ]
    if test_family == "distance_sweeps":
        lines.append(f"distance_sweep_values_AU: {format_numeric_list(HOT_SWEEP_DISTANCES_AU)}")
    if test_family in {"rocky_exoplanets_plausible", "gas_planets_plausible"}:
        lines.append(f"plausible_planet_distance_AU: {PLAUSIBLE_PLANET_DISTANCE_AU:.12g}")
    lines.extend(common_run_parameter_lines())

    for idx, system in enumerate(systems, start=1):
        lines.extend(
            [
                "",
                f"[system_{idx}]",
                f"distance_AU: {float(system.distance_au):.12g}",
            ]
        )
        lines.extend(star_parameter_lines(system.star_key))
        lines.extend(planet_parameter_lines(system.planet_key))

    output_path = standard_family_parameter_path(test_family)
    write_text_atomic("\n".join(lines) + "\n", output_path)
    return output_path


def write_p0_sweep_run_parameters() -> pathlib.Path:
    lines = [
        "Advanced P0 Sweep Run Parameters",
        "================================",
        f"test_family: {P0_SWEEP_TEST_FAMILY}",
        "sweep_parameter: P0_bar",
        f"sweep_values_bar: {format_numeric_list(P0_SWEEP_VALUES_BAR)}",
        f"distance_AU: {float(P0_SWEEP_DISTANCE_AU):.12g}",
    ]
    lines.extend(common_run_parameter_lines())
    lines.extend(["", "[base_system]"])
    lines.extend(star_parameter_lines(P0_SWEEP_STAR_KEY))
    lines.extend(planet_parameter_lines(P0_SWEEP_PLANET_KEY))

    output_path = P0_SWEEP_OUTPUT_DIR / f"{base_mass_loss.safe_name(P0_SWEEP_TEST_FAMILY)}_run_parameters.txt"
    write_text_atomic("\n".join(lines) + "\n", output_path)
    return output_path


def write_mu_sweep_run_parameters(systems: List[AdvancedSystem]) -> pathlib.Path:
    lines = [
        "Advanced Mu Sweep Run Parameters",
        "================================",
        f"test_family: {MU_SWEEP_HOT_JUPITER_FAMILY}",
        "sweep_parameter: mu_amu",
        f"sweep_values_amu: {format_numeric_list(MU_SWEEP_VALUES_AMU)}",
        f"n_systems: {len(systems)}",
    ]
    lines.extend(common_run_parameter_lines())

    for idx, system in enumerate(systems, start=1):
        lines.extend(
            [
                "",
                f"[system_{idx}]",
                f"distance_AU: {float(system.distance_au):.12g}",
            ]
        )
        lines.extend(star_parameter_lines(system.star_key))
        lines.extend(planet_parameter_lines(system.planet_key))

    output_path = MU_SWEEP_OUTPUT_DIR / "mu_sweeps_run_parameters.txt"
    write_text_atomic("\n".join(lines) + "\n", output_path)
    return output_path


def write_surface_gravity_sweep_run_parameters(system_def: AdvancedSystem) -> pathlib.Path:
    surface_gravity_values = [
        surface_gravity_m_s2_for_planet_case(build_surface_gravity_sweep_planet_case(system_def.planet_key, mass_scale))
        for mass_scale in SURFACE_GRAVITY_MASS_SCALE_VALUES
    ]
    lines = [
        "Advanced Surface-Gravity Sweep Run Parameters",
        "=============================================",
        f"test_family: {SURFACE_GRAVITY_SWEEP_FAMILY}",
        "sweep_parameter: surface_gravity_m_s2",
        f"mass_scale_values: {format_numeric_list(SURFACE_GRAVITY_MASS_SCALE_VALUES)}",
        f"surface_gravity_values_m_s2: {format_numeric_list(surface_gravity_values)}",
        f"distance_AU: {float(system_def.distance_au):.12g}",
    ]
    lines.extend(common_run_parameter_lines())
    lines.extend(["", "[base_system]"])
    lines.extend(star_parameter_lines(system_def.star_key))
    lines.extend(planet_parameter_lines(system_def.planet_key))

    output_path = SURFACE_GRAVITY_SWEEP_OUTPUT_DIR / "surface_gravity_sweeps_run_parameters.txt"
    write_text_atomic("\n".join(lines) + "\n", output_path)
    return output_path


def write_summary_txt(
    rows: List[dict],
    output_path: pathlib.Path,
    actual_star_key: str,
    planet_key: str,
    test_family: str,
    distance_au: float,
) -> None:
    total_row = next((row for row in rows if is_total_species_label(row.get("species", ""))), None)
    atomic_rows = [row for row in rows if is_atomic_species(row.get("species", ""))]
    molecular_rows = [row for row in rows if is_molecular_species(row.get("species", ""))]
    planet_case = base_mass_loss.get_planet_template(planet_key)
    exobase_rows = base_mass_loss.load_exobase_table(EXOBASE_TABLE)
    mass_info = base_mass_loss.planet_reference_mass_info(
        planet_key,
        planet_case,
        exobase_rows,
    )
    total_planet_mass_kg = planet_case["mass"].to_value(u.kg)
    total_atmosphere_mass_g, atmosphere_top_km_q, atmosphere_top_source = base_mass_loss.integrated_atmosphere_mass(
        planet_key,
        planet_case,
        exobase_rows,
    )
    total_atmosphere_mass_kg = total_atmosphere_mass_g.to_value(u.kg)
    reference_kind_key = "whole_atmosphere" if mass_info["reference_mass_kind"] == "whole_atmosphere" else "whole_planet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Advanced Mass-Loss Run\n")
        f.write("======================\n")
        f.write(f"test_family: {test_family}\n")
        f.write(f"planet: {planet_key}\n")
        f.write(f"target_stellar_teff_K: {base_mass_loss.infer_teff_from_star_template(actual_star_key)}\n")
        f.write(f"star_template: {actual_star_key}\n")
        f.write(f"actual_stellar_teff_K: {base_mass_loss.infer_teff_from_star_template(actual_star_key)}\n")
        f.write(f"distance_AU: {float(distance_au)}\n")
        f.write(f"include_stellar_gravity: {INCLUDE_STELLAR_GRAVITY}\n")
        f.write("method: advanced_trajectory_recomputed_acceleration\n")
        f.write(
            f"grid: N_RHO={N_RHO}, N_X={N_X}, COLUMN_STEPS={COLUMN_STEPS}, "
            f"RHO_GRID_POWER={RHO_GRID_POWER}\n"
        )
        f.write(
            f"trajectory: LOG_COLUMN_BIN_DEX={LOG_COLUMN_BIN_DEX}, DT_MIN_S={DT_MIN_S}, "
            f"DT_MAX_S={DT_MAX_S}, MAX_STEPS={MAX_STEPS}, MAX_TIME_S={MAX_TIME_S}\n"
        )
        f.write(f"planet_category: {mass_info['planet_category']}\n")
        f.write(f"reference_mass_kind: {mass_info['reference_mass_kind']}\n")
        f.write(f"reference_mass_g: {mass_info['reference_mass_g']:.12e}\n")
        f.write(f"reference_mass_unit_name: {mass_info['reference_mass_unit_name']}\n")
        f.write(f"reference_mass_in_unit: {mass_info['reference_mass_in_unit']:.12e}\n")
        f.write(f"total_planet_mass_kg: {total_planet_mass_kg:.12e}\n")
        f.write(f"total_atmosphere_mass_kg: {total_atmosphere_mass_kg:.12e}\n")
        f.write(f"{reference_kind_key}_mass_g: {mass_info['reference_mass_g']:.12e}\n")
        f.write(
            f"{reference_kind_key}_mass_{mass_info['reference_mass_unit_name']}: "
            f"{mass_info['reference_mass_in_unit']:.12e}\n"
        )
        f.write(f"reference_atmosphere_mass_g: {total_atmosphere_mass_g.to_value(u.g):.12e}\n")
        f.write(f"reference_atmosphere_mass_kg: {total_atmosphere_mass_kg:.12e}\n")
        if mass_info["reference_top_km"] != "":
            f.write(f"reference_atmosphere_top_km: {float(mass_info['reference_top_km']):.6f}\n")
            f.write(f"reference_atmosphere_top_source: {mass_info['reference_top_source']}\n")
        else:
            f.write(f"reference_atmosphere_top_km: {atmosphere_top_km_q.to_value(u.km):.6f}\n")
            f.write(f"reference_atmosphere_top_source: {atmosphere_top_source}\n")
        if total_row is not None:
            total_mdot_g_s = float(total_row["mass_loss_rate_g_s"])
            total_mdot_g_s_array = np.array([total_mdot_g_s], dtype=float)
            total_rate_in_planet_unit_yr = base_mass_loss.mass_loss_in_planet_unit_per_year(total_mdot_g_s_array, planet_case)[0]
            total_specific_rate_s_inv, _ = base_mass_loss.mass_loss_over_reference_mass_per_second(
                total_mdot_g_s_array,
                planet_key,
                planet_case,
                exobase_rows,
            )
            total_specific_rate_over_atmosphere_s_inv = total_mdot_g_s / total_atmosphere_mass_g.to_value(u.g)
            f.write(f"total_mass_loss_rate_g_s: {total_row['mass_loss_rate_g_s']:.12e}\n")
            f.write(f"total_mass_loss_rate_{mass_info['reference_mass_unit_name']}_yr: {total_rate_in_planet_unit_yr:.12e}\n")
            f.write(f"total_mass_loss_rate_over_reference_mass_s_inv: {float(total_specific_rate_s_inv[0]):.12e}\n")
            f.write(
                f"total_mass_loss_rate_over_{reference_kind_key}_s_inv: "
                f"{float(total_specific_rate_s_inv[0]):.12e}\n"
            )
            f.write(
                "total_mass_loss_rate_over_reference_atmosphere_s_inv: "
                f"{total_specific_rate_over_atmosphere_s_inv:.12e}\n"
            )
            f.write(f"total_escaping_shell_mass_g: {total_row['escaping_shell_mass_g']:.12e}\n")
            f.write(f"total_mean_escape_time_s: {total_row['mean_escape_time_s']:.12e}\n")
        if atomic_rows:
            atomic_mdot = float(np.nansum([float(row.get("mass_loss_rate_g_s", 0.0)) for row in atomic_rows]))
            f.write(f"total_atomic_mass_loss_rate_g_s: {atomic_mdot:.12e}\n")
        if molecular_rows:
            molecular_mdot = float(np.nansum([float(row.get("mass_loss_rate_g_s", 0.0)) for row in molecular_rows]))
            f.write(f"total_molecular_mass_loss_rate_g_s: {molecular_mdot:.12e}\n")
        f.write("\nPer-species rows are stored in the companion CSV file.\n")


def system_rows_from_checkpoint(
    rows: List[dict],
    system: AdvancedSystem,
) -> List[dict]:
    matching = [
        row
        for row in rows
        if row.get("test_family") == system.test_family
        and row.get("planet") == system.planet_key
        and row.get("star") == system.star_key
        and float(row.get("distance_AU", np.nan)) == float(system.distance_au)
        and not is_total_species_label(row.get("species", ""))
    ]
    matching.sort(key=lambda row: (row.get("species", ""),))
    return matching


def output_paths_for_system(system: AdvancedSystem) -> Tuple[pathlib.Path, pathlib.Path]:
    family_dir = OUTPUT_DIR / base_mass_loss.safe_name(system.test_family)
    output_slug = (
        f"{base_mass_loss.safe_name(system.planet_key)}_"
        f"{base_mass_loss.safe_name(system.star_key)}_"
        f"{float(system.distance_au):g}AU"
    )
    return (
        family_dir / f"{output_slug}_mass_loss_advanced.csv",
        family_dir / f"{output_slug}_mass_loss_advanced_summary.txt",
    )


def import_existing_output_rows(systems: List[AdvancedSystem], existing_rows: List[dict]) -> List[dict]:
    known_keys = completed_row_keys(existing_rows)
    imported_rows: List[dict] = []

    for system in systems:
        csv_path, _ = output_paths_for_system(system)
        if not csv_path.exists():
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                species = row.get("species", "")
                if not species or is_total_species_label(species):
                    continue
                row_key = row_key_from_row(row)
                if row_key in known_keys:
                    continue
                imported_rows.append(row)
                known_keys.add(row_key)

    if imported_rows:
        validate_checkpoint_rows(imported_rows)
    return existing_rows + imported_rows


def write_system_outputs_from_rows(
    rows: List[dict],
    system: AdvancedSystem,
) -> Tuple[pathlib.Path, pathlib.Path] | Tuple[None, None]:
    species_rows = system_rows_from_checkpoint(rows, system)
    if not species_rows:
        return None, None

    total_row = total_row_from_species_rows(species_rows)
    rows_to_write = list(species_rows)
    if total_row is not None:
        rows_to_write.append(total_row)

    family_dir = OUTPUT_DIR / base_mass_loss.safe_name(system.test_family)
    output_slug = (
        f"{base_mass_loss.safe_name(system.planet_key)}_"
        f"{base_mass_loss.safe_name(system.star_key)}_"
        f"{float(system.distance_au):g}AU"
    )
    csv_path = family_dir / f"{output_slug}_mass_loss_advanced.csv"
    txt_path = family_dir / f"{output_slug}_mass_loss_advanced_summary.txt"
    write_results_csv(rows_to_write, csv_path)
    write_summary_txt(
        rows_to_write,
        txt_path,
        system.star_key,
        system.planet_key,
        system.test_family,
        system.distance_au,
    )
    return csv_path, txt_path


def total_row_from_species_rows(species_rows: List[dict]) -> dict | None:
    if not species_rows:
        return None

    template = dict(species_rows[0])
    total_mdot_g_s = float(np.nansum([float(row.get("mass_loss_rate_g_s", 0.0)) for row in species_rows]))
    total_mdot = total_mdot_g_s * u.g / u.s
    total_mass_1myr_g = (total_mdot * (1.0e6 * u.yr)).to_value(u.g)
    total_mass_1gyr_g = (total_mdot * (1.0e9 * u.yr)).to_value(u.g)
    total_escaping_mass_g = float(np.nansum([float(row.get("escaping_shell_mass_g", 0.0)) for row in species_rows]))
    total_shell_mass_g = float(np.nansum([float(row.get("total_shell_mass_g", 0.0)) for row in species_rows]))
    min_escape_values = np.asarray(
        [float(row.get("min_escape_time_s", np.nan)) for row in species_rows],
        dtype=float,
    )
    max_step_values = np.asarray(
        [float(row.get("max_steps_any_cell", np.nan)) for row in species_rows],
        dtype=float,
    )
    finite_min_escape = min_escape_values[np.isfinite(min_escape_values)]
    finite_max_steps = max_step_values[np.isfinite(max_step_values)]

    template.update(
        {
            "species": "TOTAL_INCLUDED_SPECIES",
            "P0_bar": template.get("P0_bar", ""),
            "mu_amu": template.get("mu_amu", ""),
            "surface_gravity_m_s2": template.get("surface_gravity_m_s2", ""),
            "mass_scale": template.get("mass_scale", ""),
            "mixing_ratio": "",
            "z_exobase_km": "",
            "r_exobase_over_Rp": "",
            "total_shell_mass_g": total_shell_mass_g,
            "escaping_shell_mass_g": total_escaping_mass_g,
            "mass_loss_rate_g_s": total_mdot_g_s,
            "mass_loss_rate_kg_s": total_mdot.to_value(u.kg / u.s),
            "mass_loss_rate_Mearth_yr": total_mdot.to_value(u.M_earth / u.yr),
            "mass_lost_g_1Myr": total_mass_1myr_g,
            "mass_lost_g_1Gyr": total_mass_1gyr_g,
            "mass_lost_Mearth_1Myr": (total_mdot * (1.0e6 * u.yr)).to_value(u.M_earth),
            "mass_lost_Mearth_1Gyr": (total_mdot * (1.0e9 * u.yr)).to_value(u.M_earth),
            "mean_escape_time_s": total_escaping_mass_g / total_mdot_g_s if total_mdot_g_s > 0.0 else np.nan,
            "median_escape_time_s": np.nan,
            "min_escape_time_s": float(np.min(finite_min_escape)) if finite_min_escape.size else np.nan,
            "mass_weighted_initial_beta": "",
            "mean_steps_escaped": np.nan,
            "max_steps_any_cell": int(np.max(finite_max_steps)) if finite_max_steps.size else 0,
            "photon_pressure_cache_bins": int(np.nansum([int(row.get("photon_pressure_cache_bins", 0)) for row in species_rows])),
            "n_cells": int(np.nansum([int(row.get("n_cells", 0)) for row in species_rows])),
            "n_escape_cells": int(np.nansum([int(row.get("n_escape_cells", 0)) for row in species_rows])),
        }
    )
    return template


def p0_row_key_from_values(
    test_family: str,
    planet_key: str,
    species: str,
    actual_star_key: str,
    distance_au: float,
    p0_bar: float,
) -> Tuple[str, str, str, str, str, str]:
    return (
        test_family,
        planet_key,
        species,
        actual_star_key,
        f"{float(distance_au):.12g}",
        f"{float(p0_bar):.12g}",
    )


def p0_row_key_from_row(row: dict) -> Tuple[str, str, str, str, str, str]:
    return p0_row_key_from_values(
        row["test_family"],
        row["planet"],
        row["species"],
        row["star"],
        float(row["distance_AU"]),
        float(row["P0_bar"]),
    )


def p0_completed_row_keys(rows: List[dict]) -> set[Tuple[str, str, str, str, str, str]]:
    return {p0_row_key_from_row(row) for row in rows}


def load_p0_checkpoint_rows(path: pathlib.Path) -> List[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            row
            for row in reader
            if row.get("species")
            and not is_total_species_label(row.get("species", ""))
            and row.get("P0_bar", "") != ""
        ]

    validate_checkpoint_rows(rows)
    return rows


def p0_system_rows(rows: List[dict], p0_bar: float) -> List[dict]:
    matching = [
        row
        for row in rows
        if row.get("test_family") == P0_SWEEP_TEST_FAMILY
        and row.get("planet") == P0_SWEEP_PLANET_KEY
        and row.get("star") == P0_SWEEP_STAR_KEY
        and float(row.get("distance_AU", np.nan)) == float(P0_SWEEP_DISTANCE_AU)
        and float(row.get("P0_bar", np.nan)) == float(p0_bar)
        and not is_total_species_label(row.get("species", ""))
    ]
    matching.sort(key=lambda row: (row.get("species", ""),))
    return matching


def p0_sweep_output_slug() -> str:
    return (
        f"{base_mass_loss.safe_name(P0_SWEEP_PLANET_KEY)}_"
        f"{base_mass_loss.safe_name(P0_SWEEP_STAR_KEY)}_"
        f"{float(P0_SWEEP_DISTANCE_AU):g}AU"
    )


def write_p0_sweep_outputs(rows: List[dict]) -> None:
    if not rows:
        return

    planet_case = base_mass_loss.get_planet_template(P0_SWEEP_PLANET_KEY)
    exobase_rows = base_mass_loss.load_exobase_table(EXOBASE_TABLE)
    mass_info = base_mass_loss.planet_reference_mass_info(P0_SWEEP_PLANET_KEY, planet_case, exobase_rows)

    p0_values_all = sorted({float(row["P0_bar"]) for row in rows})
    totals = []
    for p0_bar in p0_values_all:
        total_row = total_row_from_species_rows(p0_system_rows(rows, p0_bar))
        if total_row is None:
            continue
        total_row["P0_bar"] = p0_bar
        totals.append(total_row)

    if not totals:
        return

    p0_values = [float(row["P0_bar"]) for row in totals]
    total_mdot_array = np.asarray([float(row["mass_loss_rate_g_s"]) for row in totals], dtype=float)
    matrix_g_s = total_mdot_array.reshape(-1, 1)
    unit_matrix = base_mass_loss.mass_loss_in_planet_unit_per_year(total_mdot_array, planet_case).reshape(-1, 1)
    ratio_values, _ = base_mass_loss.mass_loss_over_reference_mass_per_second(
        total_mdot_array,
        P0_SWEEP_PLANET_KEY,
        planet_case,
        exobase_rows,
    )
    ratio_matrix = ratio_values.reshape(-1, 1)

    P0_SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = p0_sweep_output_slug()
    series_value = [f"{P0_SWEEP_STAR_KEY}_{float(P0_SWEEP_DISTANCE_AU):g}AU"]
    column_name = [f"{base_mass_loss.safe_name(P0_SWEEP_STAR_KEY)}_{float(P0_SWEEP_DISTANCE_AU):g}AU"]
    base_metadata = {
        "planet": P0_SWEEP_PLANET_KEY,
        "star": P0_SWEEP_STAR_KEY,
        "distance_AU": float(P0_SWEEP_DISTANCE_AU),
        "species": "TOTAL_INCLUDED_SPECIES",
        "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
        "method": "advanced_trajectory_recomputed_acceleration",
        "n_rho": N_RHO,
        "n_x": N_X,
        "column_steps": COLUMN_STEPS,
        "rho_grid_power": RHO_GRID_POWER,
        "log_column_bin_dex": LOG_COLUMN_BIN_DEX,
        "dt_min_s": DT_MIN_S,
        "dt_max_s": DT_MAX_S,
        "max_steps": MAX_STEPS,
        "max_time_s": MAX_TIME_S,
        "planet_category": mass_info["planet_category"],
        "reference_mass_kind": mass_info["reference_mass_kind"],
        "reference_mass_g": mass_info["reference_mass_g"],
        "reference_mass_unit_name": mass_info["reference_mass_unit_name"],
        "reference_mass_in_unit": mass_info["reference_mass_in_unit"],
        "reference_top_km": mass_info["reference_top_km"],
        "reference_top_source": mass_info["reference_top_source"],
        "note": "Advanced trajectory total mass-loss rate as a function of varying P0.",
    }

    g_s_path = P0_SWEEP_OUTPUT_DIR / f"{slug}_total_mass_loss_vs_P0.txt"
    base_mass_loss.save_mass_loss_matrix_txt(
        g_s_path,
        dataset_name=f"{slug}_total_mass_loss_vs_P0",
        x_label="P0",
        x_unit="bar",
        y_label="Total included-species mass-loss rate",
        y_unit="g/s",
        x_values=p0_values,
        y_matrix=matrix_g_s,
        series_values=series_value,
        series_label="system",
        series_unit="star_distance",
        column_names=column_name,
        extra_metadata=base_metadata,
    )

    unit_suffix = mass_info["reference_mass_unit_name"]
    unit_path = P0_SWEEP_OUTPUT_DIR / f"{slug}_total_mass_loss_vs_P0_{unit_suffix}_yr.txt"
    base_mass_loss.save_mass_loss_matrix_txt(
        unit_path,
        dataset_name=f"{slug}_total_mass_loss_vs_P0_{unit_suffix}_yr",
        x_label="P0",
        x_unit="bar",
        y_label="Total included-species mass-loss rate",
        y_unit=f"{unit_suffix}/yr",
        x_values=p0_values,
        y_matrix=unit_matrix,
        series_values=series_value,
        series_label="system",
        series_unit="star_distance",
        column_names=column_name,
        extra_metadata={**base_metadata, "rate_unit_name": f"{unit_suffix}/yr"},
    )

    ratio_path = P0_SWEEP_OUTPUT_DIR / f"{slug}_total_mass_loss_ratio_vs_P0.txt"
    base_mass_loss.save_mass_loss_matrix_txt(
        ratio_path,
        dataset_name=f"{slug}_total_mass_loss_ratio_vs_P0",
        x_label="P0",
        x_unit="bar",
        y_label="Total included-species specific mass-loss rate",
        y_unit="1/s",
        x_values=p0_values,
        y_matrix=ratio_matrix,
        series_values=series_value,
        series_label="system",
        series_unit="star_distance",
        column_names=column_name,
        extra_metadata={**base_metadata, "ratio_definition": "Mdot / M_reference"},
    )

    reference_kind_key = "whole_atmosphere" if mass_info["reference_mass_kind"] == "whole_atmosphere" else "whole_planet"
    summary_path = P0_SWEEP_OUTPUT_DIR / f"{slug}_P0_sweep_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Advanced P0 Sweep Summary\n")
        f.write("=========================\n")
        f.write(f"test_family: {P0_SWEEP_TEST_FAMILY}\n")
        f.write(f"planet: {P0_SWEEP_PLANET_KEY}\n")
        f.write(f"star_template: {P0_SWEEP_STAR_KEY}\n")
        f.write(f"actual_stellar_teff_K: {base_mass_loss.infer_teff_from_star_template(P0_SWEEP_STAR_KEY)}\n")
        f.write(f"distance_AU: {float(P0_SWEEP_DISTANCE_AU)}\n")
        f.write(f"P0_values_bar: {', '.join(f'{value:.6e}' for value in p0_values)}\n")
        f.write(f"planet_category: {mass_info['planet_category']}\n")
        f.write(f"reference_mass_kind: {mass_info['reference_mass_kind']}\n")
        f.write(f"reference_mass_g: {mass_info['reference_mass_g']:.12e}\n")
        f.write(f"{reference_kind_key}_mass_g: {mass_info['reference_mass_g']:.12e}\n")
        f.write(
            f"{reference_kind_key}_mass_{mass_info['reference_mass_unit_name']}: "
            f"{mass_info['reference_mass_in_unit']:.12e}\n"
        )
        if mass_info["reference_top_km"] != "":
            f.write(f"reference_atmosphere_top_km: {float(mass_info['reference_top_km']):.6f}\n")
            f.write(f"reference_atmosphere_top_source: {mass_info['reference_top_source']}\n")
        f.write(f"rate_unit_name: {mass_info['rate_unit_name']}\n")
        f.write(f"g_s_file: {g_s_path.name}\n")
        f.write(f"planet_unit_file: {unit_path.name}\n")
        f.write(f"ratio_file: {ratio_path.name}\n")


def run_p0_sweep() -> None:
    configure_base_module()
    exobase_rows = base_mass_loss.load_exobase_table(EXOBASE_TABLE)
    system_def = build_p0_sweep_system()
    planet_base_case = base_mass_loss.get_planet_template(P0_SWEEP_PLANET_KEY)
    species_template = selected_species_for_planet(planet_base_case)
    if not species_template:
        print(f"Skipping {P0_SWEEP_TEST_FAMILY}: no species selected")
        return

    all_rows = load_family_rows(P0_SWEEP_TEST_FAMILY) if USE_CHECKPOINT else []
    completed_rows = p0_completed_row_keys(all_rows)
    if all_rows:
        print(f"Loaded {len(all_rows)} completed species rows from {family_results_path(P0_SWEEP_TEST_FAMILY)}")

    for p0_bar in P0_SWEEP_VALUES_BAR:
        planet_case = dict(planet_base_case)
        planet_case["P0"] = float(p0_bar) * u.bar
        species_list = selected_species_for_planet(planet_case)
        planet = base_mass_loss.build_planet(planet_case)
        star = build_system_star(system_def)
        system = base_mass_loss.PlanetarySystem(planet, star, P0_SWEEP_DISTANCE_AU * u.AU)

        existing = p0_system_rows(all_rows, p0_bar)
        if existing:
            print(
                f"\n--- Advanced P0 sweep: {P0_SWEEP_PLANET_KEY} / {P0_SWEEP_STAR_KEY} / "
                f"{float(P0_SWEEP_DISTANCE_AU):g} AU / P0={p0_bar:.1e} bar "
                f"({len(existing)}/{len(species_list)} species already saved) ---"
            )
            write_p0_sweep_outputs(all_rows)
        else:
            print(
                f"\n--- Advanced P0 sweep: {P0_SWEEP_PLANET_KEY}, star={P0_SWEEP_STAR_KEY} "
                f"({base_mass_loss.infer_teff_from_star_template(P0_SWEEP_STAR_KEY)} K), "
                f"distance={float(P0_SWEEP_DISTANCE_AU):g} AU, P0={p0_bar:.1e} bar ---"
            )

        pending_tasks = []
        for species in species_list:
            current_key = p0_row_key_from_values(
                P0_SWEEP_TEST_FAMILY,
                P0_SWEEP_PLANET_KEY,
                species,
                P0_SWEEP_STAR_KEY,
                P0_SWEEP_DISTANCE_AU,
                p0_bar,
            )
            if current_key in completed_rows:
                print(f"Skipping completed P0 species: {P0_SWEEP_PLANET_KEY} / {species} / P0={p0_bar:.1e}")
                continue
            pending_tasks.append(
                build_species_task(
                    system_def,
                    species,
                    planet_case,
                    extra_fields={"P0_bar": float(p0_bar)},
                )
            )

        for result in iter_species_task_results(pending_tasks, exobase_rows):
            species = result["species"]
            current_key = p0_row_key_from_values(
                P0_SWEEP_TEST_FAMILY,
                P0_SWEEP_PLANET_KEY,
                species,
                P0_SWEEP_STAR_KEY,
                P0_SWEEP_DISTANCE_AU,
                p0_bar,
            )
            if not result["ok"]:
                print(
                    f"Skipping P0 sweep {P0_SWEEP_PLANET_KEY} {species}: "
                    f"{result['error_type']}: {result['error_message']}"
                )
                continue

            row = result["row"]
            all_rows.append(row)
            completed_rows.add(current_key)
            write_family_results_txt(P0_SWEEP_TEST_FAMILY, all_rows)
            print(
                f"{species}: Mdot={row['mass_loss_rate_g_s']:.3e} g/s, "
                f"escaping_mass={row['escaping_shell_mass_g']:.3e} g, "
                f"mean_t={row['mean_escape_time_s']:.3e} s, "
                f"Myr={row['mass_lost_Mearth_1Myr']:.3e} Mearth, "
                f"Gyr={row['mass_lost_Mearth_1Gyr']:.3e} Mearth, "
                f"cache_bins={row['photon_pressure_cache_bins']}, "
                f"elapsed={result['elapsed_s']:.1f} s"
            )

        total_row = total_row_from_species_rows(p0_system_rows(all_rows, p0_bar))
        if total_row is not None:
            print(
                f"TOTAL_INCLUDED_SPECIES for {P0_SWEEP_PLANET_KEY} at P0={p0_bar:.1e} bar: "
                f"Mdot={total_row['mass_loss_rate_g_s']:.3e} g/s, "
                f"Myr={total_row['mass_lost_Mearth_1Myr']:.3e} Mearth, "
                f"Gyr={total_row['mass_lost_Mearth_1Gyr']:.3e} Mearth"
            )
    write_family_results_txt(P0_SWEEP_TEST_FAMILY, all_rows)


def scalar_row_key_from_values(
    test_family: str,
    planet_key: str,
    species: str,
    actual_star_key: str,
    distance_au: float,
    value_field: str,
    value: float,
) -> Tuple[str, str, str, str, str, str, str]:
    return (
        test_family,
        planet_key,
        species,
        actual_star_key,
        f"{float(distance_au):.12g}",
        value_field,
        f"{float(value):.12g}",
    )


def scalar_row_key_from_row(row: dict, value_field: str) -> Tuple[str, str, str, str, str, str, str]:
    return scalar_row_key_from_values(
        row["test_family"],
        row["planet"],
        row["species"],
        row["star"],
        float(row["distance_AU"]),
        value_field,
        float(row[value_field]),
    )


def scalar_completed_row_keys(rows: List[dict], value_field: str) -> set[Tuple[str, str, str, str, str, str, str]]:
    return {scalar_row_key_from_row(row, value_field) for row in rows}


def load_scalar_checkpoint_rows(path: pathlib.Path, value_field: str) -> List[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            row
            for row in reader
            if row.get("species")
            and not is_total_species_label(row.get("species", ""))
            and row.get(value_field, "") != ""
        ]

    validate_checkpoint_rows(rows)
    return rows


def scalar_system_rows(
    rows: List[dict],
    system_def: AdvancedSystem,
    value_field: str,
    value: float,
) -> List[dict]:
    matching = [
        row
        for row in rows
        if row.get("test_family") == system_def.test_family
        and row.get("planet") == system_def.planet_key
        and row.get("star") == system_def.star_key
        and float(row.get("distance_AU", np.nan)) == float(system_def.distance_au)
        and float(row.get(value_field, np.nan)) == float(value)
        and not is_total_species_label(row.get("species", ""))
    ]
    matching.sort(key=lambda row: (row.get("species", ""),))
    return matching


def build_mu_sweep_planet_case(planet_key: str, mu_amu: float) -> dict:
    planet_case = dict(base_mass_loss.get_planet_template(planet_key))
    planet_case["mu"] = float(mu_amu) * u.dimensionless_unscaled
    return planet_case


def build_surface_gravity_sweep_planet_case(planet_key: str, mass_scale: float) -> dict:
    planet_case = dict(base_mass_loss.get_planet_template(planet_key))
    planet_case["mass"] = float(mass_scale) * planet_case["mass"]
    return planet_case


def surface_gravity_m_s2_for_planet_case(planet_case: dict) -> float:
    return (const.G * planet_case["mass"] / planet_case["radius"] ** 2).to_value(u.m / u.s**2)


def scalar_sweep_output_slug(system_def: AdvancedSystem) -> str:
    return (
        f"{base_mass_loss.safe_name(system_def.planet_key)}_"
        f"{base_mass_loss.safe_name(system_def.star_key)}_"
        f"{float(system_def.distance_au):g}AU"
    )


def write_scalar_sweep_outputs(
    rows: List[dict],
    systems: List[AdvancedSystem],
    output_dir: pathlib.Path,
    value_field: str,
    file_stem: str,
    x_label: str,
    x_unit: str,
    note: str,
    planet_case_builder,
) -> None:
    if not rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    exobase_rows = base_mass_loss.load_exobase_table(EXOBASE_TABLE)

    for system_def in systems:
        system_rows_all = [
            row
            for row in rows
            if row.get("test_family") == system_def.test_family
            and row.get("planet") == system_def.planet_key
            and row.get("star") == system_def.star_key
            and float(row.get("distance_AU", np.nan)) == float(system_def.distance_au)
        ]
        if not system_rows_all:
            continue

        x_values_all = sorted({float(row[value_field]) for row in system_rows_all if row.get(value_field, "") != ""})
        totals = []
        x_values = []
        unit_values = []
        ratio_values = []
        last_mass_info = None

        for x_value in x_values_all:
            total_row = total_row_from_species_rows(scalar_system_rows(rows, system_def, value_field, x_value))
            if total_row is None:
                continue

            x_values.append(x_value)
            total_row[value_field] = x_value
            totals.append(total_row)

            planet_case = planet_case_builder(system_def.planet_key, x_value)
            last_mass_info = base_mass_loss.planet_reference_mass_info(system_def.planet_key, planet_case, exobase_rows)
            mdot_array = np.asarray([float(total_row["mass_loss_rate_g_s"])], dtype=float)
            unit_values.append(base_mass_loss.mass_loss_in_planet_unit_per_year(mdot_array, planet_case)[0])
            ratio_values.append(
                base_mass_loss.mass_loss_over_reference_mass_per_second(
                    mdot_array,
                    system_def.planet_key,
                    planet_case,
                    exobase_rows,
                )[0][0]
            )

        if not totals or last_mass_info is None:
            continue

        g_s_matrix = np.asarray([float(row["mass_loss_rate_g_s"]) for row in totals], dtype=float).reshape(-1, 1)

        slug = scalar_sweep_output_slug(system_def)
        series_value = [f"{system_def.star_key}_{float(system_def.distance_au):g}AU"]
        column_name = [f"{base_mass_loss.safe_name(system_def.star_key)}_{float(system_def.distance_au):g}AU"]
        base_metadata = {
            "planet": system_def.planet_key,
            "star": system_def.star_key,
            "distance_AU": float(system_def.distance_au),
            "species": "TOTAL_INCLUDED_SPECIES",
            "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
            "method": "advanced_trajectory_recomputed_acceleration",
            "n_rho": N_RHO,
            "n_x": N_X,
            "column_steps": COLUMN_STEPS,
            "rho_grid_power": RHO_GRID_POWER,
            "log_column_bin_dex": LOG_COLUMN_BIN_DEX,
            "dt_min_s": DT_MIN_S,
            "dt_max_s": DT_MAX_S,
            "max_steps": MAX_STEPS,
            "max_time_s": MAX_TIME_S,
            "planet_category": last_mass_info["planet_category"],
            "reference_mass_kind": last_mass_info["reference_mass_kind"],
            "reference_mass_unit_name": last_mass_info["reference_mass_unit_name"],
            "reference_mass_note": "reference mass is recomputed for each sweep value",
            "note": note,
        }

        g_s_path = output_dir / f"{slug}_total_mass_loss_vs_{file_stem}.txt"
        base_mass_loss.save_mass_loss_matrix_txt(
            g_s_path,
            dataset_name=f"{slug}_total_mass_loss_vs_{file_stem}",
            x_label=x_label,
            x_unit=x_unit,
            y_label="Total included-species mass-loss rate",
            y_unit="g/s",
            x_values=x_values,
            y_matrix=g_s_matrix,
            series_values=series_value,
            series_label="system",
            series_unit="star_distance",
            column_names=column_name,
            extra_metadata=base_metadata,
        )

        unit_suffix = last_mass_info["reference_mass_unit_name"]
        unit_path = output_dir / f"{slug}_total_mass_loss_vs_{file_stem}_{unit_suffix}_yr.txt"
        base_mass_loss.save_mass_loss_matrix_txt(
            unit_path,
            dataset_name=f"{slug}_total_mass_loss_vs_{file_stem}_{unit_suffix}_yr",
            x_label=x_label,
            x_unit=x_unit,
            y_label="Total included-species mass-loss rate",
            y_unit=f"{unit_suffix}/yr",
            x_values=x_values,
            y_matrix=np.asarray(unit_values, dtype=float).reshape(-1, 1),
            series_values=series_value,
            series_label="system",
            series_unit="star_distance",
            column_names=column_name,
            extra_metadata={**base_metadata, "rate_unit_name": f"{unit_suffix}/yr"},
        )

        ratio_path = output_dir / f"{slug}_total_mass_loss_ratio_vs_{file_stem}.txt"
        base_mass_loss.save_mass_loss_matrix_txt(
            ratio_path,
            dataset_name=f"{slug}_total_mass_loss_ratio_vs_{file_stem}",
            x_label=x_label,
            x_unit=x_unit,
            y_label="Total included-species specific mass-loss rate",
            y_unit="1/s",
            x_values=x_values,
            y_matrix=np.asarray(ratio_values, dtype=float).reshape(-1, 1),
            series_values=series_value,
            series_label="system",
            series_unit="star_distance",
            column_names=column_name,
            extra_metadata={**base_metadata, "ratio_definition": "Mdot / M_reference(parameter)"},
        )

        reference_kind_key = (
            "whole_atmosphere" if last_mass_info["reference_mass_kind"] == "whole_atmosphere" else "whole_planet"
        )
        summary_path = output_dir / f"{slug}_{file_stem}_sweep_summary.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("Advanced Parameter Sweep Summary\n")
            f.write("===============================\n")
            f.write(f"test_family: {system_def.test_family}\n")
            f.write(f"planet: {system_def.planet_key}\n")
            f.write(f"star_template: {system_def.star_key}\n")
            f.write(f"actual_stellar_teff_K: {base_mass_loss.infer_teff_from_star_template(system_def.star_key)}\n")
            f.write(f"distance_AU: {float(system_def.distance_au)}\n")
            f.write(f"x_label: {x_label}\n")
            f.write(f"x_unit: {x_unit}\n")
            f.write(f"x_values: {', '.join(f'{value:.6e}' for value in x_values)}\n")
            f.write(f"planet_category: {last_mass_info['planet_category']}\n")
            f.write(f"reference_mass_kind: {last_mass_info['reference_mass_kind']}\n")
            f.write(
                f"reference_mass_note: {reference_kind_key} mass is recomputed for each sweep value "
                "when forming the ratio\n"
            )
            f.write(f"rate_unit_name: {last_mass_info['rate_unit_name']}\n")
            f.write(f"g_s_file: {g_s_path.name}\n")
            f.write(f"planet_unit_file: {unit_path.name}\n")
            f.write(f"ratio_file: {ratio_path.name}\n")


def run_mu_sweeps() -> None:
    configure_base_module()
    exobase_rows = base_mass_loss.load_exobase_table(EXOBASE_TABLE)
    systems = build_mu_sweep_systems()
    all_rows = load_family_rows(MU_SWEEP_HOT_JUPITER_FAMILY) if USE_CHECKPOINT else []
    completed_rows = scalar_completed_row_keys(all_rows, "mu_amu")
    if all_rows:
        print(f"Loaded {len(all_rows)} completed species rows from {family_results_path(MU_SWEEP_HOT_JUPITER_FAMILY)}")

    for system_def in systems:
        species_template = selected_species_for_planet(base_mass_loss.get_planet_template(system_def.planet_key))
        if not species_template:
            print(f"Skipping {system_def.test_family}: no species selected")
            continue

        for mu_amu in MU_SWEEP_VALUES_AMU:
            planet_case = build_mu_sweep_planet_case(system_def.planet_key, mu_amu)
            species_list = selected_species_for_planet(planet_case)
            planet = base_mass_loss.build_planet(planet_case)
            star = build_system_star(system_def)
            system = base_mass_loss.PlanetarySystem(planet, star, system_def.distance_au * u.AU)
            existing = scalar_system_rows(all_rows, system_def, "mu_amu", mu_amu)
            if existing:
                print(
                    f"\n--- Advanced mu sweep: {system_def.planet_key} / {system_def.star_key} / "
                    f"{float(system_def.distance_au):g} AU / mu={mu_amu:.1f} "
                    f"({len(existing)}/{len(species_list)} species already saved) ---"
                )
            else:
                print(
                    f"\n--- Advanced mu sweep: {system_def.planet_key}, star={system_def.star_key} "
                    f"({get_system_actual_teff_k(system_def):.0f} K), "
                    f"distance={float(system_def.distance_au):g} AU, mu={mu_amu:.1f} ---"
                )

            pending_tasks = []
            for species in species_list:
                current_key = scalar_row_key_from_values(
                    system_def.test_family,
                    system_def.planet_key,
                    species,
                    system_def.star_key,
                    system_def.distance_au,
                    "mu_amu",
                    mu_amu,
                )
                if current_key in completed_rows:
                    print(f"Skipping completed mu species: {system_def.planet_key} / {species} / mu={mu_amu:.1f}")
                    continue
                pending_tasks.append(
                    build_species_task(
                        system_def,
                        species,
                        planet_case,
                        extra_fields={"mu_amu": float(mu_amu)},
                    )
                )

            for result in iter_species_task_results(pending_tasks, exobase_rows):
                species = result["species"]
                current_key = scalar_row_key_from_values(
                    system_def.test_family,
                    system_def.planet_key,
                    species,
                    system_def.star_key,
                    system_def.distance_au,
                    "mu_amu",
                    mu_amu,
                )
                if not result["ok"]:
                    print(
                        f"Skipping mu sweep {system_def.planet_key} {species}: "
                        f"{result['error_type']}: {result['error_message']}"
                    )
                    continue

                row = result["row"]
                all_rows.append(row)
                completed_rows.add(current_key)
                write_family_results_txt(MU_SWEEP_HOT_JUPITER_FAMILY, all_rows)
                print(
                    f"{species}: Mdot={row['mass_loss_rate_g_s']:.3e} g/s, "
                    f"escaping_mass={row['escaping_shell_mass_g']:.3e} g, "
                    f"mean_t={row['mean_escape_time_s']:.3e} s, "
                    f"Myr={row['mass_lost_Mearth_1Myr']:.3e} Mearth, "
                    f"Gyr={row['mass_lost_Mearth_1Gyr']:.3e} Mearth, "
                    f"cache_bins={row['photon_pressure_cache_bins']}, "
                    f"elapsed={result['elapsed_s']:.1f} s"
                )

            total_row = total_row_from_species_rows(scalar_system_rows(all_rows, system_def, "mu_amu", mu_amu))
            if total_row is not None:
                print(
                    f"TOTAL_INCLUDED_SPECIES for {system_def.planet_key} at mu={mu_amu:.1f}: "
                    f"Mdot={total_row['mass_loss_rate_g_s']:.3e} g/s, "
                    f"Myr={total_row['mass_lost_Mearth_1Myr']:.3e} Mearth, "
                    f"Gyr={total_row['mass_lost_Mearth_1Gyr']:.3e} Mearth"
                )
    write_family_results_txt(MU_SWEEP_HOT_JUPITER_FAMILY, all_rows)


def run_surface_gravity_sweep() -> None:
    configure_base_module()
    exobase_rows = base_mass_loss.load_exobase_table(EXOBASE_TABLE)
    system_def = build_surface_gravity_sweep_system()
    all_rows = load_family_rows(SURFACE_GRAVITY_SWEEP_FAMILY) if USE_CHECKPOINT else []
    completed_rows = scalar_completed_row_keys(all_rows, "surface_gravity_m_s2")
    species_template = selected_species_for_planet(base_mass_loss.get_planet_template(system_def.planet_key))
    if not species_template:
        print(f"Skipping {system_def.test_family}: no species selected")
        return

    if all_rows:
        print(
            f"Loaded {len(all_rows)} completed species rows from {family_results_path(SURFACE_GRAVITY_SWEEP_FAMILY)}"
        )

    base_g = surface_gravity_m_s2_for_planet_case(base_mass_loss.get_planet_template(system_def.planet_key))
    for mass_scale in SURFACE_GRAVITY_MASS_SCALE_VALUES:
        planet_case = build_surface_gravity_sweep_planet_case(system_def.planet_key, mass_scale)
        species_list = selected_species_for_planet(planet_case)
        planet = base_mass_loss.build_planet(planet_case)
        star = build_system_star(system_def)
        system = base_mass_loss.PlanetarySystem(planet, star, system_def.distance_au * u.AU)
        g_surface = surface_gravity_m_s2_for_planet_case(planet_case)
        existing = scalar_system_rows(all_rows, system_def, "surface_gravity_m_s2", g_surface)
        if existing:
            print(
                f"\n--- Advanced surface-gravity sweep: {system_def.planet_key} / {system_def.star_key} / "
                f"{float(system_def.distance_au):g} AU / g={g_surface:.3f} m/s^2 "
                f"({len(existing)}/{len(species_list)} species already saved) ---"
            )
        else:
            print(
                f"\n--- Advanced surface-gravity sweep: {system_def.planet_key}, star={system_def.star_key} "
                f"({get_system_actual_teff_k(system_def):.0f} K), "
                f"distance={float(system_def.distance_au):g} AU, g={g_surface:.3f} m/s^2 ---"
            )

        pending_tasks = []
        for species in species_list:
            current_key = scalar_row_key_from_values(
                system_def.test_family,
                system_def.planet_key,
                species,
                system_def.star_key,
                system_def.distance_au,
                "surface_gravity_m_s2",
                g_surface,
            )
            if current_key in completed_rows:
                print(f"Skipping completed surface-gravity species: {system_def.planet_key} / {species} / g={g_surface:.3f}")
                continue
            pending_tasks.append(
                build_species_task(
                    system_def,
                    species,
                    planet_case,
                    extra_fields={
                        "surface_gravity_m_s2": float(g_surface),
                        "mass_scale": float(mass_scale),
                    },
                )
            )

        for result in iter_species_task_results(pending_tasks, exobase_rows):
            species = result["species"]
            current_key = scalar_row_key_from_values(
                system_def.test_family,
                system_def.planet_key,
                species,
                system_def.star_key,
                system_def.distance_au,
                "surface_gravity_m_s2",
                g_surface,
            )
            if not result["ok"]:
                print(
                    f"Skipping surface-gravity sweep {system_def.planet_key} {species}: "
                    f"{result['error_type']}: {result['error_message']}"
                )
                continue

            row = result["row"]
            all_rows.append(row)
            completed_rows.add(current_key)
            write_family_results_txt(SURFACE_GRAVITY_SWEEP_FAMILY, all_rows)
            print(
                f"{species}: Mdot={row['mass_loss_rate_g_s']:.3e} g/s, "
                f"escaping_mass={row['escaping_shell_mass_g']:.3e} g, "
                f"mean_t={row['mean_escape_time_s']:.3e} s, "
                f"Myr={row['mass_lost_Mearth_1Myr']:.3e} Mearth, "
                f"Gyr={row['mass_lost_Mearth_1Gyr']:.3e} Mearth, "
                f"cache_bins={row['photon_pressure_cache_bins']}, "
                f"elapsed={result['elapsed_s']:.1f} s"
            )

        total_row = total_row_from_species_rows(
            scalar_system_rows(all_rows, system_def, "surface_gravity_m_s2", g_surface)
        )
        if total_row is not None:
            print(
                f"TOTAL_INCLUDED_SPECIES for {system_def.planet_key} at g={g_surface:.3f} m/s^2: "
                f"Mdot={total_row['mass_loss_rate_g_s']:.3e} g/s, "
                f"Myr={total_row['mass_lost_Mearth_1Myr']:.3e} Mearth, "
                f"Gyr={total_row['mass_lost_Mearth_1Gyr']:.3e} Mearth"
            )
    write_family_results_txt(SURFACE_GRAVITY_SWEEP_FAMILY, all_rows)


def mass_loss_for_species_advanced(
    test_family: str,
    planet_key: str,
    star_key: str,
    distance_au: float,
    species: str,
    planet_case: dict,
    system,
    exobase_rows: Dict[Tuple[str, str], dict],
    exobase_planet_key: str | None = None,
    target_stellar_teff_k: float | None = None,
    planet_label: str | None = None,
    star_label: str | None = None,
    spectrum_template_key: str | None = None,
    planet_source_url: str = "",
    star_source_url: str = "",
    orbit_source_url: str = "",
) -> dict:
    exobase_lookup_key = exobase_planet_key or planet_key
    z_exobase = base_mass_loss.exobase_height(exobase_lookup_key, species, exobase_rows)
    if z_exobase is None:
        raise ValueError(f"No exobase height for {exobase_lookup_key}, {species}")

    planet = system.planet
    star = system.star
    distance = system.distance.to(u.cm)
    planet_radius = planet.radius.to(u.cm)
    hill_radius = system.hill_radius().to(u.cm)
    r_exobase = planet_radius + z_exobase.to(u.cm)

    if r_exobase >= hill_radius:
        raise ValueError("Exobase is outside or at the Hill radius.")

    abundance = base_mass_loss.species_mixing_ratio(planet_case, species)
    if abundance <= 0.0:
        raise ValueError("Species abundance is zero or negative.")

    profile = get_species_profile(species)
    pp = base_mass_loss.PhotonPressure(profile, star)
    species_mass = profile.molecule.mass.to(u.g)
    photon_cache = PhotonPressureBinnedCache(pp, species_mass, planet_case["T"], system.distance, LOG_COLUMN_BIN_DEX)

    mu_planet = (const.G.cgs * planet.mass.to(u.g)).to_value(u.cm**3 / u.s**2)
    mu_star = (const.G.cgs * star.mass.to(u.g)).to_value(u.cm**3 / u.s**2)
    hill_cm = hill_radius.to_value(u.cm)
    r_exobase_cm = r_exobase.to_value(u.cm)
    distance_cm = distance.to_value(u.cm)
    species_mass_g = species_mass.to_value(u.g)

    cells = base_mass_loss.spherical_shell_cells(
        planet,
        abundance,
        species_mass_g,
        r_exobase_cm,
        hill_cm,
    )
    if len(cells["dm_g"]) == 0:
        raise ValueError("No valid shell cells were generated.")

    x_cm = cells["x_cm"]
    y_cm = cells["rho_cm"]
    r_cm = cells["r_cm"]
    dm_g = cells["dm_g"]
    initial_ncol_cm2 = cells["ncol_cm2"]

    escape_times_s = np.full(len(dm_g), np.nan, dtype=float)
    step_counts = np.zeros(len(dm_g), dtype=int)
    initial_beta = np.full(len(dm_g), np.nan, dtype=float)

    g_planet = mu_planet / r_cm**2
    for idx in range(len(dm_g)):
        result = integrate_escape_trajectory(
            x_cm[idx],
            y_cm[idx],
            initial_ncol_cm2[idx],
            planet,
            abundance,
            r_exobase_cm,
            hill_cm,
            photon_cache,
            mu_planet,
            mu_star,
            distance_cm,
        )
        step_counts[idx] = int(result["step_count"])
        initial_beta[idx] = result["initial_a_rad_cm_s2"] / g_planet[idx]
        if result["escaped"]:
            escape_times_s[idx] = float(result["escape_time_s"])

    escape_mask = np.isfinite(escape_times_s) & (escape_times_s > 0.0) & np.isfinite(dm_g) & (dm_g > 0.0)

    total_mass_g = float(np.nansum(dm_g))
    escaping_mass_g = float(np.nansum(dm_g[escape_mask])) if np.any(escape_mask) else 0.0
    mdot_g_s = float(np.nansum(dm_g[escape_mask] / escape_times_s[escape_mask])) if np.any(escape_mask) else 0.0
    beta_mass_sum = float(np.nansum(initial_beta[escape_mask] * dm_g[escape_mask])) if np.any(escape_mask) else 0.0
    mean_escape_time_s = escaping_mass_g / mdot_g_s if mdot_g_s > 0.0 else np.nan
    median_escape_time_s = float(np.nanmedian(escape_times_s[escape_mask])) if np.any(escape_mask) else np.nan
    min_escape_time_s = float(np.nanmin(escape_times_s[escape_mask])) if np.any(escape_mask) else np.nan
    mean_steps_escaped = float(np.nanmean(step_counts[escape_mask])) if np.any(escape_mask) else np.nan
    max_steps_any_cell = int(np.nanmax(step_counts)) if len(step_counts) else 0
    mass_weighted_initial_beta = beta_mass_sum / escaping_mass_g if escaping_mass_g > 0.0 else np.nan
    mdot = mdot_g_s * u.g / u.s

    return {
        "test_family": test_family,
        "planet": planet_key,
        "planet_label": planet_label or planet_key,
        "star": star_key,
        "star_label": star_label or star_key,
        "spectrum_template_key": spectrum_template_key or star_key,
        "planet_source_url": planet_source_url,
        "star_source_url": star_source_url,
        "orbit_source_url": orbit_source_url,
        "exobase_template_key": exobase_lookup_key,
        "target_stellar_teff_K": target_stellar_teff_k if target_stellar_teff_k is not None else base_mass_loss.infer_teff_from_star_template(star_key),
        "actual_stellar_teff_K": target_stellar_teff_k if target_stellar_teff_k is not None else base_mass_loss.infer_teff_from_star_template(star_key),
        "distance_AU": float(distance_au),
        "P0_bar": "",
        "mu_amu": "",
        "surface_gravity_m_s2": "",
        "mass_scale": "",
        "species": species,
        "mixing_ratio": abundance,
        "z_exobase_km": z_exobase.to_value(u.km),
        "r_exobase_over_Rp": (r_exobase / planet_radius).decompose().value,
        "hill_radius_over_Rp": (hill_radius / planet_radius).decompose().value,
        "total_shell_mass_g": total_mass_g,
        "escaping_shell_mass_g": escaping_mass_g,
        "mass_loss_rate_g_s": mdot_g_s,
        "mass_loss_rate_kg_s": mdot.to_value(u.kg / u.s),
        "mass_loss_rate_Mearth_yr": mdot.to_value(u.M_earth / u.yr),
        "mass_lost_g_1Myr": (mdot * (1.0e6 * u.yr)).to_value(u.g),
        "mass_lost_g_1Gyr": (mdot * (1.0e9 * u.yr)).to_value(u.g),
        "mass_lost_Mearth_1Myr": (mdot * (1.0e6 * u.yr)).to_value(u.M_earth),
        "mass_lost_Mearth_1Gyr": (mdot * (1.0e9 * u.yr)).to_value(u.M_earth),
        "mean_escape_time_s": mean_escape_time_s,
        "median_escape_time_s": median_escape_time_s,
        "min_escape_time_s": min_escape_time_s,
        "mass_weighted_initial_beta": mass_weighted_initial_beta,
        "mean_steps_escaped": mean_steps_escaped,
        "max_steps_any_cell": max_steps_any_cell,
        "photon_pressure_cache_bins": photon_cache.n_bins,
        "n_cells": int(len(dm_g)),
        "n_escape_cells": int(np.count_nonzero(escape_mask)),
        "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
        "method": "advanced_trajectory_recomputed_acceleration",
        "n_rho": N_RHO,
        "n_x": N_X,
        "column_steps": COLUMN_STEPS,
        "rho_grid_power": RHO_GRID_POWER,
        "log_column_bin_dex": LOG_COLUMN_BIN_DEX,
        "dt_min_s": DT_MIN_S,
        "dt_max_s": DT_MAX_S,
        "max_steps": MAX_STEPS,
        "max_time_s": MAX_TIME_S,
    }


def run_standard_systems() -> None:
    configure_base_module()
    start_time = time.perf_counter()
    exobase_rows = base_mass_loss.load_exobase_table(EXOBASE_TABLE)

    systems = build_advanced_systems()
    systems_by_family: Dict[str, List[AdvancedSystem]] = {}
    for system in systems:
        systems_by_family.setdefault(system.test_family, []).append(system)

    for test_family, family_systems in systems_by_family.items():
        family_rows = load_family_rows(test_family) if USE_CHECKPOINT else []
        completed_rows = completed_row_keys(family_rows)
        if family_rows:
            print(f"Loaded {len(family_rows)} completed species rows from {family_results_path(test_family)}")

        for system_batch in batched(family_systems, MLA_PLANET_MAX_WORKERS):
            pending_tasks = []
            batch_systems: List[AdvancedSystem] = []

            for system_def in system_batch:
                planet_key = system_def.planet_key
                star_key = system_def.star_key
                distance_au = system_def.distance_au
                planet_case = get_system_planet_case(system_def)
                species_list = selected_species_for_planet(planet_case)
                if not species_list:
                    print(f"Skipping {system_def.test_family} / {planet_key}: no species selected")
                    continue

                existing_rows = system_rows_from_checkpoint(family_rows, system_def)
                planet_label, star_label = get_system_display_names(system_def)
                actual_teff_k = get_system_actual_teff_k(system_def)

                if existing_rows:
                    print(
                        f"\n--- Advanced mass-loss system: {system_def.test_family} / {planet_label} / {star_label} / "
                        f"{distance_au:g} AU ({len(existing_rows)}/{len(species_list)} species already saved) ---"
                    )
                else:
                    print(
                        f"\n--- Advanced mass-loss system: {system_def.test_family} / {planet_label}, star={star_label} "
                        f"({actual_teff_k:.0f} K), distance={distance_au:g} AU ---"
                    )

                batch_systems.append(system_def)
                for species in species_list:
                    current_key = row_key_from_values(system_def.test_family, planet_key, species, star_key, distance_au)
                    if current_key in completed_rows:
                        print(
                            f"Skipping completed advanced species: {system_def.test_family} / {planet_label} / {species}"
                        )
                        continue
                    pending_tasks.append(build_species_task(system_def, species, planet_case))

            for result in iter_species_task_results(pending_tasks, exobase_rows, n_systems=len(batch_systems)):
                system_def = result["system_def"]
                species = result["species"]
                planet_label = result["planet_label"]
                current_key = row_key_from_values(
                    system_def.test_family,
                    system_def.planet_key,
                    species,
                    system_def.star_key,
                    system_def.distance_au,
                )
                if not result["ok"]:
                    print(
                        f"Skipping {planet_label} {species}: "
                        f"{result['error_type']}: {result['error_message']}"
                    )
                    continue

                row = result["row"]
                family_rows.append(row)
                completed_rows.add(current_key)
                txt_path = write_family_results_txt(test_family, family_rows)
                print(
                    f"{planet_label} / {species}: Mdot={row['mass_loss_rate_g_s']:.3e} g/s, "
                    f"escaping_mass={row['escaping_shell_mass_g']:.3e} g, "
                    f"mean_t={row['mean_escape_time_s']:.3e} s, "
                    f"Myr={row['mass_lost_Mearth_1Myr']:.3e} Mearth, "
                    f"Gyr={row['mass_lost_Mearth_1Gyr']:.3e} Mearth, "
                    f"cache_bins={row['photon_pressure_cache_bins']}, "
                    f"elapsed={result['elapsed_s']:.1f} s"
                )
                print(f"Updated family results: {txt_path.name}")

            for system_def in batch_systems:
                planet_label, _ = get_system_display_names(system_def)
                total_row = total_row_from_species_rows(system_rows_from_checkpoint(family_rows, system_def))
                if total_row is not None:
                    print(
                        f"TOTAL_INCLUDED_SPECIES for {system_def.test_family} / {planet_label}: "
                        f"Mdot={total_row['mass_loss_rate_g_s']:.3e} g/s, "
                        f"Myr={total_row['mass_lost_Mearth_1Myr']:.3e} Mearth, "
                        f"Gyr={total_row['mass_lost_Mearth_1Gyr']:.3e} Mearth"
                    )

    print(f"Total elapsed time: {time.perf_counter() - start_time:.1f} s")


def main() -> None:
    if family_enabled("solar_system_fixed") or family_enabled("real_reference_systems") or family_enabled("distance_sweep"):
        run_standard_systems()
    if family_enabled(P0_SWEEP_TEST_FAMILY):
        run_p0_sweep()
    if family_enabled(MU_SWEEP_HOT_JUPITER_FAMILY):
        run_mu_sweeps()
    if family_enabled(SURFACE_GRAVITY_SWEEP_FAMILY):
        run_surface_gravity_sweep()


initialize_base_namespace()


if __name__ == "__main__":
    main()
