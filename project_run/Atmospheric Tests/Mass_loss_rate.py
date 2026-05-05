import copy
import csv
import os
import pathlib
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import astropy.units as u
from astropy import constants as const

# Avoid RADIS/numba cache issues when PhotonPressure imports the molecule stack.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem
from project_classes.Star import Star
from project_func.Templates.Atoms.atom_species import ATOM_SPECIES
from project_func.exobase_table_path import resolve_exobase_table_path
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Planets.planet_templates import PLANET_TEMPLATES, get_planet_template
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template
from project_func.plotdata_to_txt import save_plotdata_txt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RUN_STANDARD_GRID = True
RUN_P0_SWEEP = True

# First-test mode: one planet over the same target-Teff stellar grid used in the
# big beta table and several distances. Set to False to run all planet templates.
RUN_SINGLE_PLANET_GRID = False
SINGLE_PLANET_KEY = "earth_like"

PLANET_KEYS = list(PLANET_TEMPLATES.keys())
TARGET_TEFFS_K = [3000, 5000, 6000, 8000, 10000, 15000, 20000, 30000, 50000]
DISTANCE_LIST_AU = [0.05, 0.1, 1.0, 10.0, 100.0]

# P0 sweep study for one representative inflated hot Jupiter. The user asked
# for 10 sweeps from 1e-8 to "P=0", interpreted here as 10^0 bar.
P0_SWEEP_PLANET_KEY = "inflated_hot_jupiter"
P0_SWEEP_VALUES_BAR = np.logspace(-8, 0, 10)

SKIP_MOLECULES = True
SELECTED_ATOMIC_SPECIES = None

# Stellar gravity is enabled for the next test. If enabled, this uses the tidal
# acceleration in the planet-centered frame, not the absolute stellar force.
INCLUDE_STELLAR_GRAVITY = True

N_RHO = 64
N_X = 200
RHO_GRID_POWER = 4.0
X_GRID_POWER = 3.0
COLUMN_STEPS = 240
COLUMN_GRID_POWER = 3.0
BETA_BATCH_SIZE = 64

WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_ATOM = 1.0 * u.km / u.s
NPTS_ATOM = 150

OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "Plots"
    / "Atmospheric test"
    / "Mass_loss_rate"
)
P0_SWEEP_OUTPUT_DIR = OUTPUT_DIR / "P0_sweeps" / P0_SWEEP_PLANET_KEY
USE_CHECKPOINT = True
CHECKPOINT_PATH = OUTPUT_DIR / "mass_loss_checkpoint.csv"
EXOBASE_TABLE = (
    resolve_exobase_table_path(pathlib.Path(__file__).resolve().parents[2])
)

M_NEPTUNE = 17.147 * u.M_earth
ROCKY_CATEGORIES = {"rocky"}
NEPTUNE_LIKE_CATEGORIES = {"mini_neptune", "sub_neptune", "neptune"}
JUPITER_LIKE_CATEGORIES = {"gas_giant"}


star_cache: Dict[str, Star] = {}
profile_cache: Dict[str, BroadeningProfile] = {}


def checkpoint_fieldnames() -> List[str]:
    return [
        "planet",
        "star",
        "stellar_teff_K",
        "distance_AU",
        "species",
        "mixing_ratio",
        "z_exobase_km",
        "r_exobase_over_Rp",
        "hill_radius_over_Rp",
        "total_torus_mass_g",
        "escaping_torus_mass_g",
        "mass_loss_rate_g_s",
        "mass_loss_rate_kg_s",
        "mass_loss_rate_Mearth_yr",
        "mean_escape_time_s",
        "min_escape_time_s",
        "mass_weighted_beta",
        "max_beta_escaping_cells",
        "mass_weighted_upstream_column_cm2",
        "max_upstream_column_cm2",
        "n_cells",
        "n_escape_cells",
        "include_stellar_gravity",
        "column_geometry",
        "n_rho",
        "n_x",
        "column_steps",
        "rho_grid_power",
    ]


# -----------------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------------
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


def select_star_keys_by_target_teff(target_teffs_k: Iterable[float]) -> List[str]:
    all_keys = list(STAR_TEMPLATES.keys())
    teff_map = {key: float(infer_teff_from_star_template(key)) for key in all_keys}
    selected_keys = []
    used_keys = set()

    for target_teff in target_teffs_k:
        remaining = [key for key in all_keys if key not in used_keys]
        if not remaining:
            break
        best_key = min(remaining, key=lambda key: abs(teff_map[key] - float(target_teff)))
        selected_keys.append(best_key)
        used_keys.add(best_key)

    return selected_keys


def iter_systems():
    star_keys = select_star_keys_by_target_teff(TARGET_TEFFS_K)
    planet_keys = [SINGLE_PLANET_KEY] if RUN_SINGLE_PLANET_GRID else PLANET_KEYS
    for planet_key in planet_keys:
        for star_key in star_keys:
            for distance_au in DISTANCE_LIST_AU:
                yield planet_key, star_key, distance_au


def get_atom_profile(species: str) -> BroadeningProfile:
    if species not in profile_cache:
        atom = Atom(species, WAVEMIN, WAVEMAX)
        profile_cache[species] = BroadeningProfile(atom, B_ATOM, NPTS_ATOM, "Voigt")
    return profile_cache[species]


def selected_species_for_planet(planet_case: dict) -> List[str]:
    selected = []
    for species in planet_case["composition"].keys():
        if species in MOLECULE_TEMPLATES:
            if SKIP_MOLECULES:
                continue
            raise NotImplementedError("Molecular mass-loss estimates are intentionally disabled for now.")

        if species not in ATOM_SPECIES:
            continue
        if SELECTED_ATOMIC_SPECIES is not None and species not in SELECTED_ATOMIC_SPECIES:
            continue
        selected.append(species)
    return selected


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
    mass_g = np.trapz(integrand.to_value(u.g / u.cm), z_grid.to_value(u.cm)) * u.g
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


def build_total_mass_loss_matrix(rows: List[dict], planet_key: str) -> Tuple[List[float], List[float], np.ndarray]:
    total_rows = [row for row in rows if row.get("species") == "TOTAL_ATOMS" and row.get("planet") == planet_key]
    teff_values = sorted({float(row["stellar_teff_K"]) for row in total_rows})
    distance_values = sorted({float(row["distance_AU"]) for row in total_rows})
    matrix = np.full((len(teff_values), len(distance_values)), np.nan, dtype=float)

    for row in total_rows:
        i = teff_values.index(float(row["stellar_teff_K"]))
        j = distance_values.index(float(row["distance_AU"]))
        matrix[i, j] = float(row["mass_loss_rate_g_s"])

    return teff_values, distance_values, matrix


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


def write_total_mass_loss_summary(
    summary_path: pathlib.Path,
    planet_key: str,
    mass_info: dict,
    g_s_path: pathlib.Path,
    unit_path: pathlib.Path,
    ratio_path: pathlib.Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Total Mass-Loss Grid Summary\n")
        f.write("===========================\n")
        f.write(f"planet: {planet_key}\n")
        f.write(f"planet_category: {mass_info['planet_category']}\n")
        f.write(f"reference_mass_kind: {mass_info['reference_mass_kind']}\n")
        f.write(f"reference_mass_g: {mass_info['reference_mass_g']:.12e}\n")
        f.write(f"reference_mass_unit_name: {mass_info['reference_mass_unit_name']}\n")
        f.write(f"reference_mass_in_unit: {mass_info['reference_mass_in_unit']:.12e}\n")
        if mass_info["reference_top_km"] != "":
            f.write(f"reference_atmosphere_top_km: {float(mass_info['reference_top_km']):.6f}\n")
            f.write(f"reference_atmosphere_top_source: {mass_info['reference_top_source']}\n")
        f.write(f"rate_unit_name: {mass_info['rate_unit_name']}\n")
        f.write(f"g_s_file: {g_s_path.name}\n")
        f.write(f"planet_unit_file: {unit_path.name}\n")
        f.write(f"ratio_file: {ratio_path.name}\n")


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------
def build_planet(planet_case: dict) -> Planet:
    return Planet(
        radius=planet_case["radius"],
        mass=planet_case["mass"],
        T=planet_case["T"],
        mu=planet_case["mu"],
        P0=planet_case["P0"],
    )


def rho_edges_between(r_inner: u.Quantity, r_outer: u.Quantity, n_rho: int) -> u.Quantity:
    fraction = np.linspace(0.0, 1.0, n_rho + 1) ** RHO_GRID_POWER
    edges_cm = r_inner.to_value(u.cm) + (r_outer - r_inner).to_value(u.cm) * fraction
    return edges_cm * u.cm


def symmetric_x_edges(x_max_cm: float, n_x: int) -> np.ndarray:
    n_half = max(1, int(n_x) // 2)
    positive_edges = x_max_cm * (np.linspace(0.0, 1.0, n_half + 1) ** X_GRID_POWER)
    return np.concatenate((-positive_edges[:0:-1], positive_edges))


def local_shell_columns_and_density(
    planet: Planet,
    x_mid_cm: np.ndarray,
    dx_cm: np.ndarray,
    rho_cm: float,
    r_exobase_cm: float,
    abundance: float,
) -> Tuple[u.Quantity, np.ndarray, np.ndarray]:
    """
    Local upstream species column for photons travelling in +x.

    The column is integrated only through the affected shell above the exobase,
    from the left-side Hill boundary to each cell center. Cells below the
    exobase are excluded from the mass integral and from the shielding column.
    """
    planet_radius_cm = planet.radius.to_value(u.cm)
    r_cm = np.sqrt(x_mid_cm**2 + rho_cm**2)
    shell_mask = r_cm >= r_exobase_cm

    n_species_cm3 = np.zeros_like(x_mid_cm, dtype=float)
    if np.any(shell_mask):
        z_shell = (r_cm[shell_mask] - planet_radius_cm) * u.cm
        n_species_cm3[shell_mask] = abundance * planet.number_density(z_shell).to_value(1 / u.cm**3)

    column_before_cell = np.concatenate(([0.0], np.cumsum(n_species_cm3[:-1] * dx_cm[:-1])))
    column_to_cell_center = column_before_cell + 0.5 * n_species_cm3 * dx_cm

    return column_to_cell_center / u.cm**2, n_species_cm3, shell_mask


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
    return float(np.trapz(n_species_cm3, x_cm))


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


def photon_acceleration_for_columns(
    pp: PhotonPressure,
    species_mass: u.Quantity,
    planet_case: dict,
    distance: u.Quantity,
    ncol_cm2: np.ndarray,
) -> np.ndarray:
    a_rad = np.full(len(ncol_cm2), np.nan, dtype=float)
    species_mass_g = species_mass.to(u.g)

    for start in range(0, len(ncol_cm2), BETA_BATCH_SIZE):
        stop = min(start + BETA_BATCH_SIZE, len(ncol_cm2))
        columns = ncol_cm2[start:stop] / u.cm**2
        force, _, _, _ = pp.calc_PhotonPressure(columns, planet_case["T"], distance)
        force_dyn = np.ravel(force.to(u.dyn))
        a_rad[start:stop] = (force_dyn / species_mass_g).to_value(u.cm / u.s**2)

    return a_rad


def stellar_tidal_acceleration(
    x_cm: np.ndarray,
    rho_cm: float,
    distance_cm: float,
    mu_star: float,
) -> Tuple[np.ndarray, np.ndarray]:
    norm = ((distance_cm + x_cm) ** 2 + rho_cm**2) ** 1.5
    a_star_x = -mu_star * (distance_cm + x_cm) / norm
    a_star_rho = -mu_star * rho_cm / norm
    a_center_x = -mu_star / distance_cm**2
    return a_star_x - a_center_x, a_star_rho


def escape_time_to_right_hill(
    x_cm: np.ndarray,
    rho_cm: float,
    ax_cm_s2: np.ndarray,
    arho_cm_s2: np.ndarray,
    hill_radius_cm: float,
) -> np.ndarray:
    accel2 = ax_cm_s2**2 + arho_cm_s2**2
    dot = x_cm * ax_cm_s2 + rho_cm * arho_cm_s2
    inside = x_cm**2 + rho_cm**2 - hill_radius_cm**2
    discriminant = (2.0 * dot) ** 2 - 4.0 * accel2 * inside

    times = np.full_like(x_cm, np.nan, dtype=float)
    valid = (accel2 > 0.0) & (discriminant >= 0.0) & (ax_cm_s2 > 0.0)
    if not np.any(valid):
        return times

    sqrt_disc = np.sqrt(discriminant[valid])
    s_escape = (-(2.0 * dot[valid]) + sqrt_disc) / (2.0 * accel2[valid])
    x_cross = x_cm[valid] + ax_cm_s2[valid] * s_escape
    valid_escape = (s_escape > 0.0) & (x_cross > 0.0)

    valid_indices = np.where(valid)[0]
    times[valid_indices[valid_escape]] = np.sqrt(2.0 * s_escape[valid_escape])
    return times


def mass_loss_for_species(
    planet_key: str,
    species: str,
    planet_case: dict,
    system: PlanetarySystem,
    exobase_rows: Dict[Tuple[str, str], dict],
) -> dict:
    z_exobase = exobase_height(planet_key, species, exobase_rows)
    if z_exobase is None:
        raise ValueError(f"No exobase height for {planet_key}, {species}")

    planet = system.planet
    star = system.star
    distance = system.distance.to(u.cm)
    planet_radius = planet.radius.to(u.cm)
    hill_radius = system.hill_radius().to(u.cm)
    r_exobase = planet_radius + z_exobase.to(u.cm)

    if r_exobase >= hill_radius:
        raise ValueError("Exobase is outside or at the Hill radius.")

    abundance = species_mixing_ratio(planet_case, species)
    if abundance <= 0.0:
        raise ValueError("Species abundance is zero or negative.")

    profile = get_atom_profile(species)
    pp = PhotonPressure(profile, star)
    species_mass = profile.molecule.mass.to(u.g)

    mu_planet = (const.G.cgs * planet.mass.to(u.g)).to_value(u.cm**3 / u.s**2)
    mu_star = (const.G.cgs * star.mass.to(u.g)).to_value(u.cm**3 / u.s**2)
    hill_cm = hill_radius.to_value(u.cm)
    r_exobase_cm = r_exobase.to_value(u.cm)
    distance_cm = distance.to_value(u.cm)
    species_mass_g = species_mass.to_value(u.g)

    cells = spherical_shell_cells(
        planet,
        abundance,
        species_mass_g,
        r_exobase_cm,
        hill_cm,
    )
    if len(cells["dm_g"]) == 0:
        raise ValueError("No valid shell cells were generated.")

    x_cm = cells["x_cm"]
    rho_cm = cells["rho_cm"]
    r_cm = cells["r_cm"]
    dm_g = cells["dm_g"]
    ncol_cm2 = cells["ncol_cm2"]
    a_rad = photon_acceleration_for_columns(pp, species_mass, planet_case, distance, ncol_cm2)

    g_planet = mu_planet / r_cm**2
    beta_local = a_rad / g_planet
    ax = a_rad - mu_planet * x_cm / r_cm**3
    arho = -mu_planet * rho_cm / r_cm**3

    if INCLUDE_STELLAR_GRAVITY:
        ax_star, arho_star = stellar_tidal_acceleration(x_cm, rho_cm, distance_cm, mu_star)
        ax = ax + ax_star
        arho = arho + arho_star

    escape_time_s = escape_time_to_right_hill(x_cm, rho_cm, ax, arho, hill_cm)
    escape_mask = np.isfinite(escape_time_s) & (escape_time_s > 0.0) & np.isfinite(dm_g) & (dm_g > 0.0)

    total_mass_g = float(np.nansum(dm_g))
    escaping_mass_g = float(np.nansum(dm_g[escape_mask])) if np.any(escape_mask) else 0.0
    mdot_g_s = float(np.nansum(dm_g[escape_mask] / escape_time_s[escape_mask])) if np.any(escape_mask) else 0.0
    beta_mass_sum = float(np.nansum(beta_local[escape_mask] * dm_g[escape_mask])) if np.any(escape_mask) else 0.0
    upstream_column_mass_sum = float(np.nansum(ncol_cm2 * dm_g))
    min_escape_time_s = float(np.nanmin(escape_time_s[escape_mask])) if np.any(escape_mask) else np.inf
    max_beta = float(np.nanmax(beta_local[escape_mask])) if np.any(escape_mask) else -np.inf
    max_upstream_column_cm2 = float(np.nanmax(ncol_cm2)) if len(ncol_cm2) else -np.inf
    n_cells = len(dm_g)
    n_escape_cells = int(np.count_nonzero(escape_mask))

    mean_escape_time_s = np.nan
    mass_weighted_beta = np.nan
    mass_weighted_upstream_column_cm2 = np.nan
    if mdot_g_s > 0.0:
        mean_escape_time_s = escaping_mass_g / mdot_g_s
    if escaping_mass_g > 0.0:
        mass_weighted_beta = beta_mass_sum / escaping_mass_g
    if total_mass_g > 0.0:
        mass_weighted_upstream_column_cm2 = upstream_column_mass_sum / total_mass_g

    return {
        "planet": planet_key,
        "species": species,
        "mixing_ratio": abundance,
        "z_exobase_km": z_exobase.to_value(u.km),
        "r_exobase_over_Rp": (r_exobase / planet_radius).decompose().value,
        "hill_radius_over_Rp": (hill_radius / planet_radius).decompose().value,
        "total_torus_mass_g": total_mass_g,
        "escaping_torus_mass_g": escaping_mass_g,
        "mass_loss_rate_g_s": mdot_g_s,
        "mass_loss_rate_kg_s": (mdot_g_s * u.g / u.s).to_value(u.kg / u.s),
        "mass_loss_rate_Mearth_yr": (mdot_g_s * u.g / u.s).to_value(u.M_earth / u.yr),
        "mean_escape_time_s": mean_escape_time_s,
        "min_escape_time_s": np.nan if not np.isfinite(min_escape_time_s) else min_escape_time_s,
        "mass_weighted_beta": mass_weighted_beta,
        "max_beta_escaping_cells": np.nan if not np.isfinite(max_beta) else max_beta,
        "mass_weighted_upstream_column_cm2": mass_weighted_upstream_column_cm2,
        "max_upstream_column_cm2": np.nan if not np.isfinite(max_upstream_column_cm2) else max_upstream_column_cm2,
        "n_cells": n_cells,
        "n_escape_cells": n_escape_cells,
    }


def total_row_from_species_rows(system_rows: List[dict]) -> dict | None:
    if not system_rows:
        return None

    template = dict(system_rows[0])
    total_mdot_g_s = float(np.nansum([float(row.get("mass_loss_rate_g_s", 0.0)) for row in system_rows]))
    total_mdot = total_mdot_g_s * u.g / u.s
    total_escaping_mass_g = float(np.nansum([float(row.get("escaping_torus_mass_g", 0.0)) for row in system_rows]))
    total_torus_mass_g = float(np.nansum([float(row.get("total_torus_mass_g", 0.0)) for row in system_rows]))

    template.update(
        {
            "species": "TOTAL_ATOMS",
            "mixing_ratio": "",
            "z_exobase_km": "",
            "r_exobase_over_Rp": "",
            "total_torus_mass_g": total_torus_mass_g,
            "escaping_torus_mass_g": total_escaping_mass_g,
            "mass_loss_rate_g_s": total_mdot_g_s,
            "mass_loss_rate_kg_s": total_mdot.to_value(u.kg / u.s),
            "mass_loss_rate_Mearth_yr": total_mdot.to_value(u.M_earth / u.yr),
            "mean_escape_time_s": total_escaping_mass_g / total_mdot_g_s if total_mdot_g_s > 0.0 else np.nan,
            "min_escape_time_s": np.nanmin([float(row.get("min_escape_time_s", np.nan)) for row in system_rows]),
            "mass_weighted_beta": "",
            "max_beta_escaping_cells": np.nanmax([float(row.get("max_beta_escaping_cells", np.nan)) for row in system_rows]),
            "mass_weighted_upstream_column_cm2": "",
            "max_upstream_column_cm2": np.nanmax([float(row.get("max_upstream_column_cm2", np.nan)) for row in system_rows]),
            "n_cells": int(np.nansum([int(row.get("n_cells", 0)) for row in system_rows])),
            "n_escape_cells": int(np.nansum([int(row.get("n_escape_cells", 0)) for row in system_rows])),
        }
    )
    return template


def save_total_grid_outputs(rows: List[dict], planet_keys: Optional[Iterable[str]] = None) -> None:
    total_rows = [row for row in rows if row.get("species") == "TOTAL_ATOMS"]
    if not total_rows:
        return

    planets = sorted({row["planet"] for row in total_rows})
    if planet_keys is not None:
        allowed = set(planet_keys)
        planets = [planet_key for planet_key in planets if planet_key in allowed]

    exobase_rows = load_exobase_table(EXOBASE_TABLE)

    for planet_key in planets:
        planet_case = get_planet_template(planet_key)
        teff_values, distance_values, matrix_g_s = build_total_mass_loss_matrix(total_rows, planet_key)
        if matrix_g_s.size == 0:
            continue

        mass_info = planet_reference_mass_info(planet_key, planet_case, exobase_rows)
        unit_matrix = mass_loss_in_planet_unit_per_year(matrix_g_s, planet_case)
        ratio_matrix, _ = mass_loss_over_reference_mass_per_second(matrix_g_s, planet_key, planet_case, exobase_rows)

        safe_planet = safe_name(planet_key)
        base_metadata = {
            "planet": planet_key,
            "species": "TOTAL_ATOMS",
            "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
            "column_geometry": "local_upstream_shell_with_exobase_tangent_cutoff",
            "n_rho": N_RHO,
            "n_x": N_X,
            "column_steps": COLUMN_STEPS,
            "rho_grid_power": RHO_GRID_POWER,
            "planet_category": mass_info["planet_category"],
            "reference_mass_kind": mass_info["reference_mass_kind"],
            "reference_mass_g": mass_info["reference_mass_g"],
            "reference_mass_unit_name": mass_info["reference_mass_unit_name"],
            "reference_mass_in_unit": mass_info["reference_mass_in_unit"],
            "reference_top_km": mass_info["reference_top_km"],
            "reference_top_source": mass_info["reference_top_source"],
            "note": "Sum of per-species atomic mass-loss rates for species present in the planet composition.",
        }

        g_s_path = OUTPUT_DIR / f"{safe_planet}_total_mass_loss_vs_teff_distance.txt"
        save_mass_loss_matrix_txt(
            g_s_path,
            dataset_name=f"{safe_planet}_total_mass_loss_vs_teff_distance",
            x_label="Stellar Teff",
            x_unit="K",
            y_label="Total atomic mass-loss rate",
            y_unit="g/s",
            x_values=teff_values,
            y_matrix=matrix_g_s,
            series_values=distance_values,
            series_label="distance",
            series_unit="AU",
            column_names=[f"distance_{value:g}_AU" for value in distance_values],
            extra_metadata=base_metadata,
        )

        unit_suffix = mass_info["reference_mass_unit_name"]
        unit_path = OUTPUT_DIR / f"{safe_planet}_total_mass_loss_vs_teff_distance_{unit_suffix}_yr.txt"
        save_mass_loss_matrix_txt(
            unit_path,
            dataset_name=f"{safe_planet}_total_mass_loss_vs_teff_distance_{unit_suffix}_yr",
            x_label="Stellar Teff",
            x_unit="K",
            y_label="Total atomic mass-loss rate",
            y_unit=f"{unit_suffix}/yr",
            x_values=teff_values,
            y_matrix=unit_matrix,
            series_values=distance_values,
            series_label="distance",
            series_unit="AU",
            column_names=[f"distance_{value:g}_AU" for value in distance_values],
            extra_metadata={**base_metadata, "rate_unit_name": f"{unit_suffix}/yr"},
        )

        ratio_path = OUTPUT_DIR / f"{safe_planet}_total_mass_loss_ratio_vs_teff_distance.txt"
        save_mass_loss_matrix_txt(
            ratio_path,
            dataset_name=f"{safe_planet}_total_mass_loss_ratio_vs_teff_distance",
            x_label="Stellar Teff",
            x_unit="K",
            y_label="Total atomic specific mass-loss rate",
            y_unit="1/s",
            x_values=teff_values,
            y_matrix=ratio_matrix,
            series_values=distance_values,
            series_label="distance",
            series_unit="AU",
            column_names=[f"distance_{value:g}_AU" for value in distance_values],
            extra_metadata={**base_metadata, "ratio_definition": "Mdot / M_reference"},
        )

        summary_path = OUTPUT_DIR / f"{safe_planet}_total_mass_loss_summary.txt"
        write_total_mass_loss_summary(summary_path, planet_key, mass_info, g_s_path, unit_path, ratio_path)
        print(f"Saved total mass-loss grid to {g_s_path}")


def unique_in_order(values: Iterable) -> List:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def current_checkpoint_config() -> dict:
    return {
        "include_stellar_gravity": bool(INCLUDE_STELLAR_GRAVITY),
        "column_geometry": "local_upstream_shell_with_exobase_tangent_cutoff",
        "n_rho": int(N_RHO),
        "n_x": int(N_X),
        "column_steps": int(COLUMN_STEPS),
        "rho_grid_power": float(RHO_GRID_POWER),
    }


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def system_key_from_values(planet_key: str, star_key: str, distance_au: float) -> Tuple[str, str, str]:
    return planet_key, star_key, f"{float(distance_au):.12g}"


def system_key_from_row(row: dict) -> Tuple[str, str, str]:
    return system_key_from_values(row["planet"], row["star"], float(row["distance_AU"]))


def validate_checkpoint_rows(rows: List[dict]) -> None:
    if not rows:
        return

    expected = current_checkpoint_config()
    for row in rows:
        actual = {
            "include_stellar_gravity": parse_bool(row.get("include_stellar_gravity", "")),
            "column_geometry": row.get("column_geometry", ""),
            "n_rho": int(float(row.get("n_rho", np.nan))),
            "n_x": int(float(row.get("n_x", np.nan))),
            "column_steps": int(float(row.get("column_steps", np.nan))),
            "rho_grid_power": float(row.get("rho_grid_power", np.nan)),
        }
        if actual != expected:
            raise ValueError(
                "Existing checkpoint was created with different numerical settings. "
                f"Delete {CHECKPOINT_PATH} or restore the old settings before resuming."
            )


def load_checkpoint_rows(path: pathlib.Path) -> List[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    validate_checkpoint_rows(rows)
    return rows


def save_checkpoint_rows(rows: List[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=checkpoint_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in checkpoint_fieldnames()})
    tmp_path.replace(path)


def completed_system_keys(rows: List[dict]) -> set[Tuple[str, str, str]]:
    return {
        system_key_from_row(row)
        for row in rows
        if row.get("species") == "TOTAL_ATOMS"
    }


def save_species_grid_outputs(rows: List[dict], planet_keys: Iterable[str] | None = None) -> None:
    species_rows = [row for row in rows if row.get("species") != "TOTAL_ATOMS"]
    if not species_rows:
        return

    planets = sorted({row["planet"] for row in species_rows})
    if planet_keys is not None:
        allowed = set(planet_keys)
        planets = [planet_key for planet_key in planets if planet_key in allowed]
    for planet_key in planets:
        planet_rows = [row for row in species_rows if row["planet"] == planet_key]
        teff_values = sorted({float(row["stellar_teff_K"]) for row in planet_rows})
        distance_values = sorted({float(row["distance_AU"]) for row in planet_rows})
        species_values = unique_in_order(row["species"] for row in planet_rows)
        series_keys = [(distance, species) for distance in distance_values for species in species_values]
        matrix = np.full((len(teff_values), len(series_keys)), np.nan, dtype=float)

        for row in planet_rows:
            i = teff_values.index(float(row["stellar_teff_K"]))
            key = (float(row["distance_AU"]), row["species"])
            j = series_keys.index(key)
            matrix[i, j] = float(row["mass_loss_rate_g_s"])

        txt_path = OUTPUT_DIR / f"{safe_name(planet_key)}_species_mass_loss_vs_teff_distance.txt"
        save_plotdata_txt(
            txt_path,
            dataset_name=f"{safe_name(planet_key)}_species_mass_loss_vs_teff_distance",
            x_label="Stellar Teff",
            x_unit="K",
            y_label="Species mass-loss rate",
            y_unit="g/s",
            x_values=np.asarray(teff_values, dtype=float),
            y_matrix=matrix,
            series_values=[f"{distance:g} AU {species}" for distance, species in series_keys],
            series_label="distance_species",
            series_unit="AU_species",
            column_names=[
                f"distance_{distance:g}_AU__{safe_name(species)}" for distance, species in series_keys
            ],
            extra_metadata={
                "planet": planet_key,
                "species": ", ".join(species_values),
                "distances_AU": ", ".join(f"{distance:g}" for distance in distance_values),
                "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
                "column_geometry": "local_upstream_shell_with_exobase_tangent_cutoff",
                "n_rho": N_RHO,
                "n_x": N_X,
                "column_steps": COLUMN_STEPS,
                "rho_grid_power": RHO_GRID_POWER,
                "note": "Per-species atomic mass-loss rates for species present in the planet composition.",
            },
        )
        print(f"Saved species mass-loss grid to {txt_path}")


def total_mass_loss_for_system(
    planet_key: str,
    star_key: str,
    distance_au: float,
    planet_case: dict,
    exobase_rows: Dict[Tuple[str, str], dict],
) -> Optional[dict]:
    planet = build_planet(planet_case)
    star = get_star(star_key)
    distance = float(distance_au) * u.AU
    system = PlanetarySystem(planet, star, distance)
    species_list = selected_species_for_planet(planet_case)
    system_rows: List[dict] = []

    for species in species_list:
        try:
            row = mass_loss_for_species(planet_key, species, planet_case, system, exobase_rows)
        except Exception as exc:
            print(f"Skipping {planet_key} {species} in P0 sweep: {type(exc).__name__}: {exc}")
            continue

        row.update(
            {
                "star": star_key,
                "stellar_teff_K": infer_teff_from_star_template(star_key),
                "distance_AU": float(distance_au),
                "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
                "column_geometry": "local_upstream_shell_with_exobase_tangent_cutoff",
                "n_rho": N_RHO,
                "n_x": N_X,
                "column_steps": COLUMN_STEPS,
                "rho_grid_power": RHO_GRID_POWER,
            }
        )
        system_rows.append(row)

    total_row = total_row_from_species_rows(system_rows)
    if total_row is None:
        return None

    total_row.update(
        {
            "planet": planet_key,
            "star": star_key,
            "stellar_teff_K": infer_teff_from_star_template(star_key),
            "distance_AU": float(distance_au),
            "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
            "column_geometry": "local_upstream_shell_with_exobase_tangent_cutoff",
            "n_rho": N_RHO,
            "n_x": N_X,
            "column_steps": COLUMN_STEPS,
            "rho_grid_power": RHO_GRID_POWER,
        }
    )
    return total_row


def save_p0_sweep_outputs(total_rows: List[dict]) -> None:
    if not total_rows:
        return

    planet_case = get_planet_template(P0_SWEEP_PLANET_KEY)
    exobase_rows = load_exobase_table(EXOBASE_TABLE)
    mass_info = planet_reference_mass_info(P0_SWEEP_PLANET_KEY, planet_case, exobase_rows)

    p0_values = sorted({float(row["P0_bar"]) for row in total_rows})
    system_keys = unique_in_order((row["star"], float(row["distance_AU"])) for row in total_rows)
    matrix_g_s = np.full((len(p0_values), len(system_keys)), np.nan, dtype=float)

    for row in total_rows:
        i = p0_values.index(float(row["P0_bar"]))
        j = system_keys.index((row["star"], float(row["distance_AU"])))
        matrix_g_s[i, j] = float(row["mass_loss_rate_g_s"])

    unit_matrix = mass_loss_in_planet_unit_per_year(matrix_g_s, planet_case)
    ratio_matrix, _ = mass_loss_over_reference_mass_per_second(
        matrix_g_s,
        P0_SWEEP_PLANET_KEY,
        planet_case,
        exobase_rows,
    )
    safe_planet = safe_name(P0_SWEEP_PLANET_KEY)
    series_values = [f"{star}_{distance:g}AU" for star, distance in system_keys]
    column_names = [f"{safe_name(star)}_{distance:g}AU" for star, distance in system_keys]

    base_metadata = {
        "planet": P0_SWEEP_PLANET_KEY,
        "species": "TOTAL_ATOMS",
        "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
        "column_geometry": "local_upstream_shell_with_exobase_tangent_cutoff",
        "n_rho": N_RHO,
        "n_x": N_X,
        "column_steps": COLUMN_STEPS,
        "rho_grid_power": RHO_GRID_POWER,
        "planet_category": mass_info["planet_category"],
        "reference_mass_kind": mass_info["reference_mass_kind"],
        "reference_mass_g": mass_info["reference_mass_g"],
        "reference_mass_unit_name": mass_info["reference_mass_unit_name"],
        "reference_mass_in_unit": mass_info["reference_mass_in_unit"],
        "reference_top_km": mass_info["reference_top_km"],
        "reference_top_source": mass_info["reference_top_source"],
        "note": "Inflated hot Jupiter total mass-loss rate as a function of varying P0.",
    }

    P0_SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g_s_path = P0_SWEEP_OUTPUT_DIR / f"{safe_planet}_total_mass_loss_vs_P0.txt"
    save_mass_loss_matrix_txt(
        g_s_path,
        dataset_name=f"{safe_planet}_total_mass_loss_vs_P0",
        x_label="P0",
        x_unit="bar",
        y_label="Total atomic mass-loss rate",
        y_unit="g/s",
        x_values=p0_values,
        y_matrix=matrix_g_s,
        series_values=series_values,
        series_label="system",
        series_unit="star_distance",
        column_names=column_names,
        extra_metadata=base_metadata,
    )

    unit_suffix = mass_info["reference_mass_unit_name"]
    unit_path = P0_SWEEP_OUTPUT_DIR / f"{safe_planet}_total_mass_loss_vs_P0_{unit_suffix}_yr.txt"
    save_mass_loss_matrix_txt(
        unit_path,
        dataset_name=f"{safe_planet}_total_mass_loss_vs_P0_{unit_suffix}_yr",
        x_label="P0",
        x_unit="bar",
        y_label="Total atomic mass-loss rate",
        y_unit=f"{unit_suffix}/yr",
        x_values=p0_values,
        y_matrix=unit_matrix,
        series_values=series_values,
        series_label="system",
        series_unit="star_distance",
        column_names=column_names,
        extra_metadata={**base_metadata, "rate_unit_name": f"{unit_suffix}/yr"},
    )

    ratio_path = P0_SWEEP_OUTPUT_DIR / f"{safe_planet}_total_mass_loss_ratio_vs_P0.txt"
    save_mass_loss_matrix_txt(
        ratio_path,
        dataset_name=f"{safe_planet}_total_mass_loss_ratio_vs_P0",
        x_label="P0",
        x_unit="bar",
        y_label="Total atomic specific mass-loss rate",
        y_unit="1/s",
        x_values=p0_values,
        y_matrix=ratio_matrix,
        series_values=series_values,
        series_label="system",
        series_unit="star_distance",
        column_names=column_names,
        extra_metadata={**base_metadata, "ratio_definition": "Mdot / M_reference"},
    )

    summary_path = P0_SWEEP_OUTPUT_DIR / f"{safe_planet}_P0_sweep_summary.txt"
    write_total_mass_loss_summary(summary_path, P0_SWEEP_PLANET_KEY, mass_info, g_s_path, unit_path, ratio_path)
    print(f"Saved P0 sweep outputs to {P0_SWEEP_OUTPUT_DIR}")


def run_p0_sweep() -> None:
    exobase_rows = load_exobase_table(EXOBASE_TABLE)
    filtered_systems = [
        (planet_key, star_key, distance_au)
        for planet_key, star_key, distance_au in iter_systems()
        if planet_key == P0_SWEEP_PLANET_KEY
    ]
    if not filtered_systems:
        print(f"No systems available for P0 sweep planet={P0_SWEEP_PLANET_KEY}")
        return

    total_rows: List[dict] = []
    for p0_bar in P0_SWEEP_VALUES_BAR:
        print(f"\n--- P0 sweep for {P0_SWEEP_PLANET_KEY}: P0={p0_bar:.1e} bar ---")
        for planet_key, star_key, distance_au in filtered_systems:
            planet_case = copy.deepcopy(get_planet_template(planet_key))
            planet_case["P0"] = float(p0_bar) * u.bar
            total_row = total_mass_loss_for_system(planet_key, star_key, distance_au, planet_case, exobase_rows)
            if total_row is None:
                continue
            total_row["P0_bar"] = float(p0_bar)
            total_rows.append(total_row)
            save_p0_sweep_outputs(total_rows)
            print(
                f"P0={p0_bar:.1e} bar | star={star_key} | distance={float(distance_au):g} AU | "
                f"Mdot={float(total_row['mass_loss_rate_g_s']):.3e} g/s"
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_standard_grid() -> None:
    start_time = time.perf_counter()
    exobase_rows = load_exobase_table(EXOBASE_TABLE)
    all_results = load_checkpoint_rows(CHECKPOINT_PATH) if USE_CHECKPOINT else []
    completed_systems = completed_system_keys(all_results)

    if all_results:
        print(
            f"Loaded checkpoint with {len(all_results)} rows and "
            f"{len(completed_systems)} completed systems from {CHECKPOINT_PATH}"
        )
        save_species_grid_outputs(all_results)
        save_total_grid_outputs(all_results)

    for planet_key, star_key, distance_au in iter_systems():
        current_system_key = system_key_from_values(planet_key, star_key, distance_au)
        if current_system_key in completed_systems:
            print(
                f"Skipping completed system: planet={planet_key}, "
                f"star={star_key}, distance={float(distance_au):g} AU"
            )
            continue

        print(f"\n--- Mass-loss system: planet={planet_key}, star={star_key}, distance={distance_au:g} AU ---")
        planet_case = get_planet_template(planet_key)
        planet = build_planet(planet_case)
        star = get_star(star_key)
        distance = float(distance_au) * u.AU
        system = PlanetarySystem(planet, star, distance)
        species_list = selected_species_for_planet(planet_case)

        system_rows = []
        for species in species_list:
            species_start = time.perf_counter()
            try:
                row = mass_loss_for_species(planet_key, species, planet_case, system, exobase_rows)
            except Exception as exc:
                print(f"Skipping {planet_key} {species}: {type(exc).__name__}: {exc}")
                continue

            row.update(
                {
                    "star": star_key,
                    "stellar_teff_K": infer_teff_from_star_template(star_key),
                    "distance_AU": float(distance_au),
                    "include_stellar_gravity": INCLUDE_STELLAR_GRAVITY,
                    "column_geometry": "local_upstream_shell_with_exobase_tangent_cutoff",
                    "n_rho": N_RHO,
                    "n_x": N_X,
                    "column_steps": COLUMN_STEPS,
                    "rho_grid_power": RHO_GRID_POWER,
                }
            )
            system_rows.append(row)
            all_results.append(row)
            print(
                f"{species}: Mdot={row['mass_loss_rate_g_s']:.3e} g/s, "
                f"escaping_mass={row['escaping_torus_mass_g']:.3e} g, "
                f"mean_t={row['mean_escape_time_s']:.3e} s, "
                f"elapsed={time.perf_counter() - species_start:.1f} s"
            )

        total_row = total_row_from_species_rows(system_rows)
        if total_row is not None:
            system_rows.append(total_row)
            all_results.append(total_row)
            print(
                f"TOTAL_ATOMS: Mdot={total_row['mass_loss_rate_g_s']:.3e} g/s, "
                f"escaping_mass={total_row['escaping_torus_mass_g']:.3e} g"
            )

        if system_rows:
            if USE_CHECKPOINT:
                save_checkpoint_rows(all_results, CHECKPOINT_PATH)
            save_species_grid_outputs(all_results, planet_keys=[planet_key])
            save_total_grid_outputs(all_results, planet_keys=[planet_key])
            completed_systems.add(current_system_key)

    if all_results:
        if USE_CHECKPOINT:
            save_checkpoint_rows(all_results, CHECKPOINT_PATH)
        save_species_grid_outputs(all_results)
        save_total_grid_outputs(all_results)

    print(f"Total elapsed time: {time.perf_counter() - start_time:.1f} s")


def main() -> None:
    if RUN_STANDARD_GRID:
        run_standard_grid()
    if RUN_P0_SWEEP:
        run_p0_sweep()


if __name__ == "__main__":
    main()
