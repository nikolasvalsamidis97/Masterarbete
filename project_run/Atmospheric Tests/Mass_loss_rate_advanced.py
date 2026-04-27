import csv
from dataclasses import dataclass
import importlib.util
import os
import pathlib
import tempfile
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import astropy.units as u
from astropy import constants as const

# Avoid RADIS/numba cache issues when PhotonPressure imports the molecule stack.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


BASE_SCRIPT_PATH = pathlib.Path(__file__).with_name("Mass_loss_rate.py")
BASE_SPEC = importlib.util.spec_from_file_location("mass_loss_rate_base", BASE_SCRIPT_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Could not load base mass-loss script from {BASE_SCRIPT_PATH}")
base_mass_loss = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base_mass_loss)

from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.Molecule import Molecule


# -----------------------------------------------------------------------------
# Advanced run configuration
# -----------------------------------------------------------------------------
SKIP_ATOMS = True
SKIP_MOLECULES = False
SELECTED_ATOMIC_SPECIES = None
SELECTED_MOLECULAR_SPECIES = None
INCLUDE_STELLAR_GRAVITY = True
USE_CHECKPOINT = True

# Optional filter for running a subset of the predefined systems.
# Use entries like:
# SELECTED_SYSTEM_KEYS = {
#     ("solar_system_fixed", "earth_like", "G1", 1.0),
#     ("distance_sweeps", "inflated_hot_jupiter", "A0", 0.05),
# }
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

OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "Plots"
    / "Atmospheric test"
    / "Mass_loss_rate_advanced"
)
CHECKPOINT_PATH = OUTPUT_DIR / "mass_loss_advanced_checkpoint.csv"
EXOBASE_TABLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "Plots"
    / "Atmospheric test"
    / "Exobase"
    / "exobase_table_planets.csv"
)

SOLAR_SYSTEM_STAR_KEY = "G1"
HOT_SWEEP_STAR_KEY = "A0"
HOT_SWEEP_DISTANCES_AU = [0.0239, 0.05, 0.1, 0.3, 1.0]


@dataclass(frozen=True)
class AdvancedSystem:
    test_family: str
    planet_key: str
    star_key: str
    distance_au: float


def build_advanced_systems() -> List[AdvancedSystem]:
    systems = [
        AdvancedSystem("solar_system_fixed", "mercury_like", SOLAR_SYSTEM_STAR_KEY, 0.387),
        AdvancedSystem("solar_system_fixed", "earth_like", SOLAR_SYSTEM_STAR_KEY, 1.0),
        AdvancedSystem("solar_system_fixed", "mars_like", SOLAR_SYSTEM_STAR_KEY, 1.524),
        AdvancedSystem("solar_system_fixed", "cold_jupiter", SOLAR_SYSTEM_STAR_KEY, 5.204),

        AdvancedSystem("rocky_exoplanets_plausible", "super_earth_rocky", "G1", 0.08),
        AdvancedSystem("rocky_exoplanets_plausible", "lava_world", "F8", 0.02),
        AdvancedSystem("rocky_exoplanets_plausible", "volatile_super_earth", "K1", 0.05),
        AdvancedSystem("rocky_exoplanets_plausible", "alkali_exosphere_rocky", "F8", 0.03),
        AdvancedSystem("rocky_exoplanets_plausible", "metal_rich_secondary", "F8", 0.05),

        AdvancedSystem("gas_planets_plausible", "mini_neptune_cool", "K1", 0.20),
        AdvancedSystem("gas_planets_plausible", "mini_neptune_warm", "G1", 0.10),
        AdvancedSystem("gas_planets_plausible", "sub_neptune", "G1", 0.08),
        AdvancedSystem("gas_planets_plausible", "warm_neptune", "K1", 0.05),
        AdvancedSystem("gas_planets_plausible", "hot_neptune", "M1", 0.0291),
        AdvancedSystem("gas_planets_plausible", "super_puff", "F8", 0.2514),
        AdvancedSystem("gas_planets_plausible", "warm_jupiter", "G1", 0.0965),
        AdvancedSystem("gas_planets_plausible", "hot_jupiter", "G1", 0.05042),
        AdvancedSystem("gas_planets_plausible", "inflated_hot_jupiter", "F8", 0.0515),
        AdvancedSystem("gas_planets_plausible", "ultra_hot_jupiter", "A0", 0.0239),
    ]

    for distance_au in HOT_SWEEP_DISTANCES_AU:
        systems.append(AdvancedSystem("distance_sweeps", "inflated_hot_jupiter", HOT_SWEEP_STAR_KEY, distance_au))
    for distance_au in HOT_SWEEP_DISTANCES_AU:
        systems.append(AdvancedSystem("distance_sweeps", "super_earth_rocky", HOT_SWEEP_STAR_KEY, distance_au))

    if SELECTED_SYSTEM_KEYS is None:
        return systems

    selected = {
        (family, planet, star, f"{float(distance):.12g}")
        for family, planet, star, distance in SELECTED_SYSTEM_KEYS
    }
    filtered = [
        system
        for system in systems
        if (system.test_family, system.planet_key, system.star_key, f"{float(system.distance_au):.12g}") in selected
    ]
    missing = selected.difference(
        {
            (system.test_family, system.planet_key, system.star_key, f"{float(system.distance_au):.12g}")
            for system in filtered
        }
    )
    if missing:
        raise ValueError(f"Unknown entries in SELECTED_SYSTEM_KEYS: {sorted(missing)}")
    return filtered


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
            localdatabase=fetch_kwargs.get("localdatabase", "exomol_data"),
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
        "star",
        "target_stellar_teff_K",
        "actual_stellar_teff_K",
        "distance_AU",
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
        if total_row is not None:
            f.write(f"total_mass_loss_rate_g_s: {total_row['mass_loss_rate_g_s']:.12e}\n")
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
            "mixing_ratio": "",
            "z_exobase_km": "",
            "r_exobase_over_Rp": "",
            "total_shell_mass_g": total_shell_mass_g,
            "escaping_shell_mass_g": total_escaping_mass_g,
            "mass_loss_rate_g_s": total_mdot_g_s,
            "mass_loss_rate_kg_s": total_mdot.to_value(u.kg / u.s),
            "mass_loss_rate_Mearth_yr": total_mdot.to_value(u.M_earth / u.yr),
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


def mass_loss_for_species_advanced(
    test_family: str,
    planet_key: str,
    star_key: str,
    distance_au: float,
    species: str,
    planet_case: dict,
    system,
    exobase_rows: Dict[Tuple[str, str], dict],
) -> dict:
    z_exobase = base_mass_loss.exobase_height(planet_key, species, exobase_rows)
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

    return {
        "test_family": test_family,
        "planet": planet_key,
        "star": star_key,
        "target_stellar_teff_K": base_mass_loss.infer_teff_from_star_template(star_key),
        "actual_stellar_teff_K": base_mass_loss.infer_teff_from_star_template(star_key),
        "distance_AU": float(distance_au),
        "species": species,
        "mixing_ratio": abundance,
        "z_exobase_km": z_exobase.to_value(u.km),
        "r_exobase_over_Rp": (r_exobase / planet_radius).decompose().value,
        "hill_radius_over_Rp": (hill_radius / planet_radius).decompose().value,
        "total_shell_mass_g": total_mass_g,
        "escaping_shell_mass_g": escaping_mass_g,
        "mass_loss_rate_g_s": mdot_g_s,
        "mass_loss_rate_kg_s": (mdot_g_s * u.g / u.s).to_value(u.kg / u.s),
        "mass_loss_rate_Mearth_yr": (mdot_g_s * u.g / u.s).to_value(u.M_earth / u.yr),
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


def main() -> None:
    configure_base_module()
    start_time = time.perf_counter()
    exobase_rows = base_mass_loss.load_exobase_table(EXOBASE_TABLE)

    systems = build_advanced_systems()
    all_rows = load_checkpoint_rows(CHECKPOINT_PATH) if USE_CHECKPOINT else []
    all_rows = import_existing_output_rows(systems, all_rows)
    completed_rows = completed_row_keys(all_rows)
    if all_rows:
        print(
            f"Loaded advanced checkpoint/output seed with {len(all_rows)} completed species rows"
        )
        for system in systems:
            write_system_outputs_from_rows(all_rows, system)

    for system_def in systems:
        planet_key = system_def.planet_key
        star_key = system_def.star_key
        distance_au = system_def.distance_au
        planet_case = base_mass_loss.get_planet_template(planet_key)
        species_list = selected_species_for_planet(planet_case)
        if not species_list:
            print(f"Skipping {system_def.test_family} / {planet_key}: no species selected")
            continue

        planet = base_mass_loss.build_planet(planet_case)
        star = base_mass_loss.get_star(star_key)
        distance = distance_au * u.AU
        planetary_system = base_mass_loss.PlanetarySystem(planet, star, distance)
        existing_rows = system_rows_from_checkpoint(all_rows, system_def)
        if existing_rows:
            print(
                f"\n--- Advanced mass-loss system: {system_def.test_family} / {planet_key} / {star_key} / {distance_au:g} AU "
                f"({len(existing_rows)}/{len(species_list)} species already saved) ---"
            )
            write_system_outputs_from_rows(all_rows, system_def)
        else:
            print(
                f"\n--- Advanced mass-loss system: {system_def.test_family} / {planet_key}, star={star_key} "
                f"({base_mass_loss.infer_teff_from_star_template(star_key)} K), distance={distance_au:g} AU ---"
            )

        for species in species_list:
            current_key = row_key_from_values(system_def.test_family, planet_key, species, star_key, distance_au)
            if current_key in completed_rows:
                print(f"Skipping completed advanced species: {system_def.test_family} / {planet_key} / {species}")
                continue

            species_start = time.perf_counter()
            try:
                row = mass_loss_for_species_advanced(
                    system_def.test_family,
                    planet_key,
                    star_key,
                    distance_au,
                    species,
                    planet_case,
                    planetary_system,
                    exobase_rows,
                )
            except Exception as exc:
                print(f"Skipping {planet_key} {species}: {type(exc).__name__}: {exc}")
                continue

            all_rows.append(row)
            completed_rows.add(current_key)
            if USE_CHECKPOINT:
                save_checkpoint_rows(all_rows, CHECKPOINT_PATH)

            csv_path, txt_path = write_system_outputs_from_rows(all_rows, system_def)
            print(
                f"{species}: Mdot={row['mass_loss_rate_g_s']:.3e} g/s, "
                f"escaping_mass={row['escaping_shell_mass_g']:.3e} g, "
                f"mean_t={row['mean_escape_time_s']:.3e} s, "
                f"cache_bins={row['photon_pressure_cache_bins']}, "
                f"elapsed={time.perf_counter() - species_start:.1f} s"
            )
            if csv_path is not None and txt_path is not None:
                print(f"Updated advanced outputs: {csv_path.name}, {txt_path.name}")

        csv_path, txt_path = write_system_outputs_from_rows(all_rows, system_def)
        if csv_path is not None and txt_path is not None:
            total_rows = system_rows_from_checkpoint(all_rows, system_def)
            total_row = total_row_from_species_rows(total_rows)
            if total_row is not None:
                print(
                    f"TOTAL_INCLUDED_SPECIES for {system_def.test_family} / {planet_key}: Mdot={total_row['mass_loss_rate_g_s']:.3e} g/s, "
                    f"escaping_mass={total_row['escaping_shell_mass_g']:.3e} g"
                )
                print(f"Saved advanced CSV to {csv_path}")
                print(f"Saved advanced summary to {txt_path}")

    if USE_CHECKPOINT and all_rows:
        save_checkpoint_rows(all_rows, CHECKPOINT_PATH)
    print(f"Total elapsed time: {time.perf_counter() - start_time:.1f} s")


if __name__ == "__main__":
    main()
