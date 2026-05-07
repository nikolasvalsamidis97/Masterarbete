

import pathlib
import sys
import gc
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import astropy.units as u

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.Molecule import Molecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_func.Templates.Atoms.atom_species import ATOM_SPECIES
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Stars.stars_templates_updated import STAR_TEMPLATES, infer_teff_from_star_template
from project_func.plotdata_to_txt import save_plotdata_txt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INCLUDE_ATOMS = False
INCLUDE_MOLECULES = True

# Leave empty to use all atoms from the template. Fill with e.g.
# ["H I", "Na I", "Fe I"] to do a small test run.
SELECTED_ATOM_SPECIES = []
SELECTED_MOLECULE_SPECIES = []

TARGET_STELLAR_TEFFS_K = [2600.0, 10000.0, 50000.0]
DISTANCE_AU = 1.0
B_KMS = 1.0
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
NPTS_ATOM = 150

# Excitation temperatures for beta(T_exc), from low to high.
T_EXC_VALUES_K = [3000, 5000, 6000, 8000, 10000, 15000, 20000, 30000, 50000]
# Use the same optically thin reference column as the tau=0 big-table study.
FIXED_NCOL_CM2 = 1e-20
CLEAR_MOLECULE_CACHES_AFTER_SPECIES = True

SAVE_TXT = True
SAVE_RAW_CSV = False

_DEFAULT_EXTERNAL_OUTPUT_DIR = pathlib.Path.home() / "DATA" / "results" / "Beta_Tgas"
_DEFAULT_OUTPUT_DIR = (
    _DEFAULT_EXTERNAL_OUTPUT_DIR
    if (pathlib.Path.home() / "DATA").exists()
    else pathlib.Path(__file__).resolve().parent
)
OUTPUT_DIR = pathlib.Path(os.environ.get("BETA_TGAS_OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR))).expanduser()
CHECKPOINT_DIR = OUTPUT_DIR / "_checkpoints"
SAVE_PROGRESS_PER_SPECIES = True
RESUME_FROM_CHECKPOINTS = True
ATOMS_TXT_NAME = "beta_vs_Texc_atoms.txt"
MOLECULES_TXT_NAME = "beta_vs_Texc_molecules.txt"
ATOMS_RAW_NAME = "beta_vs_Texc_atoms_raw.csv"
MOLECULES_RAW_NAME = "beta_vs_Texc_molecules_raw.csv"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_list(name: str, default: List[float]) -> List[float]:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _env_str_list(name: str) -> List[str]:
    value = os.environ.get(name, "").strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


INCLUDE_ATOMS = _env_flag("BETA_TGAS_INCLUDE_ATOMS", INCLUDE_ATOMS)
INCLUDE_MOLECULES = _env_flag("BETA_TGAS_INCLUDE_MOLECULES", INCLUDE_MOLECULES)
TARGET_STELLAR_TEFFS_K = _env_float_list("BETA_TGAS_TARGET_TEFFS_K", TARGET_STELLAR_TEFFS_K)
SELECTED_STAR_KEYS = _env_str_list("BETA_TGAS_STAR_KEYS")
SELECTED_ATOM_SPECIES = _env_str_list("BETA_TGAS_SELECTED_ATOM_SPECIES") or SELECTED_ATOM_SPECIES
SELECTED_MOLECULE_SPECIES = _env_str_list("BETA_TGAS_SELECTED_MOLECULE_SPECIES") or SELECTED_MOLECULE_SPECIES
SAVE_PROGRESS_PER_SPECIES = _env_flag("BETA_TGAS_SAVE_PROGRESS_PER_SPECIES", SAVE_PROGRESS_PER_SPECIES)
RESUME_FROM_CHECKPOINTS = _env_flag("BETA_TGAS_RESUME_FROM_CHECKPOINTS", RESUME_FROM_CHECKPOINTS)
CLEAR_MOLECULE_CACHES_AFTER_SPECIES = _env_flag(
    "BETA_TGAS_CLEAR_MOLECULE_CACHES_AFTER_SPECIES",
    CLEAR_MOLECULE_CACHES_AFTER_SPECIES,
)


# -----------------------------------------------------------------------------
# Species lists
# -----------------------------------------------------------------------------
MOLECULE_SPECIES = list(MOLECULE_TEMPLATES.keys())
MOLECULE_SPECIES = [species for species in MOLECULE_SPECIES if species != "O2"]

if SELECTED_ATOM_SPECIES:
    ATOM_SPECIES = [species for species in ATOM_SPECIES if species in SELECTED_ATOM_SPECIES]
if SELECTED_MOLECULE_SPECIES:
    MOLECULE_SPECIES = [species for species in MOLECULE_SPECIES if species in SELECTED_MOLECULE_SPECIES]


# -----------------------------------------------------------------------------
# Caches
# -----------------------------------------------------------------------------
star_cache: Dict[str, Star] = {}
atom_profile_cache: Dict[Tuple[str, float], BroadeningProfile] = {}
molecule_profile_cache: Dict[Tuple[str, float], BroadeningProfileMolecule] = {}
atom_pp_cache: Dict[Tuple[str, float, int], PhotonPressure] = {}
molecule_pp_cache: Dict[Tuple[str, float, int], PhotonPressure] = {}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_star(star_key: str) -> Star:
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


def select_star_keys(target_teffs_k: Iterable[float | None]) -> List[str]:
    if SELECTED_STAR_KEYS:
        invalid = [key for key in SELECTED_STAR_KEYS if key not in STAR_TEMPLATES]
        if invalid:
            available = ", ".join(sorted(STAR_TEMPLATES))
            raise KeyError(f"Unknown BETA_TGAS_STAR_KEYS entries: {invalid}. Available: {available}")
        return list(dict.fromkeys(SELECTED_STAR_KEYS))

    teff_by_key = {key: float(infer_teff_from_star_template(key)) for key in STAR_TEMPLATES}
    selected_keys: List[str] = []

    for target_teff_k in target_teffs_k:
        if target_teff_k is None:
            star_key = min(
                teff_by_key,
                key=lambda key: (teff_by_key[key], key),
            )
        else:
            star_key = min(
                teff_by_key,
                key=lambda key: (abs(teff_by_key[key] - float(target_teff_k)), key),
            )

        if star_key not in selected_keys:
            selected_keys.append(star_key)

    return selected_keys


def suffixed_output_name(base_name: str, star_key: str) -> str:
    base_path = pathlib.Path(base_name)
    return f"{base_path.stem}_{star_key}{base_path.suffix}"



def effective_b_value(b_kms: float) -> u.Quantity:
    return float(b_kms) * u.km / u.s


def clear_molecule_runtime_caches() -> None:
    molecule_pp_cache.clear()
    for profile in molecule_profile_cache.values():
        if hasattr(profile, "clear_temperature_cache"):
            profile.clear_temperature_cache(keep_current=False)
    molecule_profile_cache.clear()
    PhotonPressure.clear_molecule_flux_cache()
    gc.collect()



def get_atom_profile(species: str, b_kms: float) -> BroadeningProfile:
    cache_key = (species, float(b_kms))
    if cache_key not in atom_profile_cache:
        atom = Atom(species, WAVEMIN, WAVEMAX)
        atom_profile_cache[cache_key] = BroadeningProfile(atom, effective_b_value(b_kms), NPTS_ATOM, "Voigt")
    return atom_profile_cache[cache_key]


def get_atom_photon_pressure(species: str, b_kms: float, star: Star) -> PhotonPressure:
    cache_key = (species, float(b_kms), id(star))
    if cache_key not in atom_pp_cache:
        atom_pp_cache[cache_key] = PhotonPressure(get_atom_profile(species, b_kms), star)
    return atom_pp_cache[cache_key]



def get_molecule_profile(species: str, b_kms: float) -> BroadeningProfileMolecule:
    cache_key = (species, float(b_kms))
    if cache_key not in molecule_profile_cache:
        template = MOLECULE_TEMPLATES[species]
        fetch_kwargs = template["fetch_kwargs"]
        molecule = Molecule(species, WAVEMIN, WAVEMAX)

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

        profile = BroadeningProfileMolecule(molecule, effective_b_value(b_kms), profileType="Voigt")
        if hasattr(profile, "temp_strength_rel_cutoff"):
            profile.temp_strength_rel_cutoff = 1e-8
        molecule_profile_cache[cache_key] = profile
    return molecule_profile_cache[cache_key]


def get_molecule_photon_pressure(species: str, b_kms: float, star: Star) -> PhotonPressure:
    cache_key = (species, float(b_kms), id(star))
    if cache_key not in molecule_pp_cache:
        molecule_pp_cache[cache_key] = PhotonPressure(get_molecule_profile(species, b_kms), star)
    return molecule_pp_cache[cache_key]


def checkpoint_path(category: str, star_key: str) -> pathlib.Path:
    return CHECKPOINT_DIR / f"beta_vs_Texc_{category}_{star_key}_checkpoint.json"


def load_species_checkpoint(category: str, star_key: str, t_exc_values: np.ndarray) -> dict:
    path = checkpoint_path(category, star_key)
    if not RESUME_FROM_CHECKPOINTS or not path.exists():
        return {
            "processed_species": [],
            "kept_species": [],
            "beta_columns": [],
            "n_tau1_columns": [],
            "raw_rows": [],
            "completed": False,
        }

    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"Ignoring unreadable checkpoint {path}: {type(exc).__name__}: {exc}")
        return {
            "processed_species": [],
            "kept_species": [],
            "beta_columns": [],
            "n_tau1_columns": [],
            "raw_rows": [],
            "completed": False,
        }

    if list(map(float, data.get("t_exc_values", []))) != list(map(float, t_exc_values)):
        print(f"Ignoring checkpoint with mismatched T_exc grid: {path}")
        return {
            "processed_species": [],
            "kept_species": [],
            "beta_columns": [],
            "n_tau1_columns": [],
            "raw_rows": [],
            "completed": False,
        }

    return {
        "processed_species": list(data.get("processed_species", [])),
        "kept_species": list(data.get("kept_species", [])),
        "beta_columns": [list(col) for col in data.get("beta_columns", [])],
        "n_tau1_columns": [list(col) for col in data.get("n_tau1_columns", [])],
        "raw_rows": list(data.get("raw_rows", [])),
        "completed": bool(data.get("completed", False)),
    }


def save_species_checkpoint(
    category: str,
    star_key: str,
    t_exc_values: np.ndarray,
    processed_species: List[str],
    kept_species: List[str],
    beta_columns: List[List[float]],
    n_tau1_columns: List[List[float]],
    raw_rows: List[dict],
    completed: bool = False,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "t_exc_values": [float(value) for value in t_exc_values],
        "processed_species": processed_species,
        "kept_species": kept_species,
        "beta_columns": beta_columns,
        "n_tau1_columns": n_tau1_columns,
        "raw_rows": raw_rows,
        "completed": completed,
    }
    checkpoint_path(category, star_key).write_text(json.dumps(payload))


# -----------------------------------------------------------------------------
# N(tau = 1)
# -----------------------------------------------------------------------------
def tau_one_ncol_atom(species: str, b_kms: float, t_exc: u.Quantity, star: Star) -> u.Quantity:
    profile = get_atom_profile(species, b_kms)
    pp = get_atom_photon_pressure(species, b_kms, star)

    weights_raw = np.asarray(pp.excitation_weights(t_exc), dtype=float)
    sigma_raw = profile.sigmaArray.to(u.cm**2)

    if weights_raw.ndim == 0:
        weights_line = np.array([float(weights_raw)], dtype=float)
    elif weights_raw.ndim == 1:
        weights_line = weights_raw.astype(float)
    else:
        weights_line = np.asarray(weights_raw[:, 0], dtype=float)

    if sigma_raw.ndim == 0:
        sigma_line = np.array([float(sigma_raw.value)], dtype=float) * u.cm**2
    elif sigma_raw.ndim == 1:
        sigma_line = sigma_raw
    else:
        sigma_line = sigma_raw[:, 0]

    n_common = min(len(weights_line), len(sigma_line))
    if n_common == 0:
        return np.nan / u.cm**2

    sigma_eff = np.nanmax(sigma_line[:n_common] * weights_line[:n_common])
    sigma_eff = sigma_eff.to(u.cm**2)

    if not np.isfinite(sigma_eff.value) or sigma_eff <= 0 * u.cm**2:
        return np.nan / u.cm**2

    return (1.0 / sigma_eff).to(1 / u.cm**2)



def tau_one_ncol_molecule(species: str, b_kms: float, t_exc: u.Quantity, star: Star) -> u.Quantity:
    profile = get_molecule_profile(species, b_kms)
    profile.apply_boltzmann_weights(t_exc, verbose=False)
    sigma_total = profile.sigmaArray.to(u.cm**2)

    if sigma_total.ndim == 0:
        sigma_eff = (float(sigma_total.value)) * u.cm**2
    else:
        sigma_eff = np.nanmax(sigma_total)

    if not np.isfinite(sigma_eff.value) or sigma_eff <= 0 * u.cm**2:
        return np.nan / u.cm**2

    return (1.0 / sigma_eff).to(1 / u.cm**2)


# -----------------------------------------------------------------------------
# Beta(T_exc)
# -----------------------------------------------------------------------------
def beta_for_atom(species: str, b_kms: float, t_exc: u.Quantity, star: Star) -> Tuple[float, float, float]:
    n_tau1 = tau_one_ncol_atom(species, b_kms, t_exc, star)
    if not np.isfinite(n_tau1.value) or n_tau1 <= 0 / u.cm**2:
        return np.nan, np.nan, np.nan

    return beta_for_atom_with_ncol(species, b_kms, t_exc, star, n_tau1)


def beta_for_atom_with_ncol(
    species: str,
    b_kms: float,
    t_exc: u.Quantity,
    star: Star,
    n_col: u.Quantity,
) -> Tuple[float, float, float]:
    pp = get_atom_photon_pressure(species, b_kms, star)
    distance = DISTANCE_AU * u.AU
    n_col_array = np.array([n_col.to_value(1 / u.cm**2)], dtype=float) / u.cm**2
    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(n_col_array, t_exc, distance)
    beta, beta_err = pp.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, distance.to(u.cm))

    beta_value = float(np.ravel(beta.value)[0])
    beta_err_value = float(np.ravel(beta_err.value)[0])
    return float(n_col.to_value(1 / u.cm**2)), beta_value, beta_err_value


def beta_for_molecule(species: str, b_kms: float, t_exc: u.Quantity, star: Star) -> Tuple[float, float, float]:
    n_tau1 = tau_one_ncol_molecule(species, b_kms, t_exc, star)
    if not np.isfinite(n_tau1.value) or n_tau1 <= 0 / u.cm**2:
        return np.nan, np.nan, np.nan

    return beta_for_molecule_with_ncol(species, b_kms, t_exc, star, n_tau1)


def beta_for_molecule_with_ncol(
    species: str,
    b_kms: float,
    t_exc: u.Quantity,
    star: Star,
    n_col: u.Quantity,
) -> Tuple[float, float, float]:
    pp = get_molecule_photon_pressure(species, b_kms, star)
    distance = DISTANCE_AU * u.AU
    n_col_array = np.array([n_col.to_value(1 / u.cm**2)], dtype=float) / u.cm**2
    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(n_col_array, t_exc, distance)
    beta, beta_err = pp.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, distance.to(u.cm))

    beta_value = float(np.ravel(beta.value)[0])
    beta_err_value = float(np.ravel(beta_err.value)[0])
    return float(n_col.to_value(1 / u.cm**2)), beta_value, beta_err_value


# -----------------------------------------------------------------------------
# Data builders
# -----------------------------------------------------------------------------
def build_category_matrices(
    species_list: Iterable[str],
    category: str,
    star_key: str,
    star: Star,
    use_fixed_ncol: bool = False,
    fixed_ncol_cm2: float = FIXED_NCOL_CM2,
    progress_callback=None,
):
    t_exc_values = np.array(T_EXC_VALUES_K, dtype=float)
    checkpoint = load_species_checkpoint(category, star_key, t_exc_values)
    processed_species = list(checkpoint["processed_species"])
    beta_columns = list(checkpoint["beta_columns"])
    n_tau1_columns = list(checkpoint["n_tau1_columns"])
    kept_species = list(checkpoint["kept_species"])
    raw_rows = list(checkpoint["raw_rows"])

    if checkpoint["completed"]:
        print(f"Resuming from completed checkpoint for {category} / {star_key}; no recomputation needed.")
        if beta_columns:
            beta_matrix = np.array(beta_columns, dtype=float).T
            n_tau1_matrix = np.array(n_tau1_columns, dtype=float).T
        else:
            beta_matrix = np.empty((len(t_exc_values), 0), dtype=float)
            n_tau1_matrix = np.empty((len(t_exc_values), 0), dtype=float)
        return t_exc_values, beta_matrix, n_tau1_matrix, kept_species, raw_rows

    for species in species_list:
        if species in processed_species:
            print(f"Skipping already checkpointed {category}: {species}")
            continue

        print(f"Calculating {category}: {species}")
        beta_series = []
        n_tau1_series = []
        any_valid = False
        fixed_ncol = None

        if use_fixed_ncol:
            fixed_ncol = float(fixed_ncol_cm2) / u.cm**2

        for t_exc_k in t_exc_values:
            t_exc = t_exc_k * u.K
            try:
                if category == "atom":
                    if use_fixed_ncol:
                        if not np.isfinite(fixed_ncol.value) or fixed_ncol <= 0 / u.cm**2:
                            n_tau1_cm2, beta_value, beta_err_value = np.nan, np.nan, np.nan
                        else:
                            n_tau1_cm2, beta_value, beta_err_value = beta_for_atom_with_ncol(species, B_KMS, t_exc, star, fixed_ncol)
                    else:
                        n_tau1_cm2, beta_value, beta_err_value = beta_for_atom(species, B_KMS, t_exc, star)
                else:
                    if use_fixed_ncol:
                        if not np.isfinite(fixed_ncol.value) or fixed_ncol <= 0 / u.cm**2:
                            n_tau1_cm2, beta_value, beta_err_value = np.nan, np.nan, np.nan
                        else:
                            n_tau1_cm2, beta_value, beta_err_value = beta_for_molecule_with_ncol(species, B_KMS, t_exc, star, fixed_ncol)
                    else:
                        n_tau1_cm2, beta_value, beta_err_value = beta_for_molecule(species, B_KMS, t_exc, star)
            except Exception as exc:
                print(f"Skipping {category} species={species}, T_exc={t_exc_k:.0f} K, fixedN={use_fixed_ncol}: {type(exc).__name__}: {exc}")
                n_tau1_cm2, beta_value, beta_err_value = np.nan, np.nan, np.nan

            if np.isfinite(beta_value):
                any_valid = True

            beta_series.append(beta_value)
            n_tau1_series.append(n_tau1_cm2)
            raw_rows.append(
                {
                    "species": species,
                    "T_exc_K": float(t_exc_k),
                    "n_tau1_cm2": n_tau1_cm2,
                    "beta": beta_value,
                    "beta_err": beta_err_value,
                    "category": category,
                }
            )

        if any_valid:
            kept_species.append(species)
            beta_columns.append(beta_series)
            n_tau1_columns.append(n_tau1_series)

        processed_species.append(species)

        if SAVE_PROGRESS_PER_SPECIES:
            save_species_checkpoint(
                category,
                star_key,
                t_exc_values,
                processed_species,
                kept_species,
                beta_columns,
                n_tau1_columns,
                raw_rows,
                completed=False,
            )
            if progress_callback is not None and kept_species:
                progress_callback(
                    t_exc_values,
                    np.array(beta_columns, dtype=float).T,
                    kept_species,
                )

        if category == "molecule" and CLEAR_MOLECULE_CACHES_AFTER_SPECIES:
            clear_molecule_runtime_caches()

    if beta_columns:
        beta_matrix = np.array(beta_columns, dtype=float).T
        n_tau1_matrix = np.array(n_tau1_columns, dtype=float).T
    else:
        beta_matrix = np.empty((len(t_exc_values), 0), dtype=float)
        n_tau1_matrix = np.empty((len(t_exc_values), 0), dtype=float)

    if SAVE_PROGRESS_PER_SPECIES:
        save_species_checkpoint(
            category,
            star_key,
            t_exc_values,
            processed_species,
            kept_species,
            beta_columns,
            n_tau1_columns,
            raw_rows,
            completed=True,
        )

    return t_exc_values, beta_matrix, n_tau1_matrix, kept_species, raw_rows



def save_category_txt(
    txt_name: str,
    dataset_name: str,
    t_exc_values: np.ndarray,
    beta_matrix: np.ndarray,
    kept_species: List[str],
    star_key: str,
    star: Star,
    category: str,
    use_fixed_ncol: bool = False,
    fixed_ncol_cm2: float = FIXED_NCOL_CM2,
) -> None:
    output_path = OUTPUT_DIR / txt_name
    save_plotdata_txt(
        output_path,
        dataset_name=dataset_name,
        x_label="Excitation temperature",
        x_unit="K",
        y_label="beta",
        y_unit="dimensionless",
        x_values=t_exc_values,
        y_matrix=beta_matrix,
        series_values=kept_species,
        series_label="species",
        series_unit="label",
        column_names=[species.replace(" ", "_") for species in kept_species],
        extra_metadata={
            "star_key": star_key,
            "stellar_teff_K": infer_teff_from_star_template(star_key),
            "distance_AU": DISTANCE_AU,
            "b_km_s": B_KMS,
            "use_fixed_ncol": use_fixed_ncol,
            "fixed_ncol_mode": "constant_fixed_ncol" if use_fixed_ncol else "species_tau1_per_temperature",
            "fixed_ncol_reference_t_K": "",
            "fixed_ncol_cm2": fixed_ncol_cm2 if use_fixed_ncol else "",
            "category": category,
            "vsini_km_s": float(star.vsini.to_value(u.km / u.s)),
            "radius_rsun": float(star.radius.to_value(u.R_sun)),
            "mass_msun": float(star.mass.to_value(u.M_sun)),
            "n_series": len(kept_species),
            "note": (
                "beta calculated against stellar gravity using gas excitation temperature T_exc and species-wise N_tau=1"
                if not use_fixed_ncol
                else f"beta calculated against stellar gravity using gas excitation temperature T_exc and a fixed optically thin column density N_col={fixed_ncol_cm2:.1e} cm^-2"
            ),
        },
    )
    print(f"Saved txt data to {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    star_keys = select_star_keys(TARGET_STELLAR_TEFFS_K)
    print(f"Selected star keys: {', '.join(star_keys)}")
    print(f"Output directory: {OUTPUT_DIR}")

    if INCLUDE_ATOMS:
        for star_key in star_keys:
            star = make_star(star_key)
            t_exc_values, beta_matrix, _, kept_species, raw_rows = build_category_matrices(
                ATOM_SPECIES,
                "atom",
                star_key,
                star,
                use_fixed_ncol=True,
                fixed_ncol_cm2=FIXED_NCOL_CM2,
                progress_callback=lambda partial_t_exc, partial_beta, partial_species, star_key=star_key, star=star: save_category_txt(
                    suffixed_output_name(ATOMS_TXT_NAME, star_key),
                    dataset_name=f"beta_vs_Texc_atoms_{star_key}",
                    t_exc_values=partial_t_exc,
                    beta_matrix=partial_beta,
                    kept_species=partial_species,
                    star_key=star_key,
                    star=star,
                    category="atom",
                    use_fixed_ncol=True,
                    fixed_ncol_cm2=FIXED_NCOL_CM2,
                ),
            )
            if kept_species and SAVE_TXT:
                save_category_txt(
                    suffixed_output_name(ATOMS_TXT_NAME, star_key),
                    dataset_name=f"beta_vs_Texc_atoms_{star_key}",
                    t_exc_values=t_exc_values,
                    beta_matrix=beta_matrix,
                    kept_species=kept_species,
                    star_key=star_key,
                    star=star,
                    category="atom",
                    use_fixed_ncol=True,
                    fixed_ncol_cm2=FIXED_NCOL_CM2,
                )
            else:
                print(f"No valid atom data produced for {star_key}.")

            if SAVE_RAW_CSV and raw_rows:
                import pandas as pd
                pd.DataFrame(raw_rows).to_csv(OUTPUT_DIR / suffixed_output_name(ATOMS_RAW_NAME, star_key), index=False)

    if INCLUDE_MOLECULES:
        for star_key in star_keys:
            star = make_star(star_key)
            t_exc_values, beta_matrix, _, kept_species, raw_rows = build_category_matrices(
                MOLECULE_SPECIES,
                "molecule",
                star_key,
                star,
                use_fixed_ncol=True,
                fixed_ncol_cm2=FIXED_NCOL_CM2,
                progress_callback=lambda partial_t_exc, partial_beta, partial_species, star_key=star_key, star=star: save_category_txt(
                    suffixed_output_name(MOLECULES_TXT_NAME, star_key),
                    dataset_name=f"beta_vs_Texc_molecules_{star_key}",
                    t_exc_values=partial_t_exc,
                    beta_matrix=partial_beta,
                    kept_species=partial_species,
                    star_key=star_key,
                    star=star,
                    category="molecule",
                    use_fixed_ncol=True,
                    fixed_ncol_cm2=FIXED_NCOL_CM2,
                ),
            )
            if kept_species and SAVE_TXT:
                save_category_txt(
                    suffixed_output_name(MOLECULES_TXT_NAME, star_key),
                    dataset_name=f"beta_vs_Texc_molecules_{star_key}",
                    t_exc_values=t_exc_values,
                    beta_matrix=beta_matrix,
                    kept_species=kept_species,
                    star_key=star_key,
                    star=star,
                    category="molecule",
                    use_fixed_ncol=True,
                    fixed_ncol_cm2=FIXED_NCOL_CM2,
                )
            else:
                print(f"No valid molecule data produced for {star_key}.")

            if SAVE_RAW_CSV and raw_rows:
                import pandas as pd
                pd.DataFrame(raw_rows).to_csv(OUTPUT_DIR / suffixed_output_name(MOLECULES_RAW_NAME, star_key), index=False)

            clear_molecule_runtime_caches()

    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
