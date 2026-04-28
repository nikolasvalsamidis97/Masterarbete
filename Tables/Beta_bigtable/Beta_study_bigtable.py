import pathlib
import sys
import gc
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
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
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INCLUDE_ATOMS = False
INCLUDE_MOLECULES = True

# Leave empty to use all atoms from the template. Fill with e.g.
# ["H I", "Na I", "Fe I"] to do a small test run.
SELECTED_ATOM_SPECIES = []

DISTANCE_AU = 1.0
GAS_TEMPERATURE = 0.001 * u.K
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
NPTS_ATOM = 150
B_VALUES_KMS = [0.0, 1.0, 10.0, 100.0]
B_ZERO_EFFECTIVE = 1.0e-20 * u.km / u.s
# Use a finite grid-construction velocity only to avoid the b=0 molecule case
# collapsing to the hard dlam floor and creating a multi-billion-point grid.
# This does not change the physical b used in the line profile itself.
MOLECULE_ZERO_B_GRID_KMS = 0.1 * u.km / u.s
CLEAR_MOLECULE_CACHES_AFTER_SPECIES = True

N_SELECTED_STARS = 8
PRINT_TAU_CHECK_SUMMARY = False

# Star selection mode:
#   "even_index"  -> evenly spaced through the template list
#   "target_teff" -> choose the closest available template star to each target Teff
STAR_SELECTION_MODE = "even_index"
TARGET_TEFFS_K = []
# Example:
STAR_SELECTION_MODE = "target_teff"
TARGET_TEFFS_K = [3000, 5000, 6000, 8000, 10000, 15000, 20000, 30000, 50000]
EXCLUDED_TEFFS_K = {19000}

SAVE_RAW_TXT = True

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
RAW_ATOMS_NAME = "beta_bigtable_atoms.txt"
RAW_MOLECULES_NAME = "beta_bigtable_molecules.txt"


def save_rows_atomic(rows: List[dict], output_path: pathlib.Path) -> None:
    if not SAVE_RAW_TXT:
        return
    if not rows:
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False, sep="\t")
    tmp_path.replace(output_path)

def active_run_suffix() -> str:
    parts = []
    if INCLUDE_ATOMS:
        parts.append("atoms")
    if INCLUDE_MOLECULES:
        parts.append("molecules")
    return "_".join(parts) if parts else "none"


RUN_SUFFIX = active_run_suffix()
SELECTED_STARS_NAME = f"beta_bigtable_selected_stars_{RUN_SUFFIX}.csv"
SELECTED_STARS_TEX_NAME = f"beta_bigtable_selected_stars_{RUN_SUFFIX}.tex"


# -----------------------------------------------------------------------------
# Species lists
# -----------------------------------------------------------------------------

MOLECULE_SPECIES = list(MOLECULE_TEMPLATES.keys())
MOLECULE_SPECIES = [species for species in MOLECULE_SPECIES if species != "O2"]

if SELECTED_ATOM_SPECIES:
    ATOM_SPECIES = [species for species in ATOM_SPECIES if species in SELECTED_ATOM_SPECIES]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------



star_cache: Dict[str, Star] = {}
atom_profile_cache: Dict[Tuple[str, float], BroadeningProfile] = {}
molecule_profile_cache: Dict[Tuple[str, float], BroadeningProfileMolecule] = {}


# -----------------------------------------------------------------------------
# Star selection
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




def select_evenly_spaced_star_keys(n_select: int = N_SELECTED_STARS) -> List[str]:
    all_keys = list(STAR_TEMPLATES.keys())
    if n_select >= len(all_keys):
        return all_keys

    index_values = np.linspace(0, len(all_keys) - 1, n_select)
    indices = np.round(index_values).astype(int)

    selected = []
    used = set()
    for idx in indices:
        if idx not in used:
            selected.append(all_keys[idx])
            used.add(idx)

    if len(selected) < n_select:
        for idx, key in enumerate(all_keys):
            if idx not in used:
                selected.append(key)
                used.add(idx)
            if len(selected) == n_select:
                break

    return selected


def select_star_keys_by_target_teff(target_teffs_k: List[float]) -> List[str]:
    if not target_teffs_k:
        return select_evenly_spaced_star_keys(n_select=N_SELECTED_STARS)

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



def build_selected_stars(n_select: int = N_SELECTED_STARS) -> List[dict]:
    if STAR_SELECTION_MODE == "target_teff":
        star_keys = select_star_keys_by_target_teff(TARGET_TEFFS_K)
    else:
        star_keys = select_evenly_spaced_star_keys(n_select=n_select)
    selected = []
    for key in star_keys:
        teff_k = int(infer_teff_from_star_template(key))
        if teff_k in EXCLUDED_TEFFS_K:
            continue
        selected.append({"key": key, "teff_k": teff_k, "star": make_star(key)})
    return selected


# -----------------------------------------------------------------------------
# Profile builders
# -----------------------------------------------------------------------------
def effective_b_value(b_kms: float) -> u.Quantity:
    if float(b_kms) == 0.0:
        return B_ZERO_EFFECTIVE
    return float(b_kms) * u.km / u.s



def get_atom_profile(species: str, b_kms: float) -> BroadeningProfile:
    cache_key = (species, float(b_kms))
    if cache_key not in atom_profile_cache:
        atom = Atom(species, WAVEMIN, WAVEMAX)
        atom_profile_cache[cache_key] = BroadeningProfile(atom, effective_b_value(b_kms), NPTS_ATOM, "Voigt")
    return atom_profile_cache[cache_key]



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
                localdatabase=fetch_kwargs.get("localdatabase", "exomol_data"),
            )

        profile_kwargs = {
            "molecule": molecule,
            "b": effective_b_value(b_kms),
            "profileType": "Voigt",
        }
        if float(b_kms) == 0.0:
            rep = 0.5 * (WAVEMIN + WAVEMAX)
            doppler_sigma = (rep * (MOLECULE_ZERO_B_GRID_KMS / (299792.458 * u.km / u.s))).to(u.AA)
            profile_kwargs["dlam"] = np.maximum((doppler_sigma / 3.0).to(u.AA), 1e-5 * u.AA)

        profile = BroadeningProfileMolecule(**profile_kwargs)
        if hasattr(profile, "temp_strength_rel_cutoff"):
            profile.temp_strength_rel_cutoff = 1e-8
        molecule_profile_cache[cache_key] = profile
    return molecule_profile_cache[cache_key]


def molecule_effective_sigma(profile: BroadeningProfileMolecule, temp: u.Quantity) -> u.Quantity:
    profile.apply_boltzmann_weights(temp, verbose=False)
    sigma_total = profile.sigmaArray.to(u.cm**2)
    if sigma_total.ndim == 0:
        return float(sigma_total.value) * u.cm**2
    return np.nanmax(sigma_total)


# -----------------------------------------------------------------------------
# Tau = 1 column densities
# -----------------------------------------------------------------------------
def tau_one_ncol_atom(species: str, b_kms: float, temp: u.Quantity, star: Star) -> u.Quantity:
    profile = get_atom_profile(species, b_kms)
    pp = PhotonPressure(profile, star)

    weights_raw = np.asarray(pp.excitation_weights(temp), dtype=float)
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


# -----------------------------------------------------------------------------
# Tau checkers
# -----------------------------------------------------------------------------

def tau_check_atom(species: str, b_kms: float, temp: u.Quantity, star: Star, ncol: u.Quantity) -> float:
    """Control check: evaluate tau for the chosen Ncol using the same effective sigma."""
    profile = get_atom_profile(species, b_kms)
    pp = PhotonPressure(profile, star)

    weights_raw = np.asarray(pp.excitation_weights(temp), dtype=float)
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
        return np.nan

    sigma_eff = np.nanmax(sigma_line[:n_common] * weights_line[:n_common]).to(u.cm**2)
    if not np.isfinite(sigma_eff.value) or sigma_eff <= 0 * u.cm**2:
        return np.nan

    tau_value = (ncol * sigma_eff).decompose().value
    return float(tau_value)



def tau_one_ncol_molecule(species: str, b_kms: float, temp: u.Quantity, star: Star) -> u.Quantity:
    profile = get_molecule_profile(species, b_kms)
    sigma_eff = molecule_effective_sigma(profile, temp)

    if not np.isfinite(sigma_eff.value) or sigma_eff <= 0 * u.cm**2:
        return np.nan / u.cm**2

    return (1.0 / sigma_eff).to(1 / u.cm**2)


def tau_check_molecule(species: str, b_kms: float, temp: u.Quantity, star: Star, ncol: u.Quantity) -> float:
    """Control check: evaluate tau for the chosen Ncol using the same effective sigma."""
    profile = get_molecule_profile(species, b_kms)
    sigma_eff = molecule_effective_sigma(profile, temp)

    if not np.isfinite(sigma_eff.value) or sigma_eff <= 0 * u.cm**2:
        return np.nan

    tau_value = (ncol * sigma_eff).decompose().value
    return float(tau_value)


# -----------------------------------------------------------------------------
# Beta calculations
# -----------------------------------------------------------------------------
def beta_from_ncol(pp: PhotonPressure, star: Star, ncol: u.Quantity) -> Tuple[float, float]:
    distance = DISTANCE_AU * u.AU
    ncol_array = np.array([ncol.to_value(1 / u.cm**2)], dtype=float) / u.cm**2
    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(ncol_array, GAS_TEMPERATURE, distance)
    beta, beta_err = pp.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, distance.to(u.cm))
    beta_value = float(np.ravel(beta.value)[0])
    beta_err_value = float(np.ravel(beta_err.value)[0])
    return beta_value, beta_err_value


def beta_for_atom(species: str, b_kms: float, star_info: dict) -> Tuple[float, float, float, float, float, float, float]:
    temp = GAS_TEMPERATURE
    profile = get_atom_profile(species, b_kms)
    pp = PhotonPressure(profile, star_info["star"])

    n_tau0 = 0.0 / u.cm**2
    beta_tau0, beta_err_tau0 = beta_from_ncol(pp, star_info["star"], n_tau0)

    n_tau1 = tau_one_ncol_atom(species, b_kms, temp, star_info["star"])
    if not np.isfinite(n_tau1.value) or n_tau1 <= 0 / u.cm**2:
        return float(n_tau0.to_value(1 / u.cm**2)), beta_tau0, beta_err_tau0, np.nan, np.nan, np.nan, np.nan

    beta_tau1, beta_err_tau1 = beta_from_ncol(pp, star_info["star"], n_tau1)
    tau_check = tau_check_atom(species, b_kms, temp, star_info["star"], n_tau1)
    return (
        float(n_tau0.to_value(1 / u.cm**2)),
        beta_tau0,
        beta_err_tau0,
        float(n_tau1.to_value(1 / u.cm**2)),
        beta_tau1,
        beta_err_tau1,
        tau_check,
    )



def beta_for_molecule(species: str, b_kms: float, star_info: dict) -> Tuple[float, float, float, float, float, float, float]:
    temp = GAS_TEMPERATURE
    profile = get_molecule_profile(species, b_kms)
    pp = PhotonPressure(profile, star_info["star"])
    n_tau0 = 0.0 / u.cm**2
    beta_tau0, beta_err_tau0 = beta_from_ncol(pp, star_info["star"], n_tau0)

    n_tau1 = tau_one_ncol_molecule(species, b_kms, temp, star_info["star"])
    if not np.isfinite(n_tau1.value) or n_tau1 <= 0 / u.cm**2:
        return float(n_tau0.to_value(1 / u.cm**2)), beta_tau0, beta_err_tau0, np.nan, np.nan, np.nan, np.nan

    beta_tau1, beta_err_tau1 = beta_from_ncol(pp, star_info["star"], n_tau1)
    tau_check = tau_check_molecule(species, b_kms, temp, star_info["star"], n_tau1)
    return (
        float(n_tau0.to_value(1 / u.cm**2)),
        beta_tau0,
        beta_err_tau0,
        float(n_tau1.to_value(1 / u.cm**2)),
        beta_tau1,
        beta_err_tau1,
        tau_check,
    )


def broadening_label(b_kms: float) -> str:
    if float(b_kms) == 0.0:
        return "0"
    if float(b_kms).is_integer():
        return str(int(float(b_kms)))
    return f"{float(b_kms):g}"


def calculate_category_rows(
    species_list: Iterable[str],
    selected_stars: List[dict],
    category: str,
    progress_output_path: pathlib.Path | None = None,
) -> List[dict]:
    rows: List[dict] = []

    for species in species_list:
        print(f"Calculating {category}: {species}")
        for b_kms in B_VALUES_KMS:
            b_label = broadening_label(b_kms)
            for star_info in selected_stars:
                try:
                    if category == "atom":
                        (
                            n_tau0_cm2,
                            beta_tau0,
                            beta_err_tau0,
                            n_tau1_cm2,
                            beta_tau1,
                            beta_err_tau1,
                            tau_check,
                        ) = beta_for_atom(species, b_kms, star_info)
                    else:
                        (
                            n_tau0_cm2,
                            beta_tau0,
                            beta_err_tau0,
                            n_tau1_cm2,
                            beta_tau1,
                            beta_err_tau1,
                            tau_check,
                        ) = beta_for_molecule(species, b_kms, star_info)
                except Exception as exc:
                    print(f"Skipping {category} species={species}, b={b_label}, star={star_info['key']}: {type(exc).__name__}: {exc}")
                    n_tau0_cm2 = 0.0
                    beta_tau0, beta_err_tau0 = np.nan, np.nan
                    n_tau1_cm2, beta_tau1, beta_err_tau1, tau_check = np.nan, np.nan, np.nan, np.nan

                rows.append(
                    {
                        "species": species,
                        "b_label": b_label,
                        "b_effective_kms": float(effective_b_value(b_kms).to_value(u.km / u.s)),
                        "star_key": star_info["key"],
                        "teff_k": star_info["teff_k"],
                        "n_tau0_cm2": n_tau0_cm2,
                        "beta_tau0": beta_tau0,
                        "beta_err_tau0": beta_err_tau0,
                        "n_tau1_cm2": n_tau1_cm2,
                        "tau_check": tau_check,
                        "beta_tau1": beta_tau1,
                        "beta_err_tau1": beta_err_tau1,
                    }
                )
                if progress_output_path is not None:
                    save_rows_atomic(rows, progress_output_path)

        if category == "molecule" and CLEAR_MOLECULE_CACHES_AFTER_SPECIES:
            for profile in molecule_profile_cache.values():
                if hasattr(profile, "clear_temperature_cache"):
                    profile.clear_temperature_cache(keep_current=False)
            molecule_profile_cache.clear()
            PhotonPressure.clear_molecule_flux_cache()
            gc.collect()

    return rows




def rows_to_dataframe(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def build_selected_stars_table(selected_stars: List[dict]) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Teff (K) & Radius ($R_\odot$) & Mass ($M_\odot$) & vsini (km/s) & epsilon \\",
        r"\midrule",
    ]

    for star_info in selected_stars:
        star = star_info["star"]
        teff_k = int(star_info["teff_k"])
        radius_rsun = star.radius.to_value(u.R_sun)
        mass_msun = star.mass.to_value(u.M_sun)
        vsini_kms = star.vsini.to_value(u.km / u.s)
        epsilon_value = star.epsilon.to_value(u.dimensionless_unscaled)

        lines.append(
            f"{teff_k} & {radius_rsun:.2f} & {mass_msun:.1f} & {vsini_kms:.0f} & {epsilon_value:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_stars = build_selected_stars(n_select=N_SELECTED_STARS)
    selected_stars_df = pd.DataFrame(
        [
            {
                "star_key": s["key"],
                "teff_k": s["teff_k"],
                "radius_rsun": s["star"].radius.to_value(u.R_sun),
                "mass_msun": s["star"].mass.to_value(u.M_sun),
                "vsini_kms": s["star"].vsini.to_value(u.km / u.s),
                "epsilon": s["star"].epsilon.to_value(u.dimensionless_unscaled),
            }
            for s in selected_stars
        ]
    )
    selected_stars_df.to_csv(OUTPUT_DIR / SELECTED_STARS_NAME, index=False)

    selected_stars_tex = build_selected_stars_table(selected_stars)
    (OUTPUT_DIR / SELECTED_STARS_TEX_NAME).write_text(selected_stars_tex, encoding="utf-8")

    if INCLUDE_ATOMS:
        atom_output_path = OUTPUT_DIR / RAW_ATOMS_NAME
        atom_rows = calculate_category_rows(
            ATOM_SPECIES,
            selected_stars,
            category="atom",
            progress_output_path=atom_output_path,
        )
        atom_df = rows_to_dataframe(atom_rows)
        if PRINT_TAU_CHECK_SUMMARY and (not atom_df.empty) and ("tau_check" in atom_df.columns):
            tau_summary = atom_df["tau_check"].dropna()
            if not tau_summary.empty:
                print(
                    "Atom tau-check summary:",
                    f"min={tau_summary.min():.3e}, median={tau_summary.median():.3e}, max={tau_summary.max():.3e}"
                )
        if atom_df.empty:
            print("No atom rows were produced. Skipping atom txt output.")
        else:
            if SAVE_RAW_TXT:
                save_rows_atomic(atom_rows, atom_output_path)

    if INCLUDE_MOLECULES:
        molecule_output_path = OUTPUT_DIR / RAW_MOLECULES_NAME
        molecule_rows = calculate_category_rows(
            MOLECULE_SPECIES,
            selected_stars,
            category="molecule",
            progress_output_path=molecule_output_path,
        )
        molecule_df = rows_to_dataframe(molecule_rows)
        if PRINT_TAU_CHECK_SUMMARY and (not molecule_df.empty) and ("tau_check" in molecule_df.columns):
            tau_summary = molecule_df["tau_check"].dropna()
            if not tau_summary.empty:
                print(
                    "Molecule tau-check summary:",
                    f"min={tau_summary.min():.3e}, median={tau_summary.median():.3e}, max={tau_summary.max():.3e}"
                )
        if molecule_df.empty:
            print("No molecule rows were produced. Skipping molecule txt output.")
        else:
            if SAVE_RAW_TXT:
                save_rows_atomic(molecule_rows, molecule_output_path)

    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
