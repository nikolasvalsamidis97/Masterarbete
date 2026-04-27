import math
import pathlib
import sys
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

SAVE_RAW_CSV = True
SAVE_LATEX = True

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
RAW_ATOMS_NAME = "beta_bigtable_atoms.csv"
RAW_MOLECULES_NAME = "beta_bigtable_molecules.csv"
TEX_ATOMS_NAME = "beta_bigtable_atoms.tex"
TEX_MOLECULES_NAME = "beta_bigtable_molecules.tex"

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
        selected.append({"key": key, "teff_k": int(infer_teff_from_star_template(key)), "star": make_star(key)})
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

        profile = BroadeningProfileMolecule(molecule, effective_b_value(b_kms), profileType="Voigt")
        if hasattr(profile, "temp_strength_rel_cutoff"):
            profile.temp_strength_rel_cutoff = 1e-8
        molecule_profile_cache[cache_key] = profile
    return molecule_profile_cache[cache_key]


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
    profile.apply_boltzmann_weights(temp, verbose=False)
    sigma_total = profile.sigmaArray.to(u.cm**2)

    if sigma_total.ndim == 0:
        sigma_eff = (float(sigma_total.value)) * u.cm**2
    else:
        sigma_eff = np.nanmax(sigma_total)

    if not np.isfinite(sigma_eff.value) or sigma_eff <= 0 * u.cm**2:
        return np.nan / u.cm**2

    return (1.0 / sigma_eff).to(1 / u.cm**2)


def tau_check_molecule(species: str, b_kms: float, temp: u.Quantity, star: Star, ncol: u.Quantity) -> float:
    """Control check: evaluate tau for the chosen Ncol using the same effective sigma."""
    profile = get_molecule_profile(species, b_kms)
    profile.apply_boltzmann_weights(temp, verbose=False)
    sigma_total = profile.sigmaArray.to(u.cm**2)

    if sigma_total.ndim == 0:
        sigma_eff = (float(sigma_total.value)) * u.cm**2
    else:
        sigma_eff = np.nanmax(sigma_total)

    if not np.isfinite(sigma_eff.value) or sigma_eff <= 0 * u.cm**2:
        return np.nan

    tau_value = (ncol * sigma_eff).decompose().value
    return float(tau_value)


# -----------------------------------------------------------------------------
# Beta calculations
# -----------------------------------------------------------------------------
def beta_for_atom(species: str, b_kms: float, star_info: dict) -> Tuple[float, float, float, float]:
    temp = GAS_TEMPERATURE
    n_tau1 = tau_one_ncol_atom(species, b_kms, temp, star_info["star"])
    if not np.isfinite(n_tau1.value) or n_tau1 <= 0 / u.cm**2:
        return np.nan, np.nan, np.nan, np.nan

    profile = get_atom_profile(species, b_kms)
    pp = PhotonPressure(profile, star_info["star"])
    distance = DISTANCE_AU * u.AU
    n_tau1_array = np.array([n_tau1.to_value(1 / u.cm**2)], dtype=float) / u.cm**2
    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(n_tau1_array, temp, distance)
    beta, beta_err = pp.beta_Values(F_ph_tot, F_ph_tot_err, star_info["star"].mass, distance.to(u.cm))

    beta_value = float(np.ravel(beta.value)[0])
    beta_err_value = float(np.ravel(beta_err.value)[0])
    tau_check = tau_check_atom(species, b_kms, temp, star_info["star"], n_tau1)
    return float(n_tau1.to_value(1 / u.cm**2)), beta_value, beta_err_value, tau_check



def beta_for_molecule(species: str, b_kms: float, star_info: dict) -> Tuple[float, float, float, float]:
    temp = GAS_TEMPERATURE
    n_tau1 = tau_one_ncol_molecule(species, b_kms, temp, star_info["star"])
    if not np.isfinite(n_tau1.value) or n_tau1 <= 0 / u.cm**2:
        return np.nan, np.nan, np.nan, np.nan

    profile = get_molecule_profile(species, b_kms)
    pp = PhotonPressure(profile, star_info["star"])
    distance = DISTANCE_AU * u.AU
    n_tau1_array = np.array([n_tau1.to_value(1 / u.cm**2)], dtype=float) / u.cm**2
    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(n_tau1_array, temp, distance)
    beta, beta_err = pp.beta_Values(F_ph_tot, F_ph_tot_err, star_info["star"].mass, distance.to(u.cm))

    beta_value = float(np.ravel(beta.value)[0])
    beta_err_value = float(np.ravel(beta_err.value)[0])
    tau_check = tau_check_molecule(species, b_kms, temp, star_info["star"], n_tau1)
    return float(n_tau1.to_value(1 / u.cm**2)), beta_value, beta_err_value, tau_check


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
def broadening_label(b_kms: float) -> str:
    if float(b_kms) == 0.0:
        return "0"
    if float(b_kms).is_integer():
        return str(int(float(b_kms)))
    return f"{float(b_kms):g}"



def latex_species_name(species: str) -> str:
    return species.replace(" ", "~")



def format_pm(value: float, err: float) -> str:
    if not np.isfinite(value):
        return "-"
    if not np.isfinite(err):
        err = 0.0

    abs_value = abs(value)
    abs_err = abs(err)

    if abs_value == 0.0:
        return "$0$"

    if 1e-2 <= abs_value < 1e4:
        if abs_value < 10:
            value_fmt = f"{value:.2f}"
            err_fmt = f"{abs_err:.2f}"
        elif abs_value < 100:
            value_fmt = f"{value:.1f}"
            err_fmt = f"{abs_err:.1f}"
        else:
            value_fmt = f"{value:.0f}"
            err_fmt = f"{abs_err:.0f}"
        return f"${value_fmt} \\pm {err_fmt}$"

    exponent = int(math.floor(math.log10(abs_value)))
    scaled_value = value / (10 ** exponent)
    scaled_err = abs_err / (10 ** exponent)
    return f"$({scaled_value:.1f} \\pm {scaled_err:.1f})10^{{{exponent}}}$"


def format_value_only(value: float) -> str:
    if not np.isfinite(value):
        return "-"

    abs_value = abs(value)
    if abs_value == 0.0:
        return "$0$"

    if 1e-2 <= abs_value < 1e4:
        if abs_value < 10:
            value_fmt = f"{value:.2f}"
        elif abs_value < 100:
            value_fmt = f"{value:.1f}"
        else:
            value_fmt = f"{value:.0f}"
        return f"${value_fmt}$"

    exponent = int(math.floor(math.log10(abs_value)))
    scaled_value = value / (10 ** exponent)
    return f"${scaled_value:.1f}10^{{{exponent}}}$"


def make_longtable_block(
    rows: List[str],
    teff_headers: List[int],
    block_title: str,
    first_col_label: str = "Species",
    caption: str | None = None,
    label: str | None = None,
) -> str:
    n_cols = len(teff_headers) + 1
    col_spec = "l" + "c" * len(teff_headers)
    header_cells = " & ".join([first_col_label] + [f"{teff} K" for teff in teff_headers])

    lines = [f"\\begin{{longtable}}{{{col_spec}}}"]
    if caption is not None:
        lines.append(f"\\caption{{{caption}}}")
    if label is not None:
        lines.append(f"\\label{{{label}}} \\\\")

    lines.extend([
        "\\toprule",
        f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{block_title}}}}} \\\\",
        "\\midrule",
        header_cells + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        header_cells + " \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{r}}{{Continued on next page}} \\\\",
        "\\midrule",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
        *rows,
        "\\end{longtable}",
    ])

    return "\n".join(lines)



def build_latex_tables(df: pd.DataFrame, selected_stars: List[dict], category_label: str) -> str:
    teff_headers = [star["teff_k"] for star in selected_stars]
    if "species" not in df.columns or df.empty:
        return ""
    species_order = list(dict.fromkeys(df["species"].tolist()))
    include_errors = category_label == "Atoms"

    blocks: List[str] = []
    for b_kms in B_VALUES_KMS:
        b_label = broadening_label(b_kms)
        rows = []
        subset_b = df[df["b_label"] == b_label]

        for species in species_order:
            subset_species = subset_b[subset_b["species"] == species]
            cell_text = []
            for star in selected_stars:
                row = subset_species[subset_species["star_key"] == star["key"]]
                if row.empty:
                    cell_text.append("-")
                else:
                    beta_value = float(row.iloc[0]["beta"])
                    beta_err = float(row.iloc[0]["beta_err"])
                    if include_errors:
                        cell_text.append(format_pm(beta_value, beta_err))
                    else:
                        cell_text.append(format_value_only(beta_value))
            rows.append("{} & {} \\\\".format(latex_species_name(species), " & ".join(cell_text)))

        block_title = f"{category_label}: $b = {b_label}$ km s$^{{-1}}$ (species-wise $N_{{\\tau=1}}$)"
        species_word = "Atomic" if category_label == "Atoms" else "Molecular"
        caption = (
            f"{species_word} $\\beta$-values for selected stellar temperatures "
            f"at $b={b_label}\\ \\mathrm{{km\\,s^{{-1}}}}$."
        )
        label_prefix = "atoms" if category_label == "Atoms" else "molecules"
        label = f"tab:beta_bigtable_{label_prefix}_b{b_label}"
        blocks.append(
            "\n".join([
                "",
                make_longtable_block(
                    rows,
                    teff_headers,
                    block_title=block_title,
                    first_col_label="Ion" if category_label == "Atoms" else "Molecule",
                    caption=caption,
                    label=label,
                ),
                "",
            ])
        )

    return "\n".join(blocks)


# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------
def calculate_category_rows(species_list: Iterable[str], selected_stars: List[dict], category: str) -> List[dict]:
    rows: List[dict] = []

    for species in species_list:
        print(f"Calculating {category}: {species}")
        for b_kms in B_VALUES_KMS:
            b_label = broadening_label(b_kms)
            for star_info in selected_stars:
                try:
                    if category == "atom":
                        n_tau1_cm2, beta, beta_err, tau_check = beta_for_atom(species, b_kms, star_info)
                    else:
                        n_tau1_cm2, beta, beta_err, tau_check = beta_for_molecule(species, b_kms, star_info)
                except Exception as exc:
                    print(f"Skipping {category} species={species}, b={b_label}, star={star_info['key']}: {type(exc).__name__}: {exc}")
                    n_tau1_cm2, beta, beta_err, tau_check = np.nan, np.nan, np.nan, np.nan

                rows.append(
                    {
                        "species": species,
                        "b_label": b_label,
                        "b_effective_kms": float(effective_b_value(b_kms).to_value(u.km / u.s)),
                        "star_key": star_info["key"],
                        "teff_k": star_info["teff_k"],
                        "n_tau1_cm2": n_tau1_cm2,
                        "tau_check": tau_check,
                        "beta": beta,
                        "beta_err": beta_err,
                    }
                )

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
        atom_rows = calculate_category_rows(ATOM_SPECIES, selected_stars, category="atom")
        atom_df = rows_to_dataframe(atom_rows)
        if PRINT_TAU_CHECK_SUMMARY and (not atom_df.empty) and ("tau_check" in atom_df.columns):
            tau_summary = atom_df["tau_check"].dropna()
            if not tau_summary.empty:
                print(
                    "Atom tau-check summary:",
                    f"min={tau_summary.min():.3e}, median={tau_summary.median():.3e}, max={tau_summary.max():.3e}"
                )
        if atom_df.empty:
            print("No atom rows were produced. Skipping atom csv/latex output.")
        else:
            if SAVE_RAW_CSV:
                atom_df.to_csv(OUTPUT_DIR / RAW_ATOMS_NAME, index=False)
            if SAVE_LATEX:
                atom_tex = build_latex_tables(atom_df, selected_stars, category_label="Atoms")
                if atom_tex:
                    (OUTPUT_DIR / TEX_ATOMS_NAME).write_text(atom_tex, encoding="utf-8")

    if INCLUDE_MOLECULES:
        molecule_rows = calculate_category_rows(MOLECULE_SPECIES, selected_stars, category="molecule")
        molecule_df = rows_to_dataframe(molecule_rows)
        if PRINT_TAU_CHECK_SUMMARY and (not molecule_df.empty) and ("tau_check" in molecule_df.columns):
            tau_summary = molecule_df["tau_check"].dropna()
            if not tau_summary.empty:
                print(
                    "Molecule tau-check summary:",
                    f"min={tau_summary.min():.3e}, median={tau_summary.median():.3e}, max={tau_summary.max():.3e}"
                )
        if molecule_df.empty:
            print("No molecule rows were produced. Skipping molecule csv/latex output.")
        else:
            if SAVE_RAW_CSV:
                molecule_df.to_csv(OUTPUT_DIR / RAW_MOLECULES_NAME, index=False)
            if SAVE_LATEX:
                molecule_tex = build_latex_tables(molecule_df, selected_stars, category_label="Molecules")
                if molecule_tex:
                    (OUTPUT_DIR / TEX_MOLECULES_NAME).write_text(molecule_tex, encoding="utf-8")

    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
