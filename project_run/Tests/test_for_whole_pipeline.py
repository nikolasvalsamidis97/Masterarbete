import os
import pathlib
import sys
import time

# Script-local runtime cache setup to avoid environment-specific import/cache issues
# without changing the project classes.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIME_CACHE_DIR = REPO_ROOT / "Tables" / "runtime_cache" / "test_for_whole_pipeline"
(RUNTIME_CACHE_DIR / "mpl").mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE_DIR / "numba").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "mpl"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_DIR / "numba"))

import astropy.units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import trapezoid

sys.path.append(str(REPO_ROOT))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.Molecule import Molecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
STAR_KEY = "A5"

ATOM_SPECIES = "Na I"
MOLECULE_SPECIES = "CO"

WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
DISTANCE = 1.0 * u.AU
TEMP_ATM = 5000.0 * u.K

B_ATOM = 1.0 * u.km / u.s
B_MOLECULE = 1.0 * u.km / u.s
NPTS_ATOM = 300
PROFILE_TYPE = "Voigt"

NCOL_VALUES = np.logspace(6, 29, 15) / u.cm**2

OUTPUT_FILE = pathlib.Path(__file__).with_name("test_for_whole_pipeline_results.txt")
SWEEP_OUTPUT_FILE = pathlib.Path(__file__).with_name("test_for_whole_pipeline_temperature_sweep.txt")
TEMP_SWEEP_VALUES = np.array([500.0, 1000.0, 3000.0, 5000.0, 8000.0, 12000.0]) * u.K


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_star(star_key: str) -> Star:
    params = STAR_TEMPLATES[star_key]
    star_path = REPO_ROOT / params["path"]
    return Star(
        str(star_path),
        params["radius"],
        params["mass"],
        vsini=params["vsini"],
        epsilon=params["epsilon"],
    )


def make_atom_result(star: Star):
    t0 = time.perf_counter()
    atom = Atom(ATOM_SPECIES, WAVEMIN, WAVEMAX)
    profile = BroadeningProfile(atom, B_ATOM, NPTS_ATOM, PROFILE_TYPE)
    pp = PhotonPressure(profile, star)

    force, force_err, _, _ = pp.calc_PhotonPressure(NCOL_VALUES, TEMP_ATM, DISTANCE)
    beta, beta_err = pp.beta_Values(force, force_err, star.mass, DISTANCE)

    return {
        "species": ATOM_SPECIES,
        "kind": "atom",
        "force_N": np.ravel(force.to_value(u.N)),
        "force_err_N": np.ravel(force_err.to_value(u.N)),
        "beta": np.ravel(beta.value),
        "beta_err": np.ravel(beta_err.value),
        "runtime_s": time.perf_counter() - t0,
        "metadata": {
            "n_lines": int(np.asarray(atom.lam0).reshape(-1).shape[0]),
            "profile_type": PROFILE_TYPE,
            "b_kms": float(B_ATOM.to_value(u.km / u.s)),
        },
    }


def make_molecule_result(star: Star):
    t0 = time.perf_counter()
    cfg = MOLECULE_TEMPLATES[MOLECULE_SPECIES]
    molecule = Molecule(MOLECULE_SPECIES, WAVEMIN, WAVEMAX)

    if cfg["source"] == "exomol":
        fetch_kwargs = dict(cfg["fetch_kwargs"])
        localdatabase = fetch_kwargs.get("localdatabase")
        if localdatabase is not None:
            fetch_kwargs["localdatabase"] = str(REPO_ROOT / localdatabase)
        molecule.fetch_exomol(**fetch_kwargs)
    elif cfg["source"] == "hitran":
        fetch_kwargs = dict(cfg["fetch_kwargs"])
        localdatabase = fetch_kwargs.get("localdatabase")
        if localdatabase is not None:
            localdatabase = str(REPO_ROOT / localdatabase)
        molecule.fetch_hitran(
            molecule_name=MOLECULE_SPECIES,
            isotope=fetch_kwargs.get("isotope", 1),
            localdatabase=localdatabase,
            path=fetch_kwargs.get("path"),
            databank_name=fetch_kwargs.get("databank_name", f"HITRAN-{MOLECULE_SPECIES}"),
        )
    else:
        raise ValueError(f"Unsupported molecule source: {cfg['source']}")

    profile = BroadeningProfileMolecule(
        molecule,
        B_MOLECULE,
        profileType=PROFILE_TYPE,
    )
    pp = PhotonPressure(profile, star)

    force, force_err, _, _ = pp.calc_PhotonPressure(NCOL_VALUES, TEMP_ATM, DISTANCE)
    beta, beta_err = pp.beta_Values(force, force_err, star.mass, DISTANCE)

    return {
        "species": MOLECULE_SPECIES,
        "kind": "molecule",
        "force_N": np.ravel(force.to_value(u.N)),
        "force_err_N": np.ravel(force_err.to_value(u.N)),
        "beta": np.ravel(beta.value),
        "beta_err": np.ravel(beta_err.value),
        "runtime_s": time.perf_counter() - t0,
        "metadata": {
            "source": cfg["source"],
            "profile_type": PROFILE_TYPE,
            "b_kms": float(B_MOLECULE.to_value(u.km / u.s)),
            "grid_points": int(profile.lam_grid.shape[0]),
        },
    }


def write_header(f, star: Star):
    f.write("Whole pipeline test\n")
    f.write("===================\n\n")
    f.write(f"Star template: {STAR_KEY}\n")
    f.write(f"Star Teff (from template): {infer_teff_from_star_template(STAR_KEY)} K\n")
    f.write(f"Star path: {star.path}\n")
    f.write(f"Atom species: {ATOM_SPECIES}\n")
    f.write(f"Molecule species: {MOLECULE_SPECIES}\n")
    f.write(f"Wavelength range: {WAVEMIN} to {WAVEMAX}\n")
    f.write(f"Distance: {DISTANCE}\n")
    f.write(f"Atmospheric temperature: {TEMP_ATM}\n")
    f.write(f"Atomic b-value: {B_ATOM}\n")
    f.write(f"Molecular b-value: {B_MOLECULE}\n")
    f.write(f"Atomic velocity grid points: {NPTS_ATOM}\n")
    f.write(f"Profile type: {PROFILE_TYPE}\n")
    f.write(f"Column density grid: {NCOL_VALUES}\n\n")


def write_failure_block(f, label: str, exc: Exception):
    f.write(f"{label} FAILED\n")
    f.write(f"Reason: {type(exc).__name__}: {exc}\n\n")


def write_result_block(f, result: dict):
    f.write(f"{result['species']} ({result['kind']})\n")
    f.write("-" * (len(result["species"]) + len(result["kind"]) + 3) + "\n")
    for key, value in result["metadata"].items():
        f.write(f"{key}: {value}\n")
    f.write(f"runtime_s: {result['runtime_s']:.3f}\n\n")
    f.write(
        f"{'Ncol_cm^-2':>14}  {'force_N':>14}  {'force_err_N':>14}  {'beta':>14}  {'beta_err':>14}\n"
    )
    for ncol, force, force_err, beta, beta_err in zip(
        NCOL_VALUES.to_value(1 / u.cm**2),
        result["force_N"],
        result["force_err_N"],
        result["beta"],
        result["beta_err"],
    ):
        f.write(
            f"{ncol:14.6e}  {force:14.6e}  {force_err:14.6e}  {beta:14.6e}  {beta_err:14.6e}\n"
        )
    f.write("\n")


def legacy_atomic_lower_only_weights(atom_obj: Atom, temp_atm: u.Quantity):
    T_val = np.atleast_1d(np.asarray(temp_atm.to_value(u.K), dtype=float))
    El_val = np.asarray(atom_obj.E_l.to_value(u.eV), dtype=float).reshape(-1)
    gl_val = np.asarray(atom_obj.g_l.to_value(u.dimensionless_unscaled), dtype=float).reshape(-1)
    kb_eV_per_K = const.k_B.to_value(u.eV / u.K)

    unique_lower, inv = np.unique(np.column_stack((El_val, gl_val)), axis=0, return_inverse=True)
    E_unique = unique_lower[:, 0][:, None]
    g_unique = unique_lower[:, 1][:, None]
    boltz_unique = g_unique * np.exp(-E_unique / (kb_eV_per_K * T_val[None, :]))
    Z = np.nansum(boltz_unique, axis=0, keepdims=True)
    return boltz_unique[inv] / Z


def calc_atom_force_with_custom_weights(pp_atom: PhotonPressure, weights, column_density, distance):
    N_col = column_density.to(u.cm**(-2))
    d = distance
    R_star = pp_atom.star.radius
    omega = (R_star / d) ** 2

    sig = pp_atom.crossection_sym
    Flux = pp_atom.flux_star_interp * omega
    lam = pp_atom.lam_sym

    Flux_unit = Flux.unit
    sig_unit = sig.unit
    lam_unit = lam.unit
    force_unit = (Flux_unit * sig_unit * lam_unit / const.c.unit)

    Flux_val = np.asarray(Flux.value, dtype=np.float64)
    sig_val = np.asarray(sig.value, dtype=np.float64)
    lam_val = np.asarray(lam.value, dtype=np.float64)
    weights_val = np.asarray(weights, dtype=np.float64).reshape(sig_val.shape[0], -1)

    sigma_weighted = sig_val[:, :, None] * weights_val[:, None, :]
    N_col_val = np.asarray(N_col.to_value(1 / u.cm**2), dtype=np.float64)
    trans = np.exp(-sigma_weighted[:, :, :, None] * N_col_val[None, None, None, :])
    integrand = Flux_val[:, :, None, None] * sigma_weighted[:, :, :, None] * trans

    force_line = (
        trapezoid(integrand, lam_val[:, :, None, None], axis=1) / const.c.to_value(u.m / u.s)
    ) * force_unit
    force_total = np.nansum(force_line.to(u.N), axis=0)
    return force_total


def write_temperature_sweep():
    star = make_star(STAR_KEY)

    atom = Atom(ATOM_SPECIES, WAVEMIN, WAVEMAX)
    atom_profile = BroadeningProfile(atom, B_ATOM, NPTS_ATOM, PROFILE_TYPE)
    pp_atom = PhotonPressure(atom_profile, star)

    cfg = MOLECULE_TEMPLATES[MOLECULE_SPECIES]
    molecule = Molecule(MOLECULE_SPECIES, WAVEMIN, WAVEMAX)
    if cfg["source"] == "exomol":
        fetch_kwargs = dict(cfg["fetch_kwargs"])
        localdatabase = fetch_kwargs.get("localdatabase")
        if localdatabase is not None:
            fetch_kwargs["localdatabase"] = str(REPO_ROOT / localdatabase)
        molecule.fetch_exomol(**fetch_kwargs)
    elif cfg["source"] == "hitran":
        fetch_kwargs = dict(cfg["fetch_kwargs"])
        localdatabase = fetch_kwargs.get("localdatabase")
        if localdatabase is not None:
            localdatabase = str(REPO_ROOT / localdatabase)
        molecule.fetch_hitran(
            molecule_name=MOLECULE_SPECIES,
            isotope=fetch_kwargs.get("isotope", 1),
            localdatabase=localdatabase,
            path=fetch_kwargs.get("path"),
            databank_name=fetch_kwargs.get("databank_name", f"HITRAN-{MOLECULE_SPECIES}"),
        )
    else:
        raise ValueError(f"Unsupported molecule source: {cfg['source']}")

    molecule_profile = BroadeningProfileMolecule(
        molecule,
        B_MOLECULE,
        profileType=PROFILE_TYPE,
    )
    pp_molecule = PhotonPressure(molecule_profile, star)

    with open(SWEEP_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Whole pipeline temperature sweep\n")
        f.write("===============================\n\n")
        f.write(f"Star template: {STAR_KEY}\n")
        f.write(f"Star Teff (from template): {infer_teff_from_star_template(STAR_KEY)} K\n")
        f.write(f"Atom species: {ATOM_SPECIES}\n")
        f.write(f"Molecule species: {MOLECULE_SPECIES}\n")
        f.write(f"Temperature sweep: {TEMP_SWEEP_VALUES}\n")
        f.write(f"Distance: {DISTANCE}\n")
        f.write(f"Column density grid: {NCOL_VALUES}\n\n")

        for temp in TEMP_SWEEP_VALUES:
            atom_force_current, atom_force_err_current, _, _ = pp_atom.calc_PhotonPressure(NCOL_VALUES, temp, DISTANCE)
            atom_beta_current, _ = pp_atom.beta_Values(atom_force_current, atom_force_err_current, star.mass, DISTANCE)

            legacy_weights = legacy_atomic_lower_only_weights(atom, temp)
            atom_force_legacy = calc_atom_force_with_custom_weights(pp_atom, legacy_weights, NCOL_VALUES, DISTANCE)
            atom_zero_err = np.zeros_like(atom_force_legacy.value) * u.N
            atom_beta_legacy, _ = pp_atom.beta_Values(atom_force_legacy, atom_zero_err, star.mass, DISTANCE)

            molecule_force, molecule_force_err, _, _ = pp_molecule.calc_PhotonPressure(NCOL_VALUES, temp, DISTANCE)
            molecule_beta, _ = pp_molecule.beta_Values(molecule_force, molecule_force_err, star.mass, DISTANCE)

            atom_beta_current_val = np.ravel(atom_beta_current.value)
            atom_beta_legacy_val = np.ravel(atom_beta_legacy.value)
            molecule_beta_val = np.ravel(molecule_beta.value)
            rel_diff = np.abs(atom_beta_current_val - atom_beta_legacy_val) / np.maximum(
                np.maximum(np.abs(atom_beta_current_val), np.abs(atom_beta_legacy_val)),
                1e-300,
            )

            f.write(f"T = {temp.to_value(u.K):.0f} K\n")
            f.write("-" * 10 + "\n")
            f.write(
                f"{'Ncol_cm^-2':>14}  {'beta_Na_current':>16}  {'beta_Na_legacy':>16}  {'rel_diff':>14}  {'beta_CO_current':>16}\n"
            )
            for ncol, beta_current, beta_legacy, rdiff, beta_co in zip(
                NCOL_VALUES.to_value(1 / u.cm**2),
                atom_beta_current_val,
                atom_beta_legacy_val,
                rel_diff,
                molecule_beta_val,
            ):
                f.write(
                    f"{ncol:14.6e}  {beta_current:16.6e}  {beta_legacy:16.6e}  {rdiff:14.6e}  {beta_co:16.6e}\n"
                )
            f.write("\n")


def main():
    star = make_star(STAR_KEY)
    atom_result = None
    molecule_result = None
    atom_exc = None
    molecule_exc = None

    try:
        atom_result = make_atom_result(star)
    except Exception as exc:
        atom_exc = exc

    try:
        molecule_result = make_molecule_result(star)
    except Exception as exc:
        molecule_exc = exc

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        write_header(f, star)
        if atom_result is not None:
            write_result_block(f, atom_result)
        else:
            write_failure_block(f, ATOM_SPECIES, atom_exc)

        if molecule_result is not None:
            write_result_block(f, molecule_result)
        else:
            write_failure_block(f, MOLECULE_SPECIES, molecule_exc)

    print(f"Saved results to {OUTPUT_FILE}")
    if atom_exc is not None:
        print(f"{ATOM_SPECIES} failed: {type(atom_exc).__name__}: {atom_exc}")
    if molecule_exc is not None:
        print(f"{MOLECULE_SPECIES} failed: {type(molecule_exc).__name__}: {molecule_exc}")
    write_temperature_sweep()
    print(f"Saved temperature sweep to {SWEEP_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
