import argparse
import os
import pathlib
import sys
import time

# Script-local runtime cache setup to avoid environment-specific import/cache issues
# without changing the project classes.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNTIME_CACHE_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Tables" / "runtime_cache" / "test_molecule_temperature_array"
(RUNTIME_CACHE_DIR / "mpl").mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE_DIR / "numba").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "mpl"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_DIR / "numba"))

import astropy.units as u
import numpy as np

sys.path.append(str(REPO_ROOT))

from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.Molecule import Molecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template


STAR_KEY = "A5"
MOLECULE_SPECIES = "CO"
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
DISTANCE = 1.0 * u.AU
B_MOLECULE = 1.0 * u.km / u.s
PROFILE_TYPE = "Voigt"
TEMP_VALUES = np.array([500.0, 1000.0, 3000.0, 5000.0]) * u.K
NCOL_VALUES = np.logspace(8, 24, 9) / u.cm**2


def parse_args():
    parser = argparse.ArgumentParser(description="Test molecule temperature-array handling.")
    parser.add_argument(
        "--label",
        default="run",
        help="Label used in the output filename, e.g. before_fix or after_fix.",
    )
    return parser.parse_args()


def output_path(label: str) -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "results" / "Tables" / f"test_molecule_temperature_array_{label}.txt"


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


def make_molecule_pipeline(star: Star):
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
    return PhotonPressure(profile, star), cfg


def beta_from_pipeline(pp: PhotonPressure, star: Star, temp_atm):
    force, force_err, _, _ = pp.calc_PhotonPressure(NCOL_VALUES, temp_atm, DISTANCE)
    beta, beta_err = pp.beta_Values(force, force_err, star.mass, DISTANCE)
    return beta, beta_err


def run_array_case(star: Star):
    t0 = time.perf_counter()
    pp, cfg = make_molecule_pipeline(star)
    setup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    beta, beta_err = beta_from_pipeline(pp, star, TEMP_VALUES)
    run_time = time.perf_counter() - t1

    return {
        "cfg": cfg,
        "beta": np.asarray(beta.value, dtype=float),
        "beta_err": np.asarray(beta_err.value, dtype=float),
        "setup_time_s": setup_time,
        "run_time_s": run_time,
        "shape": tuple(beta.shape),
    }


def run_scalar_reference(star: Star):
    t0 = time.perf_counter()
    pp, cfg = make_molecule_pipeline(star)
    setup_time = time.perf_counter() - t0

    beta_rows = []
    beta_err_rows = []
    per_temp_runtimes = []

    for temp in TEMP_VALUES:
        t1 = time.perf_counter()
        beta, beta_err = beta_from_pipeline(pp, star, temp)
        per_temp_runtimes.append(time.perf_counter() - t1)
        beta_rows.append(np.ravel(beta.value))
        beta_err_rows.append(np.ravel(beta_err.value))

    return {
        "cfg": cfg,
        "beta": np.vstack(beta_rows),
        "beta_err": np.vstack(beta_err_rows),
        "setup_time_s": setup_time,
        "run_time_s": float(np.sum(per_temp_runtimes)),
        "per_temp_runtimes_s": per_temp_runtimes,
        "shape": (len(beta_rows), len(beta_rows[0])),
    }


def relative_difference(a, b):
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    return np.abs(a - b) / denom


def write_beta_table(handle, label: str, beta_values, temp_values):
    handle.write(f"{label}\n")
    handle.write("-" * len(label) + "\n")

    temp_headers = [
        f"beta_{temp.to_value(u.K):.0f}K"
        for temp in temp_values[: beta_values.shape[0]]
    ]
    header = f"{'Ncol_cm^-2':>14}  " + "  ".join(f"{name:>14}" for name in temp_headers)
    handle.write(header + "\n")

    for col_idx, ncol in enumerate(NCOL_VALUES.to_value(1 / u.cm**2)):
        row = [f"{ncol:14.6e}"]
        for temp_idx in range(beta_values.shape[0]):
            row.append(f"{beta_values[temp_idx, col_idx]:14.6e}")
        handle.write("  ".join(row) + "\n")
    handle.write("\n")


def main():
    args = parse_args()
    star = make_star(STAR_KEY)
    out_path = output_path(args.label)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    array_result = run_array_case(star)
    scalar_result = run_scalar_reference(star)
    total_runtime = time.perf_counter() - total_start

    overlapping_rows = min(array_result["beta"].shape[0], scalar_result["beta"].shape[0])
    if overlapping_rows > 0:
        rel = relative_difference(
            array_result["beta"][:overlapping_rows],
            scalar_result["beta"][:overlapping_rows],
        )
        max_rel = float(np.nanmax(rel))
    else:
        max_rel = np.nan

    missing_rows = scalar_result["beta"].shape[0] - array_result["beta"].shape[0]

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("Molecule temperature-array regression test\n")
        handle.write("=========================================\n\n")
        handle.write(f"Label: {args.label}\n")
        handle.write(f"Star template: {STAR_KEY}\n")
        handle.write(f"Star Teff (from template): {infer_teff_from_star_template(STAR_KEY)} K\n")
        handle.write(f"Molecule species: {MOLECULE_SPECIES}\n")
        handle.write(f"Molecule source: {array_result['cfg']['source']}\n")
        handle.write(f"Wavelength range: {WAVEMIN} to {WAVEMAX}\n")
        handle.write(f"Distance: {DISTANCE}\n")
        handle.write(f"Molecular b-value: {B_MOLECULE}\n")
        handle.write(f"Profile type: {PROFILE_TYPE}\n")
        handle.write(f"Temperature array: {TEMP_VALUES}\n")
        handle.write(f"Column density grid: {NCOL_VALUES}\n\n")

        handle.write("Summary\n")
        handle.write("-------\n")
        handle.write(f"Array beta shape: {array_result['shape']}\n")
        handle.write(f"Scalar reference shape: {scalar_result['shape']}\n")
        handle.write(f"Missing temperature rows in array result: {missing_rows}\n")
        handle.write(f"Max relative difference over overlapping rows: {max_rel:.6e}\n")
        handle.write(f"Array setup runtime [s]: {array_result['setup_time_s']:.3f}\n")
        handle.write(f"Array run runtime [s]: {array_result['run_time_s']:.3f}\n")
        handle.write(f"Scalar setup runtime [s]: {scalar_result['setup_time_s']:.3f}\n")
        handle.write(f"Scalar total run runtime [s]: {scalar_result['run_time_s']:.3f}\n")
        handle.write(f"Scalar per-temperature runtimes [s]: {scalar_result['per_temp_runtimes_s']}\n")
        handle.write(f"Whole script runtime [s]: {total_runtime:.3f}\n\n")

        write_beta_table(handle, "Array temperature call beta", array_result["beta"], TEMP_VALUES)
        write_beta_table(handle, "Scalar temperature reference beta", scalar_result["beta"], TEMP_VALUES)

    print(out_path)


if __name__ == "__main__":
    main()
