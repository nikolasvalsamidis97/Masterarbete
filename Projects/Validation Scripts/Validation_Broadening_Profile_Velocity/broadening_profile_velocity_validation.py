import csv
import pathlib
import sys
import time

import astropy.units as u
import numpy as np
from astropy import constants as const
from scipy.special import erf, wofz

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile


WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_PARAMETER = 1.0 * u.km / u.s
N_GRID = 1200

VALIDATION_SPECIES = ["Na I", "N I"]
PROFILE_TYPES = ["Gaussian", "Lorentz", "Voigt"]

PEAK_REL_TOL = 1.0e-10
L1_REL_TOL = 1.0e-10
GENERAL_MAPPING_TOL = 1.0e-5
VOIGT_WINDOW_REF_POINTS = 6000

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Tables"
SUMMARY_FILE = OUTPUT_DIR / "broadening_profile_velocity_validation.txt"
DETAIL_FILE = OUTPUT_DIR / "broadening_profile_velocity_validation_details.csv"


def normalize_profile_type(name: str) -> str:
    value = str(name).strip().lower()
    if value in {"gauss", "gaussian"}:
        return "gaussian"
    if value in {"lorentz", "lorentzian"}:
        return "lorentz"
    return "voigt"


def direct_wavelength_profile(profile: BroadeningProfile):
    lam0 = profile.molecule.lam0
    lam_sym = profile.half_to_symmetric_lam()
    delta_lam = lam_sym - lam0
    c_km_s = const.c.to(u.km / u.s)
    profile_type = normalize_profile_type(profile.profileType)

    if profile_type == "gaussian":
        phi_lam = (
            (c_km_s / (lam0 * profile.b * np.sqrt(np.pi)))
            * np.exp(-((c_km_s * delta_lam) / (lam0 * profile.b)) ** 2)
        ).to(1 / u.AA)
        return lam_sym, phi_lam

    gamma_lam = (lam0 * 0.5 * profile.lorentz_FWHM_v / c_km_s).to(u.AA)
    if profile_type == "lorentz":
        phi_lam = ((1.0 / np.pi) * gamma_lam / (delta_lam**2 + gamma_lam**2)).to(1 / u.AA)
        return lam_sym, phi_lam

    sigma_lam = (
        lam0
        * profile.gauss_FWHM_v
        / c_km_s
        / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    ).to(u.AA)
    z = (
        delta_lam.to_value(u.AA) + 1j * gamma_lam.to_value(u.AA)
    ) / (sigma_lam.to_value(u.AA) * np.sqrt(2.0))
    phi_val = np.real(wofz(z)) / (sigma_lam.to_value(u.AA) * np.sqrt(2.0 * np.pi))
    return lam_sym, phi_val * (1 / u.AA)


def transformed_velocity_profile_to_wavelength(profile: BroadeningProfile):
    c_over_lam0 = (const.c.to(u.km / u.s) / profile.molecule.lam0).to(1 / u.s)
    _, phi_v_sym = profile.half_to_symmetric_v(profile.profileArray)
    phi_lam_from_v = (phi_v_sym * c_over_lam0).to(1 / u.AA)
    lam_sym = profile.half_to_symmetric_lam()
    return lam_sym, phi_lam_from_v


def line_validation_metrics(profile: BroadeningProfile):
    lam_sym, phi_ref = direct_wavelength_profile(profile)
    _, phi_from_v = transformed_velocity_profile_to_wavelength(profile)

    lam_val = lam_sym.to_value(u.AA)
    phi_ref_val = phi_ref.to_value(1 / u.AA)
    phi_from_v_val = phi_from_v.to_value(1 / u.AA)
    diff_val = phi_from_v_val - phi_ref_val

    peak_ref = np.nanmax(phi_ref_val, axis=1)
    peak_rel_error = np.divide(
        np.nanmax(np.abs(diff_val), axis=1),
        peak_ref,
        out=np.zeros_like(peak_ref),
        where=peak_ref > 0.0,
    )

    area_ref = np.trapz(phi_ref_val, lam_val, axis=1)
    l1_rel_error = np.divide(
        np.trapz(np.abs(diff_val), lam_val, axis=1),
        area_ref,
        out=np.zeros_like(area_ref),
        where=area_ref > 0.0,
    )

    capture_fraction = window_capture_fraction(profile)

    line_center_error = np.abs(
        profile.lam_grid[:, 0].reshape(-1) - profile.molecule.lam0[:, 0].reshape(-1)
    ).to_value(u.AA)

    return {
        "lam0_AA": profile.molecule.lam0.to_value(u.AA).reshape(-1),
        "vlim_km_s": profile.vlim.to_value(u.km / u.s).reshape(-1),
        "line_center_error_AA": line_center_error.reshape(-1),
        "peak_rel_error": peak_rel_error.reshape(-1),
        "l1_rel_error": l1_rel_error.reshape(-1),
        "capture_fraction": capture_fraction.reshape(-1),
    }


def window_capture_fraction(profile: BroadeningProfile) -> np.ndarray:
    profile_type = normalize_profile_type(profile.profileType)
    vlim = profile.vlim.to_value(u.km / u.s).reshape(-1)
    gauss_fwhm_v = profile.gauss_FWHM_v.to_value(u.km / u.s).reshape(-1)
    lorentz_fwhm_v = profile.lorentz_FWHM_v.to_value(u.km / u.s).reshape(-1)

    if profile_type == "gaussian":
        return erf(vlim / profile.b.to_value(u.km / u.s))

    if profile_type == "lorentz":
        return (2.0 / np.pi) * np.arctan(2.0 * vlim / lorentz_fwhm_v)

    capture = np.zeros_like(vlim)
    for idx, vmax in enumerate(vlim):
        g = gauss_fwhm_v[idx]
        l = lorentz_fwhm_v[idx]
        vref = max(10.0 * vmax, 50.0 * g, 5000.0 * l, 50.0)
        v_dense = np.linspace(-vref, vref, VOIGT_WINDOW_REF_POINTS)
        sigma = g / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma = 0.5 * l
        z = (v_dense + 1j * gamma) / (sigma * np.sqrt(2.0))
        phi_dense = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
        full_area = np.trapz(phi_dense, v_dense)
        inside_mask = np.abs(v_dense) <= vmax
        inside_area = np.trapz(phi_dense[inside_mask], v_dense[inside_mask])
        capture[idx] = inside_area / full_area if full_area > 0.0 else np.nan
    return capture


def summarize_case(species: str, profile_type: str, metrics: dict) -> dict:
    lam0 = metrics["lam0_AA"]
    worst_peak_idx = int(np.argmax(metrics["peak_rel_error"]))
    worst_l1_idx = int(np.argmax(metrics["l1_rel_error"]))
    worst_capture_idx = int(np.argmin(metrics["capture_fraction"]))

    return {
        "species": species,
        "profile_type": profile_type,
        "n_lines": int(len(lam0)),
        "max_line_center_error_AA": float(np.nanmax(metrics["line_center_error_AA"])),
        "max_peak_rel_error": float(np.nanmax(metrics["peak_rel_error"])),
        "median_peak_rel_error": float(np.nanmedian(metrics["peak_rel_error"])),
        "max_l1_rel_error": float(np.nanmax(metrics["l1_rel_error"])),
        "median_l1_rel_error": float(np.nanmedian(metrics["l1_rel_error"])),
        "min_capture_fraction": float(np.nanmin(metrics["capture_fraction"])),
        "median_capture_fraction": float(np.nanmedian(metrics["capture_fraction"])),
        "worst_peak_lam0_AA": float(lam0[worst_peak_idx]),
        "worst_l1_lam0_AA": float(lam0[worst_l1_idx]),
        "worst_capture_lam0_AA": float(lam0[worst_capture_idx]),
        "mapping_pass": bool(
            np.nanmax(metrics["peak_rel_error"]) < GENERAL_MAPPING_TOL
            and np.nanmax(metrics["l1_rel_error"]) < GENERAL_MAPPING_TOL
        ),
    }


def write_detail_csv(rows):
    fieldnames = [
        "species",
        "profile_type",
        "line_index",
        "lam0_AA",
        "vlim_km_s",
        "line_center_error_AA",
        "peak_rel_error",
        "l1_rel_error",
        "capture_fraction",
    ]
    with DETAIL_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_txt(case_summaries, runtime_s: float):
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        f.write("Broadening Profile Velocity-Space Validation\n")
        f.write("===========================================\n")
        f.write(f"Species tested: {', '.join(VALIDATION_SPECIES)}\n")
        f.write(f"Profile types: {', '.join(PROFILE_TYPES)}\n")
        f.write(f"b parameter: {B_PARAMETER}\n")
        f.write(f"Velocity grid points: {N_GRID}\n")
        f.write(
            "Purpose: compare the implemented velocity-space profile against a direct "
            "wavelength-space reference on the same physical line grid.\n"
        )
        f.write(
            "Interpretation: if the mapping errors are near machine precision, the "
            "velocity-space treatment is numerically equivalent to the direct "
            "wavelength-space formulation.\n"
        )
        f.write(
            "The capture fraction reports how much of the normalized line area is "
            "contained inside the chosen finite vlim window.\n"
        )
        f.write(
            "Mapping pass criterion: max peak-relative error and max L1-relative "
            f"error both below {GENERAL_MAPPING_TOL:.1e}\n"
        )
        f.write("\n")

        for summary in case_summaries:
            f.write(f"{summary['species']} | {summary['profile_type']}\n")
            f.write(f"  lines tested: {summary['n_lines']}\n")
            f.write(f"  max |lambda(v=0) - lambda0| [AA]: {summary['max_line_center_error_AA']:.6e}\n")
            f.write(f"  max peak-relative error: {summary['max_peak_rel_error']:.6e}\n")
            f.write(f"  median peak-relative error: {summary['median_peak_rel_error']:.6e}\n")
            f.write(f"  max L1-relative error: {summary['max_l1_rel_error']:.6e}\n")
            f.write(f"  median L1-relative error: {summary['median_l1_rel_error']:.6e}\n")
            f.write(f"  min capture fraction: {summary['min_capture_fraction']:.6e}\n")
            f.write(f"  median capture fraction: {summary['median_capture_fraction']:.6e}\n")
            f.write(f"  worst peak-error line lambda0 [AA]: {summary['worst_peak_lam0_AA']:.6f}\n")
            f.write(f"  worst L1-error line lambda0 [AA]: {summary['worst_l1_lam0_AA']:.6f}\n")
            f.write(f"  worst capture line lambda0 [AA]: {summary['worst_capture_lam0_AA']:.6f}\n")
            f.write(f"  mapping validation pass: {summary['mapping_pass']}\n")
            f.write("\n")

        f.write(f"Total runtime [s]: {runtime_s:.2f}\n")
        f.write(f"Detailed per-line output: {DETAIL_FILE.name}\n")


def main():
    start = time.perf_counter()
    detail_rows = []
    case_summaries = []

    atom_cache = {}
    for species in VALIDATION_SPECIES:
        print(f"Loading atom data for {species}")
        atom_cache[species] = Atom(species, WAVEMIN, WAVEMAX)

    for species in VALIDATION_SPECIES:
        atom = atom_cache[species]
        for profile_type in PROFILE_TYPES:
            print(f"Validating {species} | {profile_type}")
            profile = BroadeningProfile(atom, B_PARAMETER, N_GRID, profile_type)
            metrics = line_validation_metrics(profile)
            summary = summarize_case(species, profile_type, metrics)
            case_summaries.append(summary)

            for idx, lam0 in enumerate(metrics["lam0_AA"]):
                detail_rows.append(
                    {
                        "species": species,
                        "profile_type": profile_type,
                        "line_index": idx,
                        "lam0_AA": float(lam0),
                        "vlim_km_s": float(metrics["vlim_km_s"][idx]),
                        "line_center_error_AA": float(metrics["line_center_error_AA"][idx]),
                        "peak_rel_error": float(metrics["peak_rel_error"][idx]),
                        "l1_rel_error": float(metrics["l1_rel_error"][idx]),
                        "capture_fraction": float(metrics["capture_fraction"][idx]),
                    }
                )

    runtime_s = time.perf_counter() - start
    write_detail_csv(detail_rows)
    write_summary_txt(case_summaries, runtime_s)
    print(f"Saved summary to {SUMMARY_FILE}")
    print(f"Saved details to {DETAIL_FILE}")
    print(f"Total runtime: {runtime_s:.2f} s")


if __name__ == "__main__":
    main()
