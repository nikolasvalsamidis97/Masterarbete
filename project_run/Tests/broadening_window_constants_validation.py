import csv
import pathlib
import sys
import time

import astropy.units as u
import numpy as np
from scipy.special import erf, wofz

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile


WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_PARAMETER = 1.0 * u.km / u.s

# Representative atomic test set for the practical Voigt-line check.
VALIDATION_SPECIES = ["Na I", "N I"]
VOIGT_REF_POINTS = 8000

GAUSSIAN_MULTIPLIERS = [1, 2, 3, 4, 5, 6, 7, 8]
LORENTZ_MULTIPLIERS = [1, 5, 10, 15, 20, 25, 30, 40, 50]
WINDOW_RULES = [
    (4, 15),
    (5, 20),
    (6, 25),
    (7, 30),
]

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
SUMMARY_FILE = OUTPUT_DIR / "broadening_window_constants_validation.txt"
DETAIL_FILE = OUTPUT_DIR / "broadening_window_constants_validation_details.csv"


def gaussian_capture_fraction(n_fwhm: float) -> float:
    return float(erf(2.0 * np.sqrt(np.log(2.0)) * float(n_fwhm)))


def lorentz_capture_fraction(n_fwhm: float) -> float:
    return float((2.0 / np.pi) * np.arctan(2.0 * float(n_fwhm)))


def symmetric_quadratic_grid(vmax_km_s: float, n_points: int) -> np.ndarray:
    base = np.linspace(0.0, 1.0, n_points) ** 2 * vmax_km_s
    return np.concatenate((-base[:0:-1], base))


def voigt_profile_velocity(v_km_s: np.ndarray, gauss_fwhm_km_s: float, lorentz_fwhm_km_s: float) -> np.ndarray:
    sigma = gauss_fwhm_km_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = 0.5 * lorentz_fwhm_km_s
    z = (v_km_s + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def voigt_capture_fraction(gauss_fwhm_km_s: float, lorentz_fwhm_km_s: float, g_mult: float, l_mult: float) -> float:
    vmax = max(g_mult * gauss_fwhm_km_s, l_mult * lorentz_fwhm_km_s)
    v_grid = symmetric_quadratic_grid(vmax, VOIGT_REF_POINTS)
    phi = voigt_profile_velocity(v_grid, gauss_fwhm_km_s, lorentz_fwhm_km_s)
    return float(np.trapz(phi, v_grid))


def summarize_rule(name: str, values: np.ndarray, lam0: np.ndarray) -> dict:
    worst_idx = int(np.argmin(values))
    return {
        "name": name,
        "min_capture_fraction": float(np.nanmin(values)),
        "median_capture_fraction": float(np.nanmedian(values)),
        "max_capture_fraction": float(np.nanmax(values)),
        "worst_lam0_AA": float(lam0[worst_idx]),
    }


def write_detail_csv(rows):
    fieldnames = [
        "species",
        "line_index",
        "lam0_AA",
        "gauss_fwhm_km_s",
        "lorentz_fwhm_km_s",
        "g_mult",
        "l_mult",
        "capture_fraction",
    ]
    with DETAIL_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_txt(gaussian_rows, lorentz_rows, case_summaries, runtime_s: float):
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        f.write("Broadening Window Constants Validation\n")
        f.write("=====================================\n")
        f.write(f"Representative atomic species: {', '.join(VALIDATION_SPECIES)}\n")
        f.write(f"b parameter for Voigt-line test: {B_PARAMETER}\n")
        f.write(
            "Purpose: validate the numerical constants in the finite integration rule\n"
        )
        f.write("         v_lim = max(6 Δv_G, 25 Δv_L)\n")
        f.write("\n")
        f.write("Pure Gaussian analytic enclosed-area fractions\n")
        f.write("---------------------------------------------\n")
        for n, frac in gaussian_rows:
            f.write(f"±{n:>2} Δv_G : {frac:.12f}\n")
        f.write("\n")
        f.write("Pure Lorentzian analytic enclosed-area fractions\n")
        f.write("------------------------------------------------\n")
        for n, frac in lorentz_rows:
            f.write(f"±{n:>2} Δv_L : {frac:.12f}\n")
        f.write("\n")
        f.write(
            "Interpretation of the analytic results:\n"
            "  The Gaussian factor 6 is extremely conservative and retains essentially\n"
            "  the full Gaussian area. The Lorentzian factor 25 retains about 98.7%\n"
            "  of the total Lorentzian area, so it should be interpreted as a practical\n"
            "  truncation of the far wings rather than a mathematically exact full-area bound.\n"
        )
        f.write("\n")
        f.write("Representative atomic Voigt-line test\n")
        f.write("-------------------------------------\n")
        f.write(
            "For each real atomic line, the Voigt profile was integrated directly in velocity\n"
            "space up to the candidate finite window. The reported capture fraction is the\n"
            "area retained inside that window.\n"
        )
        for summary in case_summaries:
            f.write(
                f"{summary['name']}: min={summary['min_capture_fraction']:.12f}, "
                f"median={summary['median_capture_fraction']:.12f}, "
                f"max={summary['max_capture_fraction']:.12f}, "
                f"worst line λ0={summary['worst_lam0_AA']:.6f} AA\n"
            )
        f.write("\n")
        chosen = next(summary for summary in case_summaries if summary["name"] == "g=6, l=25")
        f.write("Practical conclusion:\n")
        f.write(
            "  The adopted rule v_lim = max(6 Δv_G, 25 Δv_L) is strongly justified for\n"
            "  atomic Voigt profiles as a practical finite integration range. In the\n"
            "  representative test set used here, its worst-case retained fraction was\n"
            f"  {chosen['min_capture_fraction']:.6%}.\n"
        )
        f.write(f"\nDetailed per-line output: {DETAIL_FILE.name}\n")
        f.write(f"Total runtime [s]: {runtime_s:.2f}\n")


def main():
    start = time.perf_counter()

    gaussian_rows = [(n, gaussian_capture_fraction(n)) for n in GAUSSIAN_MULTIPLIERS]
    lorentz_rows = [(n, lorentz_capture_fraction(n)) for n in LORENTZ_MULTIPLIERS]

    detail_rows = []
    rule_capture_by_name = {f"g={g}, l={l}": [] for g, l in WINDOW_RULES}
    rule_lam0_by_name = {f"g={g}, l={l}": [] for g, l in WINDOW_RULES}

    for species in VALIDATION_SPECIES:
        print(f"Loading {species}")
        atom = Atom(species, WAVEMIN, WAVEMAX)
        profile = BroadeningProfile(atom, B_PARAMETER, 1200, "Voigt")

        lam0 = profile.molecule.lam0.to_value(u.AA).reshape(-1)
        gauss_fwhm = profile.gauss_FWHM_v.to_value(u.km / u.s).reshape(-1)
        lorentz_fwhm = profile.lorentz_FWHM_v.to_value(u.km / u.s).reshape(-1)

        for idx in range(len(lam0)):
            for g_mult, l_mult in WINDOW_RULES:
                capture = voigt_capture_fraction(gauss_fwhm[idx], lorentz_fwhm[idx], g_mult, l_mult)
                name = f"g={g_mult}, l={l_mult}"
                rule_capture_by_name[name].append(capture)
                rule_lam0_by_name[name].append(lam0[idx])
                detail_rows.append(
                    {
                        "species": species,
                        "line_index": idx,
                        "lam0_AA": float(lam0[idx]),
                        "gauss_fwhm_km_s": float(gauss_fwhm[idx]),
                        "lorentz_fwhm_km_s": float(lorentz_fwhm[idx]),
                        "g_mult": g_mult,
                        "l_mult": l_mult,
                        "capture_fraction": float(capture),
                    }
                )

    case_summaries = []
    for name in rule_capture_by_name:
        case_summaries.append(
            summarize_rule(
                name,
                np.asarray(rule_capture_by_name[name], dtype=float),
                np.asarray(rule_lam0_by_name[name], dtype=float),
            )
        )

    runtime_s = time.perf_counter() - start
    write_detail_csv(detail_rows)
    write_summary_txt(gaussian_rows, lorentz_rows, case_summaries, runtime_s)
    print(f"Saved summary to {SUMMARY_FILE}")
    print(f"Saved details to {DETAIL_FILE}")
    print(f"Total runtime: {runtime_s:.2f} s")


if __name__ == "__main__":
    main()
