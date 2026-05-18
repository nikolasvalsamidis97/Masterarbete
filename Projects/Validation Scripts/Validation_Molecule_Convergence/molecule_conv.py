

import sys
import pathlib
import time

import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STAR_KEY = "A5"
MOLECULE_SPECIES = "CO"
DISTANCE = 1.0 * u.AU
TEMP_ATM = 5000.0 * u.K
B_VALUE = 1.0 * u.km / u.s
NCOL_VALUES = np.logspace(6, 14, 9) / u.cm**2

WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
PROFILE_TYPE = "Voigt"

# Test the real molecule pipeline at different grid resolutions.
# The first entry uses the class default (dlam=None).
DLAM_FACTORS = [None, 2.0, 5.0, 10.0, 20.0]

MAKE_PLOT = True
SAVE_PLOT = False
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Plots"
OUTPUT_NAME = "co_molecule_convergence.pdf"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_star(star_key: str) -> Star:
    params = STAR_TEMPLATES[star_key]
    return Star(
        params["path"],
        params["radius"],
        params["mass"],
        vsini=params["vsini"],
        epsilon=params["epsilon"],
    )



def make_molecule(species: str, wavemin: u.Quantity, wavemax: u.Quantity) -> Molecule:
    mol = Molecule(species, wavemin, wavemax)
    mol.fetch_exomol(
        path="CO/12C-16O/Li2015",
        database="Li2015",
        localdatabase="exomol_data",
    )
    return mol



def default_molecule_dlam(lam_min: u.Quantity, lam_max: u.Quantity, b_value: u.Quantity) -> u.Quantity:
    rep = 0.5 * (lam_min + lam_max)
    doppler_sigma = (rep * (b_value / u.speed_of_light if hasattr(u, 'speed_of_light') else b_value / (299792.458 * u.km / u.s))).to(u.AA)
    return np.maximum((doppler_sigma / 3.0).to(u.AA), 1e-5 * u.AA)



def molecule_default_dlam(lam_min: u.Quantity, lam_max: u.Quantity, b_value: u.Quantity) -> u.Quantity:
    rep = 0.5 * (lam_min + lam_max)
    doppler_sigma = (rep * (b_value / (299792.458 * u.km / u.s))).to(u.AA)
    dlam_auto = (doppler_sigma / 3.0).to(u.AA)
    floor = 1e-5 * u.AA
    return np.maximum(dlam_auto, floor)



def dlam_from_factor(factor):
    if factor is None:
        return None, "default"

    dlam_default = molecule_default_dlam(WAVEMIN, WAVEMAX, B_VALUE)
    dlam_custom = dlam_default * (3.0 / float(factor))
    return dlam_custom, f"/{float(factor):g}"



def relative_difference(reference: np.ndarray, trial: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(reference), 1e-300)
    return np.abs(trial - reference) / denom



def run_one_case(star: Star, factor):
    dlam_input, label = dlam_from_factor(factor)

    t0 = time.perf_counter()
    molecule = make_molecule(MOLECULE_SPECIES, WAVEMIN, WAVEMAX)
    profile = BroadeningProfileMolecule(
        molecule=molecule,
        b=B_VALUE,
        lam_min=WAVEMIN,
        lam_max=WAVEMAX,
        dlam=dlam_input,
        profileType=PROFILE_TYPE,
        verbose=False,
    )
    profile.apply_boltzmann_weights(TEMP_ATM, verbose=False)

    pp_mol = PhotonPressure(profile, star)
    F_ph, F_ph_err, _, _ = pp_mol.calc_PhotonPressure_molecule(NCOL_VALUES, TEMP_ATM, DISTANCE)
    beta, beta_err = pp_mol.beta_Values(F_ph, F_ph_err, star.mass, DISTANCE.to(u.cm))
    elapsed = time.perf_counter() - t0

    beta_val = np.ravel(beta.value)
    return {
        "label": label,
        "factor": factor,
        "requested_dlam_AA": np.nan if dlam_input is None else float(dlam_input.to_value(u.AA)),
        "actual_dlam_AA": float(profile.dlam.to_value(u.AA)),
        "grid_size": int(len(profile.lam_grid)),
        "peak_sigma_cm2": float(np.nanmax(profile.sigmaArray.to_value(u.cm**2))),
        "sigma_area_cm2AA": float(np.trapz(profile.sigmaArray.to_value(u.cm**2), profile.lam_grid.to_value(u.AA))),
        "beta": beta_val,
        "runtime_s": elapsed,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    star = make_star(STAR_KEY)

    print("=" * 80)
    print(f"Real molecule convergence test for {MOLECULE_SPECIES}")
    print(f"Star key: {STAR_KEY} | T_eff = {infer_teff_from_star_template(STAR_KEY)} K")
    print(f"T_atm = {TEMP_ATM} | b = {B_VALUE} | d = {DISTANCE}")
    print(f"Ncol grid: {[float(x) for x in NCOL_VALUES.to_value(1 / u.cm**2)]}")
    print("=" * 80)

    results = []
    for factor in DLAM_FACTORS:
        print(f"Running factor={factor} ...")
        result = run_one_case(star, factor)
        results.append(result)
        print(
            f"  label={result['label']}, dlam={result['actual_dlam_AA']:.6e} AA, "
            f"grid={result['grid_size']}, runtime={result['runtime_s']:.2f} s"
        )

    reference = results[-1]["beta"]

    print("\n" + "=" * 80)
    print("Summary relative to finest grid")
    print("=" * 80)
    for result in results:
        rel = relative_difference(reference, result["beta"])
        print(
            f"{result['label']:>8} | dlam={result['actual_dlam_AA']:.6e} AA | "
            f"grid={result['grid_size']:>9} | runtime={result['runtime_s']:.2f} s | "
            f"max rel diff={np.nanmax(rel):.6e}"
        )

    print("\n" + "=" * 80)
    print(f"{'Ncol [cm^-2]':>14} | " + " | ".join([f"beta {r['label']:>8}" for r in results]))
    print("=" * 80)
    for i, ncol in enumerate(NCOL_VALUES.to_value(1 / u.cm**2)):
        row = [f"{ncol:14.6e}"]
        for result in results:
            row.append(f"{result['beta'][i]:14.6e}")
        print(" | ".join(row))

    if MAKE_PLOT:
        x = NCOL_VALUES.to_value(1 / u.cm**2)

        fig, ax = plt.subplots(figsize=(8, 5))
        for result in results:
            ax.plot(x, result["beta"], marker="o", linewidth=1.2, label=f"{result['label']} | dlam={result['actual_dlam_AA']:.2e} AA")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_{\rm col}$ [cm$^{-2}$]")
        ax.set_ylabel(r"$\beta$")
        ax.set_title(rf"{MOLECULE_SPECIES} real molecule pipeline convergence")
        ax.grid(True, which="major", alpha=0.35)
        ax.legend(framealpha=0.9)
        fig.tight_layout()

        if SAVE_PLOT:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / OUTPUT_NAME
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {output_path}")

        plt.show()


if __name__ == "__main__":
    main()
