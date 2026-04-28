import pathlib
import sys
import time
from typing import Dict, List

import astropy.units as u
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SPECIES = "CO"
STAR_KEYS = ["M9", "A0", "O0"]
DISTANCE_AU = 1.0
GAS_TEMPERATURE = 0.001 * u.K
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA

# Physical b used in the profile, matching the big-table molecule b=0 convention.
PHYSICAL_B_ZERO = 1.0e-20 * u.km / u.s

# Finite velocities used only to define the wavelength sampling step dlam.
# The first value is treated as the reference solution.
GRID_CONSTRUCTION_B_VALUES_KMS = [0.1, 0.2, 0.5]

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
OUTPUT_TXT = OUTPUT_DIR / "co_b0_grid_validation.txt"
OUTPUT_CSV = OUTPUT_DIR / "co_b0_grid_validation.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def grid_dlam_from_velocity(grid_b: u.Quantity) -> u.Quantity:
    rep = 0.5 * (WAVEMIN + WAVEMAX)
    doppler_sigma = (rep * (grid_b / (299792.458 * u.km / u.s))).to(u.AA)
    return np.maximum((doppler_sigma / 3.0).to(u.AA), 1e-5 * u.AA)


def default_pathological_grid_size() -> int:
    dlam_floor = 1e-5 * u.AA
    return int(np.floor(((WAVEMAX - WAVEMIN) / dlam_floor).decompose().value)) + 1


def make_star(star_key: str) -> Star:
    params = STAR_TEMPLATES[star_key]
    return Star(
        params["path"],
        params["radius"],
        params["mass"],
        vsini=params["vsini"],
        epsilon=params["epsilon"],
    )


def make_profile(grid_b_kms: float) -> BroadeningProfileMolecule:
    template = MOLECULE_TEMPLATES[SPECIES]
    mol = Molecule(SPECIES, WAVEMIN, WAVEMAX)
    source = template.get("source", "exomol").lower()

    if source == "hitran":
        mol.fetch_hitran(**template["fetch_kwargs"])
    else:
        fetch_kwargs = template["fetch_kwargs"]
        mol.fetch_exomol(
            path=fetch_kwargs["path"],
            database=fetch_kwargs["database"],
            localdatabase=fetch_kwargs.get("localdatabase", "exomol_data"),
        )

    dlam = grid_dlam_from_velocity(grid_b_kms * u.km / u.s)
    profile = BroadeningProfileMolecule(
        molecule=mol,
        b=PHYSICAL_B_ZERO,
        dlam=dlam,
        profileType="Voigt",
    )
    profile.temp_strength_rel_cutoff = 1e-8
    return profile


def beta_from_ncol(pp: PhotonPressure, star: Star, ncol: u.Quantity) -> Dict[str, float]:
    ncol_array = np.array([ncol.to_value(1 / u.cm**2)], dtype=float) / u.cm**2
    distance = DISTANCE_AU * u.AU
    f_ph, f_ph_err, _, _ = pp.calc_PhotonPressure(ncol_array, GAS_TEMPERATURE, distance)
    beta, beta_err = pp.beta_Values(f_ph, f_ph_err, star.mass, distance.to(u.cm))
    return {
        "beta": float(np.ravel(beta.value)[0]),
        "beta_err": float(np.ravel(beta_err.value)[0]),
    }


def molecule_effective_sigma(profile: BroadeningProfileMolecule) -> u.Quantity:
    sigma_total = profile.sigmaArray.to(u.cm**2)
    if sigma_total.ndim == 0:
        return float(sigma_total.value) * u.cm**2
    return np.nanmax(sigma_total)


def relative_difference(reference: float, trial: float) -> float:
    denom = max(abs(reference), 1e-300)
    return abs(trial - reference) / denom


def build_one_case(grid_b_kms: float, stars: Dict[str, Star]) -> List[dict]:
    t0 = time.perf_counter()
    profile = make_profile(grid_b_kms)

    t_sigma0 = time.perf_counter()
    profile.apply_boltzmann_weights(GAS_TEMPERATURE, verbose=False)
    sigma_runtime_s = time.perf_counter() - t_sigma0

    sigma_eff = molecule_effective_sigma(profile)
    n_tau1 = (1.0 / sigma_eff).to(1 / u.cm**2)
    sigma_area = np.trapz(
        profile.sigmaArray.to_value(u.cm**2),
        profile.lam_grid.to_value(u.AA),
    )

    rows = []
    for star_key, star in stars.items():
        pp = PhotonPressure(profile, star)
        beta_tau0 = beta_from_ncol(pp, star, 0.0 / u.cm**2)
        beta_tau1 = beta_from_ncol(pp, star, n_tau1)
        rows.append(
            {
                "species": SPECIES,
                "star_key": star_key,
                "teff_k": int(infer_teff_from_star_template(star_key)),
                "grid_b_kms": float(grid_b_kms),
                "physical_b_kms": float(PHYSICAL_B_ZERO.to_value(u.km / u.s)),
                "dlam_AA": float(profile.dlam.to_value(u.AA)),
                "grid_size": int(len(profile.lam_grid)),
                "sigma_runtime_s": sigma_runtime_s,
                "total_case_runtime_s": time.perf_counter() - t0,
                "sigma_peak_cm2": float(sigma_eff.to_value(u.cm**2)),
                "sigma_area_cm2AA": float(sigma_area),
                "n_tau1_cm2": float(n_tau1.to_value(1 / u.cm**2)),
                "beta_tau0": beta_tau0["beta"],
                "beta_err_tau0": beta_tau0["beta_err"],
                "beta_tau1": beta_tau1["beta"],
                "beta_err_tau1": beta_tau1["beta_err"],
            }
        )

    profile.clear_temperature_cache(keep_current=False)
    PhotonPressure.clear_molecule_flux_cache()
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stars = {key: make_star(key) for key in STAR_KEYS}
    all_rows: List[dict] = []

    for grid_b_kms in GRID_CONSTRUCTION_B_VALUES_KMS:
        print(f"Running CO validation for grid-construction b = {grid_b_kms} km/s")
        all_rows.extend(build_one_case(grid_b_kms, stars))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    ref_b = float(GRID_CONSTRUCTION_B_VALUES_KMS[0])
    ref_df = df[df["grid_b_kms"] == ref_b].set_index("star_key")

    lines = []
    lines.append("CO molecule b=0 grid-construction validation")
    lines.append("")
    lines.append(f"Physical b used in profile: {PHYSICAL_B_ZERO.to_value(u.km / u.s):.6e} km/s")
    lines.append(f"Gas temperature: {GAS_TEMPERATURE.to_value(u.K):.6f} K")
    lines.append(f"Distance: {DISTANCE_AU:.3f} AU")
    lines.append(f"Wavelength range: {WAVEMIN.to_value(u.AA):.1f} - {WAVEMAX.to_value(u.AA):.1f} AA")
    lines.append(
        "Old pathological b=0 auto-grid size at dlam floor: "
        f"{default_pathological_grid_size():,} wavelength points"
    )
    lines.append("")
    lines.append("Reference grid-construction b is the first tested value.")
    lines.append("")

    for star_key in STAR_KEYS:
        star_rows = df[df["star_key"] == star_key].sort_values("grid_b_kms")
        lines.append(f"Star {star_key} | Teff = {int(infer_teff_from_star_template(star_key))} K")
        for _, row in star_rows.iterrows():
            ref_row = ref_df.loc[star_key]
            rel_n_tau1 = relative_difference(ref_row["n_tau1_cm2"], row["n_tau1_cm2"])
            rel_beta0 = relative_difference(ref_row["beta_tau0"], row["beta_tau0"])
            rel_beta1 = relative_difference(ref_row["beta_tau1"], row["beta_tau1"])
            lines.append(
                "  "
                f"grid_b={row['grid_b_kms']:.3f} km/s | "
                f"dlam={row['dlam_AA']:.6e} AA | "
                f"grid={int(row['grid_size']):,} | "
                f"Ntau1={row['n_tau1_cm2']:.6e} | "
                f"beta0={row['beta_tau0']:.6e} | "
                f"beta1={row['beta_tau1']:.6e} | "
                f"rel(Ntau1)={rel_n_tau1:.3e} | "
                f"rel(beta0)={rel_beta0:.3e} | "
                f"rel(beta1)={rel_beta1:.3e}"
            )
        lines.append("")

    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_TXT}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
