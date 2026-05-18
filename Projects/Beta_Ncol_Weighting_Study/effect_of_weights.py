import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template
from project_utils.plotdata_to_txt import save_plotdata_txt


# -----------------------------------------------------------------------------
# Beta vs column density: effect of excitation temperature / Boltzmann weighting
# -----------------------------------------------------------------------------
# This script compares beta(N_col) for Fe I, Fe II, and Fe III for a few
# different temperatures. The calculation is done for one representative star
# and compared only against stellar gravity.

SPECIES_LIST = ["Fe I", "Fe II", "Fe III"]
STAR_KEY = "F0"
DISTANCE_AU = 0.1  # evaluate beta at a fixed distance from the star
T_EXC_LIST = [3000.0, 6000.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0] * u.K

# Line settings
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_ATOM = 1.0 * u.km / u.s
NPTS_ATOM = 150

# Column-density grid [cm^-2]
NCOL_MIN = 1e9
NCOL_MAX = 1e24
NCOL_POINTS = 300

# Output settings
SAVE_OUTPUT_TXT = True
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Beta_vs_ncol"

TEMP_COLORS = {
    3000: "tab:blue",
    6000: "tab:orange",
    10000: "tab:red",
    20000: "tab:green",
    30000: "tab:purple",
    40000: "tab:brown",
    50000: "tab:pink",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_star(star_key: str) -> Star:
    s = STAR_TEMPLATES[star_key]
    return Star(
        s["path"],
        s["radius"],
        s["mass"],
        vsini=s["vsini"],
        epsilon=s["epsilon"],
    )


def safe_name(value: str) -> str:
    return str(value).replace(" ", "").replace("/", "_")


def compute_beta_curve(species: str, star: Star, t_exc: u.Quantity):
    atom = Atom(species, WAVEMIN, WAVEMAX)
    profile = BroadeningProfile(atom, B_ATOM, NPTS_ATOM, "Voigt")
    pp = PhotonPressure(profile, star)

    ncol_values = np.logspace(np.log10(NCOL_MIN), np.log10(NCOL_MAX), NCOL_POINTS) / u.cm**2
    distance_local = DISTANCE_AU * u.AU
    r_local = distance_local.to(u.cm)

    F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(
        ncol_values,
        t_exc,
        distance_local,
    )
    beta_species, _ = pp.beta_Values(
        F_ph_tot,
        F_ph_tot_err,
        star.mass.to(u.g),
        r_local,
    )
    beta_values = np.asarray(beta_species.value, dtype=float).reshape(-1)

    valid = np.isfinite(beta_values) & (beta_values > 0)
    if not np.any(valid):
        print(
            f"Skipping curve with no positive finite beta values: "
            f"species={species}, T={t_exc.to_value(u.K):.0f} K"
        )
        return np.array([], dtype=float), np.array([], dtype=float), beta_values

    ncol_plot = ncol_values.to_value(1 / u.cm**2)[valid]
    beta_plot = beta_values[valid]
    return ncol_plot, beta_plot, beta_values


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    star = get_star(STAR_KEY)
    teff_star = infer_teff_from_star_template(STAR_KEY) * u.K

    ncol_values = np.logspace(np.log10(NCOL_MIN), np.log10(NCOL_MAX), NCOL_POINTS)

    for species in SPECIES_LIST:
        species_columns = []
        used_temperatures = []

        for t_exc in T_EXC_LIST:
            _, _, beta_values = compute_beta_curve(species, star, t_exc)
            beta_values = np.asarray(beta_values, dtype=float).reshape(-1)
            species_columns.append(beta_values)
            used_temperatures.append(float(t_exc.to_value(u.K)))

        if not species_columns:
            print(f"Skipping {species}: no output columns produced.")
            continue

        y_matrix = np.column_stack(species_columns)

        output_path = OUTPUT_DIR / f"{safe_name(species)}_beta_vs_ncol_weights.txt"

        if SAVE_OUTPUT_TXT:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            save_plotdata_txt(
                output_path,
                dataset_name=f"{safe_name(species)}_beta_vs_ncol_weights",
                x_label="Column density",
                x_unit="cm^-2",
                y_label="beta",
                y_unit="dimensionless",
                x_values=ncol_values,
                y_matrix=y_matrix,
                series_values=used_temperatures,
                series_label="temperature",
                series_unit="K",
                extra_metadata={
                    "species": species,
                    "star": STAR_KEY,
                    "stellar_teff_K": float(teff_star.to_value(u.K)),
                    "distance_AU": DISTANCE_AU,
                    "ncol_min_cm^-2": NCOL_MIN,
                    "ncol_max_cm^-2": NCOL_MAX,
                    "ncol_points": NCOL_POINTS,
                    "b_atom_km_s": float(B_ATOM.to_value(u.km / u.s)),
                },
            )
            print(f"Saved data to {output_path}")


if __name__ == "__main__":
    main()
