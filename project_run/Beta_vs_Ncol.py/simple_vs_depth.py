import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template


# -----------------------------------------------------------------------------
# Simple beta vs column density plot
# -----------------------------------------------------------------------------
# This script is meant as a clean introductory figure for the results section.
# It shows how beta changes as a function of column density for a chosen
# species around one representative star. No planet model is included here;
# the comparison is made against the stellar gravitational field.

SPECIES = "Na I"
STAR_KEYS = ["M1", "G4", "A0", "B0", "O0"]
R_OVER_RSTAR = 3.0  # evaluate beta at r = R_OVER_RSTAR * R_star

# Line settings
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_ATOM = 1 * u.km / u.s
NPTS_ATOM = 150

# Column-density grid [cm^-2]
NCOL_MIN = 1e9
NCOL_MAX = 1e28
NCOL_POINTS = 300

# Plot/output settings
FIGSIZE = (8, 5)
TITLE_SIZE = 17
AXIS_LABEL_SIZE = 19
TICK_LABEL_SIZE = 17
LEGEND_SIZE = 13
LINEWIDTH = 1.0
SAVE_FIGURE = True
SHOW_FIGURE = True
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Beta_vs_ncol"
OUTPUT_NAME = "NaI_beta_vs_ncol.pdf"


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


def log10_exponent_label(value: float, _position: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""

    exponent = np.log10(value)
    rounded = round(exponent)
    if not np.isclose(exponent, rounded, atol=1e-10):
        return ""
    return f"{int(rounded)}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    atom = Atom(SPECIES, WAVEMIN, WAVEMAX)
    profile = BroadeningProfile(atom, B_ATOM, NPTS_ATOM, "Voigt")

    ncol_values = np.logspace(np.log10(NCOL_MIN), np.log10(NCOL_MAX), NCOL_POINTS) / u.cm**2

    fig, ax = plt.subplots(figsize=FIGSIZE)

    plotted_any = False
    ncol_min_plot = None
    ncol_max_plot = None

    for star_key in STAR_KEYS:
        star = get_star(star_key)
        pp = PhotonPressure(profile, star)

        r_local = R_OVER_RSTAR * star.radius.to(u.cm)
        distance_local = r_local.to(u.AU)
        teff_star = infer_teff_from_star_template(star_key) * u.K
        g_star = ((6.67430e-11 * u.m**3 / (u.kg * u.s**2)) * star.mass.to(u.kg) / star.radius.to(u.m)**2).to(u.m / u.s**2)
        g_star_rounded = int(np.round(g_star.to_value(u.m / u.s**2) / 10.0) * 10)

        # Evaluate the radiative force for a line of sight with the chosen column
        # density and compare it to the stellar gravitational force.
        F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(
            ncol_values,
            teff_star,
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
            print(f"Skipping {star_key}: no positive finite beta values produced.")
            continue

        ncol_plot = ncol_values.to_value(1 / u.cm**2)[valid]
        beta_plot = beta_values[valid]

        ax.plot(
            ncol_plot,
            beta_plot,
            linewidth=LINEWIDTH,
            label=rf"$T_{{\rm eff}}={teff_star.to_value(u.K):.0f}$ K, $g={g_star_rounded}$ m s$^{{-2}}$",
        )
        plotted_any = True

        current_ncol_min = float(np.nanmin(ncol_plot))
        current_ncol_max = float(np.nanmax(ncol_plot))
        ncol_min_plot = current_ncol_min if ncol_min_plot is None else min(ncol_min_plot, current_ncol_min)
        ncol_max_plot = current_ncol_max if ncol_max_plot is None else max(ncol_max_plot, current_ncol_max)

    if not plotted_any:
        raise ValueError(
            "No positive finite beta values were produced for any selected stellar temperature."
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="0.4", label=r"$\beta = 1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
    ax.yaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\log_{10}\!\left(N_{\rm col}\,[\mathrm{cm}^{-2}]\right)$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\log_{10}\beta$", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(
        rf"$\beta$ vs column density | {SPECIES}",
        fontsize=TITLE_SIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 1)
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="best", framealpha=0.9, fontsize=LEGEND_SIZE)
    ax.set_xlim(ncol_min_plot, ncol_max_plot)
    #ax.set_ylim(np.nanmin(beta_plot), np.nanmax(beta_plot))
    fig.tight_layout()

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / OUTPUT_NAME
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
