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
from Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template


# -----------------------------------------------------------------------------
# Beta vs column density: effect of broadening
# -----------------------------------------------------------------------------
# This script compares beta(N_col) for one simple star-only case using:
#   1) a standard broadened line profile
#   2) a nearly unbroadened/narrow profile
# The comparison is made only against stellar gravity, with no planet model.

SPECIES = "Na I"
STAR_KEY = "F0"
R_OVER_RSTAR = 3.0  # evaluate beta at r = R_OVER_RSTAR * R_star

# Line settings
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
NPTS_ATOM = 150

# Broadening comparison cases
B_LINE = 1.0 * u.km / u.s
B_NARROW = 1.0e-20 * u.km / u.s
ROTATION_ON = True
ROTATION_OFF = False

# Column-density grid [cm^-2]
NCOL_MIN = 1e9
NCOL_MAX = 1e16
NCOL_POINTS = 300

# Plot/output settings
FIGSIZE = (8, 5)
TITLE_SIZE = 17
AXIS_LABEL_SIZE = 19
TICK_LABEL_SIZE = 17
LEGEND_SIZE = 13
LINEWIDTH = 1.5
SAVE_FIGURE = True
SHOW_FIGURE = True
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Beta_vs_ncol"
OUTPUT_NAME = f"{SPECIES.replace(' ', '')}_broadening_beta_vs_ncol.pdf"


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


def pretty_species_name(species: str) -> str:
    parts = str(species).split()
    if len(parts) != 2:
        return str(species)

    element, stage = parts
    charge_map = {
        "I": rf"$\mathrm{{{element}}}^{{0}}$",
        "II": rf"$\mathrm{{{element}}}^{{+}}$",
        "III": rf"$\mathrm{{{element}}}^{{2+}}$",
    }
    return charge_map.get(stage, str(species))


def broadening_label(b_value, vsini_value: float) -> str:
    b_display = 0.0 if b_value <= B_NARROW else b_value.to_value(u.km / u.s)
    return rf"$b = {b_display:.0f}$ km/s, $v \sin i = {vsini_value:.0f}$ km/s"


def compute_beta_curve(species: str, star_key: str, b_value, rotation_on: bool):
    star = get_star(star_key)
    atom = Atom(species, WAVEMIN, WAVEMAX)
    profile = BroadeningProfile(atom, b_value, NPTS_ATOM, "Voigt")
    pp = PhotonPressure(profile, star)

    # Explicitly choose which stellar spectrum to use in the comparison.
    # Some downstream code may read the spectrum from PhotonPressure and some
    # may read it from the Star object, so update both when possible.
    if rotation_on:
        selected_flux = getattr(star, "flux_star_rot", None)
    else:
        selected_flux = getattr(star, "flux_star_unrot", None)

    if selected_flux is not None:
        if hasattr(pp, "flux_star"):
            pp.flux_star = selected_flux

        # IMPORTANT: for atoms, PhotonPressure builds an interpolated stellar
        # spectrum during initialization and calc_PhotonPressure later uses
        # that cached interpolation. Therefore, after switching between the
        # rotated and unrotated stellar spectra, we must also rebuild the
        # interpolated stellar spectrum used in the actual calculation.
        if hasattr(pp, "get_interp_Spectra") and hasattr(pp, "flux_star_interp"):
            pp.flux_star_interp = pp.get_interp_Spectra()
    else:
        print(
            f"Warning: could not find {'rotated' if rotation_on else 'unrotated'} stellar flux "
            f"for star={star_key}."
        )

    ncol_values = np.logspace(np.log10(NCOL_MIN), np.log10(NCOL_MAX), NCOL_POINTS) / u.cm**2
    r_local = R_OVER_RSTAR * star.radius.to(u.cm)
    distance_local = r_local.to(u.AU)
    teff_star = infer_teff_from_star_template(star_key) * u.K

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

    curve_tag = (
        "line+rot" if (b_value > B_NARROW and rotation_on) else
        "line only" if (b_value > B_NARROW and not rotation_on) else
        "rot only" if (b_value <= B_NARROW and rotation_on) else
        "none"
    )
    print(
        f"Computed {curve_tag}: first beta={beta_values[0] if beta_values.size else np.nan:.6e}, "
        f"last beta={beta_values[-1] if beta_values.size else np.nan:.6e}"
    )

    valid = np.isfinite(beta_values) & (beta_values > 0)
    if not np.any(valid):
        print(
            f"Skipping curve with no positive finite beta values: "
            f"species={species}, star={star_key}, b={b_value}, rotation_on={rotation_on}."
        )
        return star, teff_star, np.array([], dtype=float), np.array([], dtype=float)

    ncol_plot = ncol_values.to_value(1 / u.cm**2)[valid]
    beta_plot = beta_values[valid]
    return star, teff_star, ncol_plot, beta_plot


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    star_ref, teff_star, ncol_none, beta_none = compute_beta_curve(
        SPECIES,
        STAR_KEY,
        B_NARROW,
        rotation_on=ROTATION_OFF,
    )
    _, _, ncol_line, beta_line = compute_beta_curve(
        SPECIES,
        STAR_KEY,
        B_LINE,
        rotation_on=ROTATION_OFF,
    )
    _, _, ncol_rot, beta_rot = compute_beta_curve(
        SPECIES,
        STAR_KEY,
        B_NARROW,
        rotation_on=ROTATION_ON,
    )
    _, _, ncol_both, beta_both = compute_beta_curve(
        SPECIES,
        STAR_KEY,
        B_LINE,
        rotation_on=ROTATION_ON,
    )

    if all(arr.size == 0 for arr in (ncol_none, ncol_line, ncol_rot, ncol_both)):
        raise ValueError("No plottable curve was produced for any broadening configuration.")


    fig, ax = plt.subplots(figsize=FIGSIZE)
    vsini_value = star_ref.vsini.to_value(u.km / u.s)
    vsini_off = 0.0
    if ncol_none.size > 0:
        ax.plot(
            ncol_none,
            beta_none,
            linewidth=LINEWIDTH,
            linestyle="-",
            label=broadening_label(B_NARROW, vsini_off),
        )
    if ncol_line.size > 0:
        ax.plot(
            ncol_line,
            beta_line,
            linewidth=LINEWIDTH,
            linestyle="--",
            label=broadening_label(B_LINE, vsini_off),
        )
    if ncol_rot.size > 0:
        ax.plot(
            ncol_rot,
            beta_rot,
            linewidth=LINEWIDTH,
            linestyle=":",
            label=broadening_label(B_NARROW, vsini_value),
        )
    if ncol_both.size > 0:
        ax.plot(
            ncol_both,
            beta_both,
            linewidth=LINEWIDTH,
            linestyle="-.",
            label=broadening_label(B_LINE, vsini_value),
        )
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black", label=r"$\beta = 1$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
    ax.yaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\log_{10}\!\left(N_{\rm col}\,[\mathrm{cm}^{-2}]\right)$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\log_{10}(\beta)$", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(
        rf"$\beta$ vs column density | {pretty_species_name(SPECIES)} | $T_{{\rm eff}}={teff_star.to_value(u.K):.0f}$ K",
        fontsize=TITLE_SIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 1)
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.3, fontsize=LEGEND_SIZE)

    x_arrays = [arr for arr in (ncol_none, ncol_line, ncol_rot, ncol_both) if arr.size > 0]
    y_arrays = [arr for arr in (beta_none, beta_line, beta_rot, beta_both) if arr.size > 0]
    x_min = min(np.nanmin(arr) for arr in x_arrays)
    x_max = max(np.nanmax(arr) for arr in x_arrays)
    y_min = min(np.nanmin(arr) for arr in y_arrays)
    y_max = max(np.nanmax(arr) for arr in y_arrays)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(10**-1, 10**2.5)

    fig.tight_layout()

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / OUTPUT_NAME
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
