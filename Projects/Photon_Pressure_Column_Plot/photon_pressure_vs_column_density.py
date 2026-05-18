import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np


OUTPUT_PATH = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Photon pressure" / "F_ph_vs_Ncol.pdf"
SPECTRUM_PATH = PROJECT_ROOT / "Templates" / "TS" / "Spectral_type" / "A" / "A6" / "lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt"
TITLE_SIZE = 17
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 12
Y_SCALE = 1e26


def log10_exponent_label(value, _position):
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = np.log10(value)
    rounded = round(exponent)
    if not np.isclose(exponent, rounded, atol=1e-10):
        return ""
    return f"{int(rounded)}"


atom = Atom("Na I", 300 * u.AA, 50000 * u.AA, 0 / u.s)

# Velocity-grid resolution
Npts = 300

# Column densities
Ncol = np.logspace(8, 15, 100) * u.cm**-2

# Evaluate multiple Doppler widths
b_values = np.array([ 0.00001, 1, 2, 3, 4, 5]) * (u.km / u.s)

# Compute photon pressure curves
Temp = 300 * u.K
Distance = 1 * u.au
vsini = 1 * u.km / u.s
epsilon = 0.0 * u.dimensionless_unscaled
star = Star(str(SPECTRUM_PATH), 
               1*const.R_sun.value * u.m, 1*const.M_sun.value * u.kg, vsini, epsilon)

fig, ax = plt.subplots(figsize=(10, 5))

for b in b_values:
    broad = BroadeningProfile(atom, b, Npts, 'Voigt')
    pp = PhotonPressure(broad, star)

    Fph, Fph_err, _, _ = pp.calc_PhotonPressure(Ncol, Temp, Distance)

    # calc_PhotonPressure returns shape (N_temp, N_col). Here N_temp=1.
    y = Fph[0].to_value(u.N) * Y_SCALE
    yerr = Fph_err[0].to_value(u.N) * Y_SCALE
    x = Ncol.to_value(1 / u.cm**2)

    ax.plot(
    x, y,
    linewidth=1.5,
    alpha=0.9,
    label=rf"$\Delta v_D$ = {b.to_value(u.km/u.s):.1f} km/s",
    )

    ax.fill_between(
        x,
        y - yerr,
        y + yerr,
        alpha=0.20,     # shadow strength
        linewidth=0,    # no edge line
    )

ax.set_xscale("log", base=10)
#ax.set_yscale("log", base=10)
ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
ax.xaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
ax.xaxis.set_minor_formatter(NullFormatter())

ax.set_xlabel(r"$\log\!\left(N_{\mathrm{col}}\,[\mathrm{cm}^{-2}]\right)$", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel(r"Photon force [$\mathrm{N}\times 10^{-26}$]", fontsize=AXIS_LABEL_SIZE)
ax.set_title("Photon force vs column density (Na I) for different Doppler widths", fontsize=TITLE_SIZE)
ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

ax.grid(True, which="both", alpha=0.3)
ax.legend(ncol=2, fontsize=LEGEND_SIZE)

fig.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.show()
