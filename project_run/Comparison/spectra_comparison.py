import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

TITLE_SIZE = 17
AXIS_LABEL_SIZE = 19
TICK_LABEL_SIZE = 17
LEGEND_SIZE = 13
OUTPUT_PATH = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Comparison" / "Spectra_comparison_betapic_fern.pdf"
BT_NEXTGEN_PATH = "TS/Spectral_type/A/A6/lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt"
FERNANDEZ_PATH = "TS/Spectra/HRspec_A5V_130.dat"


def log10_exponent_label(value, _position):
    if not np.isfinite(value) or value <= 0:
        return ""

    exponent = np.log10(value)
    rounded = round(exponent)
    if not np.isclose(exponent, rounded, atol=1e-10):
        return ""
    return f"{int(rounded)}"

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Creating synthetic star, using the same parameters as in Fernandez et al. 2006 (Teff = 8000 K, logg = 4.0, [Fe/H] = 0.0)
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
vsini = 130 * u.km / u.s
epsilon = 0.5 * u.dimensionless_unscaled                                                              # As in Fernandez et al. 2006 (e.g., Gray 1976)
my_beta_pic = Star(BT_NEXTGEN_PATH, 
               1.75*const.R_sun.value * u.m, 1.75*const.M_sun.value * u.kg, vsini, epsilon)
d_earth_to_pic = 19.3 * u.pc                                                                              # Distance to β Pic


# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Calibration made with information from Tycho catalog and Fernandez et al. 2006
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
targets_pic = {
  "TYCHO": {
    "B": 4.056, "V": 3.870
  },
}
MAGSYS_pic = {
  "TYCHO": "vegamag",
}
PHOTCALID_pic = {
  "TYCHO": {
    "B": "TYCHO/TYCHO.B/Vega",
    "V": "TYCHO/TYCHO.V/Vega",
  },
}

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Calibration of the stellar radius
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
r_old = my_beta_pic.radius
k_vals, lam_pivot = my_beta_pic.scale_factors_from_targets(targets_pic, d_earth_to_pic, MAGSYS_pic, PHOTCALID_pic, use_rot=True)
r_new = my_beta_pic.radius
print(f"Old radius: {r_old.to(u.R_sun):.3f}, New radius: {r_new.to(u.R_sun):.3f}")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
alpha=5.24e-5
dist_for_spec=1 * const.au
fern_radius = r_new
fern_beta_pic = Star(FERNANDEZ_PATH,
               fern_radius, 1.75*const.M_sun, vsini, epsilon)
fern_beta_pic.convert_from_log10()
fern_beta_pic.flux_star_rot = fern_beta_pic.flux_star_unrot * alpha * (dist_for_spec / fern_beta_pic.radius)**2  # convert to surface flux

print("MY unit:", my_beta_pic.flux_star_rot.unit, "median:", np.nanmedian(my_beta_pic.flux_star_rot.value))
print("FERN unit:", fern_beta_pic.flux_star_rot.unit, "median:", np.nanmedian(fern_beta_pic.flux_star_rot.value))

plt.figure(figsize=(10, 4))
plt.plot(my_beta_pic.lam_star, my_beta_pic.flux_star_rot, label='BT-NextGen (β Pic-like)', linewidth=0.5, color="black")
plt.plot(fern_beta_pic.lam_star, fern_beta_pic.flux_star_rot, label='Fernandez et al. (2006) (β Pic)', linewidth=0.5, color="red", alpha=0.7)

ax = plt.gca()
ax.set_yscale("log")
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
ax.yaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
ax.yaxis.set_minor_formatter(NullFormatter())
plt.xlabel(rf"Wavelength [{my_beta_pic.lam_star.unit.to_string('latex_inline')}]", fontsize=AXIS_LABEL_SIZE)
plt.ylabel(rf"$\log_{{10}}$ Stellar flux [{my_beta_pic.flux_star_rot.unit.to_string('latex_inline')}]", fontsize=AXIS_LABEL_SIZE)
plt.xlim(1500, 10000)
ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE - 1)
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH)
plt.show()
