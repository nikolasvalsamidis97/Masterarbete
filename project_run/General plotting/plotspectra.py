import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
TITLE_SIZE = 14
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 13
OUTPUT_PATH = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Spectra" / "Bt-NextGen_spectrum.pdf"
BT_NEXTGEN_PATH = "TS/Spectral_type/A/A6/lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt"
vsini = 130 * u.km / u.s
epsilon = 0.5 * u.dimensionless_unscaled                                                              # As in Fernandez et al. 2006 (e.g., Gray 1976)
star = Star(BT_NEXTGEN_PATH, 
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
r_old = star.radius
k_vals, lam_pivot = star.scale_factors_from_targets(targets_pic, d_earth_to_pic, MAGSYS_pic, PHOTCALID_pic, use_rot=True)
r_new = star.radius
print(f"Old radius: {r_old.to(u.R_sun):.3f}, New radius: {r_new.to(u.R_sun):.3f}")

plt.figure(figsize=(10, 4))
plt.plot(star.lam_star, star.flux_star_rot, label="Model spectrum", linewidth=0.5, color="black")
ax = plt.gca()
plt.xlabel(f"Wavelength [{star.lam_star.unit.to_string('latex_inline')}]", fontsize=AXIS_LABEL_SIZE)
plt.ylabel(rf"Flux [{star.flux_star_rot.unit.to_string('latex_inline')}]", fontsize=AXIS_LABEL_SIZE)
plt.title("BT-NextGen (T=8000 K, log(g)=4.0, [Fe/H]=0.0)", fontsize=TITLE_SIZE)
plt.xlim(1500, 10000)
ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH)
plt.show()
