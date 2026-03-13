import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt

vsini = 130 * u.km / u.s
epsilon = 0.5 * u.dimensionless_unscaled                                                              # As in Fernandez et al. 2006 (e.g., Gray 1976)
star = Star('TS/models_1770121505/bt-nextgen-agss2009/lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt', 
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
plt.xlabel(f"Wavelength [{star.lam_star.unit.to_string('latex_inline')}]")
plt.ylabel(f"Flux [{star.flux_star_rot.unit.to_string('latex_inline')}]")
plt.title("BT-NextGen (T=8000 K, log(g)=4.0, [Fe/H]=0.0)")
plt.xlim(1500, 10000)
plt.tight_layout()
plt.savefig("Plots/Bt-NextGen_spectrum.pdf")
plt.show()