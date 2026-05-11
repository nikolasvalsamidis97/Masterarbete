import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt


OUTPUT_PATH = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Broadening plots" / "RotBroadPlot.pdf"
# Closest local replacement for the older unavailable 5700 K BT-NextGen file.
SPECTRUM_PATH = PROJECT_ROOT / "TS" / "Spectral_type" / "F" / "F4" / "lte066-4.0-0.0a+0.2.BT-NextGen.7.dat.txt"
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 13
Y_SCALE = 1e7

############################################################################################
######################################### KODGUIDE #########################################
# 1. Hämtar molekyldata
Na = Atom('Na I', 1000 * u.AA, 9000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s        # b = v_D
vlim = 10 * u.km/u.s
Npts = 1000
Na_broadening = BroadeningProfile(Na, b, Npts, 'Voigt')
vsini = 10 * u.km / u.s
vsini2 = 20 * u.km / u.s
epsilon = 0.0 * u.dimensionless_unscaled
epsilon2 = 0.9 * u.dimensionless_unscaled

# 3. Hämta teoretiskt stjärnspectra
star = Star(str(SPECTRUM_PATH),
             const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini, epsilon)
star2 = Star(str(SPECTRUM_PATH), 
             const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini, epsilon2)

star3 = Star(str(SPECTRUM_PATH), 
             const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini2, epsilon)
star4 = Star(str(SPECTRUM_PATH), 
             const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini2, epsilon2)

flux_star_unrot = star.flux_star_unrot
lam_star = star.lam_star

flux_star_rot = star.flux_star_rot
flux_star_rot2 = star2.flux_star_rot

flux_star_rot3 = star3.flux_star_rot
flux_star_rot4 = star4.flux_star_rot

Na_5891 = 5891.583264
width = 0.8
alpha = 0.5

plt.figure(figsize=[8,5])
plt.title(rf"Rotationally broadened spectra ({Na.species}: {Na_5891:.4f} {lam_star.unit.to_string('latex_inline')})")

plt.plot(lam_star, flux_star_unrot / Y_SCALE, color="black", label="Stellar flux")

plt.plot(lam_star, flux_star_rot / Y_SCALE, color="red",label=rf"$vsini$ = {vsini}, $\epsilon$ = {epsilon}")
plt.plot(lam_star, flux_star_rot2 / Y_SCALE, color="red", linestyle = ":", alpha=alpha,label=rf"$vsini$ = {vsini}, $\epsilon$ = {epsilon2}")

plt.plot(lam_star, flux_star_rot3 / Y_SCALE, color="blue",label=rf"$vsini$ = {vsini2}, $\epsilon$ = {epsilon}")
plt.plot(lam_star, flux_star_rot4 / Y_SCALE, color="blue", linestyle = ":", alpha=alpha, label=rf"$vsini$ = {vsini2}, $\epsilon$ = {epsilon2}")


plt.xlabel(rf"Wavelength [{lam_star.unit.to_string('latex_inline')}]", fontsize=AXIS_LABEL_SIZE)
plt.ylabel(rf"Stellar flux $\times 10^{{7}}$ [{flux_star_rot.unit.to_string('latex_inline')}]", fontsize=AXIS_LABEL_SIZE)
plt.xlim(Na_5891-width, Na_5891+width)
plt.ylim(0.2, 1.6)
plt.xticks(
    [5890.8, 5891.58, 5892.4],
    ["5890.8", "5891.58", "5892.4"],
    fontsize=TICK_LABEL_SIZE,
)
plt.yticks(fontsize=TICK_LABEL_SIZE)
plt.legend()
plt.savefig(OUTPUT_PATH)
plt.show()
