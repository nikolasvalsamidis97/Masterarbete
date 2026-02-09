import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Atom import Molecule
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt

############################################################################################
######################################### KODGUIDE #########################################
# 1. Hämtar molekyldata
Na = Molecule('Na I', 1000 * u.AA, 9000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s        # b = v_D
vlim = 10 * u.km/u.s
Npts = 1000
Na_broadening = BroadeningProfile(Na, b , vlim, Npts, 'Voigt')
vsini = 10 * u.km / u.s
vsini2 = 20 * u.km / u.s
epsilon = 0.0 * u.dimensionless_unscaled
epsilon2 = 0.9 * u.dimensionless_unscaled

# 3. Hämta teoretiskt stjärnspectra
star = Star('TS/models_1769507931/bt-nextgen-agss2009/lte057-4.0-3.0a+0.4.BT-NextGen.7.dat.txt',
             const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini, epsilon)
star2 = Star('TS/models_1769507931/bt-nextgen-agss2009/lte057-4.0-3.0a+0.4.BT-NextGen.7.dat.txt', 
             const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini, epsilon2)

star3 = Star('TS/models_1769507931/bt-nextgen-agss2009/lte057-4.0-3.0a+0.4.BT-NextGen.7.dat.txt', 
             const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini2, epsilon)
star4 = Star('TS/models_1769507931/bt-nextgen-agss2009/lte057-4.0-3.0a+0.4.BT-NextGen.7.dat.txt', 
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

plt.plot(lam_star, flux_star_unrot, color="black", label="Stellar flux")

plt.plot(lam_star, flux_star_rot, color="red",label=rf"$vsini$ = {vsini}, $\epsilon$ = {epsilon}")
plt.plot(lam_star, flux_star_rot2, color="red", linestyle = ":", alpha=alpha,label=rf"$vsini$ = {vsini}, $\epsilon$ = {epsilon2}")

plt.plot(lam_star, flux_star_rot3, color="blue",label=rf"$vsini$ = {vsini2}, $\epsilon$ = {epsilon}")
plt.plot(lam_star, flux_star_rot4, color="blue", linestyle = ":", alpha=alpha, label=rf"$vsini$ = {vsini2}, $\epsilon$ = {epsilon2}")


plt.xlabel(rf"Wavelength [{lam_star.unit.to_string('latex_inline')}]")
plt.ylabel(rf"Stellar flux [{flux_star_rot.unit.to_string('latex_inline')}]")
plt.xlim(Na_5891-width, Na_5891+width)
plt.ylim(0.2e7, 0.8e7)
plt.legend()
plt.savefig("Plots/RotBroadPlot.pdf")
plt.show()