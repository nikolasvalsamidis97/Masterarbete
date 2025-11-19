import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
import numpy as np

############################################################################################
######################################### KODGUIDE #########################################
# 1. Hämtar molekyldata
Na = Molecule('Na', 5800 * u.AA, 6000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s
vlim = 10 * u.km/u.s
Npts = 1000
Na_broadening = BroadeningProfile(Na, b , vlim, Npts, 'Voigt')
## 2.5 Möjlighet att plotta profil och tvärsnitt för en linje. För att se linje: print(Na.data)
#print(Na.data)
line = 5
domain1 = 'velocity'
domain2 = 'wavelength'
Na_broadening.plot_Symmetric_Profile(line ,domain1)
Na_broadening.plot_Symmetric_Crossection(line, domain2)

# 3. Hämta teoretiskt stjärnspectra
star = Star('TS/models_1758706196/bt-nextgen-agss2009/lte063-1.0-0.0a+0.0.BT-NextGen.7.dat.xml', 1*u.au, const.R_sun.value * u.m, const.M_sun.value * u.kg)

# 4. Skapa object för strålningstryck
Na_Ph = PhotonPressure(Na_broadening, star)

# 4.5 Exempelplott för strplningstryck som funktion av kolumndensitet
Ncols = np.logspace(7, 25, 100) * u.cm**-2

F_tot = []
F_tot_err = []
for N in Ncols:
    F_ph_tot, F_ph_tot_err, _, _ = Na_Ph.calc_PhotonPressure(N)
    F_tot.append(F_ph_tot.to(u.N).value)
    F_tot_err.append(F_ph_tot_err.to(u.N).value)

# x-axis values (unitless array for mpl)
x = Ncols.to(1/u.cm**2).value
y = np.array(F_tot)
yerr = np.array(F_tot_err)


plt.figure(figsize=(7,4))
plt.errorbar(x, y, yerr=yerr, fmt='-', capsize=3, lw=1.5)
plt.xscale('log')
plt.yscale('log')   # optional; use linear if you prefer
plt.xlabel(r'$N_{\rm col}\ [{\rm cm^{-2}}]$')
plt.ylabel('Photon force per absorber [N]')
plt.title('Photon force vs column density')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("Plots/F_ph_vs_Ncol.pdf")
plt.show()

