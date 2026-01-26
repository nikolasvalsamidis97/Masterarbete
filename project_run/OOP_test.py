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
vsini = 13 * u.km / u.s
epsilon = 0.6 * u.dimensionless_unscaled
star = Star('TS/models_1758706196/bt-nextgen-agss2009/lte063-1.0-0.0a+0.0.BT-NextGen.7.dat.xml', const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini, epsilon)

# 4. Skapa object för strålningstryck
atm_Temp = 300 * u.K
Na_Ph = PhotonPressure(Na_broadening, star)

Ncols = np.logspace(7, 25, 100) * u.cm**-2

