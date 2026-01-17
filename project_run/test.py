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
Na = Molecule('Na I', 2400 * u.AA, 8100*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s        # b = v_D
vlim = 10 * u.km/u.s
Npts = 1000
Na_broadening = BroadeningProfile(Na, b , vlim, Npts, 'Voigt')

# 3. Hämta teoretiskt stjärnspectra
star = Star('TS/models_1758706196/bt-nextgen-agss2009/lte063-1.0-0.0a+0.0.BT-NextGen.7.dat.xml', 0.1*u.au, const.R_sun.value * u.m, const.M_sun.value * u.kg)



Temp = np.linspace(100, 20000, 100) * u.K
Ncol = 0 * u.cm**(-2)
beta = []

i = 0

Na_Ph = PhotonPressure(Na_broadening, star)
Na_ph_calc, _, _, _ = Na_Ph.calc_PhotonPressure(Ncol, Temp)

beta, _ = Na_Ph.beta_Values(Na_ph_calc, 0)
plt.plot(Temp, beta)
plt.show()