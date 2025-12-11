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
Na = Molecule('Na', 100 * u.AA, 9000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s
vlim = 10 * u.km/u.s
Npts = 1000
Na_broadening = BroadeningProfile(Na, b , vlim, Npts, 'Voigt')

# 3. Hämta teoretiskt stjärnspectra
star = Star('TS/models_1758706196/bt-nextgen-agss2009/lte063-1.0-0.0a+0.0.BT-NextGen.7.dat.xml', 1*u.au, const.R_sun.value * u.m, const.M_sun.value * u.kg)

# 4. Skapa object för strålningstryck
Na_Ph_100K = PhotonPressure(100 * u.K, Na_broadening, star)
Na_Ph_1000K = PhotonPressure(1000 * u.K, Na_broadening, star)
Na_Ph_10000K = PhotonPressure(10000 * u.K, Na_broadening, star)

print(Na_Ph_100K.calc_PhotonPressure(0 * u.cm**(-2))[0],
      Na_Ph_1000K.calc_PhotonPressure(0 * u.cm**(-2))[0],
      Na_Ph_10000K.calc_PhotonPressure(0 * u.cm**(-2))[0])


