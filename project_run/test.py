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
Na = Molecule('Na I', 1000 * u.AA, 9000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s        # b = v_D
vlim = 10 * u.km/u.s
Npts = 1000
Na_broadening = BroadeningProfile(Na, b , vlim, Npts, 'Voigt')
vsini = 13 * u.km / u.s
epsilon = 0 * u.dimensionless_unscaled

# 3. Hämta teoretiskt stjärnspectra
star = Star('TS/models_1769507931/bt-nextgen-agss2009/lte057-4.0-3.0a+0.4.BT-NextGen.7.dat.txt', 
            const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini, epsilon)

star.print_header()

flux_star_rot = star.flux_star_rot
flux_star_unrot = star.flux_star_unrot
lam_star = star.lam_star

distance = 0.1 * u.au
Temp = np.linspace(100, 1000, 10) * u.K
#Temp = 1000 * u.K
Ncol = np.logspace(7, 20, 100) * u.cm**(-2)
#Ncol = 0 * u.cm**(-2)

Na_Ph = PhotonPressure(Na_broadening, star)
Na_ph_calc, _, _, _ = Na_Ph.calc_PhotonPressure(Ncol, Temp, distance)
beta, _ = Na_Ph.beta_Values(Na_ph_calc, 0)

