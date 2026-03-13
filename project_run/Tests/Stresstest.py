import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Atom import Molecule
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
import numpy as np
from pytictoc import TicToc

t = TicToc()
t.tic()
############################################################################################
######################################### KODGUIDE #########################################
# 1. Hämtar molekyldata
Fe = Molecule('Fe I', 1700 * u.AA, 50000 * u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s        # b = v_D
Npts = 150
Fe_broadening = BroadeningProfile(Fe, b , Npts, 'Voigt')
vsini = 13 * u.km / u.s
epsilon = 0 * u.dimensionless_unscaled

N_lines, _ = Fe_broadening.sigmaArray.shape
print(f"Number of lines for {Fe.species}:  {N_lines}")

# 3. Hämta teoretiskt stjärnspectra
star = Star('TS/models_1769507931/bt-nextgen-agss2009/lte057-4.0-3.0a+0.4.BT-NextGen.7.dat.txt', 
            const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini, epsilon)

flux_star_rot = star.flux_star_rot
flux_star_unrot = star.flux_star_unrot
lam_star = star.lam_star

distance = 0.1 * u.au
Temp = np.linspace(100, 1000, 100) * u.K

Ncol = np.logspace(7, 30, 100) * u.cm**(-2)

Fe_Ph = PhotonPressure(Fe_broadening, star)

Fe_ph_calc, _, _, _ = Fe_Ph.calc_PhotonPressure(Ncol, Temp, distance, chunk_size=16)
beta, _ = Fe_Ph.beta_Values(Fe_ph_calc, 0)

N_lines, _, = Fe_Ph.crossection_sym.shape

t.toc()
print(f"For {N_lines} lines, {len(Temp)} temperatures an {len(Ncol)} column densities it took: {t.elapsed} seconds")

print(beta)

