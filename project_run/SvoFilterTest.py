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
Npts = 150
Na_broadening = BroadeningProfile(Na, b , Npts, 'Voigt')
vsini = 13 * u.km / u.s
epsilon = 0 * u.dimensionless_unscaled

# 3. Hämta teoretiskt stjärnspectra
star = Star('TS/models_1769507931/bt-nextgen-agss2009/lte057-4.0-3.0a+0.4.BT-NextGen.7.dat.txt', 
            const.R_sun.value * u.m, const.M_sun.value * u.kg, vsini, epsilon)

d_to_object = 10 * u.pc

# β-pic example for an comparison with observed magnitudes
targets = {
  "2MASS": {
    "J": 3.669, "H": 3.544, "Ks": 3.526
    },
  "Gaia":  {
    "G": 3.823242, "Gbp": 3.921584, "Grp": 3.660152
    },
  # "SDSS":  {
  #   "u": 12.5, "g": 11.5, "r": 10.8, "i": 10.5, "z": 10.3
  #   },
  "TESS":  {
    "TESS": 3.82
  },
}

MAGSYS = {
  "2MASS": "vegamag",
  "Gaia":  "vegamag",
  # "SDSS":  "abmag",
  "TESS":  "vegamag",
}

PHOTCALID = {
  "2MASS": {
    "J":  "2MASS/2MASS.J/Vega",
    "H":  "2MASS/2MASS.H/Vega",
    "Ks": "2MASS/2MASS.Ks/Vega",
  },
  "Gaia": {
    "G":  "GAIA/GAIA3.G/Vega",
    "Gbp": "GAIA/GAIA3.Gbp/Vega",
    "Grp": "GAIA/GAIA3.Grp/Vega",
  },
  # "SDSS": {
  #   "u": "SLOAN/SDSS.u/AB",
  #   "g": "SLOAN/SDSS.g/AB",
  #   "r": "SLOAN/SDSS.r/AB",
  #   "i": "SLOAN/SDSS.i/AB",
  #   "z": "SLOAN/SDSS.z/AB",
  # },
  "TESS": {
    "TESS": "TESS/TESS.Red/Vega",
  },
}

print("Old stellar radius before scaling:", star.radius.to(u.R_sun))

k_vals = star.scale_factors_from_targets(targets, d_to_object, magsys=MAGSYS, use_rot=True, photcalid_map=PHOTCALID)
print("Scale factors k for each band:")
for survey, k in k_vals.items():
  print(f"  {survey}: {k}")

print(f"New stellar radius after scaling: {star.radius.to(u.R_sun)}")