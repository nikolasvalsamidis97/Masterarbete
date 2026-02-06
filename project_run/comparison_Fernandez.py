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

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# This file compares the results from "Braking the gas in the β Pictoris debris disk" (Fernandez et al. 2006) with the results from this code.
# ------------------------------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Creating an atom with line data for Na I. (All available lines from NIST in the range 150-50000 Å)
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
neutral_atoms_list = [
  "Na I","Li I","Be I","B I","Mg I","Al I","Si I","Ca I","Sc I","Ti I","V I","Cr I","Mn I","Fe I","Co I","Ni I"
]
atoms = {sp: Molecule(sp, 150 * u.AA, 50000*u.AA) for sp in neutral_atoms_list}

for sp, atom in atoms.items():
  print(f"Number of lines for {sp}: {atom.lam0.shape[0]}")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Applying broadening to the atomic lines
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
b = 1 * u.km/u.s        # b = v_D
Npts = 150
broadening_profiles = {sp: BroadeningProfile(atom, b, Npts, 'Voigt') for sp, atom in atoms.items()}

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Creating synthetic star, using the same parameters as in Fernandez et al. 2006 (Teff = 8000 K, logg = 4.0, [Fe/H] = 0.0)
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
vsini = 130 * u.km / u.s
epsilon = 0.5 * u.dimensionless_unscaled                                                              # As in Fernandez et al. 2006 (e.g., Gray 1976)
beta_pic = Star('TS/models_1770121505/bt-nextgen-agss2009/lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt', 
               1.75*const.R_sun.value * u.m, 1.75*const.M_sun.value * u.kg, vsini, epsilon)
d_earth_to_pic = 19.44 * u.pc                                                                               # Distance to β Pic

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
r_old = beta_pic.radius
k_vals, lam_pivot = beta_pic.scale_factors_from_targets(targets_pic, d_earth_to_pic, MAGSYS_pic, PHOTCALID_pic, use_rot=True)
r_new = beta_pic.radius
print(f"Old radius: {r_old.to(u.R_sun):.3f}, New radius: {r_new.to(u.R_sun):.3f}")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Calculating the photon pressure for Na I and comparing with Fernandez et al. 2006
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
pps_obj = {sp: PhotonPressure(broad_prof, beta_pic) for sp, broad_prof in broadening_profiles.items()}

Temp_atm = [300] *u.K
Ncol = [0] * u.cm**(-2)
d_atom_to_pic = 100 * u.au
chunk_size = 1

pps = {sp: pp.calc_PhotonPressure(Ncol, Temp_atm, d_atom_to_pic, chunk_size=chunk_size) for sp, pp in pps_obj.items()}
for sp, (pp_calc, pp_err, _, _) in pps.items():
  print(f"{sp}: Photon Pressure = {pp_calc:} ± {pp_err}")  
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Calculating the beta values for Na I and comparing with Fernandez et al. 2006
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
beta_vals = {sp: pp.beta_Values(*pps[sp][:2], d_atom_to_pic) for sp, pp in pps_obj.items()}
for sp, (beta, beta_err) in beta_vals.items():
  print(f"{sp}: Beta = {beta} ± {beta_err}")
