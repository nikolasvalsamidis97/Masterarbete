import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Atom import Atom
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
beta_values_Fernandez = {
    "H I": (1.6e-3, 0.1e-3),
    "He I": (0.0, 0.0),

    "Li I": (900, 40),
    "Be I": (62, 7),
    "Be II": (124, 6),

    "B I": (30, 10),
    "B II": (0.07, 0.04),
    "B III": (19, 1),

    "C I": (3.3e-2, 0.1e-2),
    "C II": (2.3e-3, 0.2e-3),
    "C III": (8.5e-6, 0.9e-6),

    "N I": (2.1e-4, 0.1e-4),
    "N II": (7.5e-6, 0.5e-6),
    "N III": (7.0e-6, 1.0e-6),

    "O I": (3.3e-4, 0.2e-4),
    "O II": (3.1e-9, 0.7e-9),
    "O III": (6.5e-7, 0.6e-7),

    "F II": (3.5e-6, 0.9e-6),
    "F III": (5.0e-9, 1.0e-9),

    "Ne III": (9.0e-8, 2.0e-8),

    "Na I": (360, 20),

    "Mg I": (74, 8),
    "Mg II": (9, 2),
    "Mg III": (0.0, 0.0),

    "Al I": (53, 6),
    "Al II": (0.36, 0.05),
    "Al III": (12, 1),

    "Si I": (6.0, 0.6),
    "Si II": (9, 9),
    "Si III": (5.8e-4, 0.6e-4),

    "P I": (3.4, 0.6),
    "P II": (2.2e-3, 0.3e-3),
    "P III": (5.0e-4, 2.0e-4),

    "S I": (0.56, 0.09),
    "S II": (9.0e-5, 1.0e-5),
    "S III": (2.0e-4, 1.0e-4),

    "Cl I": (2.3e-3, 0.4e-3),
    "Cl II": (3.7e-7, 0.4e-7),
    "Cl III": (3.0e-6, 2.0e-6),

    "Ar I": (1.7e-6, 0.3e-6),
    "Ar III": (1.5e-7, 0.2e-7),

    "K I": (200, 20),
    "K III": (4.4e-4, 0.2e-4),

    "Ca I": (330, 40),
    "Ca II": (50, 10),

    "Sc I": (220, 20),
    "Sc II": (1.3e3, 0.4e3), 
    "Sc III": (9.0e-2, 3.0e-2),

    "Ti I": (97, 5),
    "Ti II": (28, 2),
    "Ti III": (5.0e-4, 0.1e-4),

    "V I": (72, 4),
    "V II": (4.4, 0.2),

    "Cr I": (93, 5),
    "Cr II": (6.0e-7, 3.0e-7),
    # "Cr III": ... excluded

    "Mn I": (28, 3),
    "Mn II": (7, 1),
    # "Mn III": ... excluded

    "Fe I": (27, 2),
    "Fe II": (5.0, 0.3),
    "Fe III": (3.0e-7, 0.6e-7),

    "Co I": (16, 1),
    "Co III": (4.0e-7, 2.0e-7),

    "Ni I": (26, 2),
    "Ni II": (7.0e-2, 2.0e-2),
    "Ni III": (3.0e-7, 2.0e-7),
}
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Creating an atom with line data for Na I. (All available lines from NIST in the range 150-50000 Å)
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
all_atoms_list = [
  "H I",
  "He I","He II",
  "Li I","Li II","Li III",
  "Be I","Be II","Be III",
  "B I","B II","B III",
  "C I","C II","C III",
  "N I","N II","N III",
  "O I","O II","O III",
  "F I","F II","F III",
  "Ne I","Ne II","Ne III",
  "Na I","Na II","Na III",
  "Mg I","Mg II","Mg III",
  "Al I","Al II","Al III",
  "Si I","Si II","Si III",
  "P I","P II","P III",
  "S I","S II","S III",
  "Cl I","Cl II","Cl III",
  "Ar I","Ar II","Ar III",
  "K I","K II","K III",
  "Ca I","Ca II","Ca III",
  "Sc I","Sc II","Sc III",
  "Ti I","Ti II","Ti III",
  "V I","V II","V III",
  "Cr I","Cr II", #"Cr III" --- IGNORE ---
  "Mn I","Mn II", #"Mn III" --- IGNORE ---
  "Fe I","Fe II", "Fe III",
  "Co I","Co II", "Co III",
  "Ni I","Ni II", "Ni III",
]
wav_min = 50 * u.AA
wav_max = 50000 * u.AA

atoms = {sp: Atom(sp, wav_min, wav_max) for sp in all_atoms_list}


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
d_earth_to_pic = 19.3 * u.pc                                                                               # Distance to β Pic

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

pps_obj = {sp: PhotonPressure(broad_prof, beta_pic) for sp, broad_prof in broadening_profiles.items()}

Temp_atm = [1] *u.K
Ncol = [0] * u.cm**(-2)
d_atom_to_pic = 100 * u.au
chunk_size = 1

pps = {sp: pp.calc_PhotonPressure(Ncol, Temp_atm, d_atom_to_pic, chunk_size=chunk_size) for sp, pp in pps_obj.items()}
beta_vals = {sp: pp.beta_Values(*pps[sp][:2], d_atom_to_pic) for sp, pp in pps_obj.items()}

common = [k for k in beta_values_Fernandez.keys() if k in beta_vals]

my_beta = np.array([beta_vals[k][0].to_value(u.dimensionless_unscaled).ravel()[0] for k in common], dtype=float)
my_err  = np.array([beta_vals[k][1].to_value(u.dimensionless_unscaled).ravel()[0] for k in common], dtype=float)

import pandas as pd
import math

# Long-form table in memory
_df = pd.DataFrame({
  "Ion": common,
  "beta": my_beta,
  "beta_err": my_err,
})

ncol = 4
rows_per_col = math.ceil(len(_df) / ncol)

blocks = []
for i in range(ncol):
    block = _df.iloc[i*rows_per_col:(i+1)*rows_per_col].reset_index(drop=True)
    # Keep identical headers for each 3-column block (so Keynote shows Ion, β, δβ in every block)
    block.columns = ["Ion", "β", "δβ"]
    blocks.append(block)

wide = pd.concat(blocks, axis=1)

outpath = "Tables/beta_thiswork_my_model.csv"
wide.to_csv(outpath, index=False, sep=";", float_format="%.2f")
print("Saved:", outpath)
