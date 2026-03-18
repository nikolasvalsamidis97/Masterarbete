import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import os
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Planet import Planet
from project_classes.Star import Star
from project_classes.PlanetarySystem import PlanetarySystem


# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Test photon pressure / beta as a function of atmospheric height
# ------------------------------------------------------------------------------------------------------------------------------------------------ #

# -- CREATE ATOM AND BROADENING PROFILE -- #
atom_species = [
  "H I",
  "He I", "He II",
  "N I", "N II", "N III",
  "O I", "O II", "O III",
  "Na I", "Na II", "Na III",
  "K I", "K II", "K III",
]

b = 1 * u.km / u.s
Npts = 150
wav_min = 150 * u.AA
wav_max = 50000 * u.AA
atoms = {species: Atom(species, wav_min, wav_max) for species in atom_species}
broad = {species: BroadeningProfile(atom, b, Npts, 'Voigt') for species, atom in atoms.items()}
# print("Defined atoms:", list(atoms.keys()))
# ---------------------------------------- #

# -- PLANET / ATMOSPHERE CASES -- #
planet_cases = {

  # Solar-system reference cases
  "Earth": {
    "radius": 1.0 * const.R_earth,
    "mass": 1.0 * const.M_earth,
    "T": 288 * u.K,
    "mu": 28.97 * u.dimensionless_unscaled,
    "P0": 1.0 * u.bar,
    "composition": {"N I": 0.7808, "O I": 0.2095, "He I": 5.24e-6, "H I": 5.5e-7, "Na I": 1e-9, "K I": 1e-10},
  },

  "Mars": {
    "radius": 3390 * u.km,
    "mass": 6.417e23 * u.kg,
    "T": 210 * u.K,
    "mu": 43.0 * u.dimensionless_unscaled,
    "P0": 6.0e-3 * u.bar,
    "composition": {"N I": 0.027, "O I": 0.0013, "He I": 1e-5, "H I": 1e-6},
  },

  "Mercury": {
    "radius": 2440 * u.km,
    "mass": 3.301e23 * u.kg,
    "T": 440 * u.K,
    "mu": 23.0 * u.dimensionless_unscaled,
    "P0": 1.0e-9 * u.bar,
    "composition": {"H I": 0.22, "He I": 0.06, "O I": 0.42, "Na I": 0.29, "K I": 0.01},
  },

  "Jupiter": {
    "radius": 69911 * u.km,
    "mass": 1.898e27 * u.kg,
    "T": 165 * u.K,
    "mu": 2.3 * u.dimensionless_unscaled,
    "P0": 1.0 * u.bar,
    "composition": {"H I": 0.898, "He I": 0.102},
  },

  "Pluto": {
    "radius": 1188.3 * u.km,
    "mass": 1.303e22 * u.kg,
    "T": 44 * u.K,
    "mu": 28.0 * u.dimensionless_unscaled,
    "P0": 1.0e-5 * u.bar,
    "composition": {"N I": 0.97, "H I": 0.01},
  },

  # Exoplanet benchmark cases
  "HD_209458_b": {
    "radius": 1.39 * const.R_jup,
    "mass": 0.73 * const.M_jup,
    "T": 1400 * u.K,
    "mu": 2.3 * u.dimensionless_unscaled,
    "P0": 1.0e-3 * u.bar,
    "composition": {"H I": 0.9, "He I": 0.1, "O I": 1e-4, "Na I": 1e-6, "K I": 1e-7},
  },

  "WASP_121_b": {
    "radius": 1.742 * const.R_jup,
    "mass": 1.17 * const.M_jup,
    "T": 2350 * u.K,
    "mu": 2.3 * u.dimensionless_unscaled,
    "P0": 1.0e-7 * u.bar,
    "composition": {"H I": 0.88, "He I": 0.09, "He II": 0.01, "O I": 0.015, "O II": 0.003, "Na I": 1.5e-3, "Na II": 3.5e-4, "K I": 1.2e-4, "K II": 3.0e-5},
  },

  "HAT_P_11_b": {
    "radius": 4.84 * const.R_earth,
    "mass": 25.0 * const.M_earth,
    "T": 880 * u.K,
    "mu": 2.3 * u.dimensionless_unscaled,
    "P0": 1.0e-3 * u.bar,
    "composition": {"H I": 0.85, "He I": 0.14, "O I": 1e-3, "Na I": 1e-5, "K I": 1e-6},
  },

  "K2_18_b": {
    "radius": 2.37 * const.R_earth,
    "mass": 8.92 * const.M_earth,
    "T": 300 * u.K,
    "mu": 2.3 * u.dimensionless_unscaled,
    "P0": 1.0e-5 * u.bar,
    "composition": {"H I": 0.8, "He I": 0.19, "O I": 0.01},
  },

  "55_Cnc_e": {
    "radius": 1.875 * const.R_earth,
    "mass": 7.99 * const.M_earth,
    "T": 2000 * u.K,
    "mu": 44.0 * u.dimensionless_unscaled,
    "P0": 1.0e-3 * u.bar,
    "composition": {"O I": 0.50, "O II": 0.15, "N I": 0.18, "N II": 0.02, "Na I": 0.08, "Na II": 0.02, "K I": 0.04, "K II": 0.01},
  },

}

planet_sources = {
  "Earth": "NASA Earth facts / JPL planetary parameters",
  "Mars": "NASA Mars facts",
  "Mercury": "NASA Mercury facts",
  "Jupiter": "NASA Jupiter facts",
  "Pluto": "NASA Pluto facts",
  "HD_209458_b": "NASA Exoplanet Catalog",
  "WASP_121_b": "NASA Exoplanet Catalog",
  "HAT_P_11_b": "NASA Exoplanet Catalog",
  "K2_18_b": "NASA Exoplanet Catalog",
  "55_Cnc_e": "NASA Exoplanet Catalog",
}

# Create Planet instances for each case exluding "distance" and "z_max" which are not needed for the Planet class
planets = {
  case: Planet(
    params["radius"],
    params["mass"],
    params["T"],
    params["mu"],
    params["P0"],
  )
  for case, params in planet_cases.items()
}
# print("Defined planets:", list(planets.keys()))
# -------------------------------- #

# -- STELLAR MODELS -- #
stellar_models = {

  "O0": {"path": "TS/Spectral_type/O/O0/lte500-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 12 * const.R_sun, "mass": 60 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O1": {"path": "TS/Spectral_type/O/O1/lte480-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 11 * const.R_sun, "mass": 50 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O2": {"path": "TS/Spectral_type/O/O2/lte460-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 10 * const.R_sun, "mass": 45 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O3": {"path": "TS/Spectral_type/O/O3/lte440-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 9.5 * const.R_sun, "mass": 40 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O4": {"path": "TS/Spectral_type/O/O4/lte420-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 9 * const.R_sun, "mass": 35 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O5": {"path": "TS/Spectral_type/O/O5/lte400-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 8.5 * const.R_sun, "mass": 30 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O6": {"path": "TS/Spectral_type/O/O6/lte380-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 8 * const.R_sun, "mass": 28 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O7": {"path": "TS/Spectral_type/O/O7/lte360-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 7.5 * const.R_sun, "mass": 25 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O8": {"path": "TS/Spectral_type/O/O8/lte340-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 7 * const.R_sun, "mass": 22 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},
  "O9": {"path": "TS/Spectral_type/O/O9/lte320-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 6.5 * const.R_sun, "mass": 20 * const.M_sun, "vsini": 100 * u.km/u.s, "epsilon": 0.3 * u.dimensionless_unscaled},

  "B0": {"path": "TS/Spectral_type/B/B0/lte300-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 6 * const.R_sun, "mass": 18 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B1": {"path": "TS/Spectral_type/B/B1/lte250-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 5.5 * const.R_sun, "mass": 14 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B2": {"path": "TS/Spectral_type/B/B2/lte220-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 5 * const.R_sun, "mass": 10 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B3": {"path": "TS/Spectral_type/B/B3/lte190-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 4.5 * const.R_sun, "mass": 8 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B4": {"path": "TS/Spectral_type/B/B4/lte170-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 4.3 * const.R_sun, "mass": 7 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B5": {"path": "TS/Spectral_type/B/B5/lte150-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 4.1 * const.R_sun, "mass": 6 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B6": {"path": "TS/Spectral_type/B/B6/lte140-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.9 * const.R_sun, "mass": 5.5 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B7": {"path": "TS/Spectral_type/B/B7/lte130-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.7 * const.R_sun, "mass": 5 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B8": {"path": "TS/Spectral_type/B/B8/lte120-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.4 * const.R_sun, "mass": 4 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B9": {"path": "TS/Spectral_type/B/B9/lte110-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.1 * const.R_sun, "mass": 3 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},

  "A0": {"path": "TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.5 * const.R_sun, "mass": 2.5 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A1": {"path": "TS/Spectral_type/A/A1/lte094-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.3 * const.R_sun, "mass": 2.4 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A2": {"path": "TS/Spectral_type/A/A2/lte090-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.2 * const.R_sun, "mass": 2.3 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A3": {"path": "TS/Spectral_type/A/A3/lte088-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.1 * const.R_sun, "mass": 2.2 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A4": {"path": "TS/Spectral_type/A/A4/lte086-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.0 * const.R_sun, "mass": 2.1 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A5": {"path": "TS/Spectral_type/A/A5/lte082-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.9 * const.R_sun, "mass": 2.0 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A6": {"path": "TS/Spectral_type/A/A6/lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.85 * const.R_sun, "mass": 1.95 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A7": {"path": "TS/Spectral_type/A/A7/lte078-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.8 * const.R_sun, "mass": 1.9 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A8": {"path": "TS/Spectral_type/A/A8/lte076-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.75 * const.R_sun, "mass": 1.85 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A9": {"path": "TS/Spectral_type/A/A9/lte074-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.7 * const.R_sun, "mass": 1.8 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},

  "F0": {"path": "TS/Spectral_type/F/F0/lte072-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.6 * const.R_sun, "mass": 1.6 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "F1": {"path": "TS/Spectral_type/F/F1/lte070-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.55 * const.R_sun, "mass": 1.55 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "F2": {"path": "TS/Spectral_type/F/F2/lte068-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.5 * const.R_sun, "mass": 1.5 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "F4": {"path": "TS/Spectral_type/F/F4/lte066-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 1.45 * const.R_sun, "mass": 1.45 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "F5": {"path": "TS/Spectral_type/F/F5/lte064-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.4 * const.R_sun, "mass": 1.4 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "F6": {"path": "TS/Spectral_type/F/F6/lte062-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.35 * const.R_sun, "mass": 1.35 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "F8": {"path": "TS/Spectral_type/F/F8/lte060-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 1.25 * const.R_sun, "mass": 1.25 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},

  "G1": {"path": "TS/Spectral_type/G/G1/lte058-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.05 * const.R_sun, "mass": 1.05 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "G4": {"path": "TS/Spectral_type/G/G4/lte056-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.98 * const.R_sun, "mass": 0.98 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "G6": {"path": "TS/Spectral_type/G/G6/lte054-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.93 * const.R_sun, "mass": 0.93 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "G8": {"path": "TS/Spectral_type/G/G8/lte052-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.88 * const.R_sun, "mass": 0.88 * const.M_sun, "vsini": 5 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},

  "K1": {"path": "TS/Spectral_type/K/K1/lte050-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.82 * const.R_sun, "mass": 0.82 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K3": {"path": "TS/Spectral_type/K/K3/lte048-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.78 * const.R_sun, "mass": 0.78 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K4": {"path": "TS/Spectral_type/K/K4/lte046-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.74 * const.R_sun, "mass": 0.74 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K5": {"path": "TS/Spectral_type/K/K5/lte044-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.70 * const.R_sun, "mass": 0.70 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K6": {"path": "TS/Spectral_type/K/K6/lte042-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.68 * const.R_sun, "mass": 0.68 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K7": {"path": "TS/Spectral_type/K/K7/lte040-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.65 * const.R_sun, "mass": 0.65 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K9": {"path": "TS/Spectral_type/K/K9/lte038-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.60 * const.R_sun, "mass": 0.60 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},

  "M1": {"path": "TS/Spectral_type/M/M1/lte036-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 0.50 * const.R_sun, "mass": 0.50 * const.M_sun, "vsini": 2 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "M2": {"path": "TS/Spectral_type/M/M2/lte034-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 0.45 * const.R_sun, "mass": 0.45 * const.M_sun, "vsini": 2 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "M4": {"path": "TS/Spectral_type/M/M4/lte032-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 0.35 * const.R_sun, "mass": 0.35 * const.M_sun, "vsini": 2 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "M6": {"path": "TS/Spectral_type/M/M6/lte030-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.25 * const.R_sun, "mass": 0.25 * const.M_sun, "vsini": 2 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "M8": {"path": "TS/Spectral_type/M/M8/lte028-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.15 * const.R_sun, "mass": 0.15 * const.M_sun, "vsini": 2 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "M9": {"path": "TS/Spectral_type/M/M9/lte026-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.10 * const.R_sun, "mass": 0.10 * const.M_sun, "vsini": 2 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},

}

stars = {stype: Star(**params) for stype, params in stellar_models.items()}
# print("Defined stellar models:", list(stars.keys()))
# -------------------- #

# -- System cases -- #
systems = {
  "Earth-Sun": {"planet": planets["Earth"], "star": stars["G1"], "distance": 1.0 * u.AU},
  "Mars-Sun": {"planet": planets["Mars"], "star": stars["G1"], "distance": 1.524 * u.AU},
  "Mercury-Sun": {"planet": planets["Mercury"], "star": stars["G1"], "distance": 0.387 * u.AU},
  "Jupiter-Sun": {"planet": planets["Jupiter"], "star": stars["G1"], "distance": 5.2 * u.AU},
  "Pluto-Sun": {"planet": planets["Pluto"], "star": stars["G1"], "distance": 39.0 * u.AU},
  "HD_209458_b": {"planet": planets["HD_209458_b"], "star": stars["G1"], "distance": 0.04707 * u.AU},
  "WASP_121_b": {"planet": planets["WASP_121_b"], "star": stars["F0"], "distance": 0.02571 * u.AU},
  "HAT_P_11_b": {"planet": planets["HAT_P_11_b"], "star": stars["K1"], "distance": 0.05258 * u.AU},
  "K2_18_b": {"planet": planets["K2_18_b"], "star": stars["M1"], "distance": 0.1429 * u.AU},
  "55_Cnc_e": {"planet": planets["55_Cnc_e"], "star": stars["G8"], "distance": 0.01544 * u.AU},
}
planetary_systems = {name: PlanetarySystem(**params) for name, params in systems.items()}
# print("Defined planetary systems:", list(planetary_systems.keys()))
# -------------------- #  

# -- CALCULATE AND PLOT BETA VS HEIGHT -- #
def calc_beta_vs_height(system_name, n_z=10000, z_max_type="hill"):

  system = planetary_systems[system_name]
  planet_key = next(name for name, planet_obj in planets.items() if planet_obj is system.planet)
  allowed_species = list(planet_cases[planet_key]["composition"].keys())

  if z_max_type == "grav":
    z = system.z_grid_gravity_equal(n_z=n_z)
  elif z_max_type == "hill":
    z = system.z_grid_hill(n_z=n_z)
  elif z_max_type == "roche":
    z = system.z_grid_roche(n_z=n_z)
  else:
    raise ValueError("z_max_type must be 'grav', 'hill', or 'roche'")

  Temp_atm = system.planet.T

  Ncol_z = np.array([
    system.planet.slant_column_density(zi).to_value(1 / u.cm**2) for zi in z
  ]) / u.cm**2

  beta_results = []

  for species in allowed_species:
    pp = PhotonPressure(broad[species], system.star)

    Fph, Fph_err, _, _ = pp.calc_PhotonPressure(Ncol_z, Temp_atm, system.distance)
    beta, beta_err = pp.beta_Values(Fph, Fph_err, system.planet.mass, z + system.planet.radius)

    beta_results.append({
      "system_name": system_name,
      "species": species,
      "z": z,
      "beta": beta,
      "beta_err": beta_err,
      "Fph": Fph,
      "Fph_err": Fph_err,
      "z_max_grav": system.max_height_gravity_equal().to(u.km),
      "z_max_roche": system.max_height_roche().to(u.km),
      "z_max_hill": system.max_height_hill().to(u.km),
    })

  return beta_results

def plot_beta_vs_height(beta_results, save_path=None):

  fig, ax = plt.subplots(figsize=(8, 5))

  # use the first result for shared x-grid and vertical limits
  z = beta_results[0]["z"]
  z_km = z.to_value(u.km)

  z_max_grav = beta_results[0]["z_max_grav"].to_value(u.km)
  z_max_roche = beta_results[0]["z_max_roche"].to_value(u.km)
  z_max_hill = beta_results[0]["z_max_hill"].to_value(u.km)

  for beta_result in beta_results:
    beta = beta_result["beta"]
    beta_err = beta_result["beta_err"]

    beta_y = beta[0].to_value(u.dimensionless_unscaled)
    beta_yerr = beta_err[0].to_value(u.dimensionless_unscaled)

    ax.plot(z_km, beta_y, linewidth=0.8, label=beta_result["species"])

    # ax.fill_between(
    #   z_km,
    #   np.maximum(beta_y - beta_yerr, 1e-300),
    #   beta_y + beta_yerr,
    #   alpha=0.12,
    #   linewidth=0
    # )

  ax.axvline(z_max_grav, linestyle='--', linewidth=1, label="Planet = Star gravity")
  ax.axvline(z_max_roche, linestyle='-.', linewidth=1, label="Roche-lobe limit")
  # ax.axvline(z_max_hill, linestyle=':', linewidth=1.5, label="Hill radius")
  ax.axhline(1.0, linestyle='-', color='gray', linewidth=1, label=r"$\beta = 1$")

  ax.set_xlim(z_km[0], z_max_hill)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Height [km]")
  ax.set_ylabel(r"$\beta$")
  ax.set_title(f"Beta vs height | {beta_results[0]['system_name']}")

  ax.legend(fontsize=8, ncol=2)
  fig.tight_layout()

  if save_path is not None:
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


# New function to save plots for all systems
def save_beta_plots_all_systems(system_names=None, n_z=10000, z_max_type="hill"):

  output_dir = "Plots/Atmospheric test/Beta_vs_height_system"
  os.makedirs(output_dir, exist_ok=True)

  if system_names is None:
    system_names = list(planetary_systems.keys())

  for system_name in system_names:
    beta_results = calc_beta_vs_height(system_name, n_z=n_z, z_max_type=z_max_type)
    save_path = os.path.join(output_dir, f"{system_name}_beta_vs_height.pdf")
    plot_beta_vs_height(beta_results, save_path=save_path)
    print(f"Saved plot: {save_path}")

save_beta_plots_all_systems(n_z=10000, z_max_type="hill")

# ----------------------------------------# 