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


Na = Molecule('Na', 5800 * u.AA, 6000*u.AA)
b = 1 * u.km/u.s
vlim = 10 * u.km/u.s
Npts = 1000

# Plot Gaussian
Na_broadening_Gauss = BroadeningProfile(Na, b , vlim, Npts, 'lorentz')

Na_broadening_Gauss.plot_Symmetric_Profile(5,"wavelength")

