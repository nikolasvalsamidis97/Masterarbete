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

############################################################################################
######################################### KODGUIDE #########################################
# 1. Hämtar molekyldata
Na = Atom('Na I', 1000 * u.AA, 9000*u.AA)

print(Na.data)
