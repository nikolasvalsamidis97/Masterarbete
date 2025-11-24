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

# 1. Hämtar molekyldata
Na = Molecule('Na', 5800 * u.AA, 6000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
bval = [0.1, 0.3] * u.km/u.s
colors = ["black", "black"]
vlim = 2 * u.km/u.s
Npts = 1000

## 2.5 Möjlighet att plotta profil och tvärsnitt för en linje. För att se linje: print(Na.data)
#print(Na.data)
line = 5
domain1 = 'velocity'
domain2 = 'wavelength'

Ncols = np.logspace(8, 16, 10) * u.cm**-2

def normIntensity(N):
  return np.exp(-(N*sig))


for i in range(len(bval)):
  alpha = 1
  b = bval[i]
  color = colors[i]

  plt.figure(figsize=(8, 8))
  plt.title(rf"Voigt-broadened normalized intensity, $\Delta v_D = ${b}")
  plt.xlabel(rf"Relative velocity {bval.unit}")

  Na_broadening = BroadeningProfile(Na, b , vlim, Npts, 'Voigt')
  sig = Na_broadening.sigmaArray_sym[line,:]
  v = Na_broadening.v_grid_sym[0,:]
  label = rf"\Delta v_D = {b}"
  
  for N in Ncols:
    alpha -= 0.08
    plt.plot(v, normIntensity(N), linewidth = 0.7,color = color, alpha = alpha)

  plt.ylabel(rf"Normalized intensity $I = e^{{-(N \, \sigma_v)}}$")
  plt.savefig(rf"Plots/BroadeningProfiles_Ncol_{b.value}.pdf")
  plt.show()





