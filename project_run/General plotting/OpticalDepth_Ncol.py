import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
import numpy as np


OUTPUT_PATH = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Broadening plots" / "BroadeningProfiles_combined.pdf"
SUPTITLE_SIZE = 17
PANEL_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14

# 1. Hämtar molekyldata
Na = Atom('Na I', 5800 * u.AA, 6000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
bval = [0.1, 0.3] * u.km/u.s
colors = ["black", "black"]
vlim = 2 * u.km/u.s
Npts = 1000

## 2.5 Möjlighet att plotta profil och tvärsnitt för en linje. För att se linje: print(Na.data)
#print(Na.data)
line = 0
domain1 = 'velocity'
domain2 = 'wavelength'

Ncols = np.logspace(8, 16, 10) * u.cm**-2

def normIntensity(N):
  return np.exp(-(N*sig))

from matplotlib.ticker import MaxNLocator

fig, axes = plt.subplots(1, len(bval), figsize=(10, 4), sharey=True)

# Common title for the whole figure
fig.suptitle(r"Voigt-broadened normalized intensity", fontsize=SUPTITLE_SIZE)

for ax, b, color in zip(axes, bval, colors):
    alpha = 1
    Na_broadening = BroadeningProfile(Na, b, Npts, 'Voigt')
    sig = Na_broadening.sigmaArray_sym[line, :]
    v   = Na_broadening.v_grid_sym[0, :]

    for N in Ncols:
        alpha -= 0.09
        ax.plot(v, normIntensity(N), linewidth=0.7, color=color, alpha=alpha)

    # Subtitle for each panel: broadening parameter
    ax.set_title(rf"$\Delta v_D = {b}$", fontsize=PANEL_TITLE_SIZE)
    ax.set_xlabel(rf"Relative velocity {bval.unit}", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    # Fewer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(5))

# Shared y-label on the left
axes[0].set_ylabel(r"Normalized intensity $I = e^{-(N \, \sigma_v)}$", fontsize=AXIS_LABEL_SIZE)

# Let matplotlib handle spacing
fig.tight_layout()

fig.savefig(OUTPUT_PATH)
plt.show()
