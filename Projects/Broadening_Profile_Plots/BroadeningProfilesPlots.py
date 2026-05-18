import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Voigt1D


OUTPUT_PATH = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Broadening plots" / "BroadeningProfiles.pdf"
TITLE_SIZE = 17
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 14

# ------------------ Grid and basic params ------------------
# Physical velocity grid [km/s]
v = np.linspace(-10, 10, 10000)
fwhm_common = 1.0    # common FWHM in km/s

# Dimensionless x-axis: (v - v0)/FWHM
x = (v) / fwhm_common

# ------------------ Profiles ------------------
def gaussian(v):
    fac = ((2*np.sqrt(np.log(2)))/ fwhm_common)
    return np.exp(-(fac * v)**2)

def lorentzian(v):
    gamma = fwhm_common / 2.0
    return 1.0 / (1.0 + (v / gamma)**2)

def measure_fwhm(v, y):
    """Return FWHM of a 1D profile y(v)."""
    y = y / np.max(y)
    half = 0.5

    # indices around the maximum
    i_max = np.argmax(y)

    # left half-maximum
    i1 = np.where(y[:i_max] < half)[0][-1]
    i2 = i1 + 1
    v_left = np.interp(half, y[i1:i2+1][::-1], v[i1:i2+1][::-1])

    # right half-maximum
    j1 = i_max + np.where(y[i_max:] < half)[0][0] - 1
    j2 = j1 + 1
    v_right = np.interp(half, y[j1:j2+1], v[j1:j2+1])

    return v_right - v_left

phi_G = gaussian(v)
phi_L = lorentzian(v)

fwhm_V = fwhm_common / (np.sqrt(1 + (8 * np.pi**2 * np.log(2))))

print(fwhm_V)
# Voigt profile using Voigt1D with same FWHM scale
voigt_model = Voigt1D(x_0=0, amplitude_L=1,
                      fwhm_L=fwhm_common, fwhm_G= fwhm_common)
phi_V_raw = voigt_model(v)
phi_V = phi_V_raw / np.max(phi_V_raw)

fwhm_voigt = measure_fwhm(v, phi_V)

print(fwhm_voigt)
x_V = v * (fwhm_common / fwhm_voigt)  # rescaled Voigt x

# ------------------ Plot ------------------
plt.figure(figsize=(8, 5))

plt.plot(x, phi_G, linewidth=0.9, color='brown',        label="Gaussian")
plt.plot(x, phi_L, linewidth=0.9, color='darkgoldenrod', label="Lorentzian")
plt.plot(x_V, phi_V, linewidth=0.9, color='lightseagreen', label="Voigt (Voigt1D)")

plt.xlabel(r"$\mathrm{FWHM}$", fontsize=AXIS_LABEL_SIZE)
plt.ylabel(r"Broadening profiles $\phi(v)$", fontsize=AXIS_LABEL_SIZE)
plt.title("Gaussian, Lorentzian and Voigt profiles", fontsize=TITLE_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.yticks([])
plt.xticks(fontsize=TICK_LABEL_SIZE)

plt.xlim(-4, 4)

plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.show()
