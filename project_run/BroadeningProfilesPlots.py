import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Voigt1D

# ------------------ Grid and basic params ------------------
# Physical velocity grid [km/s]
v = np.linspace(-5, 5, 1000)
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

phi_G = gaussian(v)
phi_L = lorentzian(v)

# Voigt profile using Voigt1D with same FWHM scale
voigt_model = Voigt1D(x_0=0, amplitude_L=1,
                      fwhm_L=fwhm_common, fwhm_G=fwhm_common)
phi_V_raw = voigt_model(v)
phi_V = phi_V_raw / np.max(phi_V_raw)

# ------------------ Plot ------------------
plt.figure(figsize=(8, 5))

plt.plot(x, phi_G, linewidth=0.9, color='brown',        label="Gaussian")
plt.plot(x, phi_L, linewidth=0.9, color='darkgoldenrod', label="Lorentzian")
plt.plot(x, phi_V, linewidth=0.9, color='lightseagreen', label="Voigt (Voigt1D)")

plt.xlabel(r"$\mathrm{FWHM}$")
plt.ylabel(r"Broadening profiles $\phi(v)$")
plt.title("Gaussian, Lorentzian and Voigt profiles")
plt.legend()
plt.yticks([])

plt.tight_layout()
plt.savefig("Plots/BroadeningProfiles.pdf")
plt.show()
