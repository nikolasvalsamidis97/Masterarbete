import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Voigt1D

# ------------------ Grid and basic params ------------------
# Physical velocity grid [km/s]
v = np.linspace(-50, 50, 1000)
v0 = 0.0              # line centre [km/s]
fwhm_common = 10.0    # common FWHM in km/s

# Dimensionless x-axis: (v - v0)/FWHM
x = (v - v0) / fwhm_common

# ------------------ Profiles ------------------
def lorentzian(v, v0, fwhm):
    gamma = fwhm / 2.0  # HWHM
    return (1.0 / np.pi) * (gamma / ((v - v0)**2 + gamma**2))

def gaussian(v, v0, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # from FWHM
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(v - v0)**2 / (2.0 * sigma**2))

phi_G = gaussian(v, v0, fwhm_common)
phi_L = lorentzian(v, v0, fwhm_common)

# Voigt profile using Voigt1D with same FWHM scale
voigt_model = Voigt1D(x_0=v0, amplitude_L=1.0,
                      fwhm_L=fwhm_common, fwhm_G=fwhm_common)
phi_V_raw = voigt_model(v)

# Normalize Voigt to unit area
area_V = np.trapz(phi_V_raw, v)
phi_V = phi_V_raw / area_V

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
