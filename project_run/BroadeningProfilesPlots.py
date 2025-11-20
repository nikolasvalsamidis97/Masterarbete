import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfile import BroadeningProfile
from astropy import units as u
from matplotlib import pyplot as plt
import numpy as np

Na   = Molecule('Na', 5800*u.AA, 6000*u.AA)
b    = 0.01 * u.km/u.s          # Doppler parameter
vlim = 10 * u.km/u.s
Npts = 1000
iline = 5                       # which line to plot

# --- profiles ---
prof_G = BroadeningProfile(Na, b, vlim, Npts, 'gauss')
prof_L = BroadeningProfile(Na, b, vlim, Npts, 'lorentz')
prof_V = BroadeningProfile(Na, b, vlim, Npts, 'voigt')

lam_G = prof_G.lam_sym[iline, :].to(u.AA).value
phi_G = prof_G.sigmaArray_sym[iline, :].value
lam_L = prof_L.lam_sym[iline, :].to(u.AA).value
phi_L = prof_L.sigmaArray_sym[iline, :].value
lam_V = prof_V.lam_sym[iline, :].to(u.AA).value
phi_V = prof_V.sigmaArray_sym[iline, :].value

lam0 = Na.lam0[iline][0].to(u.AA).value   # line-centre wavelength
lim = b.value/10

plt.figure(figsize=(8, 5)) 
plt.plot(lam_G, phi_G, lw=0.9, alpha=1, label='Gaussian', color = 'goldenrod')
plt.plot(lam_L, phi_L, lw=0.9, alpha=1, label='Lorentzian', color = 'teal')
plt.plot(lam_V, phi_V, lw=0.9, alpha=1, label='Voigt', color = 'darkred')
plt.xlabel(r'Wavelength $\lambda$ [Å]')
plt.ylabel(r'Normalized Profile $\phi(\lambda)$')
plt.title(rf'Broadening profiles for Na, b = {b}')
plt.xlim(lam0 - lim, lam0 + lim)       # only ±0.001 Å around line centre
plt.xticks([lam0-lim, lam0, lam0+lim], [rf"-{lim}", rf"$\lambda_0= {lam0}$", rf"+{lim}"])
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.savefig("Plots/BroadeningProfiles.pdf")
plt.show()


plt.figure(figsize=(8, 5))

