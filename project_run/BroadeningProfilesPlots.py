import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfile import BroadeningProfile
from astropy import units as u
from matplotlib import pyplot as plt


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

lam0 = Na.lam0[iline].to(u.AA).value   # line-centre wavelength

plt.figure()
plt.plot(lam_G, phi_G, label='Gaussian', color = 'grey')
plt.plot(lam_L, phi_L, label='Lorentzian', color = 'lightgrey')
plt.plot(lam_V, phi_V, label='Voigt', color = 'black')
plt.xlabel(r'Wavelength $\lambda$ [Å]')
plt.ylabel(r'Profile $\phi(\lambda)$')
plt.title('Gaussian, Lorentzian and Voigt profiles')
plt.xlim(lam0 - 0.001, lam0 + 0.001)       # only ±0.1 Å around line centre
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
b_values = [0.01, 0.005, 0.001]   # km/s, broad → very narrow
#werer
for b_val in b_values:
    b_here = b_val * u.km/u.s
    prof_V = BroadeningProfile(Na, b_here, vlim, Npts, 'voigt')

    lam = prof_V.lam_sym[iline, :].to(u.AA).value
    phi = prof_V.sigmaArray_sym[iline, :].value

    plt.plot(lam, phi, label=fr'$b = {b_val:.3f}\,\mathrm{{km\,s^{{-1}}}}$')

plt.plot(lam_G, phi_G, label='Gaussian', color='grey')
plt.plot(lam_L, phi_L, label='Lorentzian', color='lightgrey')

plt.xlabel(r'Wavelength $\lambda$ [Å]')
plt.ylabel(r'Voigt profile $\phi_V(\lambda)$')
plt.title('Voigt profiles for different Doppler parameter $b$')
plt.xlim(lam0 - 0.001, lam0 + 0.001)       # zoom in
plt.legend()
plt.tight_layout()
plt.show()


