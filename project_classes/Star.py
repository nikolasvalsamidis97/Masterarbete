from project_func.errors import _not_quantity
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.table import Table
from matplotlib import pyplot as plt


class Star:
  
  def __init__(self, path: str, distance, radius, mass, vsini, epsilon):
    """
    path:       str               Filepath for theoretical spectra
    distance:   Quantity          Distance to star
    radius:     Quantity          Radius of star
    mass:       Quantity          Mass of star
    vsini:      Quantity          Projected rotational velocity
    epsilon:    Quantity          Limb darkening
    """
    self.path = path
    self.distance = distance.to(u.au) if isinstance(distance, u.Quantity) else _not_quantity("distance")
    self.radius = radius.to(u.m) if isinstance(radius, u.Quantity) else _not_quantity("radius")
    self.mass = mass.to(u.kg) if isinstance(mass, u.Quantity) else _not_quantity("mass")
    self.vsini = vsini.to(u.km / u.s) if isinstance(vsini, u.Quantity) else _not_quantity("vsini")
    self.epsilon = epsilon.to(u.dimensionless_unscaled) if isinstance(epsilon, u.Quantity) else _not_quantity("epsilon")

    self.lam_star, self.flux_star = self.read_Spectra()

    
  def read_Spectra(self):
    """
    Reads a spectra from a file and returns the flux in vacuum
    """
    VOtab = Table.read(self.path, format='votable')

    lam = VOtab['WAVELENGTH'].value          #u.AA
    flux = VOtab['FLUX'].value              #(u.erg/u.s/(u.cm**2)/u.AA)
    flux = flux * (u.erg/u.s/(u.cm**2)/u.AA)
    lam = self.air_to_vacuum(lam) * u.AA

    lam, flux_rot = self.rot_kernel(lam, flux)

    omega = ((self.radius.to(u.m)/self.distance.to(u.m))**2).to_value(u.dimensionless_unscaled)
    flux = flux_rot * omega

    return lam, flux
  
  def air_to_vacuum(self, lam_air_A):
    s2 = (1e4/lam_air_A)**2
    n_minus_1 = 1e-8*(8342.13 + 2406030/(130 - s2) + 15997/(38.9 - s2))
    n = 1 + n_minus_1
    return lam_air_A * n
  
  def rot_kernel(self, lam, flux):
    vsini = self.vsini.to_value(u.km/u.s)
    eps = self.epsilon.to_value(u.dimensionless_unscaled)
    c_kms = const.c.to_value(u.km / u.s)
    
    lam_star = lam.to_value(u.AA)
    flux_star = flux.to_value(u.erg/u.s/(u.cm**2)/u.AA)

    mask = np.isfinite(lam_star) & (lam_star > 0)
    lam_star = lam_star[mask]
    flux_star = flux_star[mask]
    sort = np.argsort(lam_star)
    lam_star = lam_star[sort]
    flux_star = flux_star[sort]

    lnlam_star = np.log(lam_star)
    dlnlam_star = np.median(np.diff(lnlam_star)) / 4      # Double the resolution of the stars spectra
    N = int(np.floor((lnlam_star[-1] - lnlam_star[0]) / dlnlam_star)) + 1
    lnlam = lnlam_star[0] + dlnlam_star * np.arange(N)

    flux_star_interp = np.interp(lnlam, lnlam_star, flux_star)

    dv = (c_kms * dlnlam_star)
    half = int(np.ceil(vsini / dv))
    dv_axis = (np.arange(-half, half+1) * dv)              # The grid of dv the kernel will use in km/s

    x = dv_axis / vsini
    g = np.zeros_like(x)

    m = np.abs(x) <= 1
    xm = x[m]

    g[m] = (2*(1-eps)*np.sqrt(1-xm**2) + 0.5*np.pi*eps*(1-xm**2)) / (np.pi*vsini*(1-eps/3))

    g_weights = g * dv      
    g_weights /= g_weights.sum()

    flux_rot_interp = np.convolve(flux_star_interp, g_weights, mode="same")
    flux_rot_star = np.interp(lnlam_star, lnlam, flux_rot_interp)

    plt.plot(lam_star, flux_star)
    plt.plot(lam_star, flux_rot_star)
    plt.xlim(5890, 5900)
    plt.ylim(0, 1.5e7)
    plt.show()

    return lam_star * u.AA, flux_rot_star * u.erg/u.s/(u.cm**2)/u.AA


    

