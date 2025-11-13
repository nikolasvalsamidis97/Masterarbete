from project_func.errors import _not_quantity
from astropy import units as u
from astropy.table import Table


class Star:
  
  def __init__(self, path: str, distance, radius, mass):
    """
    path: str       Filepath for theoretical spectra
    """
    self.path = path
    self.distance = distance.to(u.au) if isinstance(distance, u.Quantity) else _not_quantity("distance")
    self.radius = radius.to(u.m) if isinstance(radius, u.Quantity) else _not_quantity("radius")
    self.lam_star, self.flux_star = self.read_Spectra()
    self.mass = mass.to(u.kg) if isinstance(mass, u.Quantity) else _not_quantity("mass")
    
  def read_Spectra(self):
    """
    Reads a spectra from a file and returns the flux in vacuum
    """
    VOtab = Table.read(self.path, format='votable')

    lam = VOtab['WAVELENGTH'].value          #u.AA
    flux = VOtab['FLUX'].value              #(u.erg/u.s/(u.cm**2)/u.AA)
    flux = flux * (u.erg/u.s/(u.cm**2)/u.AA)
    lam = self.air_to_vacuum(lam) * u.AA
    omega = (self.radius.to(u.m)/self.distance.to(u.m))**2
    flux *= omega

    return lam, flux 
  
  def air_to_vacuum(self, lam_air_A):
    s2 = (1e4/lam_air_A)**2
    n_minus_1 = 1e-8*(8342.13 + 2406030/(130 - s2) + 15997/(38.9 - s2))
    n = 1 + n_minus_1
    return lam_air_A * n
