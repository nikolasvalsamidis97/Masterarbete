from project_func.errors import _not_quantity
from astropy import units as u
from astropy import constants as const
import numpy as np

class Planet:
  def __init__(self, radius, mass, distance, T, mu, n0):
    self.radius = radius.to(u.m) if isinstance(radius, u.Quantity) else _not_quantity("radius")
    self.mass = mass.to(u.kg) if isinstance(mass, u.Quantity) else _not_quantity("mass")
    self.distance = distance.to(u.m) if isinstance(distance, u.Quantity) else _not_quantity("distance")
    self.T = T.to(u.K) if isinstance(T, u.Quantity) else _not_quantity("T")
    self.mu = mu.to(u.dimensionless_unscaled) if isinstance(mu, u.Quantity) else _not_quantity("mu")
    self.n0 = n0.to(1 / u.cm**3) if isinstance(n0, u.Quantity) else _not_quantity("n0")

  def gravity(self):
    """Surface gravity assuming constant g."""
    return (const.G * self.mass / self.radius**2).to(u.cm / u.s**2)

  def scale_height(self):
    """Isothermal scale height."""
    m_particle = self.mu * const.u
    H = (const.k_B * self.T / (m_particle * self.gravity())).to(u.cm)
    return H

  def number_density(self, z):
    """Number density at height z."""
    z = z.to(u.cm) if isinstance(z, u.Quantity) else _not_quantity("z")
    H = self.scale_height()
    return (self.n0 * np.exp(-(z / H).decompose().value)).to(1 / u.cm**3)

  def slant_column_density(self, z):
    """
    Slant column density along a side‑on stellar ray.

    Geometry:
    The ray passes the planet with impact parameter

      b = R_p + z

    and we integrate the density along the path coordinate s:

      N = ∫ n(r(s)) ds

    where

      r(s) = sqrt(b^2 + s^2)

    and

      z(s) = r(s) − R_p

    """

    z = z.to(u.cm) if isinstance(z, u.Quantity) else _not_quantity("z")

    Rp = self.radius
    b = Rp + z

    # integrate out to several scale heights where density becomes negligible
    H = self.scale_height()
    s_max = 10 * H

    s = np.linspace(-s_max.value, s_max.value, 2000) * s_max.unit

    r = np.sqrt(b**2 + s**2)
    z_local = r - Rp

    n = self.number_density(z_local)

    N = np.trapz(n, s)

    return N.to(1 / u.cm**2)