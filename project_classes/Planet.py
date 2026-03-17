from project_func.errors import _not_quantity
from astropy import units as u
from astropy import constants as const
import numpy as np

class Planet:
  def __init__(self, radius, mass, T, mu, P0):
    self.radius = radius.to(u.m) if isinstance(radius, u.Quantity) else _not_quantity("radius")
    self.mass = mass.to(u.kg) if isinstance(mass, u.Quantity) else _not_quantity("mass")
    self.T = T.to(u.K) if isinstance(T, u.Quantity) else _not_quantity("T")
    self.mu = mu.to(u.dimensionless_unscaled) if isinstance(mu, u.Quantity) else _not_quantity("mu")
    self.P0 = P0.to(u.Pa) if isinstance(P0, u.Quantity) else _not_quantity("P0")
    self.n0 = (self.P0 / (const.k_B * self.T)).to(1 / u.m**3)

  def gravity(self, z=0 * u.cm):
    """Gravitational acceleration at height z above the planetary surface."""
    z = z.to(u.cm) if isinstance(z, u.Quantity) else _not_quantity("z")
    r = self.radius + z
    return (const.G * self.mass / r**2).to(u.cm / u.s**2)

  def scale_height(self, z=0 * u.cm):
    """Local isothermal scale height at height z."""
    z = z.to(u.cm) if isinstance(z, u.Quantity) else _not_quantity("z")
    m_particle = self.mu * const.u
    H = (const.k_B * self.T / (m_particle * self.gravity(z))).to(u.cm)
    return H

  def number_density(self, z):
    """
    Number density at height z for an isothermal hydrostatic atmosphere
    with distance-dependent gravity.

    Hydrostatic equation:
      dP/dr = -rho * G M / r^2

    With constant T and P = n k_B T, this integrates to:
      n(r) = n0 * exp[-A * (1/Rp - 1/r)]

    where
      A = mu * m_u * G M / (k_B T)
      r = Rp + z
    """
    z = z.to(u.cm) if isinstance(z, u.Quantity) else _not_quantity("z")

    r = self.radius + z
    m_particle = self.mu * const.u
    A = (m_particle * const.G * self.mass / (const.k_B * self.T)).to(u.cm)

    exponent = (-A * ((1 / self.radius) - (1 / r))).decompose().value
    return (self.n0 * np.exp(exponent)).to(1 / u.cm**3)

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
    H0 = self.scale_height(z)
    s_max = 10 * H0

    s = np.linspace(-s_max.value, s_max.value, 2000) * s_max.unit

    r = np.sqrt(b**2 + s**2)
    z_local = r - Rp

    n = self.number_density(z_local)

    N = np.trapz(n, s)

    return N.to(1 / u.cm**2)