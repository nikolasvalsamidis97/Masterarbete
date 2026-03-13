from project_func.errors import _not_quantity
from astropy import units as u
from astropy import constants as const
import numpy as np

from project_classes.Planet import Planet
from project_classes.Star import Star


class PlanetarySystem:
  def __init__(self, planet: Planet, star: Star, distance, eccentricity=0 * u.dimensionless_unscaled):
    self.planet = planet if isinstance(planet, Planet) else _not_quantity("planet")
    self.star = star if isinstance(star, Star) else _not_quantity("star")
    self.distance = distance.to(u.cm) if isinstance(distance, u.Quantity) else _not_quantity("distance")
    self.eccentricity = eccentricity.to(u.dimensionless_unscaled) if isinstance(eccentricity, u.Quantity) else _not_quantity("eccentricity")

  def gravity_equal_radius(self):
    """
    Radius where the planet's gravity equals the star's gravity.

    R_eq = a * (M_p / M_*)^(1/2)
    """
    return (self.distance * (self.planet.mass / self.star.mass)**(1/2)).to(u.cm)
  
  def max_height_gravity_equal(self):
    """
    Maximum atmospheric height before the planet's gravity equals the star's gravity.
    """
    return (self.gravity_equal_radius() - self.planet.radius).to(u.cm)
  
  def hill_radius(self):
    """
    Hill radius of the planet.

    R_H = a * (M_p / (3 M_*))^(1/3)
    """
    return (self.distance * (self.planet.mass / (3 * self.star.mass))**(1/3)).to(u.cm)

  def roche_lobe_radius(self):
    """
    Approximate Roche-lobe radius using the Eggleton formula.

    R_L / a = 0.49 q^(2/3) / [0.6 q^(2/3) + ln(1 + q^(1/3))]
    where q = M_p / M_*
    """
    q = (self.planet.mass / self.star.mass).decompose().value
    q23 = q**(2/3)
    q13 = q**(1/3)
    factor = 0.49 * q23 / (0.6 * q23 + np.log(1 + q13))
    return (factor * self.distance).to(u.cm)

  def max_height_hill(self):
    """
    Maximum atmospheric height before reaching the Hill radius.
    """
    return (self.hill_radius() - self.planet.radius).to(u.cm)

  def max_height_roche(self):
    """
    Maximum atmospheric height before reaching the Roche-lobe radius.
    """
    return (self.roche_lobe_radius() - self.planet.radius).to(u.cm)
  
  def z_grid_gravity_equal(self, n_z=10000):
    """
    Log-spaced height grid from the surface up to the gravity-equal radius.
    """
    z_max = self.max_height_gravity_equal().to(u.km)
    return np.logspace(0, np.log10(z_max.value), n_z) * u.km
    
  def z_grid_hill(self, n_z=10000):
    """
    Log-spaced height grid from the surface up to the Hill limit.
    """
    z_max = self.max_height_hill().to(u.km)
    return np.logspace(0, np.log10(z_max.value), n_z) * u.km

  def z_grid_roche(self, n_z=10000):
    """
    Log-spaced height grid from the surface up to the Roche-lobe limit.
    """
    z_max = self.max_height_roche().to(u.km)
    return np.logspace(0, np.log10(z_max.value), n_z) * u.km