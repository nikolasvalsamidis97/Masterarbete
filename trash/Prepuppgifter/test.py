from astropy import constants as const
from astropy import units as u
import numpy as np

sig = 1 * u.cm**2 * u.AA
Flux = 1 * u.erg / u.cm**2 / u.s / u.AA
sol = 1 * u.cm / u.s

F = (sig * Flux) / sol

print(F.to(u.N))