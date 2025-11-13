# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 08:43:20 2025

@author: Alexis Brandeker
"""

import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.modeling.physical_models import BlackBody


m_Na = 22.99*u.u
M_s = 1*u.M_sun


def Na_grav(distance):
    """Returns gravity forceon Na atoms from Sun, given distance
    """
    F_G = const.G * m_Na * M_s/distance**2

    return F_G.to(u.N)

# Gravity force
F_G01 = Na_grav(0.1*u.au)
F_G1 = Na_grav(1*u.au)
F_G10 = Na_grav(10*u.au)
 

# Cross sections (data from https://physics.nist.gov/PhysRefData/ASD/lines_form.html )

w_5890 = 589.1583*u.nm
gu_5890 = 4
gl_5890 = 2
A_5890 = 6.16e7/u.s

w_5896 = 589.7558*u.nm
gu_5896 = 2
gl_5896 = 2
A_5896 = 6.14e7/u.s

def cross(wavelength, Aul, gu, gl):
    sigma = Aul*wavelength**4*(gu/gl)/(8*np.pi*const.c)
    return sigma.to(u.cm**2 * u.AA)
    
sigma_5890 = cross(w_5890, A_5890, gu_5890, gl_5890)
sigma_5896 = cross(w_5896, A_5896, gu_5896, gl_5896)

# Solar flux at 590 nm; assume 5778 K black body

def flux_sun(distance, wavelength=590*u.nm):
    """Given distance from Sun, returns the flux assuming blackbody
    Wavelength is defaulted to 590 nm
    """
    bb = BlackBody(temperature=5778*u.K, scale=1*u.erg/u.cm**2/u.s/u.AA/u.sr)
    omega_sun = np.pi*(1*u.R_sun/distance)**2*u.sr
    flux = bb(wavelength)*omega_sun
    
    return flux.to(u.erg/u.cm**2/u.s/u.AA)


def Na_rad(distance):
    """Given distance to Sun, computes radiation force on Na atoms
    """
    F_R = (sigma_5890 + sigma_5896)/const.c*flux_sun(distance)
    return F_R.to(u.N)


# Radiation force

F_R01 = Na_rad(0.1*u.au)
F_R1 = Na_rad(1*u.au)
F_R10 = Na_rad(10*u.au)

print("At 0.1 AU: Gravity {:.2e}, Rad.force {:.2e}, beta={:.2f}".format(
    F_G01, F_R01, F_R01/F_G01))

print("At   1 AU: Gravity {:.2e}, Rad.force {:.2e}, beta={:.2f}".format(
    F_G1, F_R1, F_R1/F_G1))

print("At  10 AU: Gravity {:.2e}, Rad.force {:.2e}, beta={:.2f}".format(
    F_G10, F_R10, F_R10/F_G10))


